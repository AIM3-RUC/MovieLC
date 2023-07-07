import logging
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.autograd import Variable

from modules.util_module import PreTrainedModel, LayerNorm, CrossEn, MILNCELoss, MaxMarginRankingLoss, ACT2FN
from modules.module_kg import KGModel, KGConfig
from modules.module_danmaku import DanmakuModel, DanmakuConfig
from modules.module_visual import VisualModel, VisualConfig
from modules.module_cross import CrossModel, CrossConfig, CrossOnlyMLMHead
from modules.module_decoder import DecoderModel, DecoderConfig
from modules.module_bert import BertModel, BertConfig
import copy
import random

logger = logging.getLogger(__name__)

class KLVCGPretrainedModel(PreTrainedModel, nn.Module):
    def __init__(self, kg_config, danmaku_config, visual_config, cross_config, decoder_config, bert_config, *inputs, **kwargs):
        super(KLVCGPretrainedModel, self).__init__(cross_config)
        self.kg_config = kg_config
        self.danmamku_config = danmaku_config
        self.visual_config = visual_config
        self.cross_config = cross_config
        self.decoder_config = decoder_config
        self.bert_config = bert_config
    
    @classmethod    
    def from_pretrained(cls, state_dict=None, cache_dir=None, type_vocab_size=5, *inputs, **kwargs):
        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0
        kg_config, _ = KGConfig.get_config('kg-base', cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        bert_config, _ = BertConfig.get_config('bert-base', cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        danmaku_config, _ = DanmakuConfig.get_config('danmaku-base', cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        visual_config, _ = VisualConfig.get_config('visual-base', cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        cross_config, _ = CrossConfig.get_config('cross-base', cache_dir, 4, state_dict=None, task_config=task_config)
        decoder_config, _ = DecoderConfig.get_config('decoder-base', cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        model = cls(kg_config, danmaku_config, visual_config, cross_config, decoder_config, bert_config, *inputs, **kwargs)

        if state_dict is not None: # load parameters
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model


class NormalizeVideo(nn.Module):
    def __init__(self, video_dim):
        super(NormalizeVideo, self).__init__()
        self.visual_norm2d = LayerNorm(video_dim)

    def forward(self, video):
        video = self.visual_norm2d(video)
        return video


class KLVCG(KLVCGPretrainedModel):
    def __init__(self, kg_config, danmaku_config, visual_config, cross_config, decoder_config, bert_config, task_config):
        super(KLVCG, self).__init__(kg_config, danmaku_config, visual_config, cross_config, decoder_config, bert_config)
        self.task_config = task_config
        self.vocab_size = danmaku_config.vocab_size
        # Text Embedding 
        # input [batch_size, max_comment_context_len]
        # output [batch_size, max_comment_context_len, hidden_size]
        self.bert_encoder = BertModel(bert_config)
        bert_word_embeddings_weight = self.bert_encoder.embeddings.word_embeddings.weight
        bert_position_embeddings_weight = self.bert_encoder.embeddings.position_embeddings.weight
        # the danmaku encoder and visual encoder share the same position embeddings
        self.position_embeddings = nn.Embedding(visual_config.max_position_embeddings, visual_config.hidden_size)
        self.danmaku_encoder = DanmakuModel(danmaku_config, self.position_embeddings)
        self.kg_encoder = KGModel(kg_config)
        # Visual Embedding
        self.normalize_video = NormalizeVideo(task_config.video_dim)
        self.visual_linear = nn.Linear(task_config.video_dim, kg_config.hidden_size)
        self.visual_encoder = VisualModel(visual_config, self.position_embeddings)
        # Cross Encoder
        self.cross_encoder = CrossModel(cross_config)
        # decoder
        self.decoder = DecoderModel(decoder_config, bert_word_embeddings_weight, bert_position_embeddings_weight)

        if self.task_config.do_pretrain:
            self.mlm_loss = CrossEntropyLoss(ignore_index=-1)
        else:
            self.decoder_loss_fct = CrossEntropyLoss(ignore_index=0) 
            self.ranking_loss = nn.CrossEntropyLoss(reduce=False, size_average=False, ignore_index=0) 


    def forward(self, I, K_token_ids, K_pos_ids, K_attn_mask, D_token_ids, D_pos_ids, D_attn_mask, Y, Y_attn_mask, loc_time, Y_token_label=None):
        I_emb = self.visual_linear(self.normalize_video(I)) 
        visual_output, _ = self.visual_encoder(I_emb, output_all_encoded_layers=False)
        _, time_cls =  self.bert_encoder(D_token_ids, pos_ids=D_pos_ids, attention_mask=D_attn_mask, output_all_encoded_layers=False)
        danmaku_output, _ = self.danmaku_encoder(time_cls, output_all_encoded_layers=False)
        kg_output, _ = self.kg_encoder(K_token_ids, pos_ids=K_pos_ids, attention_mask=K_attn_mask, q_input=time_cls, output_all_encoded_layers=False)
        cross_emb = torch.cat((visual_output, danmaku_output, kg_output), dim=1) 
        # prepare token type embedding
        batch_size = cross_emb.size(0)
        visual_type = torch.ones((batch_size, self.task_config.time_range * 2 + 1), dtype=torch.long).to(cross_emb.device) # type code for visual token is 1
        danmaku_type = 2 * torch.ones((batch_size, self.task_config.time_range * 2), dtype=torch.long).to(cross_emb.device) # type code for danmaku token is 2
        kg_type = 3 * torch.ones((batch_size, self.task_config.time_range * 2), dtype=torch.long).to(cross_emb.device) # type code for knowledge token is 3
        
        cls_token = torch.mean(cross_emb, dim=1).unsqueeze(1) # cls token
        cls_type = torch.zeros(cls_token.size()[0:2], dtype=torch.long).to(cross_emb.device) # cls type token
        
        cross_emb = torch.cat((cls_token, cross_emb), dim=1)
        cross_type = torch.cat((cls_type, visual_type, danmaku_type, kg_type), dim=1) 
        
        # attention mask for cross encoder
        sl = cross_emb.size(1)
        attention_mask = torch.ones((batch_size, sl, sl), dtype=torch.long).to(cross_emb.device)
        # cross_output
        loc_time = loc_time.repeat(1, 6 * self.task_config.time_range + 2) # add the period information
        cross_output, _ = self.cross_encoder(concat_input=cross_emb, concat_type=cross_type, attention_mask=attention_mask, output_all_encoded_layers=False, loc_type=loc_time) 
        
        if self.task_config.do_pretrain:
            loss = self.mlm_task(cross_output, attention_mask, Y, Y_attn_mask, Y_token_label)
        else:
            decoder_output_Y = Y[:, 1:]
            scores = self.decoder(input_ids=Y[:,:-1], encoder_outs=cross_output, answer_mask=Y_attn_mask[:,:-1,:-1], encoder_mask=attention_mask[:,:self.task_config.max_output_comment_len-1,:])
            loss = self.decoder_loss_fct(scores.view(-1, self.vocab_size), decoder_output_Y.contiguous().view(-1))  

        return loss

    def mlm_task(self, cross_output, attention_mask, Y, Y_attn_mask, Y_token_label):
        scores = self.decoder(input_ids=Y[:,:-1], encoder_outs=cross_output, answer_mask=Y_attn_mask[:,:-1,:-1], encoder_mask=attention_mask[:,:self.task_config.max_output_comment_len-1,:])
        loss = self.mlm_loss(scores.view(-1, self.vocab_size), Y_token_label[:, :-1].contiguous().view(-1))
        return loss 
    
    def ranking(self, I, K_token_ids, K_pos_ids, K_attn_mask, D_token_ids, D_pos_ids, D_attn_mask, Y, Y_attn_mask, loc_time): 
        # testing by ranking the loss of candidate comments, use teacher forcing strategy
        num_candidates = len(Y)
        I = I.unsqueeze(0) 
        K_token_ids = K_token_ids.unsqueeze(0)
        K_pos_ids = K_pos_ids.unsqueeze(0) 
        K_attn_mask = K_attn_mask.unsqueeze(0) 
        D_token_ids = D_token_ids.unsqueeze(0) 
        D_pos_ids = D_pos_ids.unsqueeze(0) 
        D_attn_mask = D_attn_mask.unsqueeze(0)
        loc_time = loc_time.unsqueeze(0)

        I_emb = self.visual_linear(self.normalize_video(I)) 
        visual_output, _ = self.visual_encoder(I_emb, output_all_encoded_layers=False)
        _, time_cls =  self.bert_encoder(D_token_ids, pos_ids=D_pos_ids, attention_mask=D_attn_mask, output_all_encoded_layers=False)
        danmaku_output, _ = self.danmaku_encoder(time_cls, output_all_encoded_layers=False)
        kg_output, _ = self.kg_encoder(K_token_ids, pos_ids=K_pos_ids, attention_mask=K_attn_mask, q_input=time_cls, output_all_encoded_layers=False)
        # prepare token type embedding
        cross_emb = torch.cat((visual_output, danmaku_output, kg_output), dim=1) 
        batch_size = cross_emb.size(0)
        visual_type = torch.ones((batch_size, self.task_config.time_range * 2 + 1), dtype=torch.long).to(cross_emb.device) # type code for visual token is 1
        danmaku_type = 2 * torch.ones((batch_size, self.task_config.time_range * 2), dtype=torch.long).to(cross_emb.device) # type code for danmaku token is 2
        kg_type = 3 * torch.ones((batch_size, self.task_config.time_range * 2), dtype=torch.long).to(cross_emb.device) # type code for knowledge token is 3
        
        cls_token = torch.mean(cross_emb, dim=1).unsqueeze(1) # [batch_size, 1, 768]
        cls_type = torch.zeros(cls_token.size()[0:2], dtype=torch.long).to(cross_emb.device) # type为0
        
        cross_emb = torch.cat((cls_token, cross_emb), dim=1) # [batch_size, 32, 768]
        cross_type = torch.cat((cls_type, visual_type, danmaku_type, kg_type), dim=1) # [batch_size, 32, 768]
        # attention mask for cross encoder
        sl = cross_emb.size(1)
        attention_mask = torch.ones((batch_size, sl, sl), dtype=torch.long).to(cross_emb.device)
        # cross_output
        loc_time = loc_time.repeat(1, 6 * self.task_config.time_range + 2) # add the period information
        cross_output, _ = self.cross_encoder(concat_input=cross_emb, concat_type=cross_type, attention_mask=attention_mask, output_all_encoded_layers=False, loc_type=loc_time) 
        
        cross_output = cross_output.repeat(num_candidates, 1, 1)
        attention_mask = attention_mask.repeat(num_candidates, 1, 1)

        scores = self.decoder(input_ids=Y[:,:-1], encoder_outs=cross_output, answer_mask=Y_attn_mask[:,:-1,:-1], encoder_mask=attention_mask[:,:self.task_config.max_output_comment_len - 1,:])
        scores = scores.transpose(0,1)
        
        # count the number of words for each candidate comment
        Y_valid_num_list = []
        for yi in Y:
            l = 0
            for yii in yi:
                if yii == 0:
                    break
                l += 1
            Y_valid_num_list.append(l)
        Y_valid_num = torch.LongTensor(Y_valid_num_list).to(Y.device)

        decoder_output_Y = Y[:, 1:].t()
        
        loss = self.ranking_loss(scores.contiguous().view(-1, self.vocab_size), decoder_output_Y.contiguous().view(-1))
        loss = loss.view(-1, num_candidates).sum(0) 
        loss = loss / Y_valid_num # average the loss of each token in one comment
        
        return torch.sort(loss, dim=0, descending=False)[1] # ascending, the smaller to loss, the closer to groundtruth

    def subsequent_mask(self, batch, size):
        # make triangular matrix for left-to-right generation mask
        attn_shape = (batch, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0   

    def generation(self, I, K_token_ids, K_pos_ids, K_attn_mask, D_token_ids, D_pos_ids, D_attn_mask, BOS_token, EOS_token, beam_size, loc_time):
        # visualize the generated comments on test set
        I = I.unsqueeze(0) 
        K_token_ids = K_token_ids.unsqueeze(0) 
        K_pos_ids = K_pos_ids.unsqueeze(0) 
        K_attn_mask = K_attn_mask.unsqueeze(0) 
        D_token_ids = D_token_ids.unsqueeze(0) 
        D_pos_ids = D_pos_ids.unsqueeze(0)
        D_attn_mask = D_attn_mask.unsqueeze(0) 
        loc_time = loc_time.unsqueeze(0)
        
        I_emb = self.visual_linear(self.normalize_video(I)) 
        visual_output, _ = self.visual_encoder(I_emb, output_all_encoded_layers=False)

        _, time_cls =  self.bert_encoder(D_token_ids, pos_ids=D_pos_ids, attention_mask=D_attn_mask, output_all_encoded_layers=False)
        danmaku_output, _ = self.danmaku_encoder(time_cls, output_all_encoded_layers=False)

        kg_output, _ = self.kg_encoder(K_token_ids, pos_ids=K_pos_ids, attention_mask=K_attn_mask, q_input=time_cls, output_all_encoded_layers=False)
        
        cross_emb = torch.cat((visual_output, danmaku_output, kg_output), dim=1) 
        batch_size = cross_emb.size(0)
        visual_type = torch.ones((batch_size, self.task_config.time_range * 2 + 1), dtype=torch.long).to(cross_emb.device) 
        danmaku_type = 2 * torch.ones((batch_size, self.task_config.time_range * 2), dtype=torch.long).to(cross_emb.device) 
        kg_type = 3 * torch.ones((batch_size, self.task_config.time_range * 2), dtype=torch.long).to(cross_emb.device) 
        
        cls_token = torch.mean(cross_emb, dim=1).unsqueeze(1) 
        cls_type = torch.zeros(cls_token.size()[0:2], dtype=torch.long).to(cross_emb.device) 
        
        cross_emb = torch.cat((cls_token, cross_emb), dim=1) 
        cross_type = torch.cat((cls_type, visual_type, danmaku_type, kg_type), dim=1) 

        sl = cross_emb.size(1)
        attention_mask = torch.ones((batch_size, sl, sl), dtype=torch.long).to(cross_emb.device)

        loc_time = loc_time.repeat(1, 6 * self.task_config.time_range + 2)
        cross_output, _ = self.cross_encoder(concat_input=cross_emb, concat_type=cross_type, attention_mask=attention_mask, output_all_encoded_layers=False, loc_type=loc_time) 
        comments = self.beam_search(cross_output, BOS_token, EOS_token, beam_size, attention_mask)
        return comments
    
    def beam_search(self, cross_output, BOS_token, EOS_token, beam_size, attention_mask):
        LENGTH_NORM = True
        batch_size = cross_output.size(0)
        startTokenArray = Variable(torch.LongTensor(batch_size, 1).fill_(BOS_token)).to(cross_output.device)
        #print('Start matrix ', startTokenArray.size()) # [1,1]

        backVector = torch.LongTensor(beam_size).to(cross_output.device) # [5]
        torch.arange(0, beam_size, out=backVector) # backvector [0,1,2,3,4]
        backVector = backVector.unsqueeze(0).repeat(batch_size, 1) # [[0,1,2,3,4]]
        backVector = Variable(backVector)
        #print('Back matrix ', backVector.size())

        tokenArange = torch.LongTensor(self.vocab_size).to(cross_output.device)
        torch.arange(0, self.vocab_size, out=tokenArange)
        tokenArange = Variable(tokenArange) # [0,1,2,3,……, 30005]
        #print('Token matrix ', tokenArange.size())

        # [1,5,19]
        beamTokensTable = torch.LongTensor(batch_size, beam_size, self.task_config.max_output_comment_len).fill_(EOS_token)
        beamTokensTable = Variable(beamTokensTable.to(cross_output.device))
        #print('beam Table ', beamTokensTable.size())

        backIndices = torch.LongTensor(batch_size, beam_size, self.task_config.max_output_comment_len).fill_(-1)
        backIndices = Variable(backIndices.to(cross_output.device))
        #print('Back Indices ', backIndices.size())

        aliveVector = beamTokensTable[:, :, 0].eq(EOS_token).unsqueeze(2)
        #print('AliveVector ', aliveVector.size())

        for i in range(self.task_config.max_output_comment_len - 1):
            if i  == 0:
                Cap = startTokenArray
                mask = Variable(self.subsequent_mask(Cap.size(0), Cap.size(1))).to(cross_output.device)
                # out = self.decode(Cap, cross_output, mask)
                out = self.decoder(input_ids=Cap, encoder_outs=cross_output, answer_mask=mask, encoder_mask=attention_mask[:,:Cap.size(1),:])
                #print('Out ', out.size())
                probs = out[:, -1]
                topProbs, topIdx = probs.topk(beam_size, dim=1)
                beamTokensTable[:, :, 0] = topIdx.data
                ProbSums = topProbs
            else:
                Cap = beamTokensTable[:, :, :i].squeeze(0)
                mask = Variable(self.subsequent_mask(Cap.size(0), Cap.size(1))).to(cross_output.device)
                # out = self.decode(Cap, cross_output, mask)
                out = self.decoder(input_ids=Cap, encoder_outs=cross_output, answer_mask=mask, encoder_mask=attention_mask[:,:Cap.size(1),:])
                probCurrent = out[:, -1,:].view(batch_size, beam_size, self.vocab_size)

                if LENGTH_NORM:
                    probs = probCurrent * (aliveVector.float() / (i+1))
                    coeff_ = aliveVector.eq(0).float() + (aliveVector.float() * i / (i+1))
                    probs += ProbSums.unsqueeze(2) * coeff_
                else:
                    probs = probCurrent * (aliveVector.float())
                    probs += ProbSums.unsqueeze(2)

                mask_ = aliveVector.eq(0).repeat(1, 1, self.vocab_size)
                mask_[:, :, 0] = 0
                minus_infinity_ = torch.min(probs).item()

                probs.data.masked_fill_(mask_.data, minus_infinity_)
                probs = probs.view(batch_size, -1)

                tokensArray = tokenArange.unsqueeze(0).unsqueeze(0).repeat(batch_size, beam_size, 1)
                tokensArray.masked_fill_(aliveVector.eq(0), 2)
                tokensArray = tokensArray.view(batch_size, -1)
                backIndexArray = backVector.unsqueeze(2).repeat(1, 1, self.vocab_size).view(batch_size, -1)

                topProbs, topIdx = probs.topk(beam_size, dim=1)
                ProbSums = topProbs
                beamTokensTable[:, :, i] = tokensArray.gather(1, topIdx)
                backIndices[:, :, i] = backIndexArray.gather(1, topIdx)

            aliveVector = beamTokensTable[:, :, i:i + 1].ne(2)
            aliveBeams = aliveVector.data.long().sum()
            finalLen = i
            if aliveBeams == 0:
                break

        beamTokensTable = beamTokensTable.data
        backIndices = backIndices.data

        RECOVER_TOP_BEAM_ONLY = True
        tokenIdx = finalLen
        backID = backIndices[:, :, tokenIdx]
        tokens = []
        while tokenIdx >= 0:
            tokens.append(beamTokensTable[:, :, tokenIdx].gather(1, backID).unsqueeze(2))
            backID = backIndices[:, :, tokenIdx].gather(1, backID)
            tokenIdx = tokenIdx - 1

        tokens.append(startTokenArray.unsqueeze(2).repeat(1, beam_size, 1).data)
        tokens.reverse()
        tokens = torch.cat(tokens, 2)
        seqLen = tokens.ne(2).long().sum(dim=2)

        if RECOVER_TOP_BEAM_ONLY:
            tokens = tokens[:, 0]
            seqLen = seqLen[:, 0]
            
        return Variable(tokens) 



