import os
import argparse
import time
import numpy as np
import random
import json
import torch
from torch.utils.data import (SequentialSampler)
import metrics
from datasets.datasets import Dataset, PretrainDataset
from torch.utils.data import DataLoader
from modules.modeling import KLVCG
from modules.optimization import BertAdam
import utils

torch.distributed.init_process_group(backend="nccl")
global logger

def get_args(description='comment generation'):
    parser = argparse.ArgumentParser(description=description)
    # distributed training
    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training") 
    parser.add_argument('--n_gpu', type=int, default=4, help="Number of GPUs, Changed in the execute process.")
    parser.add_argument('--num_thread_reader', type=int, default=2, help='distribted training')
    # training control
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run pre-training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training/fine-tunning")
    parser.add_argument("--do_test", action='store_true', help="Whether to run eval on test set.")
    parser.add_argument("--do_gen", action='store_true', help="Whether to run generation on test set.")
    parser.add_argument('--n_display', type=int, default=100, help='Display the train loss at each n_display steps') # 每几个step输出一次 train loss
    parser.add_argument('--n_report', type=int, default=1, help='Do evaluation and report eval loss at each n_report steps') 
    # input and output path
    parser.add_argument('--data_path', type=str, default='/data7/cjt/danmaku/Github_MovieLC_data/processed_MovieLC',
                        help='the input data path')
    parser.add_argument('--setup_path', type=str, default='setup', help='the input setup path. e.g., "setup", "setup_knowledge_enhanced"')
    parser.add_argument('--video_dim', type=int, default=2048, help='video feature dimension')
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--output_dir", default='/data7/cjt/danmaku/Github_MovieLC_data/output', type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--runid', type=str, default='temp', help='a unique id for running')
    # training parameters
    parser.add_argument('--time_range', type=int, default=5, help='left and right time range, should be <= 5') 
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_comment_context_len', type=int, default=100, help='the maximum number of words of the input comment context')
    parser.add_argument('--max_kg_len', type=int, default=80, help='the maximum number of words of the input knowledge facts')
    parser.add_argument('--max_output_comment_len', type=int, default=20, help='the maximum number of words of the output comment') 
    parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size in concept, the true batch size should be divided by gradient_accumulation_steps')
    parser.add_argument('--batch_size_val', type=int, default=64, help='batch size eval')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--beam_size', type=int, default=5, help="beam size for beam search") 
    # others 
    parser.add_argument("--cache_dir", default="./modules", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3/ the config file path")
    args = parser.parse_args()

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_test and not args.do_pretrain and not args.do_gen:
        raise ValueError("At least one of `do_train` or `do_test` or `do_gen` or `do_pretrain` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps) # get the true batch size

    args.output_dir = os.path.join(args.output_dir, args.runid)
    
    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank) # set default gpu
    args.world_size = world_size

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = utils.get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = KLVCG.from_pretrained(cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device)

    return model

def get_params_num(model):
    # Find all parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, total_trainable_params

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):
    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)] # no_decay[]以外的参数
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)] # no_decay[]以内的参数
    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_decay_param_tp], 'weight_decay': 0.01},
        {'params': [p for n, p in decay_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, 
                         t_total=num_train_optimization_steps, weight_decay=0.01,
                         max_grad_norm=1.0)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model


def save_model(args, model, type_name=""):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}".format("" if type_name=="" else type_name))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = KLVCG.from_pretrained(cache_dir=cache_dir, type_vocab_size=args.max_danmakus + 1, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def get_pretrain_data(args, datas, split_type, vocabs, rev_vocabs, img_ft, vid_duration):
    if split_type == 'train':
        dm_dataset = PretrainDataset(
            datas=datas,
            vocabs=vocabs,
            rev_vocabs=rev_vocabs,
            img_ft=img_ft,
            max_kg_len=args.max_kg_len,
            max_comment_context_len=args.max_comment_context_len,
            max_output_comment_len=args.max_output_comment_len,
            split_type=split_type,
            time_range=args.time_range,
            vid_duration=vid_duration,
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(dm_dataset)

        dataloader = DataLoader(
            dm_dataset,
            batch_size=args.batch_size // args.n_gpu,
            num_workers=args.num_thread_reader,
            pin_memory=False,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=True,
        ) 
        return dataloader, len(dm_dataset), train_sampler

    if split_type == 'dev':
        dm_dataset = PretrainDataset(
            datas=datas,
            vocabs=vocabs,
            rev_vocabs=rev_vocabs,
            img_ft=img_ft,
            max_kg_len=args.max_kg_len,
            max_comment_context_len=args.max_comment_context_len,
            max_output_comment_len=args.max_output_comment_len,
            split_type=split_type,
            time_range=args.time_range,
            vid_duration=vid_duration,
        )

        test_sampler = SequentialSampler(dm_dataset)

        dataloader = DataLoader(
            dm_dataset,
            sampler=test_sampler,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )
        return dataloader, len(dm_dataset)    

def get_data(args, datas, split_type, vocabs, rev_vocabs, img_ft, vid_duration):
    if split_type == 'train':
        dm_dataset = Dataset(
            datas=datas,
            vocabs=vocabs,
            rev_vocabs=rev_vocabs,
            img_ft=img_ft,
            max_kg_len=args.max_kg_len,
            max_comment_context_len=args.max_comment_context_len,
            max_output_comment_len=args.max_output_comment_len,
            split_type=split_type,
            time_range=args.time_range,
            vid_duration=vid_duration,
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(dm_dataset)

        dataloader = DataLoader(
            dm_dataset,
            batch_size=args.batch_size // args.n_gpu,
            num_workers=args.num_thread_reader,
            pin_memory=False,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=True,
        ) 
        return dataloader, len(dm_dataset), train_sampler

    if split_type == 'dev':
        dm_dataset = Dataset(
            datas=datas,
            vocabs=vocabs,
            rev_vocabs=rev_vocabs,
            img_ft=img_ft,
            max_kg_len=args.max_kg_len,
            max_comment_context_len=args.max_comment_context_len,
            max_output_comment_len=args.max_output_comment_len,
            split_type=split_type,
            time_range=args.time_range,
            vid_duration=vid_duration,
        )

        test_sampler = SequentialSampler(dm_dataset)
        
        dataloader = DataLoader(
            dm_dataset,
            sampler=test_sampler,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )
        return dataloader, len(dm_dataset)   
    
    if split_type == 'test':
        dm_dataset = Dataset(
            datas=datas,
            vocabs=vocabs,
            rev_vocabs=rev_vocabs,
            img_ft=img_ft,
            max_kg_len=args.max_kg_len,
            max_comment_context_len=args.max_comment_context_len,
            max_output_comment_len=args.max_output_comment_len,
            split_type=split_type,
            time_range=args.time_range,
            vid_duration=vid_duration,
        )
        return dm_dataset 


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler,
                global_step, nlgEvalObj=None, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display # report a log at each log_step
    start_time = time.time()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        if args.do_pretrain:
            I, K_token_ids, K_pos_ids, K_attn_mask, D_token_ids, D_pos_ids, D_attn_mask, Y, Y_attn_mask, Y_token_label, loc_time = batch
            loss = model(I, K_token_ids, K_pos_ids, K_attn_mask, D_token_ids, D_pos_ids, D_attn_mask, Y, Y_attn_mask, loc_time, Y_token_label)
        else:
            I, K_token_ids, K_pos_ids, K_attn_mask, D_token_ids, D_pos_ids, D_attn_mask, Y, Y_attn_mask, loc_time  = batch
            loss = model(I, K_token_ids, K_pos_ids, K_attn_mask, D_token_ids, D_pos_ids, D_attn_mask, Y, Y_attn_mask, loc_time)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward() # calculate the gradient

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # clip the gradient to avoid gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule
            optimizer.step() # update the parameters
            optimizer.zero_grad() # clear the gradient
            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()
    total_loss = total_loss / len(train_dataloader) 
    return total_loss, global_step


def eval(args, model, dev_dataloader, device, n_gpu):
    # do evaluation during training and get the evaluation loss
    if hasattr(model, 'module'):
        model = model.module.to(device)
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for step, batch in enumerate(dev_dataloader):
            batch = tuple(t.to(device, non_blocking=True) for t in batch)
            if args.do_pretrain:
                I, K_token_ids, K_pos_ids, K_attn_mask, D_token_ids, D_pos_ids, D_attn_mask, Y, Y_attn_mask, Y_token_label, loc_time = batch
                loss = model(I, K_token_ids, K_pos_ids, K_attn_mask, D_token_ids, D_pos_ids, D_attn_mask, Y, Y_attn_mask, loc_time, Y_token_label)
            else:
                I, K_token_ids, K_pos_ids, K_attn_mask, D_token_ids, D_pos_ids, D_attn_mask, Y, Y_attn_mask, loc_time  = batch
                loss = model(I, K_token_ids, K_pos_ids, K_attn_mask, D_token_ids, D_pos_ids, D_attn_mask, Y, Y_attn_mask, loc_time)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            total_loss += float(loss)
    total_loss = total_loss / len(dev_dataloader) 
    return total_loss

def test_ranking(args, model, test_set, device, n_gpu):
    # evaluate the result on test set and get the ranking metrics like Recall@k, MR, MRR
    if hasattr(model, 'module'):
        model = model.module.to(device)
    model.eval()
    predictions, references = [], [] # references means groundtruth
    start_time = time.time()
    with torch.no_grad():
        for i in range(len(test_set)):
            I, K_token_ids, K_pos_ids, K_attn_mask, D_token_ids, D_pos_ids, D_attn_mask, Y, Y_attn_mask, loc_time, data = test_set.get_test_data(i)
            I = I.to(device=device, non_blocking=True) 
            K_token_ids = K_token_ids.to(device=device, non_blocking=True) 
            K_pos_ids = K_pos_ids.to(device=device, non_blocking=True)
            K_attn_mask = K_attn_mask.to(device=device, non_blocking=True)
            D_token_ids = D_token_ids.to(device=device, non_blocking=True) 
            D_pos_ids = D_pos_ids.to(device=device, non_blocking=True)
            D_attn_mask = D_attn_mask.to(device=device, non_blocking=True)
            Y = Y.to(device=device, non_blocking=True) # [num_candidates, max_output_comment_len]
            Y_attn_mask = Y_attn_mask.to(device=device, non_blocking=True)
            loc_time = loc_time.to(device=device, non_blocking=True)
            ids = model.ranking(I, K_token_ids, K_pos_ids, K_attn_mask, D_token_ids, D_pos_ids, D_attn_mask, Y, Y_attn_mask, loc_time).data

            candidate = []
            comments = list(data['candidate'].keys())
            for id in ids:
                candidate.append(comments[id]) 
            predictions.append(candidate)
            references.append(data['candidate'])

            if i % 1000 == 0:
                print(i)
    
    recall_1 = metrics.recall(predictions, references, 1)
    recall_5 = metrics.recall(predictions, references, 5)
    recall_10 = metrics.recall(predictions, references, 10)
    mr = metrics.mean_rank(predictions, references)
    mrr = metrics.mean_reciprocal_rank(predictions, references)
    print(recall_1, recall_5, recall_10, mr, mrr)
    print("testing time:", time.time() - start_time) 
    f = open(os.path.join(args.output_dir, 'log_test.txt'), 'a')
    f.write(str(recall_1) + ' ' + str(recall_5) + ' ' + str(recall_10) + ' ' + str(mr) + ' ' + str(mrr) +'\n')
    f.close()   

def transform(ids, vocabs, rev_vocabs):
    # transform the output comment word ids into words
    sentences = []
    for wid in ids:
        if wid == vocabs['<BOS>']:
            continue
        if wid == vocabs['<EOS>']:
            break
        sentences.append(rev_vocabs[str(wid)])
    return ''.join(sentences)

def test_generation(args, model, test_set, device, n_gpu, vocabs, rev_vocabs):
    # visualization the generated comments on test set
    if hasattr(model, 'module'):
        model = model.module.to(device)
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        with open(os.path.join(args.output_dir, "out.txt"), 'w', encoding='utf-8') as fout:
            for i in range(len(test_set)):
                if i % 1000 == 0:
                    print(i) 
                I, K_token_ids, K_pos_ids, K_attn_mask, D_token_ids, D_pos_ids, D_attn_mask, Y, Y_attn_mask, loc_time, data = test_set.get_test_data(i)
                I = I.to(device=device, non_blocking=True) 
                K_token_ids = K_token_ids.to(device=device, non_blocking=True) 
                K_pos_ids = K_pos_ids.to(device=device, non_blocking=True)
                K_attn_mask = K_attn_mask.to(device=device, non_blocking=True)
                D_token_ids = D_token_ids.to(device=device, non_blocking=True) 
                D_pos_ids = D_pos_ids.to(device=device, non_blocking=True)
                D_attn_mask = D_attn_mask.to(device=device, non_blocking=True)
                loc_time = loc_time.to(device=device, non_blocking=True)
                comment_ids = model.generation(I, K_token_ids, K_pos_ids, K_attn_mask, D_token_ids, D_pos_ids, D_attn_mask, vocabs['<BOS>'], vocabs['<EOS>'], args.beam_size, loc_time).data.tolist()
                comment = transform(comment_ids[0], vocabs, rev_vocabs)
                sample = {'video': data['video'],
                          'time': data['time'],
                          'generation': comment}
                term = json.dumps(sample, ensure_ascii=False)
                fout.write(str(term)+'\n')
        fout.close()


def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    start_time = time.time()
    # load input data
    setup = args.setup_path
    train_path, test_path, dev_path = utils.set_data_path(os.path.join(args.data_path, setup))
    # load vocab
    vocabs, rev_vocabs = utils.load_dict(os.path.join(args.data_path, 'dict.json')) 
    vocab_len = len(vocabs)
    # load video duration for period awareness
    vid_duration = json.load(open(os.path.join(args.data_path, 'vid_duration.json'), 'r')) 
    # load img feature pkl
    img_path = utils.set_img_path(args.data_path)
    img_ft = torch.load(open(img_path, 'rb'))

    if args.local_rank == 0:
        logger.info('**********Load setup: %s', setup)
        logger.info('**********Load vocabs: %d', vocab_len)
        logger.info('**************Finish loading img feature')
        logger.info('**************load data time cost: %d', int(time.time() - start_time))

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    model = init_model(args, device, n_gpu, args.local_rank)

    if args.do_pretrain or args.do_train:
        # load dev data
        if args.do_pretrain:
            dev_datas = utils.load_from_json(dev_path)
            dev_dataloader, dev_length = get_pretrain_data(args, dev_datas, 'dev', vocabs, rev_vocabs, img_ft, vid_duration)
            logger.info("finish load from json: dev")
        else:
            dev_datas = utils.load_from_json(dev_path)
            dev_dataloader, dev_length = get_data(args, dev_datas, 'dev', vocabs, rev_vocabs, img_ft, vid_duration)
            logger.info("finish load from json: dev")
        if args.local_rank == 0:
            logger.info("***** Dev set *****")
            logger.info("  Num examples = %d", dev_length)
            logger.info("  Batch size = %d", args.batch_size_val)
            logger.info("  Num steps = %d", len(dev_dataloader))
        # load train data
        if args.do_pretrain:
            train_datas = utils.load_from_json(train_path)
            train_dataloader, train_length, train_sampler = get_pretrain_data(args, train_datas, 'train', vocabs, rev_vocabs, img_ft, vid_duration)            
            logger.info("finish load from json: train")
        else:
            train_datas = utils.load_from_json(train_path)
            train_dataloader, train_length, train_sampler = get_data(args, train_datas, 'train', vocabs, rev_vocabs, img_ft, vid_duration)            
            logger.info("finish load from json: train")            
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs
        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=args.coef_lr)

        best_loss = 100000
        best_output_model_file = None
        global_step = 0

        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch) 

            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, local_rank=args.local_rank)
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                output_model_file = save_model(args, model, type_name=str(epoch+1))
                if (epoch + 1) % args.n_report == 0:
                    eval_loss = eval(args, model, dev_dataloader, device, n_gpu)
                    if best_loss >= eval_loss:
                        best_loss = eval_loss
                        best_output_model_file = save_model(args, model, type_name="BEST")
                    logger.info("Evaluation ************ The loss is: {:.4f}".format(best_loss))
                else:
                    logger.warning("Skip the evaluation after {}-th epoch.".format(epoch+1))       
        
        if args.local_rank == 0:
            model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
            eval_epoch(args, model, dev_dataloader, device, n_gpu)
    
    elif args.do_test:
        if args.local_rank == 0:
            # # get params num
            # total_num, total_train_num = get_params_num(model)
            # print('total_num: ', total_num)
            # print('total_train_num: ', total_train_num)
            # exit()
            test_datas = utils.load_from_json(test_path)
            logger.info("finish load from json: test")
            test_data = get_data(args, test_datas, 'test', vocabs, rev_vocabs, img_ft, vid_duration)
            logger.info("***** Running testing ranking *****")
            logger.info("  Num examples = %d", len(test_data))
            logger.info("  Batch size = 1")
        test_ranking(args, model, test_data, device, n_gpu)
    elif args.do_gen:
        if args.local_rank == 0:
            test_datas = utils.load_from_json(test_path)
            logger.info("finish load from json: test")
            test_data = get_data(args, test_datas, 'test', vocabs, rev_vocabs, img_ft, vid_duration)
            logger.info("***** Running testing genration *****")
            logger.info("  Num examples = %d", len(test_data))
            logger.info("  Batch size = 1")
        test_generation(args, model, test_data, device, n_gpu, vocabs, rev_vocabs)
    else:
        print('Stage type error')

    
if __name__ == "__main__":
    main()