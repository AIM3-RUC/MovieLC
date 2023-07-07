import os
import time
import json

import torch
import numpy as np
import utils
import random
import copy

class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, datas, vocabs, rev_vocabs, img_ft, max_kg_len, max_comment_context_len, max_output_comment_len, split_type, time_range, vid_duration):
        self.datas = datas
        self.vocabs = vocabs
        self.vocab_len = len(vocabs)
        self.rev_vocabs = rev_vocabs
        self.img_ft = img_ft
        self.max_kg_len = max_kg_len
        self.max_comment_context_len = max_comment_context_len
        self.max_output_comment_len = max_output_comment_len
        self.split_type = split_type
        self.time_range = time_range
        self.vid_duration = vid_duration
        if 'kg' in self.datas[0]:
            self.kg_fact = 1
        else:
            self.kg_fact = 0

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        video_id, video_time = data['video'], data['time']
        I = self.load_imgs(video_id, video_time)
        if self.kg_fact:
            K_token_ids, K_pos_ids, K_attn_mask = self.load_kg(data['kg'])
        else:
            K_token_ids, K_pos_ids, K_attn_mask = self.load_kg('')
        D_token_ids, D_pos_ids, D_attn_mask = self.load_danmakus(data['context'])
        Y, Y_attn_mask, Y_token_label = self.load_target(data['comment'])
        # for video period 
        loc_pct = float(video_time) / float(self.vid_duration[video_id])
        if loc_pct <= 0.2:
            loc = 0
        elif loc_pct <= 0.4:
            loc = 1
        elif loc_pct <= 0.6:
            loc = 2
        elif loc_pct <= 0.8:
            loc = 3
        else: 
            loc = 4
        loc_time = torch.Tensor([loc]).long()
        return I, K_token_ids, K_pos_ids, K_attn_mask, D_token_ids, D_pos_ids, D_attn_mask, Y, Y_attn_mask, Y_token_label, loc_time

    def load_target(self, comment):
        comment = comment.split(' ')
        if len(comment) > self.max_output_comment_len - 2:
            comment = comment[: self.max_output_comment_len - 2]
        Y_comment = list(map(lambda t: self.vocabs.get(t, 3), comment))
        length = len(Y_comment)
        # mask
        Y_token_label = []
        for i in range(length):
            prob = random.random()
            if prob < 0.30:
                Y_token_label.append(Y_comment[i])
                prob /= 0.30
                if prob < 0.8:
                    Y_comment[i] = 4
                elif prob < 0.9:
                    Y_comment[i] = random.randint(0, self.vocab_len - 1)
            else:
                Y_token_label.append(-1)

        Y_comment = torch.LongTensor([1] + Y_comment + [2] + [0] * (self.max_output_comment_len - 2 - length))
        Y_attn_mask = torch.LongTensor([1] * (length + 2) + [0] * (self.max_output_comment_len - 2 - length)).unsqueeze(0).repeat(self.max_output_comment_len, 1)
        Y_token_label = torch.LongTensor([-1] + Y_token_label + [-1] + [-1] * (self.max_output_comment_len - 2 - length))
        return Y_comment, Y_attn_mask, Y_token_label

    def load_imgs(self, video_id, video_time):
        img_list = []
        for time in range(video_time - self.time_range, video_time + self.time_range + 1): 
            if time not in self.img_ft[video_id]:
                print('Image Wrong. Video: ',video_id, ' time: ',time)
            img_list.append(torch.from_numpy(self.img_ft[video_id][time]))
        return torch.stack(img_list)

    def load_kg(self, kg):
        kg = kg.split(' ')
        if len(kg) > self.max_kg_len:
            kg = kg[:self.max_kg_len]
        K_token_ids = list(map(lambda t: self.vocabs.get(t, 3), kg))
        length = len(K_token_ids)
        K_token_ids = torch.cat([torch.LongTensor(K_token_ids), torch.zeros(self.max_kg_len - length).long()]) # padding
        K_attn_mask = torch.LongTensor([1] * length + [0] * (self.max_kg_len - length)).unsqueeze(0).repeat(self.max_kg_len, 1)
        K_pos_ids = torch.arange(self.max_kg_len, dtype=torch.long)
        return K_token_ids, K_pos_ids, K_attn_mask

    def load_danmakus(self, context):
        danmaku_list = []
        danmaku_pos_list = []
        danmaku_attn_mask_list = []
        for time in context:
            time_context = context[time]
            time_context = time_context.split(' ')
            if len(time_context) > self.max_comment_context_len:
                time_context = time_context[: self.max_comment_context_len]
            D_context = list(map(lambda t: self.vocabs.get(t, 3), time_context))
            length = len(D_context)
            D_token_ids = torch.cat([torch.LongTensor(D_context), torch.zeros(self.max_comment_context_len - length).long()])
            D_pos_ids = torch.arange(self.max_comment_context_len, dtype=torch.long)
            # +1 for cls token
            D_context_attn_mask = torch.LongTensor([1] * (length + 1) + [0] * (self.max_comment_context_len - length))
            D_attn_mask = D_context_attn_mask.unsqueeze(0).repeat(self.max_comment_context_len + 1, 1)
            danmaku_list.append(D_token_ids)
            danmaku_pos_list.append(D_pos_ids)
            danmaku_attn_mask_list.append(D_attn_mask)
        return torch.stack(danmaku_list), torch.stack(danmaku_pos_list), torch.stack(danmaku_attn_mask_list)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, datas, vocabs, rev_vocabs, img_ft, max_kg_len, max_comment_context_len, max_output_comment_len, split_type, time_range, vid_duration):
        self.datas = datas
        self.vocabs = vocabs
        self.rev_vocabs = rev_vocabs
        self.img_ft = img_ft
        self.max_kg_len = max_kg_len
        self.max_comment_context_len = max_comment_context_len
        self.max_output_comment_len = max_output_comment_len
        self.split_type = split_type
        self.time_range = time_range
        self.vid_duration = vid_duration
        if 'kg' in self.datas[0]:
            self.kg_fact = 1
        else:
            self.kg_fact = 0

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        # Load data for train mode
        data = self.datas[index]
        video_id, video_time = data['video'], data['time']
        I = self.load_imgs(video_id, video_time)
        if self.kg_fact:
            K_token_ids, K_pos_ids, K_attn_mask = self.load_kg(data['kg'])
        else:
            K_token_ids, K_pos_ids, K_attn_mask = self.load_kg('')
        D_token_ids, D_pos_ids, D_attn_mask = self.load_danmakus(data['context'])
        Y, Y_attn_mask = self.load_target(data['comment'])
        loc_pct = float(video_time) / float(self.vid_duration[video_id])
        # for video period
        if loc_pct <= 0.2:
            loc = 0
        elif loc_pct <= 0.4:
            loc = 1
        elif loc_pct <= 0.6:
            loc = 2
        elif loc_pct <= 0.8:
            loc = 3
        else: 
            loc = 4
        loc_time = torch.Tensor([loc]).long()
        return I, K_token_ids, K_pos_ids, K_attn_mask, D_token_ids, D_pos_ids, D_attn_mask, Y, Y_attn_mask, loc_time

    def get_test_data(self, index):
        # Load data for test mode
        data = self.datas[index]
        video_id, video_time = data['video'], data['time']
        I = self.load_imgs(video_id, video_time)
        if self.kg_fact:
            K_token_ids, K_pos_ids, K_attn_mask = self.load_kg(data['kg'])
        else:
            K_token_ids, K_pos_ids, K_attn_mask = self.load_kg('')
        D_token_ids, D_pos_ids, D_attn_mask = self.load_danmakus(data['context'])
        Y_list = []
        Y_attn_mask_list = []
        for c in data['candidate']:
            Y, Y_attn_mask = self.load_target(c)
            Y_list.append(Y)
            Y_attn_mask_list.append(Y_attn_mask)
        loc_pct = float(video_time) / float(self.vid_duration[video_id])
        # for video period
        if loc_pct <= 0.2:
            loc = 0
        elif loc_pct <= 0.4:
            loc = 1
        elif loc_pct <= 0.6:
            loc = 2
        elif loc_pct <= 0.8:
            loc = 3
        else: 
            loc = 4
        loc_time = torch.Tensor([loc]).long()
        return I, K_token_ids, K_pos_ids, K_attn_mask, D_token_ids, D_pos_ids, D_attn_mask, torch.stack(Y_list), torch.stack(Y_attn_mask_list), loc_time, data # stack就是把list变成正常的tensor维度矩阵

    def load_imgs(self, video_id, video_time):
        img_list = []
        for time in range(video_time - self.time_range, video_time + self.time_range + 1): 
            if time not in self.img_ft[video_id]:
                print('Image Wrong. Video: ',video_id, ' time: ',time)
            img_list.append(torch.from_numpy(self.img_ft[video_id][time]))
        return torch.stack(img_list)

    def load_kg(self, kg):
        kg = kg.split(' ')
        if len(kg) > self.max_kg_len:
            kg = kg[:self.max_kg_len]
        K_token_ids = list(map(lambda t: self.vocabs.get(t, 3), kg))
        length = len(K_token_ids)
        K_token_ids = torch.cat([torch.LongTensor(K_token_ids), torch.zeros(self.max_kg_len - length).long()]) # padding
        K_attn_mask = torch.LongTensor([1] * length + [0] * (self.max_kg_len - length)).unsqueeze(0).repeat(self.max_kg_len, 1)
        K_pos_ids = torch.arange(self.max_kg_len, dtype=torch.long)
        return K_token_ids, K_pos_ids, K_attn_mask

    def load_target(self, comment):
        comment = comment.split(' ')
        if len(comment) > self.max_output_comment_len - 2:
            comment = comment[: self.max_output_comment_len - 2]
        Y_comment = list(map(lambda t: self.vocabs.get(t, 3), comment))
        length = len(Y_comment)
        Y_comment = torch.LongTensor([1] + Y_comment + [2] + [0] * (self.max_output_comment_len - 2 - length))
        Y_comment_attn_mask = torch.LongTensor([1] * (length + 2) + [0] * (self.max_output_comment_len - 2 - length)).unsqueeze(0).repeat(self.max_output_comment_len, 1)
        # make a triangular matrix for left-to-right generation
        down_tri_mask = torch.tril(torch.ones((self.max_output_comment_len, self.max_output_comment_len), dtype=torch.long))
        Y_attn_mask = torch.mul(Y_comment_attn_mask, down_tri_mask)
        return Y_comment, Y_attn_mask

    def load_danmakus(self, context):
        danmaku_list = []
        danmaku_pos_list = []
        danmaku_attn_mask_list = []
        for time in context:
            time_context = context[time]
            time_context = time_context.split(' ')
            if len(time_context) > self.max_comment_context_len:
                time_context = time_context[: self.max_comment_context_len]
            D_context = list(map(lambda t: self.vocabs.get(t, 3), time_context))
            length = len(D_context)
            D_token_ids = torch.cat([torch.LongTensor(D_context), torch.zeros(self.max_comment_context_len - length).long()])
            D_pos_ids = torch.arange(self.max_comment_context_len, dtype=torch.long)
            D_context_attn_mask = torch.LongTensor([1] * (length + 1) + [0] * (self.max_comment_context_len - length))
            D_attn_mask = D_context_attn_mask.unsqueeze(0).repeat(self.max_comment_context_len + 1, 1)
            danmaku_list.append(D_token_ids)
            danmaku_pos_list.append(D_pos_ids)
            danmaku_attn_mask_list.append(D_attn_mask)
        return torch.stack(danmaku_list), torch.stack(danmaku_pos_list), torch.stack(danmaku_attn_mask_list)
