'''
    @ Date:     20211108
    @ Author:   Jieting Chen
    @ Function: add highlight score into setup
'''

import os
import json
import copy
import jieba
import jieba.posseg as psg
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer 
import re
import random

# # for videoic
# def count_danmaku_stat(): 
#     vid_class = {}
#     test_list = []
#     splits_path = '/data7/cjt/danmaku/VideoIC/tasks/comments_generation/division'
#     splits = os.listdir(splits_path)
#     for split in splits:
#         infile = open(os.path.join(splits_path, split), 'r', encoding='utf-8').readlines()
#         for line in infile:
#             jterm = json.loads(line.strip())
#             vid = jterm['aid']
#             c = jterm['class']
#             vid_class[vid] = c
#             if split == 'test.json':
#                 test_list.append(vid)
    
#     class_list = ['life', 'entertainment', 'culture_art', 'science_education', 'movie_tv', 'games']
#     danmaku_stat = {} # {'life': {"xxxxxxxx": {"No0016": [45,50,124], "No0932": [6,10,24]}, "xxxxxx22222": {}, ...}, ...}
#     for c in class_list:
#         danmaku_stat[c] = {}
#     vids = os.listdir('/data7/cjt/danmaku/VideoIC/danmakus_word/')
#     total = len(vids)
#     count = 0
#     # 正好100个标点符号
#     punct = ['！','？','。','＂','＃','＄','％','＆','＇','（','）','＊','－','／','：','；','＜','＝','＞','＠','［','＼','］','＾',\
#     '＿','｀','｛','｜','｝','～','｟','｠','｢','｣','､','、','〃','《','》','「','」','『','』','【','】','〔','〕','〖','〗','〘','〙','〚',\
#     '〛','〜','〝','〞','〟','〰','，','〿','–—','‘','’','‛','“','”','„','‟','…','‧','﹏','.','!','#','$','%','&','(',')','*','+',',',\
#     '-','.','/',':',';','<','=','>','?','@','[',']','^','_','`','{','|','}','~', '...']
#     for fvid in vids:
#         vid = fvid[:-5]
#         if vid not in test_list:
#             c = vid_class[vid]
#             danmaku_file = json.load(open('/data7/cjt/danmaku/VideoIC/danmakus_word/' + vid + '.json', 'r', encoding='utf-8'))
#             for time in danmaku_file:
#                 for dd in danmaku_file[time]:
#                     d_tokens = dd.split(' ')
#                     temp_tokens = []
#                     for tk in d_tokens:
#                         if tk not in punct:
#                             temp_tokens.append(tk)
#                     if len(temp_tokens) == 0:
#                         d = dd
#                     else:
#                         d = ' '.join(temp_tokens)
#                     if d not in danmaku_stat[c]:
#                         danmaku_stat[c][d] = {}
#                     if vid not in danmaku_stat[c][d]:
#                         danmaku_stat[c][d][vid] = []
#                     danmaku_stat[c][d][vid].append(int(time))
#         else:
#             print(vid)
#         count += 1
#         print(count, '/', total)
#     json.dump(danmaku_stat, open('/data7/cjt/danmaku/assist/data/videoic_danmaku_stat.json', 'w'), ensure_ascii=False)
#     # for c in danmaku_stat:
#     #     print(c)

# def count_danmaku_stat(): # for movie
#     danmaku_stat = {} # {"xxxxxxxx": {"No0016": [45,50,124], "No0932": [6,10,24]}, "xxxxxx22222": {}, ...}
#     vids = os.listdir('/data7/cjt/danmaku/data/danmakus_word')
#     total = len(vids)

    # # exclude test set
    # test_list = ["No0059", "No0577", "No0084", "No0627", "No0597", "No0008", "No0515", "No0115", "No0122"]
    # count = 0
    # # 正好100个标点符号
    # punct = ['！','？','。','＂','＃','＄','％','＆','＇','（','）','＊','－','／','：','；','＜','＝','＞','＠','［','＼','］','＾',\
    # '＿','｀','｛','｜','｝','～','｟','｠','｢','｣','､','、','〃','《','》','「','」','『','』','【','】','〔','〕','〖','〗','〘','〙','〚',\
    # '〛','〜','〝','〞','〟','〰','，','〿','–—','‘','’','‛','“','”','„','‟','…','‧','﹏','.','!','#','$','%','&','(',')','*','+',',',\
    # '-','.','/',':',';','<','=','>','?','@','[',']','^','_','`','{','|','}','~', '...']
    # for fvid in vids:
    #     vid = fvid[:-5]
    #     if vid not in test_list:
    #         danmaku_file = json.load(open('/data7/cjt/danmaku/data/danmakus_word/' + vid + '.json', 'r', encoding='utf-8'))
    #         for time in danmaku_file:
    #             for dd in danmaku_file[time]:
    #                 d_tokens = dd.split(' ')
    #                 temp_tokens = []
    #                 for tk in d_tokens:
    #                     if tk not in punct:
    #                         temp_tokens.append(tk)
    #                 if len(temp_tokens) == 0:
    #                     d = dd
    #                 else:
    #                     d = ' '.join(temp_tokens)
    #                 if d not in danmaku_stat:
    #                     danmaku_stat[d] = {}
    #                 if vid not in danmaku_stat[d]:
    #                     danmaku_stat[d][vid] = []
    #                 danmaku_stat[d][vid].append(int(time))
    #     else:
    #         print(vid)
    #     count += 1
    #     print(count, '/', total)
    # json.dump(danmaku_stat, open('/data7/cjt/danmaku/assist/data/movie_danmaku_stat_wotest.json', 'w'), ensure_ascii=False)

# def count_danmaku_stat(): # for livebot
#     danmaku_stat = {} # {"xxxxxxxx": {"No0016": [45,50,124], "No0932": [6,10,24]}, "xxxxxx22222": {}, ...}
#     vids = os.listdir('/data7/cjt/danmaku/data/Livebot_ori/word/')
#     total = len(vids)

#     # exclude test set
#     test_list = []
#     test_file = open('/data7/cjt/danmaku/data/Livebot_ori/split/test.txt', 'r').readlines()
#     for line in test_file:
#         test_list.append(line.strip())

#     count = 0
#     # 正好100个标点符号
#     punct = ['！','？','。','＂','＃','＄','％','＆','＇','（','）','＊','－','／','：','；','＜','＝','＞','＠','［','＼','］','＾',\
#     '＿','｀','｛','｜','｝','～','｟','｠','｢','｣','､','、','〃','《','》','「','」','『','』','【','】','〔','〕','〖','〗','〘','〙','〚',\
#     '〛','〜','〝','〞','〟','〰','，','〿','–—','‘','’','‛','“','”','„','‟','…','‧','﹏','.','!','#','$','%','&','(',')','*','+',',',\
#     '-','.','/',':',';','<','=','>','?','@','[',']','^','_','`','{','|','}','~', '...']
#     for fvid in vids:
#         vid = fvid[:-5]
#         if vid not in test_list:
#             danmaku_file = open('/data7/cjt/danmaku/data/Livebot_ori/word/' + vid + '.json', 'r', encoding='utf-8').readlines()
#             for line in danmaku_file:
#                 item = json.loads(line)
#                 time = item['time']
#                 for dd in item["comment"]:
#                     d_tokens = dd.split(' ')
#                     temp_tokens = []
#                     for tk in d_tokens:
#                         if tk not in punct:
#                             temp_tokens.append(tk)
#                     if len(temp_tokens) == 0:
#                         d = dd
#                     else:
#                         d = ' '.join(temp_tokens)
#                     if d not in danmaku_stat:
#                         danmaku_stat[d] = {}
#                     if vid not in danmaku_stat[d]:
#                         danmaku_stat[d][vid] = []
#                     danmaku_stat[d][vid].append(int(time))
#         else:
#             print(vid)
#         count += 1
#         print(count, '/', total)
#     json.dump(danmaku_stat, open('/data7/cjt/danmaku/assist/data/livebot_danmaku_stat_wotest.json', 'w'), ensure_ascii=False)

# # version 1.1 for % context
# def add_for_context(context, vid, time, context_tokens, max_len, tfidf, danmaku_stat):
#     new_context_tokens = copy.deepcopy(context_tokens)
#     c_sent = 0
#     if len(tfidf) > 50:
#         tfidf = tfidf[:50]
#     inter_tks = list(set(context_tokens).intersection(set(tfidf)))
#     if len(inter_tks) > 1:
#         for item in danmaku_stat:
#             if vid in danmaku_stat[item]: # 相同视频的通通不要
#                 continue
#             if item in context:  # 防泄漏，引入更多元的一些
#                 continue
#             item_tks = item.split(' ')
#             inter_inter = list(set(inter_tks).intersection(set(item_tks)))
#             if len(inter_inter) > 1:
#                 new_context_tokens.append('<SEP>')
#                 new_context_tokens += item_tks
#                 c_sent = 1
#             if len(new_context_tokens) >= max_len:
#                 new_context_tokens = new_context_tokens[:max_len]
#                 break
#     add_tokens = len(new_context_tokens) - len(context_tokens)
#     new_context = ' '.join(new_context_tokens)
#     # if add_tokens > 0:
#     #     print('*****************')
#     #     print(' '.join(context_tokens))
#     #     print(new_context)
#     return add_tokens, c_sent, new_context

# version 1.1，不允许找到的弹幕在原始context里出现过，target时刻不引入， inter_tokens >=2, tfidf 50
def add_for_context(context, vid, time, context_tokens, max_len, tfidf, danmaku_stat):
    new_context_tokens = copy.deepcopy(context_tokens)
    c_sent = 0
    if len(tfidf) > 50:
        tfidf = tfidf[:50]
    inter_tks = list(set(context_tokens).intersection(set(tfidf))) # 有点强化主题的意思在hhh
    if len(inter_tks) > 1:
        for item in danmaku_stat:
            if item in context:  # 防泄漏
                continue

            item_tks = item.split(' ') # 相交词要>=2
            inter_inter = list(set(inter_tks).intersection(set(item_tks)))
            if len(inter_inter) <= 1:
                continue

            if vid in danmaku_stat[item]:
                flag = 1
                for t in danmaku_stat[item][vid]: # 防泄漏
                    # print('time', type(time)) # int
                    # print('t', type(t)) # int
                    if t >= time - 5 and t <= time + 5: # 防泄漏
                        flag = 0
                        break
                if flag == 0:
                    continue

            new_context_tokens.append('<SEP>')
            new_context_tokens += item_tks
            c_sent = 1
            if len(new_context_tokens) >= max_len:
                new_context_tokens = new_context_tokens[:max_len]
                break
    add_tokens = len(new_context_tokens) - len(context_tokens)
    new_context = ' '.join(new_context_tokens)
    # if add_tokens > 0:
    #     print('*****************')
    #     print(' '.join(context_tokens))
    #     print(new_context)
    return add_tokens, c_sent, new_context


if __name__ == '__main__':
    # count_danmaku_stat()
    # exit()

    # # for videoic
    # vid_class = {}
    # splits_path = '/data7/cjt/danmaku/VideoIC/tasks/comments_generation/division'
    # splits = os.listdir(splits_path)
    # for split in splits:
    #     infile = open(os.path.join(splits_path, split), 'r', encoding='utf-8').readlines()
    #     for line in infile:
    #         jterm = json.loads(line.strip())
    #         vid = jterm['aid']
    #         c = jterm['class']
    #         vid_class[vid] = c

    all_new_context = {}

    max_len = 100 # max len for the total context modality

    tfidf_dict = json.load(open('/data7/cjt/danmaku/assist/data/livebot_tfidf_test20.json', 'r'))

    danmaku_stat = json.load(open('/data7/cjt/danmaku/assist/data/livebot_danmaku_stat_wotest.json', 'r'))
    
    file_to_change = 'test-more.json' # dev-all.json, test-all-2.json, train-all.json

    fpath = os.path.join('/data7/cjt/danmaku/data/Livebot_processed/setup_2022/', file_to_change)
    opath = os.path.join('/data7/cjt/danmaku/data/Livebot_processed/setup_2022_addcontext3/', file_to_change)

    all_sent = 0
    change_sent = 0
    
    ori_file = []
    with open(fpath, 'r', encoding='utf-8') as fin:
        for lidx, line in enumerate(fin):
            jterm = json.loads(line.strip())
            ori_file.append(jterm)
    
    count = 0
    total = len(ori_file)
    new_file = []
    total_all_tokens = 0
    total_add_tokens = 0
    for item in ori_file:
        count += 1
        print(count, '/', total)
        # if count > 100:
        #     break
        new = copy.deepcopy(item)
        vid = item["video"]
        time = item["time"]
        # add knowledge
        context = item["context"]
        for time_context in context:
            if context[time_context] != '':
                context_tokens = context[time_context].split(' ')
                num = len(context_tokens)
                if num <= 100:
                    flag = 0
                    if vid in all_new_context:
                        if time_context in all_new_context[vid]:
                            add_tokens, c_sent, new_context = all_new_context[vid][time_context]
                            flag = 1
                    if flag == 0:
                        # for livebot and movie
                        add_tokens, c_sent, new_context = add_for_context(context[time_context], vid, time, context_tokens, max_len, tfidf_dict[vid], danmaku_stat)
                        # # for videoic
                        # add_tokens, c_sent, new_context = add_for_context(context[time_context], vid, time, context_tokens, max_len, tfidf_dict[vid], danmaku_stat[vid_class[vid]])
                        if vid not in all_new_context:
                            all_new_context[vid] = {}
                        all_new_context[vid][time_context] = [add_tokens, c_sent, new_context]
                    
                    new["context"][time_context] = new_context
                    total_add_tokens += add_tokens
                total_all_tokens += num
                change_sent += c_sent
            all_sent += 1

        new_file.append(new)

    print('total_add_tokens/total_all_tokens', total_add_tokens, '/', total_all_tokens)   
    print('change_sent/all_sent', change_sent, '/', all_sent)        

    with open(opath, 'w') as fout:
        for data in new_file:
            jterm = json.dumps(data, ensure_ascii=False)
            fout.write(jterm+'\n')

    
        
        
    