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

# 1.2
def filter_knowledge(path, out_path, tfidf_dict):
    # ori_f = open(path, 'r', encoding ="UTF-8").readlines()
    # count = 0
    # total_r = len(ori_f)
    # print('ori relation num', total_r)
    # f = {}
    # for line in ori_f:
    #     if count % 10000 == 0:
    #         print(count, '/', total_r)
    #     count += 1
    #     # if count >= 10000:
    #     #     break
    #     try:
    #         e1, r, e2 = line.strip().split(',')
    #         # 去括号
    #         temp_e1 = re.sub(u"\（.*?\）|\\(.*?\\)|\\{.*?\\}|\\[.*?\\]|\\<.*?\\>", "", e1)
    #         e1 = temp_e1
    #         temp_e2 = re.sub(u"\（.*?\）|\\(.*?\\)|\\{.*?\\}|\\[.*?\\]|\\<.*?\\>", "", e2)
    #         e2 = temp_e2
    #         if e1 in f:
    #             f[e1].append([r, e2])
    #         else:
    #             f[e1] = [[r, e2]]
    #     except:
    #         continue
    # json.dump(f, open('/data7/cjt/danmaku/assist/data/all_KG/ownthink_e1dict.json', 'w'), ensure_ascii=False)

    f = json.load(open('/data7/cjt/danmaku/assist/data/all_KG/ownthink_e1dict.json', 'r', encoding='utf-8'))

    tfidf_words = []
    for vid in tfidf_dict:
        tfidf_words += tfidf_dict[vid]
    tfidf_words = list(set(tfidf_words))
    print('word size of tfidf_dict', len(tfidf_words))

    out_json = {}
    filter_r_list = ['体重', '出生日期', '身高', '星座', '血型', '逝世日期', '发行时间', 'Language', 'Website', 'Starring', '时区', '出生地', \
    '语言', '拼音', '注音', '歧义关系', '笔画', '笔划', '部首笔划',  '部首 笔划', '总笔画', '部首笔画', '郑码', '词性', '制作成本', '成立时间', '公司名称', '年营业额' \
    '汉字结构', '五笔', '部首', '歧义权重', '部外', '部外 笔画', '歧义 权重', '每集长度', '上映时间', '片长', '发行公司', '对白语言', '经纪公司', '出品公司', '拍摄日期' \
    , '票房', '拍摄地点', '制片地区', 'IMDB评分']
    count = 0
    total = len(f)
    final = 0
    for e1 in f:
        if count % 1000 == 0:
            print(count, '/', total)
        count += 1
        # if count > 10000: # small
        #     break
        if e1 in tfidf_words: # 根据后面add_kg的规则，这样过滤就可以了
            for r, e2 in f[e1]:
                if e1 == "" or (e1 == e2) or e2 == "语言" or e1 == "电影" or e1 == "导演" or e1 == "李子恒" or (r in filter_r_list) or e1.isdigit():
                    continue
                else:
                    cut_e1_list = ' '.join(jieba.cut(e1)).split(' ')
                    cut_r_list = ' '.join(jieba.cut(r)).split(' ')
                    cut_e2_list = ' '.join(jieba.cut(e2)).split(' ')
                    final += 1
                    cut_e1 = ' '.join(cut_e1_list)
                    cut_r_e2 = ' '.join(cut_r_list + cut_e2_list)
                    if cut_e1 not in out_json:
                        out_json[cut_e1] = []
                    out_json[cut_e1].append(cut_r_e2) 
    json.dump(out_json, open(out_path, 'w'), ensure_ascii=False)
    print('relation: ', final)    
    print('filtered entities: ', len(out_json))

def make_test_tfidf():
    # stop word load
    stop_word_list = []
    stop_word_file = open('/data7/cjt/danmaku/assist/data/stop_words.txt', 'r').readlines()
    for w in stop_word_file:
        stop_word_list.append(w.strip())
    bad_tk = ['电影', '这个', '真的', '老师', '哥哥', '一个', '就是', '哈哈哈', '没有', '我们', \
    '不是', '什么', '哈哈哈', '哈哈', '一块', '好看', '可以', '妹妹', '如果', '付费', '一天', '回去', \
    '看不懂', '好看', '你们', '他们', '666', 'vip', '自己', '弹幕', '厉害', '这么', '这样', '觉得', \
    '两个', '感觉', '一起', '知道', '那个', '这部', '为什么', '因为', '发来', '经典', '怎么', '正常', \
    '弟弟', '这女', '一所', '姐姐', '故事', '这是', '那是', '<SEP>']
    stop_word_list += bad_tk

    # 输出为一个dict，vid: [w1, w2, w3, ……] 按照tf-idf值从高到低排序
    print('start count tf-idf')
    corpus = []
    vid_list = []
    # 注意下面这个路径一定要是分词后的
    root_path = '/data7/cjt/danmaku/data/VideoIC_processed/setup_2022'
    infile = 'test-all-80.json'
    lines = open(os.path.join(root_path, infile), 'r', encoding='utf-8').readlines()
    for line in lines:
        jterm = json.loads(line.strip())
        vid = jterm['video']
        temp_context = ''
        for t in jterm['context']:
            if jterm['context'][t] != '':
                temp_context += jterm['context'][t] + ' '
        if vid not in vid_list:
            vid_list.append(vid)
            corpus.append(temp_context)
        else:
            idx = vid_list.index(vid)
            corpus[idx] += temp_context

    vectorizer=CountVectorizer()
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
    word=vectorizer.get_feature_names()
    weight=tfidf.toarray()
    result = {}
    for i in range(len(weight)): # 遍历所有文本
        print(i, '/', len(weight))
        i_tfidf = {}
        for j in range(len(word)): # 遍历所有词语
            i_tfidf[word[j]] = weight[i][j]
        sorted_i_tfidf = sorted(i_tfidf.items(), key=lambda x:x[1], reverse=True) # reverse为True时是从大到小
        result[vid_list[i]] = []
        temp_count = 0
        for tp in sorted_i_tfidf:
            if temp_count >= 300:
                break
            temp_count += 1
            tks = tp[0]
            re1 = psg.cut(tks)
            for t in re1:
                if tks not in stop_word_list and not(tks.isdigit()) and (('n' in t.flag) or (t.flag == 'i') or (t.flag == 's')):
                    result[vid_list[i]].append(tks)

    json.dump(result, open('/data7/cjt/danmaku/assist/data/videoic_tfidf_test80.json', 'w'), ensure_ascii=False)
    print('finish count tf-idf')

def make_tfidf():
    # stop word load
    stop_word_list = []
    stop_word_file = open('/data7/cjt/danmaku/assist/data/stop_words.txt', 'r').readlines()
    for w in stop_word_file:
        stop_word_list.append(w.strip())
    bad_tk = ['电影', '这个', '真的', '老师', '哥哥', '一个', '就是', '哈哈哈', '没有', '我们', \
    '不是', '什么', '哈哈哈', '哈哈', '一块', '好看', '可以', '妹妹', '如果', '付费', '一天', '回去', \
    '看不懂', '好看', '你们', '他们', '666', 'vip', '自己', '弹幕', '厉害', '这么', '这样', '觉得', \
    '两个', '感觉', '一起', '知道', '那个', '这部', '为什么', '因为', '发来', '经典', '怎么', '正常', \
    '弟弟', '这女', '一所', '姐姐', '故事', '这是', '那是']
    stop_word_list += bad_tk

    # 输出为一个dict，vid: [w1, w2, w3, ……] 按照tf-idf值从高到低排序
    print('start count tf-idf')
    corpus = []
    # 注意下面这个路径一定要是分词后的
    root_path = '/data7/cjt/danmaku/VideoIC/danmakus_word/'
    # root_path = '/data7/cjt/danmaku/data/Livebot_ori/word/'
    vid_list = []
    vids = os.listdir(root_path)
    for fvid in vids:
        # if len(vid_list) > 20:
        #     break
        vid = fvid[:-5]
        vid_list.append(vid)
        one_video_context = ''
        # for videoic
        danmaku_file = json.load(open(os.path.join(root_path, vid + '.json'), 'r', encoding='utf-8'))
        for time in danmaku_file:
            for d in danmaku_file[time]:
                one_video_context += d + ' '
        
        # # for livebot
        # danmaku_file = open(os.path.join(root_path, vid + '.json'), 'r').readlines()
        # for line in danmaku_file:
        #     item = json.loads(line.strip())
        #     for d in item["comment"]:
        #         one_video_context += d + ' '

        corpus.append(one_video_context)
    vectorizer=CountVectorizer()
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
    word=vectorizer.get_feature_names()
    weight=tfidf.toarray()
    result = {}
    for i in range(len(weight)): # 遍历所有文本
        print(i, '/', len(weight))
        i_tfidf = {}
        for j in range(len(word)): # 遍历所有词语
            i_tfidf[word[j]] = weight[i][j]
        sorted_i_tfidf = sorted(i_tfidf.items(), key=lambda x:x[1], reverse=True) # reverse为True时是从大到小
        result[vid_list[i]] = []
        temp_count = 0
        for tp in sorted_i_tfidf:
            if temp_count >= 300:
                break
            temp_count += 1
            tks = tp[0]
            re1 = psg.cut(tks)
            for t in re1:
                if tks not in stop_word_list and not(tks.isdigit()) and (('n' in t.flag) or (t.flag == 'i') or (t.flag == 's')):
                    result[vid_list[i]].append(tks)

    json.dump(result, open('/data7/cjt/danmaku/assist/data/videoic_tfidf.json', 'w'), ensure_ascii=False)
    print('finish count tf-idf')


def add_KG(s_tokens, knowledge, max_len, tfidf, word2id): # add KG like K-BERT
    ss_tokens = list(set(s_tokens))

    knowledge_small = {}
    one_tks = []
    for tk in ss_tokens:
        if tk in tfidf:
            if tk in knowledge:
                knowledge_small[tk] = knowledge[tk]
                one_tks.append(tk)

    s_rlt_list = []
    for e1 in knowledge_small:
        for rlt in knowledge_small[e1]:
            relation = e1 + ' ' + rlt
            s_rlt_list.append(relation)
    random.shuffle(s_rlt_list)

    # 加入检索到的三元组
    kg_s = ''
    kg_s_tokens = []
    for rlt in s_rlt_list:
        rlt_tokens = rlt.split(' ')
        temp_rlt_tokens = list(set(rlt_tokens))
        rlt_tokens = []
        for tks in temp_rlt_tokens:
            if tks not in stop_word_list and tks in word2id:
                rlt_tokens.append(tks)
        rlt_tokens.append('<SEP>')
        kg_s_tokens += rlt_tokens
        if len(kg_s_tokens) > max_len:
            kg_s_tokens = kg_s_tokens[:max_len]
            break
    kg_s += ' '.join(kg_s_tokens)

    # 二跳
    kg_s_tokens = kg_s.split(' ')
    kkg_s_tokens = list(set(kg_s_tokens))
    if  len(kg_s_tokens) >= 1:
        knowledge_small = {}
        for tk in kkg_s_tokens:
            if tk in tfidf and tk not in one_tks:
                if tk in knowledge:
                    knowledge_small[tk] = knowledge[tk]
    
        s_rlt_list = []
        for e1 in knowledge_small:
            for rlt in knowledge_small[e1]:
                relation = e1 + ' ' + rlt
                s_rlt_list.append(relation)
        random.shuffle(s_rlt_list)

        # 加入检索到的三元组
        kg_s_2_tokens = ['<SEP>']
        for rlt in s_rlt_list:
            rlt_tokens = rlt.split(' ')
            temp_rlt_tokens = list(set(rlt_tokens))
            rlt_tokens = []
            for tks in temp_rlt_tokens:
                if tks not in stop_word_list and tks in word2id:
                    rlt_tokens.append(tks)
            rlt_tokens.append('<SEP>')
            kg_s_2_tokens += rlt_tokens
            if len(kg_s_2_tokens) > 30:
                kg_s_2_tokens = kg_s_2_tokens[:30]
                break

        if len(kg_s_2_tokens) > 20:
            if len(kg_s_tokens) > 51:
                kg_s_tokens = kg_s_tokens[:50]
            # print('&&&&&&&&&&&&&&&')
            # print('kg_s_tokens', kg_s_tokens)
            # print('kg_s_2_tokens', kg_s_2_tokens)
            kg_s = ' '.join(kg_s_tokens + kg_s_2_tokens)

    kg_s_tokens = kg_s.split(' ')
    num_kg_tokens = len(kg_s_tokens)

    if num_kg_tokens > 1:
        c_sent = 1
    else:
        c_sent = 0

    # comment and kg interaction，返回comment的序号
    comment_kg_inter_pos = []
    comment_kg_inter_count = 0
    # print('***********************************')
    # print(num_kg_tokens)
    # print('s_tokens', s_tokens)
    # print('kg_s', kg_s)
    return num_kg_tokens, c_sent, kg_s, comment_kg_inter_pos, comment_kg_inter_count


if __name__ == '__main__':
    max_len = 80 # max len for the total kg modality

    # # get % test tfidf for 3 datasets
    # make_test_tfidf()
    # exit()

    # # get context and count tf-idf
    # tfidf_dict = make_tfidf()
    # exit()

    # # filter knowledge
    # tfidf_dict = json.load(open('/data7/cjt/danmaku/assist/data/videoic_tfidf.json', 'r'))
    # knowledge_path = '/data7/cjt/danmaku/assist/data/all_KG/ownthink_v2.csv'
    # knowledge = filter_knowledge(knowledge_path, '/data7/cjt/danmaku/assist/data/all_KG/filtered_KG_for_videoic.json', tfidf_dict)
    # exit()

    tfidf_dict = json.load(open('/data7/cjt/danmaku/assist/data/livebot_tfidf.json', 'r'))
    
    knowledge = json.load(open('/data7/cjt/danmaku/assist/data/all_KG/filtered_KG_for_livebot.json', 'r', encoding ="UTF-8"))
    
    dict_json = json.load(open('/data7/cjt/danmaku/data/Livebot_processed/dict.json', 'r', encoding ="UTF-8"))
    word2id = dict_json["word2id"]

    # stop word load
    stop_word_list = []
    stop_word_file = open('/data7/cjt/danmaku/assist/data/stop_words.txt', 'r').readlines()
    for w in stop_word_file:
        stop_word_list.append(w.strip())
    punct = ['！','？','。','＂','＃','％','＆','＇','（','）','＊','－','／','：','；','＜','＝','＞','＠','［','＼','］','＾',\
    '＿','｀','｛','｜','｝','～','｟','｠','｢','｣','､','、','〃','「','」','『','』','【','】','〔','〕','〖','〗','〘','〙','〚',\
    '〛','〜','〝','〞','〟','〰','，','〿','–—','‘','’','‛','“','”','„','‟','…','‧','﹏','.','!','#','%','&','(',')','*','+',',',\
    '-','.','/',':',';','<','=','>','?','@','[',']','^','_','`','{','|','}','~', '...']
    stop_word_list += punct
    stop_word_list += ['<SEP>', '']

    file_to_change = 'test-more.json' # dev-all.json, test-all-2.json, train-all.json

    fpath = os.path.join('/data7/cjt/danmaku/data/Livebot_processed/setup_2022', file_to_change)
    opath = os.path.join('/data7/cjt/danmaku/data/Livebot_processed/setup_2022_kg3/', file_to_change)

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
    total_kg_tokens = 0
    # # 清理，后面用a
    # fout = open(opath, 'w')
    # fout.close()
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
        concat_context = ''
        for time_context in context:
            if context[time_context] != '':
                concat_context += context[time_context] + ' <SEP> '
        if concat_context != '':
            concat_context = concat_context[:-7]
        concat_context_tokens = concat_context.split(' ')
        num = len(concat_context_tokens)
        if num > 300:
            start = random.randint(0, num - 301)
            concat_context_tokens = concat_context_tokens[start:start+300]
        
        kg_tokens, c_sent, kg_s, comment_kg_inter_pos, comment_kg_inter_count = add_KG(concat_context_tokens, knowledge, max_len, tfidf_dict[vid], word2id)

        new["kg"] = kg_s
        new["comment_kg_inter_pos"] = comment_kg_inter_pos

        total_all_tokens += num
        total_kg_tokens += kg_tokens
        change_sent += c_sent
        all_sent += 1

        new_file.append(new)

        # if count % 500 == 0:
        #     with open(opath, 'a') as fout:
        #         for data in new_file:
        #             jterm = json.dumps(data, ensure_ascii=False)
        #             fout.write(jterm+'\n')
        #         fout.flush()
        #         del new_file
        #         new_file = []

    print('total_kg_tokens/total_all_tokens', total_kg_tokens, '/', total_all_tokens)   
    print('change_sent/all_sent', change_sent, '/', all_sent) 
     

    with open(opath, 'w') as fout:
        for data in new_file:
            jterm = json.dumps(data, ensure_ascii=False)
            fout.write(jterm+'\n')

    
        
        
    