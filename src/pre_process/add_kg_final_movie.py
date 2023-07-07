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
def filter_knowledge(path, out_path, word2id, tfidf_dict):
    tfidf_words = []
    for vid in tfidf_dict:
        tfidf_words += tfidf_dict[vid]
    tfidf_words = list(set(tfidf_words))
    print('word size of tfidf_dict', len(tfidf_words))

    f = json.load(open(path, 'r', encoding ="UTF-8"))
    out_json = {}
    filter_r_list = ['体重', '出生日期', '身高', '星座', '血型', '逝世日期', '发行时间', 'Language', 'Website', 'Starring', '时区', '出生地', \
    '语言', '拼音', '注音', '歧义关系', '笔画', '笔划', '部首笔划',  '部首 笔划', '总笔画', '部首笔画', '郑码', '词性', '制作成本', '成立时间', '公司名称', '年营业额' \
    '汉字结构', '五笔', '部首', '歧义权重', '部外', '部外 笔画', '歧义 权重', '每集长度', '上映时间', '片长', '发行公司', '对白语言', '经纪公司', '出品公司', '拍摄日期' \
    , '票房', '拍摄地点', '制片地区', 'IMDB评分']
    other_name_r = ['别名', '昵称', '中文名', '外文名', '其它译名', '其他名', '其他名称', 'Othername', '名字']
    count = 0
    total = len(f)
    final = 0
    print('ori entity num', total)
    for e1 in f:
        other_e1 = []
        for item in f[e1]:
            count += 1
            if count % 1000 == 0:
                print(count)
            # if count > 100000: # small
            #     break
            e1, r, e2 = item
            # 去括号
            temp_e1 = re.sub(u"\（.*?\）|\\(.*?\\)|\\{.*?\\}|\\[.*?\\]|\\<.*?\\>", "", e1)
            e1 = temp_e1
            temp_e2 = re.sub(u"\（.*?\）|\\(.*?\\)|\\{.*?\\}|\\[.*?\\]|\\<.*?\\>", "", e2)
            e2 = temp_e2
            if e1 == "" or (e1 == e2) or e2 == "语言" or e1 == "电影" or e1 == "导演" or e1 == "李子恒" or (r in filter_r_list) or e1.isdigit():
                continue
            else:
                flag = 0
                in_tfidf = 0
                re1 = psg.cut(e1)
                for i in re1:
                    # 保留词性有：有a的，有n的，i:成语，s:处所词)
                    if ('n' in i.flag) or ('a' in i.flag) or (i.flag == 'i') or (i.flag == 's'):
                        flag = 1
                    if i.word in tfidf_words:
                        in_tfidf = 1
                    if flag == 1 and in_tfidf == 1:
                        break
                if flag == 1 and in_tfidf == 1:
                    cut_e1_list = ' '.join(jieba.cut(e1)).split(' ')
                    cut_r_list = ' '.join(jieba.cut(r)).split(' ')
                    cut_e2_list = ' '.join(jieba.cut(e2)).split(' ')
                    check_list = cut_e1_list + cut_e2_list
                    in_count  = 0
                    for tk in check_list:
                        if tk in word2id:
                            in_count += 1
                    if in_count > (len(check_list) // 2) or r in other_name_r:
                        final += 1
                        cut_e1 = ' '.join(cut_e1_list)
                        cut_r_e2 = ' '.join(cut_r_list + cut_e2_list)
                        if cut_e1 not in out_json:
                            out_json[cut_e1] = []
                        out_json[cut_e1].append(cut_r_e2) 
                        if r in other_name_r:
                            if '/' in e2:
                                e2_list = e2.split('/')
                                for it in e2_list:
                                    if it != "":
                                        other_e1.append(' '.join(jieba.cut(it.strip())))
                            elif '、' in e2:
                                e2_list = e2.split('、')
                                for it in e2_list:
                                    if it != "":
                                        other_e1.append(' '.join(jieba.cut(it.strip())))
                            else:
                                oe = ' '.join(cut_e2_list)
                                if oe != "":
                                    other_e1.append(oe)
        for e1 in other_e1:
            if e1 not in out_json:
                out_json[e1] = out_json[' '.join(cut_e1_list)]
           
    print('relation: ', final, '/', count)    
    json.dump(out_json, open(out_path, 'w'), ensure_ascii=False)
    print('filtered entities: ', len(out_json))

def make_tfidf():
    # 输出为一个dict，vid: [w1, w2, w3, ……] 按照tf-idf值从高到低排序
    print('start count tf-idf')
    corpus = []
    vids = os.listdir('/data7/cjt/danmaku/data/danmakus_word/')
    vid_list = []
    for fvid in vids:
        vid = fvid[:-5]
        vid_list.append(vid)
        one_video_context = ''
        danmaku_file = json.load(open('/data7/cjt/danmaku/data/danmakus_word/' + vid + '.json', 'r'))
        for time in danmaku_file:
            for d in danmaku_file[time]:
                one_video_context += d + ' '
        corpus.append(one_video_context)
    vectorizer=CountVectorizer()
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
    word=vectorizer.get_feature_names()
    weight=tfidf.toarray()
    result = {}
    for i in range(len(weight)): # 遍历所有文本
        i_tfidf = {}
        for j in range(len(word)): # 遍历所有词语
            i_tfidf[word[j]] = weight[i][j]
        sorted_i_tfidf = sorted(i_tfidf.items(), key=lambda x:x[1], reverse=True) # reverse为True时是从大到小
        result[vid_list[i]] = []
        for tp in sorted_i_tfidf:
            result[vid_list[i]].append(tp[0])
    json.dump(result, open('/data7/cjt/danmaku/assist/data/video_tfidf.json', 'w'), ensure_ascii=False)
    print('finish count tf-idf')

def remade_kg_wo_digit(): # 去digit操作已经包含在filtered函数中，现在可以不用了
    knowledge = json.load(open('/data7/cjt/danmaku/assist/data/film_KG/filtered_KG.json', 'r', encoding ="UTF-8"))
    new_knowledge = copy.deepcopy(knowledge)
    for e1 in knowledge:
        e1_tokens = e1.split(' ')
        if ''.join(e1_tokens).isdigit():
            new_knowledge.pop(e1)
    json.dump(new_knowledge, open('/data7/cjt/danmaku/assist/data/film_KG/filtered_KG_wodigit.json', 'w'), ensure_ascii=False)


def remade_tfidf():
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

    tfidf = json.load(open('/data7/cjt/danmaku/assist/data/full_video_tfidf.json', 'r'))
    new_tfidf = {}
    for vid in tfidf:
        new_tfidf[vid] = []
        video_tfidf = tfidf[vid][:300]
        for tks in video_tfidf:
            re1 = psg.cut(tks)
            for i in re1:
                if tks not in stop_word_list and not(tks.isdigit()) and (('n' in i.flag) or (i.flag == 'i') or (i.flag == 's')):
                    new_tfidf[vid].append(tks)
    json.dump(new_tfidf, open('/data7/cjt/danmaku/assist/data/video_tfidf2.json', 'w'), ensure_ascii=False)


def add_KG(s_tokens, knowledge, max_len, tfidf, word2id): # add KG like K-BERT
    # 改进规则。e1和s_tokens能对上两个，且至少有一个在tfidf里才能引入。引入时筛掉dict外的词，筛掉停用词。同时找二跳，加入一部分二跳内容进入。
    knowledge_small = {}
    e1_all_tokens = []
    for e1 in knowledge:
        e1_tokens = e1.split(' ')
        for e1tks in e1_tokens:
            if e1tks in tfidf:
                inter_tks = list(set(e1_tokens).intersection(set(s_tokens)))
                if len(inter_tks) >= 2:
                    knowledge_small[e1] = knowledge[e1]
                    e1_all_tokens.append(e1_tokens)
                    break

    s_rlt_list = []
    info_rlt_list = []
    for e1 in knowledge_small:
        e1_tokens = e1.split(' ')
        for rlt in knowledge_small[e1]:
            rlt_tokens = rlt.split(' ')
            if 'information' in rlt or 'Information' in rlt:
                if len(rlt_tokens) >= 41:
                    start = random.randint(1, len(rlt_tokens) - 40)
                    relation = e1 + ' ' + 'information ' + ' '.join(rlt_tokens[start: start + 40])
                else:
                    relation = e1 + ' ' + ' '.join(rlt_tokens)
                info_rlt_list.insert(0, relation)
            else:     
                relation = e1 + ' ' + ' '.join(rlt_tokens)
                s_rlt_list.append(relation)
    

    random.shuffle(info_rlt_list)
    random.shuffle(s_rlt_list)
    s_rlt_list = info_rlt_list + s_rlt_list

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
    if len(kg_s_tokens) >= 1:
        knowledge_small = {}
        for e1 in knowledge:
            e1_tokens = e1.split(' ')
            if e1_tokens in e1_all_tokens:
                continue
            else:
                for e1tks in e1_tokens:
                    if e1tks in tfidf:
                        inter_tks = list(set(e1_tokens).intersection(set(kg_s_tokens)))
                        if len(inter_tks) >= 2:
                            knowledge_small[e1] = knowledge[e1]
                            break  
            
        s_rlt_list = []
        info_rlt_list = []
        for e1 in knowledge_small:
            for rlt in knowledge_small[e1]:
                rlt_tokens = rlt.split(' ')
                if 'information' in rlt or 'Information' in rlt:
                    if len(rlt_tokens) >= 41:
                        start = random.randint(1, len(rlt_tokens) - 40)
                        relation = e1 + ' ' + 'information ' + ' '.join(rlt_tokens[start: start + 40])
                    else:
                        relation = e1 + ' ' + ' '.join(rlt_tokens)
                    info_rlt_list.insert(0, relation)
                else:     
                    relation = e1 + ' ' + ' '.join(rlt_tokens)
                    s_rlt_list.append(relation) 

        random.shuffle(info_rlt_list)
        random.shuffle(s_rlt_list)
        s_rlt_list = info_rlt_list + s_rlt_list

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
    
    # remade_kg_wo_digit()
    # remade_tfidf()

    # # get context and count tf-idf
    # tfidf_dict = make_tfidf()
    tfidf_dict = json.load(open('/data7/cjt/danmaku/assist/data/movie_tfidf.json', 'r'))

    # knowledge_path = '/data7/cjt/danmaku/assist/data/film_KG/kb_film.json'
    dict_json = json.load(open('/data7/cjt/danmaku/data/processed/dict.json', 'r', encoding ="UTF-8"))
    word2id = dict_json["word2id"]
    # knowledge = filter_knowledge(knowledge_path, '/data7/cjt/danmaku/assist/data/film_KG/filtered_KG_3.json', word2id, tfidf_dict)
    # exit()

    knowledge = json.load(open('/data7/cjt/danmaku/assist/data/film_KG/filtered_KG_3.json', 'r', encoding ="UTF-8"))
    
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
    
    file_to_change = 'test-all.json' # dev-all.json, test-all-2.json, train-all.json

    fpath = os.path.join('/data7/cjt/danmaku/data/processed/setup_2022/', file_to_change)
    opath = os.path.join('/data7/cjt/danmaku/data/processed/setup_2022_kg3/', file_to_change)

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
        # if count > 10000:
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

    
        
        
    