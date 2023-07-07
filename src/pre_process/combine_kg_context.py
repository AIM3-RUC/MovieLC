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

if __name__ == '__main__':
    file_to_change = 'test-more.json' # dev-all.json, test-all-2.json, train-all.json

    fpath_context = os.path.join('/data7/cjt/danmaku/data/Livebot_processed/setup_2022_addcontext3/', file_to_change)
    fpath_kg = os.path.join('/data7/cjt/danmaku/data/Livebot_processed/setup_2022_kg3/', file_to_change)
    opath = os.path.join('/data7/cjt/danmaku/data/Livebot_processed/setup_2022_kg3_context3/', file_to_change)

    ori_file_context = []
    with open(fpath_context, 'r', encoding='utf-8') as fin:
        for lidx, line in enumerate(fin):
            jterm = json.loads(line.strip())
            ori_file_context.append(jterm)

    ori_file_kg = []
    with open(fpath_kg, 'r', encoding='utf-8') as fin:
        for lidx, line in enumerate(fin):
            jterm = json.loads(line.strip())
            ori_file_kg.append(jterm)

    count = 0
    total = len(ori_file_context)
    new_file = []
    for (item, jtem) in zip(ori_file_context, ori_file_kg):
        count += 1
        print(count, '/', total)
        # if count > 1000:
        #     break
        new = copy.deepcopy(item)
        new['kg'] = jtem['kg']

        new_file.append(new)     

    with open(opath, 'w') as fout:
        for data in new_file:
            jterm = json.dumps(data, ensure_ascii=False)
            fout.write(jterm+'\n')