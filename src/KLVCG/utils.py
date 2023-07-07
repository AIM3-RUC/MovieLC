import os
import json
import torch
import torch.nn as nn
import logging

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG) # ERROR, WARN, INFO, DEBUG
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger

def load_from_json(infile):
    datas = []
    with open(infile, 'r', encoding='utf-8') as fin:
        for line in fin:
            data = json.loads(line.strip())
            datas.append(data)
    return datas

def load_dict(vocab_path):
    vocabs = json.load(open(vocab_path, 'r', encoding='utf-8'))['word2id']
    rev_vocabs = json.load(open(vocab_path, 'r', encoding='utf-8'))['id2word']
    return vocabs, rev_vocabs

def set_data_path(datapath):
    train_path = os.path.join(datapath, 'train-all.json')
    test_path = os.path.join(datapath, 'test-all.json')
    dev_path = os.path.join(datapath, 'dev-all.json')
    return train_path, test_path, dev_path

def set_img_path(imgpath):
    imgf_path = ''
    for file in os.listdir(imgpath):
        if file.endswith('101.pkl') and 'audio' not in file:
            imgf_path = os.path.join(imgpath, file)
            break
    print('Image path', imgf_path)
    return imgf_path