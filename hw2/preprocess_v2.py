#!/usr/bin/env python
# coding: utf-8

import os
import torch
import random
import numpy as np
from tqdm import tqdm
torch.cuda.empty_cache()

# Preparing Data
def load_feat(path):
    feat = torch.load(path)
    return feat

def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]

    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)

def preprocess_data(split, feat_dir, phone_path, train_ratio=0.8, random_seed=1124):
    
    t_label = []
    t_feat = []
    
    class_num = 41 # NOTE: pre-computed, should not need change
    if split == 'train' or split == 'val':
        mode = 'train'
    elif split == 'test':
        mode = 'test'
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    label_dict = {}
    
    if mode == 'train':
        for line in open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines():
            line = line.strip('\n').split(' ')
            label_dict[line[0]] = [int(p) for p in line[1:]]

        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(random_seed)
        random.shuffle(usage_list)
        train_len = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:train_len] if split == 'train' else usage_list[train_len:]
    
    elif mode == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    total_len = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        t_feat.append(feat)
        total_len += cur_len
        
        if mode == 'train':
          label = torch.LongTensor(label_dict[fname])
          t_label.append(label)
    print(f'[INFO] {split} set')
    if mode == 'train':
        return t_feat,t_label,total_len
    else:
        return t_feat,total_len
