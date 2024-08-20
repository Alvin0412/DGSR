#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/17 4:32
# @Author : ZM7
# @File : new_main
# @Software: PyCharm

import datetime
import pathlib
import pathlib
from multiprocessing import freeze_support

import torch
from sys import exit
import pandas as pd
import numpy as np
from DGSR import DGSR, collate, collate_test
from dgl import load_graphs
import pickle
from utils import myFloder
import warnings
import argparse
import os
import sys
import functools
from functools import partial
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from DGSR_utils import eval_metric, mkdir_if_not_exist, Logger, init_parser


def collate_test_with_data_neg(x, data_neg):  # to avoid pickle error
    return collate_test(x, data_neg)


def toggle_cuda(x: torch.Tensor) -> torch.tensor:
    if torch.cuda.is_available():
        return x.cuda()
    return x


def model_test():
    parser = init_parser()
    opt = parser.parse_args()
    args, extras = parser.parse_known_args()
    # Mac support: Check if CUDA is available, otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps')
    #     print("CUDA is not available. Using Metal instead.")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU instead.")

    print(opt)
    data = pd.read_csv('./Data/' + opt.data + '.csv')
    user = data['user_id'].unique()
    item = data['item_id'].unique()
    user_num = len(user)
    item_num = len(item)
    test_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/test/'
    test_set = myFloder(test_root, load_graphs)
    f = open(opt.data + '_neg', 'rb')
    data_neg = pickle.load(f)  # 用于评估测试集
    test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize,
                           collate_fn=partial(collate_test_with_data_neg, data_neg=data_neg), pin_memory=True,
                           num_workers=8)
    model = DGSR(user_num=user_num, item_num=item_num, input_dim=opt.hidden_size, item_max_length=opt.item_max_length,
                 user_max_length=opt.user_max_length, feat_drop=opt.feat_drop, attn_drop=opt.attn_drop,
                 user_long=opt.user_long, user_short=opt.user_short,
                 item_long=opt.item_long, item_short=opt.item_short, user_update=opt.user_update,
                 item_update=opt.item_update, last_item=opt.last_item,
                 layer_num=opt.layer_num, require_recommendation=True)
    model_file = f'{opt.data}_ba_{opt.batchSize}_G_{opt.gpu}_dim_{opt.hidden_size}_ulong_{opt.user_long}_ilong_{opt.item_long}_' \
                 f'US_{opt.user_short}_IS_{opt.item_short}_La_{args.last_item}_UM_{opt.user_max_length}_IM_{opt.item_max_length}_K_{opt.k_hop}' \
                 f'_layer_{opt.layer_num}_l2_{opt.l2}'
    model_file = pathlib.Path().cwd() / "save_models" / f"{model_file}.pkl"
    parameters = torch.load(model_file)
    model.load_state_dict(parameters)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    loss_func = nn.CrossEntropyLoss()
    best_result = [0, 0, 0, 0, 0, 0]  # hit5,hit10,hit20,mrr5,mrr10,mrr20
    best_epoch = [0, 0, 0, 0, 0, 0]
    stop_num = 0
    for epoch in range(opt.epoch):
        print('start predicting: ', datetime.datetime.now())
        all_top, all_label, all_length = [], [], []
        iter = 0
        all_loss = []
        with torch.no_grad():
            for user, batch_graph, label, last_item, neg_tar in test_data:
                iter += 1
                score, top, recommendation = model(batch_graph.to(device), user.to(device), last_item.to(device),
                                   neg_tar=torch.cat([label.unsqueeze(1), neg_tar], -1).to(device), is_training=False)

                test_loss = loss_func(score, toggle_cuda(label))
                all_loss.append(test_loss.item())
                all_top.append(top.detach().cpu().numpy())
                all_label.append(label.numpy())
                if iter % 100 == 0:
                    print('Iter {}, test_loss {:.4f}'.format(iter, np.mean(all_loss)), datetime.datetime.now())
                    print(f"Recommendation: {recommendation}")


if __name__ == '__main__':
    freeze_support()
    model_test()