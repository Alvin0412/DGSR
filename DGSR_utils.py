#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/19 10:54
# @Author : ZM7
# @File : DGSR_utils
# @Software: PyCharm

import numpy as np
import argparse
import sys


def init_parser():
    parser = argparse.ArgumentParser()


    parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0005, help='l2 penalty')
    parser.add_argument('--tag_num_gnn_layers', default=2, help="Number of tag generating model layers")
    parser.add_argument('--tag_negative_slope', default=0.01, help='negative slope for leaky ReLU')
    parser.add_argument('--feat_drop', type=float, default=0.3, help='drop_out')
    parser.add_argument('--attn_drop', type=float, default=0.3, help='drop_out')
    parser.add_argument('--layer_num', type=int, default=3, help='GNN layer')
    """Above are the parameters that need to be tuned"""

    parser.add_argument('--data', default='Movies', help='data name')
    parser.add_argument('--batchSize', type=int, default=256, help='input batch size')  # 不需要调整
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden state size')  # 不需要调整
    # parser.add_argument('--hidden_size', type=int, default=300, help='hidden state size')
    parser.add_argument('--user_update', default='rnn')  # 不需要调整
    parser.add_argument('--item_update', default='rnn')  # 不需要调整
    parser.add_argument('--user_long', default='orgat')  # 不用调整
    parser.add_argument('--item_long', default='orgat')  # 不用调整
    parser.add_argument('--user_short', default='att')  # 不用调整
    parser.add_argument('--item_short', default='att')  # 不用调整
    parser.add_argument('--item_max_length', type=int, default=50, help='the max length of item sequence')  # 不用调整
    parser.add_argument('--user_max_length', type=int, default=50, help='the max length of use sequence')  # 不用调整
    parser.add_argument('--k_hop', type=int, default=2, help='sub-graph size')  # 不用调整

    parser.add_argument('--gpu', default='0')
    parser.add_argument('--patience', default='1') # 没啥耐心
    parser.add_argument('--last_item', action='store_true', help='aggreate last item')
    parser.add_argument("--record", action='store_true', default=True, help='record experimental results')
    parser.add_argument("--val", action='store_true', default=False)
    parser.add_argument("--model_record", action='store_true', default=True, help='record model')
    return parser


def eval_metric(all_top, random_rank=True):
    recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = [], [], [], [], [], []
    data_l = np.zeros((100, 7))
    for index in range(len(all_top)):
        prediction = (-all_top[index]).argsort(1).argsort(1)
        predictions = prediction[:, 0]
        for i, rank in enumerate(predictions):
            # data_l[per_length[i], 6] += 1
            if rank < 20:
                ndgg20.append(1 / np.log2(rank + 2))
                recall20.append(1)
            else:
                ndgg20.append(0)
                recall20.append(0)
            if rank < 10:
                ndgg10.append(1 / np.log2(rank + 2))
                recall10.append(1)
            else:
                ndgg10.append(0)
                recall10.append(0)
            if rank < 5:
                ndgg5.append(1 / np.log2(rank + 2))
                recall5.append(1)
            else:
                ndgg5.append(0)
                recall5.append(0)
    return np.mean(recall5), np.mean(recall10), np.mean(recall20), np.mean(ndgg5), np.mean(ndgg10), np.mean(ndgg20)


def mkdir_if_not_exist(file_name):
    import os
    import shutil

    dir_name = os.path.dirname(file_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


class Logger(object):
    """
    这个类的目的是尽可能不改变原始代码的情况下, 使得程序的输出同时打印在控制台和保存在文件中
    用法: 只需在程序中加入一行 `sys.stdout = Logger(log_file_path)` 即可
    """

    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
