#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/1/31 11:15
# @Author : ZM7
# @File : generate_neg
# @Software: PyCharm
import pandas as pd
import pickle
import argparse
from utils import myFloder, pickle_loader, collate, trans_to_cuda, \
    eval_metric, collate_test, user_neg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Games',
                        help='dataset name: Games')
    opt = parser.parse_args()
    dataset = opt.dataset
    data = pd.read_csv('./Data/' + dataset + '.csv')
    user = data['user_id'].unique()
    item = data['item_id'].unique()
    user_num = len(user)
    item_num = len(item)
    data_neg = user_neg(data, item_num)

    f = open(dataset + '_neg', 'wb')
    pickle.dump(data_neg, f)
    f.close()
