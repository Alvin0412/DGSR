#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/17 4:32
# @Author : ZM7
# @File : new_main
# @Software: PyCharm

import datetime
import os
import pathlib
import copy
import pickle
import sys
import warnings
from functools import partial
from multiprocessing import freeze_support

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from dgl import load_graphs
from optree import functools
from torch.utils.data import DataLoader

from DGSR import DGSR, collate, collate_test
from DGSR_utils import eval_metric, mkdir_if_not_exist, Logger, init_parser
# from KG import KGAT_KG, KGDataRetriever
from Interim_models import KGDataRetriever, TaggingItems
from utils import myFloder


def collate_test_with_data_neg(x, data_neg):  # to avoid pickle error
    return collate_test(x, data_neg)


def toggle_cuda(x: torch.Tensor) -> torch.tensor:
    if torch.cuda.is_available():
        return x.cuda()
    return x


def load_dataset(opt):
    data = pd.read_csv('./Data/' + opt.data + '.csv')
    user = data['user_id'].unique()
    item = data['item_id'].unique()
    user_num = len(user)
    item_num = len(item)
    train_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/train/'
    test_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/test/'
    val_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/val/'
    train_set = myFloder(train_root, load_graphs)
    test_set = myFloder(test_root, load_graphs)
    val_set = None
    if opt.val:
        val_set = myFloder(val_root, load_graphs)
    print('train number:', train_set.size)
    print('test number:', test_set.size)
    print('user number:', user_num)
    print('item number:', item_num)
    return user_num, item_num, train_set, test_set, val_set


def create_data_loader(opt, train_set=None, test_set=None, val_set=None):
    f = open(opt.data + '_neg', 'rb')
    data_neg = pickle.load(f)  # 用于评估测试集

    train_data = DataLoader(dataset=train_set, batch_size=opt.batchSize, collate_fn=collate, shuffle=True,
                            pin_memory=True, num_workers=12)
    test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize,
                           collate_fn=partial(collate_test_with_data_neg, data_neg=data_neg), pin_memory=True,
                           num_workers=8)
    val_data = None
    if opt.val:
        val_data = DataLoader(dataset=val_set, batch_size=opt.batchSize,
                              collate_fn=partial(collate_test_with_data_neg, data_neg=data_neg), pin_memory=True,
                              num_workers=2)
    return data_neg, train_data, test_data, val_data


def train(opt):
    warnings.filterwarnings('ignore')

    # Mac support: Check if CUDA is available, otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU instead.")

    print(opt)

    if opt.record:
        log_file = f'results/{opt.data}_ba_{opt.batchSize}_G_{opt.gpu}_dim_{opt.hidden_size}_ulong_{opt.user_long}_ilong_{opt.item_long}_' \
                   f'US_{opt.user_short}_IS_{opt.item_short}_La_{opt.last_item}_UM_{opt.user_max_length}_IM_{opt.item_max_length}_K_{opt.k_hop}' \
                   f'_layer_{opt.layer_num}_l2_{opt.l2}'
        mkdir_if_not_exist(log_file)
        sys.stdout = Logger(log_file)
        print(f'Logging to {log_file}')
    if opt.model_record:
        model_file = f'{opt.data}_ba_{opt.batchSize}_G_{opt.gpu}_dim_{opt.hidden_size}_ulong_{opt.user_long}_ilong_{opt.item_long}_' \
                     f'US_{opt.user_short}_IS_{opt.item_short}_La_{opt.last_item}_UM_{opt.user_max_length}_IM_{opt.item_max_length}_K_{opt.k_hop}' \
                     f'_layer_{opt.layer_num}_l2_{opt.l2}_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'

    # loading data
    print(f"[{datetime.datetime.now()}] Start loading data...")
    user_num, item_num, train_set, test_set, val_set = load_dataset(opt)
    data_neg, train_data, test_data, val_data = create_data_loader(opt, train_set, test_set, val_set)
    print(f"[{datetime.datetime.now()}] All data loaded!")

    # 初始化模型
    model = DGSR(user_num=user_num, item_num=item_num, input_dim=opt.hidden_size, item_max_length=opt.item_max_length,
                 user_max_length=opt.user_max_length, feat_drop=opt.feat_drop, attn_drop=opt.attn_drop,
                 user_long=opt.user_long, user_short=opt.user_short,
                 item_long=opt.item_long, item_short=opt.item_short, user_update=opt.user_update,
                 item_update=opt.item_update, last_item=opt.last_item,
                 layer_num=opt.layer_num)
    kg_data_retriever = KGDataRetriever(n_users=user_num, n_items=item_num, data_name=opt.data)
    kg_model = TaggingItems(item_num=item_num, hidden_size=opt.hidden_size, tag_num_gnn_layers=opt.tag_num_gnn_layers,
                            feat_drop=opt.feat_drop, attn_drop=opt.attn_drop, tag_negative_slope=opt.tag_negative_slope,
                            tag_vocab=kg_data_retriever.tad_id_mapping,
                            )
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    kg_optimizer = optim.Adam(kg_model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    loss_func = nn.CrossEntropyLoss()
    best_result = [0, 0, 0, 0, 0, 0]  # hit5,hit10,hit20,mrr5,mrr10,mrr20
    best_epoch = [0, 0, 0, 0, 0, 0]
    stop_num = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch_loss = 0
    test_loss = 0
    patience = opt.patience  # 设置早停的耐心值
    best_test_loss = float('inf')  # 追踪验证集上的最佳loss
    epochs_no_improve = 0  # 未提升的epoch计数器
    for epoch in range(opt.epoch):
        stop = True
        epoch_loss = 0
        iter = 0
        print('start training: ', datetime.datetime.now())
        model.train()
        for user, batch_graph, label, last_item in train_data:
            # print(f"user: {user}")
            batch_graph = batch_graph.to(device)
            user = user.to(device)
            last_item = last_item.to(device)
            label = label.to(device)

            iter += 1
            kg_graph = kg_model.tag_item_graph_constructor(
                kg_data_retriever.kg_heterograph, batch_graph
            )
            upd_embd = kg_model(kg_graph, batch_graph.nodes['item'].data['item_id'])
            # print(f"Done generate tagging embedding: {datetime.datetime.now()}")
            score = model(
                batch_graph.to(device),
                user.to(device),
                last_item.to(device),
                is_training=True,
                tag_item_embedding=upd_embd
            )
            # print(f"loss - start {datetime.datetime.now()}")
            loss = loss_func(score, label.to(device))
            # print(f"loss - end {datetime.datetime.now()}")
            optimizer.zero_grad()
            kg_optimizer.zero_grad()
            # print(f"backprop - start {datetime.datetime.now()}")
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            #     loss.backward()
            loss.backward()
            # print(prof.key_averages().table(sort_by="cuda_time_total"))
            # print(f"backprop - end {datetime.datetime.now()}")
            optimizer.step()
            kg_optimizer.step()
            epoch_loss += loss.item()
            print('[Epoch #{}] Iter {}, loss {:.4f}'.format(epoch, iter, epoch_loss / iter),
                  datetime.datetime.now())
            # if iter % 400 == 0:
            #     print('[Epoch #{}] Iter {}, loss {:.4f}'.format(epoch, iter, epoch_loss / iter),
            #           datetime.datetime.now())
            # else:
            #     print('[Epoch #{}] Iter {}'.format(epoch, iter),
            #           datetime.datetime.now())
        epoch_loss /= iter
        model.eval()
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss), '=============================================')

        # val
        if opt.val:
            print('start validation: ', datetime.datetime.now())
            val_loss_all, top_val = [], []
            with torch.no_grad():
                for user, batch_graph, label, last_item, neg_tar in val_data:
                    score, top = model(batch_graph.to(device), user.to(device), last_item.to(device),
                                       neg_tar=torch.cat([label.unsqueeze(1), neg_tar], -1).to(device),
                                       is_training=False)
                    val_loss = loss_func(score, toggle_cuda(label))
                    val_loss_all.append(val_loss.append(val_loss.item()))
                    top_val.append(top.detach().cpu().numpy())
                recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(top_val)
                print('train_loss:%.4f\tval_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tNDGG@5:%.4f'
                      '\tNDGG10@10:%.4f\tNDGG@20:%.4f' %
                      (epoch_loss, np.mean(val_loss_all), recall5, recall10, recall20, ndgg5, ndgg10, ndgg20))

        # test
        print('start predicting: ', datetime.datetime.now())
        all_top, all_label, all_length = [], [], []
        iter = 0
        all_loss = []
        with torch.no_grad():
            for user, batch_graph, label, last_item, neg_tar in test_data:
                iter += 1
                score, top = model(batch_graph.to(device), user.to(device), last_item.to(device),
                                   neg_tar=torch.cat([label.unsqueeze(1), neg_tar], -1).to(device), is_training=False)

                test_loss = loss_func(score, toggle_cuda(label))
                all_loss.append(test_loss.item())
                all_top.append(top.detach().cpu().numpy())
                all_label.append(label.numpy())
                if iter % 100 == 0:
                    print('Iter {}, test_loss {:.4f}'.format(iter, np.mean(all_loss)), datetime.datetime.now())
            recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(all_top)
            try:
                mean_test_loss = np.mean(all_loss)
                if mean_test_loss < best_test_loss:
                    best_test_loss = mean_test_loss
                    epochs_no_improve = 0  # 如果测试集损失下降，重置计数器
                    stop = False
                else:
                    epochs_no_improve += 1
                    print(f'Epochs without improvement: {epochs_no_improve}')
                    if epochs_no_improve >= patience:
                        print(f'Early stopping at epoch {epoch} due to no improvement in test loss.')
                        break
            except Exception as e:
                print(f"Early stop mechanism went wrong because of {e}")
            if recall5 > best_result[0]:
                best_result[0] = recall5
                best_epoch[0] = epoch
                stop = False
            if recall10 > best_result[1]:
                if opt.model_record:
                    # save_dir = pathlib.Path('save_models/')
                    # save_dir.mkdir(parents=True, exist_ok=True)
                    # torch.save(model.state_dict(), save_dir / f'{model_file}.pkl')
                    save_dir = pathlib.Path('save_models/') / f'{model_file}'
                    save_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_dir / f'DGSR_param.pkl')
                    torch.save(kg_model.state_dict(), save_dir / f'tagging.pkl')
                best_result[1] = recall10
                best_epoch[1] = epoch
                stop = False
            if recall20 > best_result[2]:
                best_result[2] = recall20
                best_epoch[2] = epoch
                stop = False
                # ------select Mrr------------------
            if ndgg5 > best_result[3]:
                best_result[3] = ndgg5
                best_epoch[3] = epoch
                stop = False
            if ndgg10 > best_result[4]:
                best_result[4] = ndgg10
                best_epoch[4] = epoch
                stop = False
            if ndgg20 > best_result[5]:
                best_result[5] = ndgg20
                best_epoch[5] = epoch
                stop = False
            if stop:
                stop_num += 1
            else:
                stop_num = 0
            print('train_loss:%.4f\ttest_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tNDGG@5:%.4f'
                  '\tNDGG10@10:%.4f\tNDGG@20:%.4f\tEpoch:%d,%d,%d,%d,%d,%d' %
                  (epoch_loss, np.mean(all_loss), best_result[0], best_result[1], best_result[2], best_result[3],
                   best_result[4], best_result[5], best_epoch[0], best_epoch[1],
                   best_epoch[2], best_epoch[3], best_epoch[4], best_epoch[5]))
    # return {
    #     "train_loss": epoch_loss,
    #     "Recall@5": best_result[0],
    #     "Recall@10": best_result[1],
    #     "Recall@20": best_result[2],
    #     "NDGG@5": best_result[3],
    #     "NDGG@10": best_result[4],
    #     "NDGG@20": best_result[5],
    # }
    return (epoch_loss + test_loss) / 2


def save_hyperparameters(trial: optuna.Trial, opt, filename, loss=None):
    # 获取保存文件的路径
    save_dir = pathlib.Path(__file__).parent / "hyperparameters"
    save_dir.mkdir(parents=True, exist_ok=True)

    file_path = save_dir / filename

    # 将超参数转换为字典
    hyperparameters = {
        "trial_number": trial.number,
        "lr": opt.lr,
        "l2": opt.l2,
        "tag_num_gnn_layers": opt.tag_num_gnn_layers,
        "tag_negative_slope": opt.tag_negative_slope,
        "feat_drop": opt.feat_drop,
        "attn_drop": opt.attn_drop,
        "layer_num": opt.layer_num,
        "loss": loss
    }

    # 将字典转为 DataFrame
    df = pd.DataFrame([hyperparameters])

    # 检查文件是否已经存在，决定是新建文件还是追加
    if file_path.exists():
        # 追加到现有文件中，不写入列名
        df.to_csv(file_path, mode='a', header=False, index=False, float_format='%.6e')
    else:
        # 创建新文件，写入列名

        df.to_csv(file_path, mode='w', header=True, index=False, float_format='%.6e')


def bayesian_objective(trial: optuna.Trial, base_opt):
    # 复制原始的 opt，避免影响原始对象
    print(f"Trail: {trial.number}")
    opt = copy.deepcopy(base_opt)

    # 使用 trial 来动态调整超参数
    # opt.epoch = trial.suggest_int('epoch', 5, 50)  # 优化epoch的范围在5到50之间
    opt.lr = trial.suggest_float('lr', 5e-4, 5e-3, log=True)  # 学习率的对数均匀分布
    opt.l2 = trial.suggest_float('l2', 2e-4, 5e-3, log=True)  # l2正则化项的范围
    opt.tag_num_gnn_layers = trial.suggest_int('tag_num_gnn_layers', 2, 5)  # 标签生成模型层数
    opt.tag_negative_slope = trial.suggest_float('tag_negative_slope', 0.001, 0.1)  # Leaky ReLU 负斜率
    opt.feat_drop = trial.suggest_float('feat_drop', 0.1, 0.5)  # 特征dropout的范围
    opt.attn_drop = trial.suggest_float('attn_drop', 0.1, 0.5)  # 注意力dropout的范围
    opt.layer_num = trial.suggest_int('layer_num', 3, 5)  # GNN层数

    loss = train(opt)
    try:
        save_hyperparameters(trial, opt, filename=f"{trial.study.study_name}.csv", loss=loss)
    except Exception as e:
        print(f"Save parameters failed because of: {e}")
        print(f"hyperparameters(loss: {loss}): \n {opt.__dict__}")
    return loss


def main_logic(opt, tuning=False):
    if tuning:
        objective = functools.partial(bayesian_objective, base_opt=opt)
        name = f"hyperparameter_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        study = optuna.create_study(direction='minimize', study_name=name)
        study.optimize(objective, n_trials=25)
    else:
        train(opt)


def experiment(opt):
    """Load model parameters and """


if __name__ == '__main__':
    freeze_support()
    parser = init_parser()
    opt = parser.parse_args()
    args, extras = parser.parse_known_args()
    main_logic(opt, tuning=True)
