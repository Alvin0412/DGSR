import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import collections
import logging
import pathlib
import pandas as pd
import dgl.function as fn
from dgl.nn.pytorch import GraphConv, SAGEConv

ARGS = {
    'seed': 2019,
    'data_name': 'amazon-book',
    'data_dir': 'datasets/',
    'use_pretrain': 1,
    'pretrain_embedding_dir': 'datasets/pretrain/',
    'pretrain_model_path': 'trained_model/model.pth',
    'kg_batch_size': 2048,
    'test_batch_size': 10000,
    'embed_dim': 64,
    'relation_dim': 64,
    'laplacian_type': 'random-walk',
    'aggregation_type': 'bi-interaction',
    'conv_dim_list': '[64, 32, 16]',
    'mess_dropout': '[0.1, 0.1, 0.1]',
    'kg_l2loss_lambda': 1e-5,
    'lr': 0.0001,
    'n_epoch': 1000,
    'stopping_steps': 10,
    'kg_print_every': 1,
    'evaluate_every': 10,
    'Ks': '[20, 40, 60, 80, 100]'
}


# class BaseDataRetriever:
#
#     def __init__(self, args, logging):
#         self.args = args
#         self.data_name = args.data_name
#         self.use_pretrain = args.use_pretrain
#         self.pretrain_embedding_dir = args.pretrain_embedding_dir
#
#         self.data_dir = os.path.join(args.data_dir, args.data_name)
#         self.train_file = os.path.join(self.data_dir, 'train.txt')
#         self.test_file = os.path.join(self.data_dir, 'test.txt')
#         self.kg_file = os.path.join(self.data_dir, "kg_final.txt")
#
#         self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file)
#         self.cf_test_data, self.test_user_dict = self.load_cf(self.test_file)
#         self.statistic_cf()
#
#         if self.use_pretrain == 1:
#             self.load_pretrained_data()
#
#     def load_cf(self, filename):
#         user = []
#         item = []
#         user_dict = dict()
#
#         lines = open(filename, 'r').readlines()
#         for l in lines:
#             tmp = l.strip()
#             inter = [int(i) for i in tmp.split()]
#
#             if len(inter) > 1:
#                 user_id, item_ids = inter[0], inter[1:]
#                 item_ids = list(set(item_ids))
#
#                 for item_id in item_ids:
#                     user.append(user_id)
#                     item.append(item_id)
#                 user_dict[user_id] = item_ids
#
#         user = np.array(user, dtype=np.int32)
#         item = np.array(item, dtype=np.int32)
#         return (user, item), user_dict
#
#     def statistic_cf(self):
#         self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0])) + 1
#         self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1])) + 1
#         self.n_cf_train = len(self.cf_train_data[0])
#         self.n_cf_test = len(self.cf_test_data[0])
#
#     def load_kg(self, filename):
#         kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python')
#         kg_data = kg_data.drop_duplicates()
#         return kg_data
#
#     def sample_pos_items_for_u(self, user_dict, user_id, n_sample_pos_items):
#         pos_items = user_dict[user_id]
#         n_pos_items = len(pos_items)
#
#         sample_pos_items = []
#         while True:
#             if len(sample_pos_items) == n_sample_pos_items:
#                 break
#
#             pos_item_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
#             pos_item_id = pos_items[pos_item_idx]
#             if pos_item_id not in sample_pos_items:
#                 sample_pos_items.append(pos_item_id)
#         return sample_pos_items
#
#     def sample_neg_items_for_u(self, user_dict, user_id, n_sample_neg_items):
#         pos_items = user_dict[user_id]
#
#         sample_neg_items = []
#         while True:
#             if len(sample_neg_items) == n_sample_neg_items:
#                 break
#
#             neg_item_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
#             if neg_item_id not in pos_items and neg_item_id not in sample_neg_items:
#                 sample_neg_items.append(neg_item_id)
#         return sample_neg_items
#
#     def generate_cf_batch(self, user_dict, batch_size):
#         exist_users = user_dict.keys()
#         if batch_size <= len(exist_users):
#             batch_user = random.sample(exist_users, batch_size)
#         else:
#             batch_user = [random.choice(exist_users) for _ in range(batch_size)]
#
#         batch_pos_item, batch_neg_item = [], []
#         for u in batch_user:
#             batch_pos_item += self.sample_pos_items_for_u(user_dict, u, 1)
#             batch_neg_item += self.sample_neg_items_for_u(user_dict, u, 1)
#
#         batch_user = torch.LongTensor(batch_user)
#         batch_pos_item = torch.LongTensor(batch_pos_item)
#         batch_neg_item = torch.LongTensor(batch_neg_item)
#         return batch_user, batch_pos_item, batch_neg_item
#
#     def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
#         pos_triples = kg_dict[head]
#         n_pos_triples = len(pos_triples)
#
#         sample_relations, sample_pos_tails = [], []
#         while True:
#             if len(sample_relations) == n_sample_pos_triples:
#                 break
#
#             pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
#             tail = pos_triples[pos_triple_idx][0]
#             relation = pos_triples[pos_triple_idx][1]
#
#             if relation not in sample_relations and tail not in sample_pos_tails:
#                 sample_relations.append(relation)
#                 sample_pos_tails.append(tail)
#         return sample_relations, sample_pos_tails
#
#     def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx):
#         pos_triples = kg_dict[head]
#
#         sample_neg_tails = []
#         while True:
#             if len(sample_neg_tails) == n_sample_neg_triples:
#                 break
#
#             tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
#             if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
#                 sample_neg_tails.append(tail)
#         return sample_neg_tails
#
#     def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
#         exist_heads = kg_dict.keys()
#         if batch_size <= len(exist_heads):
#             batch_head = random.sample(exist_heads, batch_size)
#         else:
#             batch_head = [random.choice(exist_heads) for _ in range(batch_size)]
#
#         batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
#         for h in batch_head:
#             relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
#             batch_relation += relation
#             batch_pos_tail += pos_tail
#
#             neg_tail = self.sample_neg_triples_for_h(kg_dict, h, relation[0], 1, highest_neg_idx)
#             batch_neg_tail += neg_tail
#
#         batch_head = torch.LongTensor(batch_head)
#         batch_relation = torch.LongTensor(batch_relation)
#         batch_pos_tail = torch.LongTensor(batch_pos_tail)
#         batch_neg_tail = torch.LongTensor(batch_neg_tail)
#         return batch_head, batch_relation, batch_pos_tail, batch_neg_tail
#
#     def load_pretrained_data(self):
#         pre_model = 'mf'
#         pretrain_path = '%s/%s/%s.npz' % (self.pretrain_embedding_dir, self.data_name, pre_model)
#         pretrain_data = np.load(pretrain_path)
#         self.user_pre_embed = pretrain_data['user_embed']
#         self.item_pre_embed = pretrain_data['item_embed']
#
#         assert self.user_pre_embed.shape[0] == self.n_users
#         assert self.item_pre_embed.shape[0] == self.n_items
#         assert self.user_pre_embed.shape[1] == self.args.embed_dim
#         assert self.item_pre_embed.shape[1] == self.args.embed_dim

"""KGRT"""
# class KGDataRetriever:
#     def __init__(self, n_users: int, n_items: int, laplacian_type: str = "random-walk",
#                  data_name="Movies", data_path="./Data", kg_batch_size=2048, test_batch_size=10000):
#         # super().__init__(args, logging)
#         self.kg_batch_size = kg_batch_size
#         self.test_batch_size = test_batch_size
#         self.data_name = data_name
#         self.n_users = n_users
#         self.n_items = n_items
#
#         self.data_path = pathlib.Path(data_path)
#         if not self.data_path.exists():
#             raise FileNotFoundError(f"{self.data_path} does not exist!")
#         self.kg_file = self.ensure_kg_file()
#         kg_data: pd.DataFrame = self.load_kg(self.kg_file)
#         self.n_tags = kg_data['tag_id'].unique().shape[0]
#         self.kg_hetrograph = self.convert_to_dgl_heterograph(kg_data)
#
#         # self.construct_data(kg_data)
#         # self.print_info(logging)
#         #
#         # self.laplacian_type = laplacian_type
#         # self.adjacency_dict = self.create_adjacency_dict()
#         # self.laplacian_dict = self.create_laplacian_dict()
#         # self.A_in = self.convert_coo2tensor(sum(self.laplacian_dict.values()).tocoo())
#
#     def ensure_kg_file(self):
#         file = self.data_path / f"{self.data_name}_tags.csv"
#         if not file.exists():
#             raise FileNotFoundError(f"{self.data_name}_kg_final.txt does not exist!")
#         return file
#
#     @staticmethod
#     def load_kg(filename):
#         kg_data = pd.read_csv(filename, names=['item_id', 'tag_id'], engine='python', header=0)
#         kg_data = kg_data.drop_duplicates()
#         return kg_data
#
#     @staticmethod
#     def convert_to_dgl_heterograph(kg_data):
#         src = kg_data['item_id'].values
#         dst = kg_data['tag_id'].values
#
#         # 获取所有唯一的节点和关系类型
#         node_types = set(src).union(set(dst))
#
#         data_dict = {}
#         data_dict[('item', 'as', 'entity')] = (src, dst)
#
#         hetero_graph = dgl.heterograph(data_dict)
#         return hetero_graph
#
#     def construct_data(self, kg_data):
#         # add inverse kg data
#         n_relations = max(kg_data['r']) + 1
#         inverse_kg_data = kg_data.copy()
#         inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
#         inverse_kg_data['r'] += n_relations
#         kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)
#
#         # re-map user id
#         kg_data['r'] += 2
#         self.n_relations = max(kg_data['r']) + 1
#         self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
#         self.n_users_entities = self.n_users + self.n_entities
#
#         # 构建知识图谱训练数据
#         self.kg_train_data = kg_data
#         self.n_kg_train = len(self.kg_train_data)
#
#         # construct kg dict
#         h_list = []
#         t_list = []
#         r_list = []
#
#         self.train_kg_dict = collections.defaultdict(list)
#         self.train_relation_dict = collections.defaultdict(list)
#
#         for row in self.kg_train_data.iterrows():
#             h, r, t = row[1]
#             h_list.append(h)
#             t_list.append(t)
#             r_list.append(r)
#
#             self.train_kg_dict[h].append((t, r))
#             self.train_relation_dict[r].append((h, t))
#
#         self.h_list = torch.LongTensor(h_list)
#         self.t_list = torch.LongTensor(t_list)
#         self.r_list = torch.LongTensor(r_list)
#
#     def convert_coo2tensor(self, coo):
#         values = coo.data
#         indices = np.vstack((coo.row, coo.col))
#
#         i = torch.LongTensor(indices)
#         v = torch.FloatTensor(values)
#         shape = coo.shape
#         return torch.sparse.FloatTensor(i, v, torch.Size(shape))
#
#     def create_adjacency_dict(self):
#         adjacency_dict = {}
#         for r, ht_list in self.train_relation_dict.items():
#             rows = [e[0] for e in ht_list]
#             cols = [e[1] for e in ht_list]
#             vals = [1] * len(rows)
#             adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
#             adjacency_dict[r] = adj
#         return adjacency_dict
#
#     def create_laplacian_dict(self):
#         def symmetric_norm_lap(adj):
#             rowsum = np.array(adj.sum(axis=1))
#
#             d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#             d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
#             d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#
#             norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
#             return norm_adj.tocoo()
#
#         def random_walk_norm_lap(adj):
#             rowsum = np.array(adj.sum(axis=1))
#
#             d_inv = np.power(rowsum, -1.0).flatten()
#             d_inv[np.isinf(d_inv)] = 0
#             d_mat_inv = sp.diags(d_inv)
#
#             norm_adj = d_mat_inv.dot(adj)
#             return norm_adj.tocoo()
#
#         if self.laplacian_type == 'symmetric':
#             norm_lap_func = symmetric_norm_lap
#         elif self.laplacian_type == 'random-walk':
#             norm_lap_func = random_walk_norm_lap
#         else:
#             raise NotImplementedError
#
#         laplacian_dict = {}
#         for r, adj in self.adjacency_dict.items():
#             laplacian_dict[r] = norm_lap_func(adj)
#         return laplacian_dict
#
#     def print_info(self, logging):
#         logging.info('n_users:           %d' % self.n_users)
#         logging.info('n_items:           %d' % self.n_items)
#         logging.info('n_entities:        %d' % self.n_entities)
#         logging.info('n_users_entities:  %d' % self.n_users_entities)
#         logging.info('n_relations:       %d' % self.n_relations)
#
#         logging.info('n_h_list:          %d' % len(self.h_list))
#         logging.info('n_t_list:          %d' % len(self.t_list))
#         logging.info('n_r_list:          %d' % len(self.r_list))
#
#         logging.info('n_kg_train:        %d' % self.n_kg_train)


import torch
import torch.nn as nn
import torch.nn.functional as F


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Aggregator(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

        if self.aggregator_type == 'gcn':
            self.linear = nn.Linear(self.in_dim, self.out_dim)  # W in Equation (6)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'graphsage':
            self.linear = nn.Linear(self.in_dim * 2, self.out_dim)  # W in Equation (7)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'bi-interaction':
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)  # W1 in Equation (8)
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)  # W2 in Equation (8)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)

        else:
            raise NotImplementedError

    def forward(self, ego_embeddings, A_in):
        """
        ego_embeddings:  (n_users + n_entities, in_dim)
        A_in:            (n_users + n_entities, n_users + n_entities), torch.sparse.FloatTensor
        """
        # Equation (3)
        side_embeddings = torch.matmul(A_in, ego_embeddings)

        if self.aggregator_type == 'gcn':
            # Equation (6) & (9)
            embeddings = ego_embeddings + side_embeddings
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'graphsage':
            # Equation (7) & (9)
            embeddings = torch.cat([ego_embeddings, side_embeddings], dim=1)
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'bi-interaction':
            # Equation (8) & (9)
            sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
            bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))
            embeddings = bi_embeddings + sum_embeddings

        embeddings = self.message_dropout(embeddings)  # (n_users + n_entities, out_dim)
        return embeddings


class KGAT(nn.Module):
    def __init__(self, n_users, n_entities, n_relations, A_in=None, user_pre_embed=None, item_pre_embed=None,
                 embed_dim=64, relation_dim=64, aggregation_type='bi-interaction', conv_dim_list="[64, 32, 16]",
                 mess_dropout='[0.1, 0.1, 0.1]', kg_l2loss_lambda=1e-5):
        super(KGAT, self).__init__()

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = embed_dim
        self.relation_dim = relation_dim

        self.aggregation_type = aggregation_type
        self.conv_dim_list = [embed_dim] + eval(conv_dim_list)
        self.mess_dropout = eval(mess_dropout)
        self.n_layers = len(eval(conv_dim_list))

        self.kg_l2loss_lambda = kg_l2loss_lambda

        self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))

        if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.embed_dim))
            nn.init.xavier_uniform_(other_entity_embed)
            entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
            self.entity_user_embed.weight = nn.Parameter(entity_user_embed)
        else:
            nn.init.xavier_uniform_(self.entity_user_embed.weight)

        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.trans_M)

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(
                Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k],
                           self.aggregation_type))

        self.A_in = nn.Parameter(
            torch.sparse.FloatTensor(self.n_users + self.n_entities, self.n_users + self.n_entities))
        if A_in is not None:
            self.A_in.data = A_in
        self.A_in.requires_grad = False

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)  # (kg_batch_size, relation_dim)
        W_r = self.trans_M[r]  # (kg_batch_size, embed_dim, relation_dim)

        h_embed = self.entity_user_embed(h)  # (kg_batch_size, embed_dim)
        pos_t_embed = self.entity_user_embed(pos_t)  # (kg_batch_size, embed_dim)
        neg_t_embed = self.entity_user_embed(neg_t)  # (kg_batch_size, embed_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)  # (kg_batch_size)

        # Equation (2)
        # kg_loss = F.softplus(pos_score - neg_score)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(
            r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def update_attention_batch(self, h_list, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx]
        W_r = self.trans_M[r_idx]

        h_embed = self.entity_user_embed.weight[h_list]
        t_embed = self.entity_user_embed.weight[t_list]

        # Equation (4)
        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)
        return v_list

    def update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        # Equation (5)
        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)

    def forward(self, *input, mode):
        if mode == 'train_kg':
            return self.calc_kg_loss(*input)
        if mode == 'update_att':
            return self.update_attention(*input)


class KGAT_KG(nn.Module):
    def __init__(self, n_item, n_tag, item_embedding: nn.Embedding, hidden_size,
                 ):
        super(KGAT_KG, self).__init__()
        self.n_item = n_item
        self.n_tag = n_tag
        self.hidden_size = hidden_size
        self.n_output = n_item

        self.item_embedding = item_embedding
        self.tag_embedding = nn.Embedding(self.n_tag, self.hidden_size)

        # 定义图卷积层
        self.sage_item_tag = SAGEConv(self.hidden_size, self.hidden_size, 'mean')
        self.final = nn.Sequential(
            nn.Linear(self.hidden_size, self.n_output, bias=False),
            nn.Tanh()
        )
        self.reset_parameters()

    @classmethod
    def kg_graph_constructor(cls, item_tag_g: dgl.DGLHeteroGraph, user_item_subg: dgl.DGLHeteroGraph):
        """获得当前batch的图结构"""
        # 获取 user_item_subg 中的 item_id
        user_item_ids = user_item_subg.nodes('item')

        # 筛选 item_tag_g 中的 item_id 在 user_item_subg 中存在的部分
        item_tag_g_filtered = dgl.node_subgraph(item_tag_g, {'item': user_item_ids})

        # 合并两个子图
        combined_g = dgl.compact_graphs([user_item_subg, item_tag_g_filtered])

        return combined_g

    def message_propagation(self, g):
        """在图上进行消息传递"""
        with g.local_scope():
            # 项目-标签关系的消息传递
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'), etype=('item', 'tagged_as', 'tag'))
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'), etype=('tag', 'tagged_as_by', 'item'))

            # 使用图卷积层更新项目的嵌入
            g.nodes['item'].data['h'] = self.sage_item_tag(g, g.nodes['item'].data['h'])
            g.nodes['tag'].data['h'] = self.sage_item_tag(g, g.nodes['tag'].data['h'])

            return g

    def forward(self, g):
        # 初始化嵌入
        g.nodes['item'].data['h'] = self.item_embedding(g.nodes['item'].data['item_id'])
        g.nodes['tag'].data['h'] = self.tag_embedding(g.nodes['tag'].data['tag_id'])

        # 进行消息传递和聚合
        g = self.message_propagation(g)

        # 获得更新后的项目嵌入
        updated_item_emb = self.final(g.nodes['item'].data['h'])
        return updated_item_emb

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight, gain=gain)


if __name__ == '__main__':
    retriever = KGDataRetriever(logging, n_users=31013, n_items=23715, )
    ...
