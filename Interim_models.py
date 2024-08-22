import pathlib

import dgl
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv, HeteroGraphConv
from gensim.models import KeyedVectors


class CustomSAGEConv(nn.Module):
    def __init__(self, in_feats, out_feats, aggregator_type='lstm', feat_drop=0.2, attn_drop=0.2, negative_slope=0.2,
                 use_attention=False):
        super(CustomSAGEConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.use_attention = use_attention

        # LSTM 聚合器初始化
        self.lstm = nn.LSTM(in_feats, out_feats, batch_first=True)

        # 基础的 SAGEConv 初始化
        self.sageconv = SAGEConv(in_feats, out_feats, aggregator_type=aggregator_type)

        # Attention 机制相关的参数（仅在 use_attention 为 True 时使用）
        if self.use_attention:
            self.attn_fc = nn.Linear(2 * out_feats, 1, bias=False)  # 用于计算 attention 的线性变换
            self.leaky_relu = nn.LeakyReLU(negative_slope)

        # Dropout
        self.feat_drop = nn.Dropout(feat_drop)
        if self.use_attention:
            self.attn_drop = nn.Dropout(attn_drop)

    def edge_attention(self, edges):
        # 边的 attention 计算：对邻居和中心节点的特征拼接，计算注意力分数
        z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        a = self.attn_fc(z2)
        return {'e': self.leaky_relu(a)}

    def message_func(self, edges):
        # 在使用 attention 时，传递带有权重的消息
        if self.use_attention:
            return {'z': edges.src['h'], 'e': edges.data['alpha']}
        else:
            return {'z': edges.src['h']}

    def reduce_func(self, nodes):
        if self.use_attention:
            # 使用 attention 权重聚合邻居节点的消息
            alpha = F.softmax(nodes.mailbox['e'], dim=1)
            alpha = self.attn_drop(alpha)
            h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        else:
            # 使用 LSTM 聚合邻居节点的消息
            h_neigh = nodes.mailbox['z']  # shape: [batch_size, num_neighbors, feature_size]
            batch_size = h_neigh.size(0)
            num_neighbors = h_neigh.size(1)

            # LSTM 需要的输入为 [batch_size, num_neighbors, feature_size]，保证输入的维度是正确的
            h_lstm, (h_final, _) = self.lstm(h_neigh)
            h = h_final[-1]  # 最终的 LSTM 输出

        return {'h': h}

    def forward(self, graph, inputs):
        with graph.local_scope():
            # SAGEConv 的基础消息传递和聚合
            h = self.sageconv(graph, inputs)
            graph.ndata['h'] = h

            if self.use_attention:
                # 计算 attention score 并使用自定义的消息传递与聚合
                graph.apply_edges(self.edge_attention)
                graph.update_all(self.message_func, self.reduce_func)
                return graph.ndata.pop('h')
            else:
                # 如果不使用 attention，则直接使用 LSTM 聚合后的结果
                graph.update_all(self.message_func, self.reduce_func)
                return graph.ndata.pop('h')


class Word2VecEmbedding(nn.Module):
    def __init__(self, vocab_size,
                 embedding_dim,
                 word2vec_model_path: pathlib.Path,
                 tag_vocab,
                 extract_word2vec_embedding=False):
        super(Word2VecEmbedding, self).__init__()
        self.word2vec_model_path = word2vec_model_path
        embedding_matrix = torch.randn(vocab_size, embedding_dim)

        if not extract_word2vec_embedding:
            if not self.word2vec_model_path.exists():
                raise FileNotFoundError(f"Word2Vec model not found at {word2vec_model_path}")
            print(f"Loading Word2Vec model from {self.word2vec_model_path}")
            self.word2vec = KeyedVectors.load_word2vec_format(str(self.word2vec_model_path), binary=True)
            for i, tag in tag_vocab.items():
                if tag in self.word2vec:
                    embedding_matrix[i] = torch.tensor(self.word2vec[tag])
            self.save_embeddings(tag_vocab, embedding_matrix, word2vec_model_path.parent)
        else:
            print(f"Loading Word2Vec model from the cached word embedding file of {self.word2vec_model_path}")
            load_bag, load_embd = self.load_embeddings_from_csv(
                word2vec_model_path.parent / f"{self.word2vec_model_path.stem}_tag_embeddings.csv")
            for i, tag in tag_vocab.items():
                if tag in load_bag:
                    embedding_matrix[i] = torch.tensor(load_embd[i])

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)  # freeze=False 表示嵌入是可学习的

    @staticmethod
    def load_embeddings_from_csv(csv_path):
        df = pd.read_csv(csv_path)

        tags = df['tags'].tolist()  # 标签列
        embeddings = df.drop(columns=['tags']).values  # 嵌入向量部分

        tag_vocab = {tag: idx for idx, tag in enumerate(tags)}

        # 将嵌入矩阵转换为 torch.Tensor
        embedding_matrix = torch.tensor(embeddings, dtype=torch.float)

        return tag_vocab, embedding_matrix

    def save_embeddings(self, tag_vocab, embedding_matrix, save_dir):
        embeddings = []
        header = ['tags']
        for i in range(embedding_matrix.shape[1]):
            header.append(f"col_{i}")
        for i, tag in tag_vocab.items():
            vector = embedding_matrix[i].detach().numpy()
            embeddings.append([tag] + vector.tolist())

        save_path = save_dir / f"{self.word2vec_model_path.stem}_tag_embeddings.csv"

        df = pd.DataFrame(embeddings, columns=header)
        df.to_csv(save_path, index=False, header=True)
        print(f"Embeddings saved to {save_path}")

    def forward(self, tag_ids):
        return self.embedding(tag_ids)


class KGDataRetriever:
    def __init__(self, n_users: int, n_items: int, laplacian_type: str = "random-walk",
                 data_name: str = "Movies", data_path: str = "./Data",
                 kg_batch_size: int = 2048, test_batch_size: int = 10000):
        """
        Initialize the KGDataRetriever class.
        """
        self.kg_batch_size = kg_batch_size
        self.test_batch_size = test_batch_size
        self.data_name = data_name
        self.n_users = n_users
        self.n_items = n_items

        self.data_path = pathlib.Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path {self.data_path} does not exist!")

        self.kg_file = self.ensure_kg_file()
        self.kg_data = self.load_kg(self.kg_file)
        self.n_tags = self.kg_data['tag_id'].nunique()
        self.tad_id_mapping = self.load_mapping()
        self.kg_heterograph = self.convert_to_dgl_heterograph(self.kg_data)

    def load_mapping(self) -> dict[int, str]:
        file = self.data_path / f"{self.data_name}_tags_name.csv"
        if not file.exists():
            raise FileNotFoundError(f"Knowledge graph file {file} does not exist!")

        data = pd.read_csv(file)
        mapping = data.set_index('tag_id')['tag'].to_dict()

        return mapping

    def ensure_kg_file(self) -> pathlib.Path:
        """
        Ensures that the knowledge graph file exists.
        """
        file = self.data_path / f"{self.data_name}_tags.csv"
        if not file.exists():
            raise FileNotFoundError(f"Knowledge graph file {file} does not exist!")
        return file

    @staticmethod
    def load_kg(filename: pathlib.Path) -> pd.DataFrame:
        """
        Load knowledge graph data from a CSV file and remove duplicates.
        """
        kg_data: pd.DataFrame = pd.read_csv(filename, names=['item_id', 'tag_id'], engine='python', header=0)
        kg_data = kg_data.drop_duplicates()
        return kg_data

    @staticmethod
    def convert_to_dgl_heterograph(kg_data: pd.DataFrame) -> dgl.DGLHeteroGraph:
        """
        Convert the knowledge graph data to a DGL bipartite heterograph.
        """
        src = kg_data['item_id'].values
        dst = kg_data['tag_id'].values

        data_dict = {
            ('item', 'as', 'tag'): (src, dst),
            ('tag', 'ras', 'item'): (dst, src),
        }
        hetero_graph = dgl.heterograph(data_dict)
        hetero_graph.nodes["item"].data['item_id'] = torch.LongTensor(np.unique(src))
        hetero_graph.nodes["tag"].data['tag_id'] = torch.LongTensor(np.unique(dst))
        return hetero_graph


class TaggingItems(torch.nn.Module):
    ...
    """
    Model purpose: Generate item embeddings that can give content-based features to a collaborative model 
    
    Implementation steps
    0. Initialization method
    1. build a item-tag graph from subgraph of user-item bipartite graph
    2. Determine input&output, layers and embeddings
    3. Message propagation and aggregation
    4. Weighted integration of embeddings & set up the back prop rules
    """

    def __init__(self, item_num, tag_vocab: dict,
                 num_gnn_layers=4, feat_drop=0.2, attn_drop=0.2, negative_slope=0.01, hidden_size=300,
                 word2vec_model_path: pathlib.Path | None = pathlib.Path(
                     __file__).parent / "pretrained" / "GoogleNews-vectors-negative300.bin"):
        super(TaggingItems, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        # Item embedding 和 Tag embedding 的初始化
        self.hidden_size = hidden_size
        self.n_output = self.hidden_size  # TODO: refine it
        self.item_embedding = nn.Embedding(item_num, hidden_size).to(device)

        self.tag_vocab = tag_vocab
        self.tag_num = len(self.tag_vocab.keys())
        self._use_tag_pretrain = False
        if word2vec_model_path is not None and word2vec_model_path.exists():
            # TODO: investigate that whether it will disappear
            print("Using word2vec pretrained model from: ", word2vec_model_path)
            self.tag_embedding = Word2VecEmbedding(self.tag_num, hidden_size, word2vec_model_path, tag_vocab,
                                                   extract_word2vec_embedding=True).to(device)
            self._use_tag_pretrain = True
        else:
            print("Using default tag embedding")
            self.tag_embedding = nn.Embedding(self.tag_num, hidden_size).to(device)
        # GNN 层的初始化 (GraphSAGE 或 GCN)
        # self.gnn_layers = nn.ModuleList(  # TODO: Add custom GNN option
        #     [SAGEConv(hidden_size, hidden_size, aggregator_type="lstm") for _ in range(num_gnn_layers)])

        self.gnn_layers = nn.ModuleList().to(device)
        for _ in range(num_gnn_layers):
            conv_layer = HeteroGraphConv(
                {
                    'as': SAGEConv(hidden_size, hidden_size, aggregator_type="lstm").to(device),
                    'ras': SAGEConv(hidden_size, hidden_size, aggregator_type="lstm").to(device),
                },
                aggregate='sum'
            ).to(device)
            self.gnn_layers.append(conv_layer)

        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_size).to(device) for _ in range(num_gnn_layers)])

        # 其他相关参数
        self.feat_drop = nn.Dropout(feat_drop).to(device)
        self.attn_drop = nn.Dropout(attn_drop).to(device)

        self.final = nn.Sequential(
            nn.Linear(self.hidden_size, self.n_output, bias=False),
            nn.LeakyReLU(negative_slope)
        ).to(device)

        self.negative_slope = negative_slope
        self.reset_parameters()

    def tag_item_graph_constructor(self, item_tag_g: dgl.DGLHeteroGraph, user_item_subg: dgl.DGLHeteroGraph):
        """通过用户-物品交互图中的物品节点，采样物品-标签关系子图"""
        # user_item_ids = user_item_subg.nodes['item'].data['_ID'].unique()
        # item_tag_g_filtered = dgl.node_subgraph(item_tag_g, {
        #     'item': user_item_ids
        # })
        #
        # return item_tag_g_filtered
        # Get unique item IDs from the user-item interaction subgraph
        user_item_ids = user_item_subg.nodes['item'].data['_ID'].unique().to(self.device)

        # Get the edges where the source node is an item and find connected tags
        item_tag_edges = item_tag_g.edges(etype='as')  # Assuming edge type is 'item-tag'

        # Find the tags connected to the filtered items
        mask = torch.isin(item_tag_edges[0].to(self.device), user_item_ids.to(self.device)).to(self.device)
        filtered_item_ids = item_tag_edges[0][mask].to(self.device)
        connected_tag_ids = item_tag_edges[1][mask].to(self.device)

        # Create a new subgraph with both the filtered item and tag nodes
        item_tag_g_filtered = dgl.node_subgraph(item_tag_g, {
            'item': filtered_item_ids.unique(),
            'tag': connected_tag_ids.unique()
        }).to(self.device)

        return item_tag_g_filtered

    # def forward(self, item_tag_graph, items):
    #     item_embed = self.item_embedding(items.unique())  # 从 embedding 层获取物品 embedding
    #     tag_embed = self.tag_embedding(item_tag_graph.nodes('tag').unique())  # 获取标签 embedding
    #
    #     h = {'item': item_embed, 'tag': tag_embed}
    #     for i, gnn_layer in enumerate(self.gnn_layers):
    #         h = gnn_layer(item_tag_graph, h)
    #
    #         h['item'] = self.norm_layers[i](item_embed)
    #         h['item'] = self.feat_drop(item_embed)
    #
    #     final_item_embed = self.final(h['item'])
    #
    #     return final_item_embed

    def forward(self, item_tag_graph, items):
        items = items.to(self.device)
        unique_items, inverse_indices = torch.unique(items,
                                                     return_inverse=True)
        item_embed = self.item_embedding(unique_items)
        tag_embed = self.tag_embedding(item_tag_graph.nodes('tag').unique())

        h = {'item': item_embed, 'tag': tag_embed}

        for i, gnn_layer in enumerate(self.gnn_layers):
            h = gnn_layer(item_tag_graph, h)

            h['item'] = self.norm_layers[i](h['item'])
            h['item'] = self.feat_drop(h['item'])

        final_item_embed = self.final(h['item'])
        reconstructed_item_embed = final_item_embed[inverse_indices]

        return reconstructed_item_embed

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', param=self.negative_slope)

        for name, weight in self.named_parameters():
            if not ('tag_embedding' in name and self._use_tag_pretrain):
                if len(weight.shape) > 1:
                    nn.init.xavier_normal_(weight, gain=gain)


if __name__ == '__main__':
    user_item_data = pd.read_csv('./Data/' + "Movies" + '.csv')
    users = user_item_data['user_id'].unique()
    items = user_item_data['item_id'].unique()
    kg_retriever = KGDataRetriever(
        len(users), len(items), data_name="Movies"
    )
    tagging_model = TaggingItems(item_num=len(items),
                                 tag_vocab=kg_retriever.tad_id_mapping
                                 )
