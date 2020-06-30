import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers import SpGraphAttentionLayer, ConvKB

CUDA = torch.cuda.is_available()  # checking cuda availability

# 支持sitiation
def get_situation_id(Corpus_, batch_inputs):
    head_situation = []
    tail_situation = []
    for i, tp in enumerate(batch_inputs):
        head_situation.append(Corpus_.entity2situation[Corpus_.id2entity[int(tp[0])]])
        tail_situation.append(Corpus_.entity2situation[Corpus_.id2entity[int(tp[2])]])

    situation_id = []
    for i, situation1 in enumerate(head_situation):
        situation2 = tail_situation[i]

        if situation1 == situation2 and situation1 in Corpus_.situation2id.keys():
            situation_id.append(Corpus_.situation2id[situation1])
        else:
            situation_id.append(len(Corpus_.situation2id))

    return situation_id

class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 relation_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = SpGraphAttentionLayer(num_nodes, nhid * nheads,
                                             nheads * nhid, nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             )

    def forward(self, Corpus_, batch_inputs, entity_embeddings, relation_embed,
                edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop):
        x = entity_embeddings

        # 相当于把两种关系加起来作为2-hop的关系？
        edge_embed_nhop = relation_embed[
            edge_type_nhop[:, 0]] + relation_embed[edge_type_nhop[:, 1]]

        # 多头操作
        x = torch.cat([att(x, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop)
                       for att in self.attentions], dim=1)
        x = self.dropout_layer(x)

        # 对 relationEmbeding 做转换
        out_relation_1 = relation_embed.mm(self.W)

        edge_embed = out_relation_1[edge_type]
        edge_embed_nhop = out_relation_1[
            edge_type_nhop[:, 0]] + out_relation_1[edge_type_nhop[:, 1]]

        # 最后一层不多头
        x = F.elu(self.out_att(x, edge_list, edge_embed,
                               edge_list_nhop, edge_embed_nhop))
        return x, out_relation_1


class SpKBGATModified(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, alpha, nheads_GAT):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.alpha = alpha      # For leaky relu

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.entity_embeddings = nn.Parameter(initial_entity_emb)
        self.relation_embeddings = nn.Parameter(initial_relation_emb)

        self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,
                                  self.drop_GAT, self.alpha, self.nheads_GAT_1)

        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

    def forward(self, Corpus_, adj, batch_inputs, train_indices_nhop):
        # getting edge list
        # 这里 adj_matrix = (adj_indices, adj_values)
        edge_list = adj[0]
        edge_type = adj[1]

        edge_list_nhop = torch.cat(
            (train_indices_nhop[:, 3].unsqueeze(-1), train_indices_nhop[:, 0].unsqueeze(-1)), dim=1).t()
        edge_type_nhop = torch.cat(
            [train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)

        if(CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()
            edge_list_nhop = edge_list_nhop.cuda()
            edge_type_nhop = edge_type_nhop.cuda()

        edge_embed = self.relation_embeddings[edge_type]

        start = time.time()

        # 对entity_embeddings.data进行L2正则化
        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()

        # self.relation_embeddings.data = F.normalize(
        #     self.relation_embeddings.data, p=2, dim=1)

        # 通过GAT得到新的实体和关系嵌入
        out_entity_1, out_relation_1 = self.sparse_gat_1(
            Corpus_, batch_inputs, self.entity_embeddings, self.relation_embeddings,
            edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop)

        # 新的实体嵌入和初始嵌入相加
        mask_indices = torch.unique(batch_inputs[:, 2]).cuda()
        mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()
        mask[mask_indices] = 1.0

        entities_upgraded = self.entity_embeddings.mm(self.W_entities)
        out_entity_1 = entities_upgraded + \
            mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1

        # 相加后做L2正则化
        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)

        # 更新参数
        self.final_entity_embeddings.data = out_entity_1.data
        self.final_relation_embeddings.data = out_relation_1.data

        return out_entity_1, out_relation_1


class SpKBGATConvOnly(nn.Module):
    def __init__(self, situation2id, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, drop_conv, alpha, alpha_conv, nheads_GAT, conv_out_channels):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.drop_conv = drop_conv
        self.alpha = alpha      # For leaky relu
        self.alpha_conv = alpha_conv
        self.conv_out_channels = conv_out_channels

        # 支持situations
        self.situation_num = len(situation2id)
        self.situation_embeddings = nn.Parameter(
            torch.randn(self.situation_num + 1, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        # 原本是3 加了situation变成4
        self.convKB = ConvKB(self.entity_out_dim_1 * self.nheads_GAT_1, 4, 1,
                             self.conv_out_channels, self.drop_conv, self.alpha_conv)

    def forward(self, Corpus_, adj, batch_inputs):
        situation_id = get_situation_id(Corpus_, batch_inputs)
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1), 
            self.situation_embeddings[situation_id].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv

    def batch_test(self, Corpus_,batch_inputs):
        situation_id = get_situation_id(Corpus_, batch_inputs)
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1), 
            self.situation_embeddings[situation_id].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv
