import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.autograd import Variable


CUDA = torch.cuda.is_available()


class ConvKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super().__init__()

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((input_dim) * out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):
    
        batch_size, length, dim = conv_input.size()
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)
        print(conv_input.shape)
        out_conv = self.dropout(
            self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output


class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        # assert indices.requires_grad == False
        # torch.sparse_coo_tensor(indices, values, size=None）-> tensor
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(CUDA):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
            # print("Grad Outputs-> ", grad_output)
            # print("Grad values-> ", grad_values)
        return None, grad_values, None, None, None


class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, num_nodes, in_features, out_features, nrela_dim, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat
        self.nrela_dim = nrela_dim

        self.a = nn.Parameter(torch.zeros(
            size=(out_features, 2 * in_features + nrela_dim)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.a_2 = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a_2.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()

    # 这里输入的是图的全量edge，为了做attention
    def forward(self, input, edge, edge_embed, edge_list_nhop, edge_embed_nhop):
        # N为图的 node数
        N = input.size()[0]

        # Self-attention on the nodes - Shared attention mechanism
        edge = torch.cat((edge[:, :], edge_list_nhop[:, :]), dim=1)
        edge_embed = torch.cat(
            (edge_embed[:, :], edge_embed_nhop[:, :]), dim=0)

        # edge_h 其实就是三元组的嵌入
        edge_h = torch.cat(
            (input[edge[0, :], :], input[edge[1, :], :], edge_embed[:, :]), dim=1).t()
        # edge_h: (2*in_dim + nrela_dim) x E

        # 对三元组嵌入做非线性变换
        edge_m = self.a.mm(edge_h)
        # edge_m: D * E

        # to be checked later
        # 得到注意力系数
        powers = -self.leakyrelu(self.a_2.mm(edge_m).squeeze())
        edge_e = torch.exp(powers).unsqueeze(1)
        assert not torch.isnan(edge_e).any()
        # edge_e: E （E 应该是edge的总数）

        # 计算出每个节点相关的注意力系数的和 shape = （N，1）
        e_rowsum = self.special_spmm_final(
            edge, edge_e, N, edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        e_rowsum = e_rowsum
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        # edge_e: E ，即随机把部分注意力系数设置为0

        # 分配权重给每个三元组嵌入
        edge_w = (edge_e * edge_m).t()
        # edge_w: E * D

        # 计算出三元组的新嵌入
        h_prime = self.special_spmm_final(
            edge, edge_w, N, edge_w.shape[0], self.out_features)

        assert not torch.isnan(h_prime).any()
        # h_prime: N x out， 做归一化
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out

        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
