import torch

from models import SpKBGATModified, SpKBGATConvOnly
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy

from preprocess import read_entity_from_id, read_relation_from_id, init_embeddings, build_data
from create_batch import Corpus
from utils import save_model

import random
import argparse
import os
import sys
import logging
import time
import pickle

# %%
# %%from torchviz import make_dot, make_dot_from_trace
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-data", "--data",
                      default="/home/devsdb/MovieGraphs/ych_result/MG_DATA_ArcFace_Complete_with_Situ/", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=1500, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=120, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=5e-6, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=True, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=50, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=True)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=True)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)
    args.add_argument("-outfolder", "--output_folder",
                      default="/home/devsdb/MovieGraphs/ych_result/MG_RESULT_C_SITUATION/", help="Folder name to save the models.")

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=50000, help="Batch size for GAT")
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float, 
                      default=0.3, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[100, 200], help="Entity output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[2, 2], help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float,
                      default=5, help="Margin used in hinge loss")

    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=128, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=40,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=500,
                      help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.0, help="Dropout probability for convolution layer")

    args = args.parse_args()
    return args


args = parse_args()
# %%
relation_num = 9

def load_data(args):
    train_data,test_data, entity2id, headTailSelector, unique_entities_train, entity2situation, situation2id= build_data(
        args.data, is_unweigted=False, directed=True, relation_num = 9)

    if args.pretrained_emb:
        entity_embeddings = init_embeddings(os.path.join(args.data, 'features.txt'))
        print("Initialised entities from Resnet")

    else:
        entity_embeddings = np.random.randn(
            len(entity2id), args.embedding_size)
        print("Initialised entities randomly")

    # 随机初始化关系嵌入    
    relation_embeddings = np.random.randn(relation_num, args.embedding_size)
    

    corpus = Corpus(args, train_data, test_data, entity2id, headTailSelector, entity2situation, situation2id, 
                    args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, args.get_2hop)

    return corpus, torch.FloatTensor(entity_embeddings), torch.FloatTensor(relation_embeddings)


Corpus_, entity_embeddings, relation_embeddings = load_data(args)

# 保存
if(args.get_2hop):
    file = args.data + "/2hop.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(Corpus_.node_neighbors_2hop, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


if(args.use_2hop):
    print("Opening node_neighbors pickle object")
    file = args.data + "/2hop.pickle"
    with open(file, 'rb') as handle:
        node_neighbors_2hop = pickle.load(handle)

# deepcopy https://blog.csdn.net/u010712012/article/details/79754132
entity_embeddings_copied = deepcopy(entity_embeddings)
relation_embeddings_copied = deepcopy(relation_embeddings)

print("Initial entity dimensions {} , relation dimensions {}".format(
    entity_embeddings.size(), relation_embeddings.size()))
# %%

CUDA = torch.cuda.is_available()


def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed):
    len_pos_triples = int(
        train_indices.shape[0] / (int(args.valid_invalid_ratio_gat) + 1))

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio_gat), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    # 对于pos的三元组， x越接近0 越好
    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=1, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    # 对于neg的三元组， x越大越好
    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=1, dim=1)

    y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples).cuda()

    # 这里如果pos_norm 对应位置比 neg_norm 越小，loss会越低
    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss


def train_gat(args):

    # Creating the gat model here.
    ####################################

    print("Defining model")

    print(
        "\nModel type -> GAT layer with {} heads used , Initital Embeddings training".format(args.nheads_GAT[0]))
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT)

    if CUDA:
        model_gat.cuda()

    optimizer = torch.optim.Adam(
        model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)

    # schaduler用于动态调整学习率
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.5, last_epoch=-1)

    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)

    current_batch_2hop_indices = torch.tensor([])
    if(args.use_2hop):
        current_batch_2hop_indices = Corpus_.get_batch_nhop_neighbors_all(args,
                                                                          Corpus_.unique_entities_train, node_neighbors_2hop)

    if CUDA:
        current_batch_2hop_indices = Variable(
            torch.LongTensor(current_batch_2hop_indices)).cuda()
    else:
        current_batch_2hop_indices = Variable(
            torch.LongTensor(current_batch_2hop_indices))

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs_gat))

    for epoch in range(args.epochs_gat):
        print("\nepoch-> ", epoch)
        # 随机排序
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_gat.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(Corpus_.train_indices) % args.batch_size_gat == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_size_gat
        else:
            num_iters_per_epoch = (
                len(Corpus_.train_indices) // args.batch_size_gat) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            # 这里的train_indices and values 已经造出了neg三元组
            train_indices, train_values = Corpus_.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            # forward pass
            entity_embed, relation_embed = model_gat(
                Corpus_, Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices)

            optimizer.zero_grad()

            loss = batch_gat_loss(
                gat_loss_func, train_indices, entity_embed, relation_embed)

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item()))

        scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        
        if epoch > args.epochs_gat - 20:
            save_model(model_gat, args.data, epoch, args.output_folder)


def train_conv(args):

    # Creating convolution model here.
    ####################################

    print("Defining model")
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT)
    print("Only Conv model trained")
    model_conv = SpKBGATConvOnly(Corpus_.situation2id, entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)

    if CUDA:
        model_conv.cuda()
        model_gat.cuda()

    model_gat.load_state_dict(torch.load(
        '{}/trained_{}.pth'.format(args.output_folder, args.epochs_gat - 1)))
    
    # todo：这个地方有意思
    model_conv.final_entity_embeddings = model_gat.final_entity_embeddings
    model_conv.final_relation_embeddings = model_gat.final_relation_embeddings

    Corpus_.batch_size = args.batch_size_conv
    Corpus_.invalid_valid_ratio = int(args.valid_invalid_ratio_conv)

    optimizer = torch.optim.Adam(
        model_conv.parameters(), lr=args.lr * 0.3, weight_decay=args.weight_decay_conv)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=25, gamma=0.5, last_epoch=-1)

    margin_loss = torch.nn.SoftMarginLoss()

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs_conv))

    for epoch in range(args.epochs_conv):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_conv.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(Corpus_.train_indices) % args.batch_size_conv == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_size_conv
        else:
            num_iters_per_epoch = (
                len(Corpus_.train_indices) // args.batch_size_conv) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            # conv 其实相当与一个打分函数（decoder）
            preds = model_conv(
                Corpus_, Corpus_.train_adj_matrix, train_indices)

            optimizer.zero_grad()

            # values是一个取值为1或-1的目标tensor
            loss = margin_loss(preds.view(-1), train_values.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item()))

        scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        if not os.path.exists(args.output_folder + "conv/"):
            os.makedirs(args.output_folder + "conv/")

        if epoch > args.epochs_conv - 20:
            save_model(model_conv, args.data, epoch, args.output_folder + "conv/")


def evaluate_conv(args, unique_entities):
    model_conv = SpKBGATConvOnly(Corpus_.situation2id, entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)
    model_conv.load_state_dict(torch.load(
        '{0}conv/trained_{1}.pth'.format(args.output_folder, args.epochs_conv - 1)))

    model_conv.cuda()
    model_conv.eval()
    with torch.no_grad():
        Corpus_.get_validation_pred(Corpus_, args, model_conv, unique_entities)


train_gat(args)

train_conv(args)
evaluate_conv(args, Corpus_.unique_entities_train)
