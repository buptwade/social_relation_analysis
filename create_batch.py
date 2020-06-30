import torch
import numpy as np
from collections import defaultdict
import time
import queue
import random

def is_triple_valid(name1_id, name2_id, id2entity):
    name1 = id2entity[name1_id]
    name2 = id2entity[name2_id]
    name1 = name1.split("-")
    name2 = name2.split("-")
    if name1[0] == name2[0] and name1[1] == name2[1]:
        return True
    return False

def is_test_triple_valid(triple, gen_triple, id2entity):
    (name1_id, relation, name2_id) = triple
    (g_name1_id, g_relation, g_name2_id) = gen_triple
    if is_same_person(name1_id, g_name1_id, id2entity) and is_same_person(name2_id, g_name2_id, id2entity): 
        return True
    return False

def is_same_person(name1_id, name2_id, id2entity):
    name1 = id2entity[name1_id]
    name2 = id2entity[name2_id]
    name1 = name1.split("-")
    name2 = name2.split("-")
    if name1[0] == name2[0] and name1[2] == name2[2]:
        return True
    # hack
    if name1[0] == name2[0] and name1[1] == name2[1]:
        return True
    return False

class Corpus:
    def __init__(self, args, train_data, test_data, entity2id, headTailSelector, entity2situation, situation2id, 
                 batch_size, valid_to_invalid_samples_ratio, unique_entities_train, get_2hop=False):
        # train_data = (train_triples, train_adjacency_mat)
        self.train_triples = train_data[0]
        # Converting to sparse tensor
        # train_adjacency_mat = (rows, cols, data) 这里的data 根据isWeighted 为1或者relationId
        adj_indices = torch.LongTensor(
            [train_data[1][0], train_data[1][1]])  # rows and columns
        adj_values = torch.LongTensor(train_data[1][2])
        self.train_adj_matrix = (adj_indices, adj_values)

        # adjacency matrix is needed for train_data only, as GAT is trained for
        # training data
        self.test_triples = test_data[0]

        self.headTailSelector = headTailSelector  # for selecting random entities
        self.entity2id = entity2id
        self.id2entity = {v: k for k, v in self.entity2id.items()}

        # 支持situations
        self.entity2situation = entity2situation
        self.situation2id = situation2id
        self.id2situation = {v: k for k, v in self.situation2id.items()}

        self.batch_size = batch_size
        # ratio of valid to invalid samples per batch for training ConvKB Model
        self.invalid_valid_ratio = int(valid_to_invalid_samples_ratio)

        if(get_2hop):
            self.graph = self.get_graph()
            self.node_neighbors_2hop = self.get_further_neighbors()

        self.unique_entities_train = [self.entity2id[i]
                                      for i in unique_entities_train]

        self.train_indices = np.array(
            list(self.train_triples)).astype(np.int32)
        # These are valid triples, hence all have value 1
        self.train_values = np.array(
            [[1]] * len(self.train_triples)).astype(np.float32)

        self.test_indices = np.array(list(self.test_triples)).astype(np.int32)
        self.test_values = np.array(
            [[1]] * len(self.test_triples)).astype(np.float32)

        self.valid_triples_dict = {j: i for i, j in enumerate(
            self.train_triples + self.test_triples)}
        print("Total triples count {}, training triples {}, test_triples {}".format(len(self.valid_triples_dict), len(self.train_indices),
                                                                                                            len(self.test_indices)))

        # For training purpose
        self.batch_indices = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
        self.batch_values = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

    def get_iteration_batch(self, iter_num):
        if (iter_num + 1) * self.batch_size <= len(self.train_indices):
            self.batch_indices = np.empty(
                (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty(
                (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * iter_num,
                            self.batch_size * (iter_num + 1))
            
            # values存放三元组是valid还是invalid
            self.batch_indices[:self.batch_size,
                               :] = self.train_indices[indices, :]
            self.batch_values[:self.batch_size,
                              :] = self.train_values[indices, :]

            last_index = self.batch_size

            if self.invalid_valid_ratio > 0:
                # 获得 last_index * self.invalid_valid_ratio 个 0～num（entities）之间的随机数
                random_entities = np.random.randint(
                    0, len(self.entity2id), last_index * self.invalid_valid_ratio)

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

                # 造neg三元组，一半随机替换source node，一半随机替换target node
                for i in range(last_index):
                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = i * (self.invalid_valid_ratio // 2) + j

                        while is_triple_valid(random_entities[current_index], 
                                self.batch_indices[last_index + current_index, 2], self.id2entity):
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           0] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = last_index * \
                            (self.invalid_valid_ratio // 2) + \
                            (i * (self.invalid_valid_ratio // 2) + j)

                        while is_triple_valid(random_entities[current_index], 
                                self.batch_indices[last_index + current_index, 2], self.id2entity):
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           2] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]
                # 返回的已经是一个batch的pos三元组 + 造出的neg三元组
                return self.batch_indices, self.batch_values

            return self.batch_indices, self.batch_values

        else:
            last_iter_size = len(self.train_indices) - \
                self.batch_size * iter_num
            self.batch_indices = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * iter_num,
                            len(self.train_indices))
            self.batch_indices[:last_iter_size,
                               :] = self.train_indices[indices, :]
            self.batch_values[:last_iter_size,
                              :] = self.train_values[indices, :]

            last_index = last_iter_size

            if self.invalid_valid_ratio > 0:
                random_entities = np.random.randint(
                    0, len(self.entity2id), last_index * self.invalid_valid_ratio)

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

                for i in range(last_index):
                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = i * (self.invalid_valid_ratio // 2) + j

                        while is_triple_valid(random_entities[current_index], 
                                self.batch_indices[last_index + current_index, 2], self.id2entity):
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           0] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = last_index * \
                            (self.invalid_valid_ratio // 2) + \
                            (i * (self.invalid_valid_ratio // 2) + j)

                        while is_triple_valid(random_entities[current_index], 
                                self.batch_indices[last_index + current_index, 2], self.id2entity):
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           2] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                return self.batch_indices, self.batch_values

            return self.batch_indices, self.batch_values

    def get_iteration_batch_nhop(self, current_batch_indices, node_neighbors, batch_size):

        self.batch_indices = np.empty(
            (batch_size * (self.invalid_valid_ratio + 1), 4)).astype(np.int32)
        self.batch_values = np.empty(
            (batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)
        indices = random.sample(range(len(current_batch_indices)), batch_size)

        self.batch_indices[:batch_size,
                           :] = current_batch_indices[indices, :]
        self.batch_values[:batch_size,
                          :] = np.ones((batch_size, 1))

        last_index = batch_size

        if self.invalid_valid_ratio > 0:
            random_entities = np.random.randint(
                0, len(self.entity2id), last_index * self.invalid_valid_ratio)

            # Precopying the same valid indices from 0 to batch_size to rest
            # of the indices
            self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
            self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

            for i in range(last_index):
                for j in range(self.invalid_valid_ratio // 2):
                    current_index = i * (self.invalid_valid_ratio // 2) + j

                    self.batch_indices[last_index + current_index,
                                       0] = random_entities[current_index]
                    self.batch_values[last_index + current_index, :] = [0]

                for j in range(self.invalid_valid_ratio // 2):
                    current_index = last_index * \
                        (self.invalid_valid_ratio // 2) + \
                        (i * (self.invalid_valid_ratio // 2) + j)

                    self.batch_indices[last_index + current_index,
                                       3] = random_entities[current_index]
                    self.batch_values[last_index + current_index, :] = [0]

            return self.batch_indices, self.batch_values

        return self.batch_indices, self.batch_values

    def get_graph(self):
        graph = {}
        # train_adj_matrix = (adj_indices（索引）, adj_values（值）)
        all_tiples = torch.cat([self.train_adj_matrix[0].transpose(
            0, 1), self.train_adj_matrix[1].unsqueeze(1)], dim=1)

        for data in all_tiples:
            source = data[1].data.item()
            target = data[0].data.item()
            value = data[2].data.item()

            if(source not in graph.keys()):
                graph[source] = {}
                graph[source][target] = value
            else:
                graph[source][target] = value
        print("Graph created")
        # 这里相当于建立了一个二维字典 表示图
        return graph
    
    # 获得一个节点不同距离的邻居节点
    # 感觉可以优化
    def bfs(self, graph, source, nbd_size=2):
        visit = {}
        distance = {}
        parent = {}
        distance_lengths = {}

        visit[source] = 1
        distance[source] = 0
        parent[source] = (-1, -1)

        q = queue.Queue()
        q.put((source, -1))

        # 广度搜索 获取到所有和source节点距离为2的节点和路径
        while(not q.empty()):
            top = q.get()
            if top[0] in graph.keys():
                for target in graph[top[0]].keys():
                    if(target in visit.keys()):
                        continue
                    else:
                        q.put((target, graph[top[0]][target]))

                        distance[target] = distance[top[0]] + 1

                        visit[target] = 1
                        if distance[target] > 2:
                            continue
                        parent[target] = (top[0], graph[top[0]][target])

                        if distance[target] not in distance_lengths.keys():
                            distance_lengths[distance[target]] = 1

        neighbors = {}
        for target in visit.keys():
            if(distance[target] != nbd_size):
                continue
            edges = [-1, parent[target][1]]
            relations = []
            entities = [target]
            temp = target
            # 追溯回 source 节点 ，parent[temp] = (nodeId, relationId) 即 source 和 temp节点有关系
            # 在step = 2 的情况下 relations里会记录 source 到 target 的两个关系
            while(parent[temp] != (-1, -1)):
                relations.append(parent[temp][1])
                entities.append(parent[temp][0])
                temp = parent[temp][0]
            
            # 获取到不同 distance[target] 距离的所有关系和实体
            if(distance[target] in neighbors.keys()):
                neighbors[distance[target]].append(
                    (tuple(relations), tuple(entities[:-1])))
            else:
                neighbors[distance[target]] = [
                    (tuple(relations), tuple(entities[:-1]))]
        # neighbor字典，key为与source节点的距离，value是所有关系和实体
        return neighbors

    # 获得一个图里所有节点的不同距离的邻居节点
    def get_further_neighbors(self, nbd_size=2):
        neighbors = {}
        start_time = time.time()
        print("length of graph keys is ", len(self.graph.keys()))
        for source in self.graph.keys():
            # st_time = time.time()
            temp_neighbors = self.bfs(self.graph, source, nbd_size)
            for distance in temp_neighbors.keys():
                if(source in neighbors.keys()):
                    if(distance in neighbors[source].keys()):
                        neighbors[source][distance].append(
                            temp_neighbors[distance])
                    else:
                        neighbors[source][distance] = temp_neighbors[distance]
                else:
                    neighbors[source] = {}
                    neighbors[source][distance] = temp_neighbors[distance]

        print("time taken ", time.time() - start_time)

        print("length of neighbors dict is ", len(neighbors))
        # 二维字典，第一级key表示某个节点id，第二级key是距离，value是（relations，entities）
        return neighbors

    def get_batch_nhop_neighbors_all(self, args, batch_sources, node_neighbors, nbd_size=2):
        batch_source_triples = []
        print("length of unique_entities ", len(batch_sources))
        count = 0
        for source in batch_sources:
            # randomly select from the list of neighbors
            if source in node_neighbors.keys():
                nhop_list = node_neighbors[source][nbd_size]

                for i, tup in enumerate(nhop_list):
                    if(args.partial_2hop and i >= 1):
                        break

                    count += 1
                    batch_source_triples.append([source, nhop_list[i][0][-1], nhop_list[i][0][0],
                                                 nhop_list[i][1][0]])

        return np.array(batch_source_triples).astype(np.int32)

    def transe_scoring(self, batch_inputs, entity_embeddings, relation_embeddings):
        source_embeds = entity_embeddings[batch_inputs[:, 0]]
        relation_embeds = relation_embeddings[batch_inputs[:, 1]]
        tail_embeds = entity_embeddings[batch_inputs[:, 2]]
        x = source_embeds - tail_embeds
        x = torch.norm(x, p=1, dim=1)
        return x

    def generate_key(self, entity_id1, entity_id2, relation_id):
        entity1 = self.id2entity[entity_id1]
        entity2 = self.id2entity[entity_id2]

        entity1_split = entity1.split("-")
        video = entity1_split[0]
        scene = entity1_split[1]
        name1 = entity1_split[3]

        entity2_split = entity2.split("-")
        name2 = entity2_split[3]

        key = video + "-" + scene + "-" + name1 + "-" + name2 + "-" + str(relation_id)
        return key

    def calculate_relation_acc(self, appear_dict, achieve_dict):
        record = []
        relation_acc_dict = {}
        for key, val in appear_dict.items():
            relation_id = key.split("-")[4]
            if key not in achieve_dict.keys():
                record.append(0)

                if relation_id not in relation_acc_dict.keys():
                    relation_acc_dict[relation_id] = [0]
                else:
                    relation_acc_dict[relation_id].append(0)
                continue
            
            if achieve_dict[key] / val > 0.65:
                record.append(1)
                if relation_id not in relation_acc_dict.keys():
                    relation_acc_dict[relation_id] = [1]
                else:
                    relation_acc_dict[relation_id].append(1)
            else:
                record.append(0)
                if relation_id not in relation_acc_dict.keys():
                    relation_acc_dict[relation_id] = [0]
                else:
                    relation_acc_dict[relation_id].append(0)

        return sum(record) / len(record), relation_acc_dict

    def get_validation_pred(self, Corpus_, args, model, unique_entities):
        average_hits_at_100_head, average_hits_at_100_tail = [], []
        average_hits_at_ten_head, average_hits_at_ten_tail = [], []
        average_hits_at_three_head, average_hits_at_three_tail = [], []
        average_hits_at_one_head, average_hits_at_one_tail = [], []
        average_mean_rank_head, average_mean_rank_tail = [], []
        average_mean_recip_rank_head, average_mean_recip_rank_tail = [], []

        situation_hits_at_one_head_dict, situation_hits_at_one_tail_dict = {}, {}
        situation_hits_at_three_head_dict, situation_hits_at_three_tail_dict = {}, {}
        situation_hits_at_ten_head_dict, situation_hits_at_ten_tail_dict = {}, {}
        situation_hits_at_100_head_dict, situation_hits_at_100_tail_dict = {}, {}
        situations_hit = {}
        test_situations = []

        relation_hits_at_one_head_dict, relation_hits_at_one_tail_dict = {}, {}
        relation_hits_at_three_head_dict, relation_hits_at_three_tail_dict = {}, {}
        relation_hits_at_ten_head_dict, relation_hits_at_ten_tail_dict = {}, {}
        relation_hits_at_100_head_dict, relation_hits_at_100_tail_dict = {}, {}
        relations_hit = {}
        for iters in range(1):
            start_time = time.time()

            indices = [i for i in range(len(self.test_indices))]
            batch_indices = self.test_indices[indices, :]
            print("Sampled indices")
            print("test set length ", len(self.test_indices))
            entity_list = [j for i, j in self.entity2id.items()]

            ranks_head, ranks_tail = [], []
            reciprocal_ranks_head, reciprocal_ranks_tail = [], []
            hits_at_100_head, hits_at_100_tail = 0, 0
            hits_at_ten_head, hits_at_ten_tail = 0, 0
            hits_at_three_head, hits_at_three_tail = 0, 0
            hits_at_one_head, hits_at_one_tail = 0, 0

            for i in range(batch_indices.shape[0]):
                situation_name = self.entity2situation[self.id2entity[int(batch_indices[i, 0])]]
                if situation_name in self.situation2id.keys():
                    test_situations.append(situation_name)
                else:
                    test_situations.append("others")

                print(len(ranks_head))
                start_time_it = time.time()
                new_x_batch_head = np.tile(
                    batch_indices[i, :], (len(self.entity2id), 1))
                new_x_batch_tail = np.tile(
                    batch_indices[i, :], (len(self.entity2id), 1))

                if(batch_indices[i, 0] not in unique_entities or batch_indices[i, 2] not in unique_entities):
                    continue

                new_x_batch_head[:, 0] = entity_list
                new_x_batch_tail[:, 2] = entity_list

                last_index_head = []  # array of already existing triples
                last_index_tail = []
                for tmp_index in range(len(new_x_batch_head)):
                    temp_triple_head = (new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][1],
                                        new_x_batch_head[tmp_index][2])
                    if is_test_triple_valid(batch_indices[i], temp_triple_head, self.id2entity):
                        last_index_head.append(tmp_index)

                    temp_triple_tail = (new_x_batch_tail[tmp_index][0], new_x_batch_tail[tmp_index][1],
                                        new_x_batch_tail[tmp_index][2])
                    if is_test_triple_valid(batch_indices[i], temp_triple_tail, self.id2entity):
                        last_index_tail.append(tmp_index)

                # Deleting already existing triples, leftover triples are invalid, according
                # to train, validation and test data
                # Note, all of them maynot be actually invalid
                new_x_batch_head = np.delete(
                    new_x_batch_head, last_index_head, axis=0)
                new_x_batch_tail = np.delete(
                    new_x_batch_tail, last_index_tail, axis=0)
                # print(new_x_batch_head.shape)
                # new_x_batch_head = np.array(random.sample(list(new_x_batch_head), 28000))
                # new_x_batch_tail = np.array(random.sample(list(new_x_batch_tail), 28000))
                # print(new_x_batch_head.shape)
                # adding the current valid triples to the top, i.e, index 0
                new_x_batch_head = np.insert(
                    new_x_batch_head, 0, batch_indices[i], axis=0)
                new_x_batch_tail = np.insert(
                    new_x_batch_tail, 0, batch_indices[i], axis=0)

                import math
                # Have to do this, because it doesn't fit in memory

                if 'MG' in args.data:
                    num_triples_each_shot = int(
                        math.ceil(new_x_batch_head.shape[0] / 5))

                    scores1_head = model.batch_test(Corpus_, torch.LongTensor(
                        new_x_batch_head[:num_triples_each_shot, :]).cuda())
                    scores2_head = model.batch_test(Corpus_, torch.LongTensor(
                        new_x_batch_head[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
                    scores3_head = model.batch_test(Corpus_, torch.LongTensor(
                        new_x_batch_head[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
                    scores4_head = model.batch_test(Corpus_, torch.LongTensor(
                        new_x_batch_head[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())
                    scores5_head = model.batch_test(Corpus_, torch.LongTensor(
                        new_x_batch_head[4 * num_triples_each_shot: 5 * num_triples_each_shot, :]).cuda())
                    # scores5_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[4 * num_triples_each_shot: 5 * num_triples_each_shot, :]).cuda())
                    # scores6_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[5 * num_triples_each_shot: 6 * num_triples_each_shot, :]).cuda())
                    # scores7_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[6 * num_triples_each_shot: 7 * num_triples_each_shot, :]).cuda())
                    # scores8_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[7 * num_triples_each_shot: 8 * num_triples_each_shot, :]).cuda())
                    # scores9_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[8 * num_triples_each_shot: 9 * num_triples_each_shot, :]).cuda())
                    # scores10_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[9 * num_triples_each_shot:, :]).cuda())

                    scores_head = torch.cat(
                        [scores1_head, scores2_head, scores3_head, scores4_head, scores5_head], dim=0)
                    #scores5_head, scores6_head, scores7_head, scores8_head,
                    # cores9_head, scores10_head], dim=0)
                else:
                    scores_head = model.batch_test(Corpus_, new_x_batch_head)

                # 将scores从大到小排序
                sorted_scores_head, sorted_indices_head = torch.sort(
                    scores_head.view(-1), dim=-1, descending=True)
                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_head.append(
                    np.where(sorted_indices_head.cpu().numpy() == 0)[0][0] + 1)
                reciprocal_ranks_head.append(1.0 / ranks_head[-1])

                # Tail part here

                if 'MG' in args.data:
                    num_triples_each_shot = int(
                        math.ceil(new_x_batch_tail.shape[0] / 5))

                    scores1_tail = model.batch_test(Corpus_, torch.LongTensor(
                        new_x_batch_tail[:num_triples_each_shot, :]).cuda())
                    scores2_tail = model.batch_test(Corpus_, torch.LongTensor(
                        new_x_batch_tail[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
                    scores3_tail = model.batch_test(Corpus_, torch.LongTensor(
                        new_x_batch_tail[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
                    scores4_tail = model.batch_test(Corpus_, torch.LongTensor(
                        new_x_batch_tail[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())
                    scores5_tail = model.batch_test(Corpus_, torch.LongTensor(
                        new_x_batch_tail[4 * num_triples_each_shot: 5 * num_triples_each_shot, :]).cuda())
                    # scores5_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[4 * num_triples_each_shot: 5 * num_triples_each_shot, :]).cuda())
                    # scores6_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[5 * num_triples_each_shot: 6 * num_triples_each_shot, :]).cuda())
                    # scores7_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[6 * num_triples_each_shot: 7 * num_triples_each_shot, :]).cuda())
                    # scores8_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[7 * num_triples_each_shot: 8 * num_triples_each_shot, :]).cuda())
                    # scores9_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[8 * num_triples_each_shot: 9 * num_triples_each_shot, :]).cuda())
                    # scores10_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[9 * num_triples_each_shot:, :]).cuda())

                    scores_tail = torch.cat(
                        [scores1_tail, scores2_tail, scores3_tail, scores4_tail, scores5_tail], dim=0)
                    #     scores5_tail, scores6_tail, scores7_tail, scores8_tail,
                    #     scores9_tail, scores10_tail], dim=0)

                else:
                    scores_tail = model.batch_test(Corpus_, new_x_batch_tail)

                sorted_scores_tail, sorted_indices_tail = torch.sort(
                    scores_tail.view(-1), dim=-1, descending=True)

                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_tail.append(
                    np.where(sorted_indices_tail.cpu().numpy() == 0)[0][0] + 1)
                reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])
                print(batch_indices[i])
                print("sample - ", ranks_head[-1], ranks_tail[-1])

            for i in range(len(ranks_head)):
                if test_situations[i] not in situations_hit.keys():
                    situations_hit[test_situations[i]] = 1
                else:
                    situations_hit[test_situations[i]] += 1

                entity_id1, relation_id, entity_id2 = batch_indices[i]
                pair_key = self.generate_key(entity_id1, entity_id2, relation_id)
                if pair_key not in relations_hit.keys():
                    relations_hit[pair_key] = 1
                else:
                    relations_hit[pair_key] += 1

                if ranks_head[i] <= 100:
                    if test_situations[i] not in situation_hits_at_100_head_dict.keys():
                        situation_hits_at_100_head_dict[test_situations[i]] = 1
                    else:
                        situation_hits_at_100_head_dict[test_situations[i]] += 1

                    if pair_key not in relation_hits_at_100_head_dict.keys():
                        relation_hits_at_100_head_dict[pair_key] = 1
                    else:
                        relation_hits_at_100_head_dict[pair_key] += 1

                    hits_at_100_head = hits_at_100_head + 1

                if ranks_head[i] <= 10:
                    if test_situations[i] not in situation_hits_at_ten_head_dict.keys():
                        situation_hits_at_ten_head_dict[test_situations[i]] = 1
                    else:
                        situation_hits_at_ten_head_dict[test_situations[i]] += 1

                    if pair_key not in relation_hits_at_ten_head_dict.keys():
                        relation_hits_at_ten_head_dict[pair_key] = 1
                    else:
                        relation_hits_at_ten_head_dict[pair_key] += 1

                    hits_at_ten_head = hits_at_ten_head + 1

                if ranks_head[i] <= 3:
                    if test_situations[i] not in situation_hits_at_three_head_dict.keys():
                        situation_hits_at_three_head_dict[test_situations[i]] = 1
                    else:
                        situation_hits_at_three_head_dict[test_situations[i]] += 1

                    if pair_key not in relation_hits_at_three_head_dict.keys():
                        relation_hits_at_three_head_dict[pair_key] = 1
                    else:
                        relation_hits_at_three_head_dict[pair_key] += 1

                    hits_at_three_head = hits_at_three_head + 1

                if ranks_head[i] == 1:
                    if test_situations[i] not in situation_hits_at_one_head_dict.keys():
                        situation_hits_at_one_head_dict[test_situations[i]] = 1
                    else:
                        situation_hits_at_one_head_dict[test_situations[i]] += 1

                    if pair_key not in relation_hits_at_one_head_dict.keys():
                        relation_hits_at_one_head_dict[pair_key] = 1
                    else:
                        relation_hits_at_one_head_dict[pair_key] += 1

                    hits_at_one_head = hits_at_one_head + 1

            for i in range(len(ranks_tail)):
                entity_id1, relation_id, entity_id2 = batch_indices[i]
                pair_key = self.generate_key(entity_id1, entity_id2, relation_id)

                if ranks_tail[i] <= 100:
                    if test_situations[i] not in situation_hits_at_100_tail_dict.keys():
                        situation_hits_at_100_tail_dict[test_situations[i]] = 1
                    else:
                        situation_hits_at_100_tail_dict[test_situations[i]] += 1

                    if pair_key not in relation_hits_at_100_tail_dict.keys():
                        relation_hits_at_100_tail_dict[pair_key] = 1
                    else:
                        relation_hits_at_100_tail_dict[pair_key] += 1

                    hits_at_100_tail = hits_at_100_tail + 1

                if ranks_tail[i] <= 10:
                    if test_situations[i] not in situation_hits_at_ten_tail_dict.keys():
                        situation_hits_at_ten_tail_dict[test_situations[i]] = 1
                    else:
                        situation_hits_at_ten_tail_dict[test_situations[i]] += 1

                    if pair_key not in relation_hits_at_ten_tail_dict.keys():
                        relation_hits_at_ten_tail_dict[pair_key] = 1
                    else:
                        relation_hits_at_ten_tail_dict[pair_key] += 1

                    hits_at_ten_tail = hits_at_ten_tail + 1
                    
                if ranks_tail[i] <= 3:
                    if test_situations[i] not in situation_hits_at_three_tail_dict.keys():
                        situation_hits_at_three_tail_dict[test_situations[i]] = 1
                    else:
                        situation_hits_at_three_tail_dict[test_situations[i]] += 1

                    if pair_key not in relation_hits_at_three_tail_dict.keys():
                        relation_hits_at_three_tail_dict[pair_key] = 1
                    else:
                        relation_hits_at_three_tail_dict[pair_key] += 1

                    hits_at_three_tail = hits_at_three_tail + 1

                if ranks_tail[i] == 1:
                    if test_situations[i] not in situation_hits_at_one_tail_dict.keys():
                        situation_hits_at_one_tail_dict[test_situations[i]] = 1
                    else:
                        situation_hits_at_one_tail_dict[test_situations[i]] += 1

                    if pair_key not in relation_hits_at_one_tail_dict.keys():
                        relation_hits_at_one_tail_dict[pair_key] = 1
                    else:
                        relation_hits_at_one_tail_dict[pair_key] += 1

                    hits_at_one_tail = hits_at_one_tail + 1

            assert len(ranks_head) == len(reciprocal_ranks_head)
            assert len(ranks_tail) == len(reciprocal_ranks_tail)
            print("here {}".format(len(ranks_head)))
            print("\nCurrent iteration time {}".format(time.time() - start_time))
            print("Stats for replacing head are -> ")
            print("Current iteration Hits@100 are {}".format(
                hits_at_100_head / float(len(ranks_head))))
            print("Current iteration Hits@10 are {}".format(
                hits_at_ten_head / len(ranks_head)))
            print("Current iteration Hits@3 are {}".format(
                hits_at_three_head / len(ranks_head)))
            print("Current iteration Hits@1 are {}".format(
                hits_at_one_head / len(ranks_head)))
            print("Current iteration Mean rank {}".format(
                sum(ranks_head) / len(ranks_head)))
            print("Current iteration Mean Reciprocal Rank {}".format(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)))

            print("\nStats for replacing tail are -> ")
            print("Current iteration Hits@100 are {}".format(
                hits_at_100_tail / len(ranks_head)))
            print("Current iteration Hits@10 are {}".format(
                hits_at_ten_tail / len(ranks_head)))
            print("Current iteration Hits@3 are {}".format(
                hits_at_three_tail / len(ranks_head)))
            print("Current iteration Hits@1 are {}".format(
                hits_at_one_tail / len(ranks_head)))
            print("Current iteration Mean rank {}".format(
                sum(ranks_tail) / len(ranks_tail)))
            print("Current iteration Mean Reciprocal Rank {}".format(
                sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)))

            average_hits_at_100_head.append(
                hits_at_100_head / len(ranks_head))
            average_hits_at_ten_head.append(
                hits_at_ten_head / len(ranks_head))
            average_hits_at_three_head.append(
                hits_at_three_head / len(ranks_head))
            average_hits_at_one_head.append(
                hits_at_one_head / len(ranks_head))
            average_mean_rank_head.append(sum(ranks_head) / len(ranks_head))
            average_mean_recip_rank_head.append(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head))

            average_hits_at_100_tail.append(
                hits_at_100_tail / len(ranks_head))
            average_hits_at_ten_tail.append(
                hits_at_ten_tail / len(ranks_head))
            average_hits_at_three_tail.append(
                hits_at_three_tail / len(ranks_head))
            average_hits_at_one_tail.append(
                hits_at_one_tail / len(ranks_head))
            average_mean_rank_tail.append(sum(ranks_tail) / len(ranks_tail))
            average_mean_recip_rank_tail.append(
                sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail))

        # head
        print("\nsituation hit 1 for replacing head are")
        for key, val in situation_hits_at_one_head_dict.items():
            print("{}: {}. hits:{}, situation appear:{}".format(key, val / situations_hit[key], val, situations_hit[key]))

        print("\nsituation hit 3 for replacing head are")
        for key, val in situation_hits_at_three_head_dict.items():
            print("{}: {}. hits:{}, situation appear:{}".format(key, val / situations_hit[key], val, situations_hit[key]))

        print("\nsituation hit 10 for replacing head are")
        for key, val in situation_hits_at_ten_head_dict.items():
            print("{}: {}. hits:{}, situation appear:{}".format(key, val / situations_hit[key], val, situations_hit[key]))

        print("\nsituation hit 100 for replacing head are")
        for key, val in situation_hits_at_100_head_dict.items():
            print("{}: {}. hits:{}, situation appear:{}".format(key, val / situations_hit[key], val, situations_hit[key]))

        # tail
        print("\nsituation hit 1 for replacing tail are")
        for key, val in situation_hits_at_one_tail_dict.items():
            print("{}: {}. hits:{}, situation appear:{}".format(key, val / situations_hit[key], val, situations_hit[key]))

        print("\nsituation hit 3 for replacing tail are")
        for key, val in situation_hits_at_three_tail_dict.items():
            print("{}: {}. hits:{}, situation appear:{}".format(key, val / situations_hit[key], val, situations_hit[key]))

        print("\nsituation hit 10 for replacing tail are")
        for key, val in situation_hits_at_ten_tail_dict.items():
            print("{}: {}. hits:{}, situation appear:{}".format(key, val / situations_hit[key], val, situations_hit[key]))

        print("\nsituation hit 100 for replacing tail are")
        for key, val in situation_hits_at_100_tail_dict.items():
            print("{}: {}. hits:{}, situation appear:{}".format(key, val / situations_hit[key], val, situations_hit[key]))

        # head
        # print("\nhit 1 relation predict acc for replacing head are: {}"
        #         .format(self.calculate_relation_acc(relations_hit, relation_hits_at_one_head_dict)))
        # print("hit 3 relation predict acc for replacing head are: {}"
        #         .format(self.calculate_relation_acc(relations_hit, relation_hits_at_three_head_dict)))
        # print("hit 10 relation predict acc for replacing head are: {}"
        #         .format(self.calculate_relation_acc(relations_hit, relation_hits_at_ten_head_dict)))
        # print("hit 100 relation predict acc for replacing head are: {}"
        #         .format(self.calculate_relation_acc(relations_hit, relation_hits_at_100_head_dict)))
        self.print_result(1, "head", relations_hit, relation_hits_at_one_head_dict)
        self.print_result(3, "head", relations_hit, relation_hits_at_three_head_dict)
        self.print_result(10, "head", relations_hit, relation_hits_at_ten_head_dict)
        self.print_result(100, "head", relations_hit, relation_hits_at_100_head_dict)

        # tail
        # print("\nhit 1 relation predict acc for replacing tail are: {}"
        #         .format(self.calculate_relation_acc(relations_hit, relation_hits_at_one_tail_dict)))
        # print("hit 3 relation predict acc for replacing tail are: {}"
        #         .format(self.calculate_relation_acc(relations_hit, relation_hits_at_three_tail_dict)))
        # print("hit 10 relation predict acc for replacing tail are: {}"
        #         .format(self.calculate_relation_acc(relations_hit, relation_hits_at_ten_tail_dict)))
        # print("hit 100 relation predict acc for replacing tail are: {}"
        #         .format(self.calculate_relation_acc(relations_hit, relation_hits_at_100_tail_dict)))
        self.print_result(1, "tail", relations_hit, relation_hits_at_one_tail_dict)
        self.print_result(3, "tail", relations_hit, relation_hits_at_three_tail_dict)
        self.print_result(10, "tail", relations_hit, relation_hits_at_ten_tail_dict)
        self.print_result(100, "tail", relations_hit, relation_hits_at_100_tail_dict)

        print("\nAveraged stats for replacing head are -> ")
        print("Hits@100 are {}".format(
            sum(average_hits_at_100_head) / len(average_hits_at_100_head)))
        print("Hits@10 are {}".format(
            sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)))
        print("Hits@3 are {}".format(
            sum(average_hits_at_three_head) / len(average_hits_at_three_head)))
        print("Hits@1 are {}".format(
            sum(average_hits_at_one_head) / len(average_hits_at_one_head)))
        print("Mean rank {}".format(
            sum(average_mean_rank_head) / len(average_mean_rank_head)))
        print("Mean Reciprocal Rank {}".format(
            sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head)))

        print("\nAveraged stats for replacing tail are -> ")
        print("Hits@100 are {}".format(
            sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)))
        print("Hits@10 are {}".format(
            sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)))
        print("Hits@3 are {}".format(
            sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)))
        print("Hits@1 are {}".format(
            sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)))
        print("Mean rank {}".format(
            sum(average_mean_rank_tail) / len(average_mean_rank_tail)))
        print("Mean Reciprocal Rank {}".format(
            sum(average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)))

        cumulative_hits_100 = (sum(average_hits_at_100_head) / len(average_hits_at_100_head)
                               + sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)) / 2
        cumulative_hits_ten = (sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)
                               + sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)) / 2
        cumulative_hits_three = (sum(average_hits_at_three_head) / len(average_hits_at_three_head)
                                 + sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)) / 2
        cumulative_hits_one = (sum(average_hits_at_one_head) / len(average_hits_at_one_head)
                               + sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)) / 2
        cumulative_mean_rank = (sum(average_mean_rank_head) / len(average_mean_rank_head)
                                + sum(average_mean_rank_tail) / len(average_mean_rank_tail)) / 2
        cumulative_mean_recip_rank = (sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head) + sum(
            average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)) / 2

        print("\nCumulative stats are -> ")
        print("Hits@100 are {}".format(cumulative_hits_100))
        print("Hits@10 are {}".format(cumulative_hits_ten))
        print("Hits@3 are {}".format(cumulative_hits_three))
        print("Hits@1 are {}".format(cumulative_hits_one))
        print("Mean rank {}".format(cumulative_mean_rank))
        print("Mean Reciprocal Rank {}".format(cumulative_mean_recip_rank))

    def print_result(self, hit, pos, appear_dict, achieve_dict):

        acc, relation_acc_dict = self.calculate_relation_acc(appear_dict, achieve_dict)
        print("\nhit {} relation predict acc for replacing {} are: {}".format(hit, pos, acc))
        for key, val in relation_acc_dict.items():
            print("relation {} acc:{}".format(key, sum(relation_acc_dict[key])/len(relation_acc_dict[key])))
        print("\n")