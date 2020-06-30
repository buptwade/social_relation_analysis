import torch
import os
import numpy as np

def read_situations(filename, limit = 50):
    situation2id = {}

    idx = 1
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split(":")) > 1:
                situation2id[line.strip().split(":")[0].strip()] = idx
                idx += 1
            
            if idx > limit:
                break
    return situation2id

# txt里每一行表示一个entity。用空白字符分割entity name 和 id
def read_entity_from_id(filename='/home/devsdb/MovieGraphs/ych_result/MG_DATA/entity2id.txt'):
    entity2id = {}
    entity2situation = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split(";")) > 1:
                entity, entity_id, scene_name = line.strip().split(";")[0].strip(), line.strip().split(";")[1].strip(), line.strip().split(";")[2].strip()
                entity2id[entity] = int(entity_id)
                entity2situation[entity] = scene_name
    return entity2id, entity2situation

# 同理
# deprecated
def read_relation_from_id(filename='./data/WN18RR/relation2id.txt'):
    relation2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                relation, relation_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                relation2id[relation] = int(relation_id)
    return relation2id

# 读文件获取embedding
def init_embeddings(entity_file):
    entity_emb = []

    with open(entity_file) as f:
        for line in f:
            entity_emb.append([float(val) for val in line.strip().split()])

    return np.array(entity_emb, dtype=np.float32)


def parse_line(line):
    line = line.strip().split(";")
    e1, e2, relation = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2


# 这里 entity和relation 的map是数据集里提前准备好的， 这个函数就是将一个file里的所有三元组解析成id的形式并返回，且用
# rows 和 cols 两个list 记录边的方向，(rows, cols, data)可以代表adjMatrix，并用一个uniset记录实体数量
def load_data(filename, entity2id, is_unweigted=False, directed=True):
    with open(filename) as f:
        lines = f.readlines()

    # this is list for relation triples
    triples_data = []

    # for sparse tensor, rows list contains corresponding row of sparse tensor, cols list contains corresponding
    # columnn of sparse tensor, data contains the type of relation
    # Adjacecny matrix of entities is undirected, as the source and tail entities should know, the relation
    # type they are connected with
    rows, cols, data = [], [], []
    unique_entities = set()
    for line in lines:
        e1, relation, e2 = parse_line(line)
        unique_entities.add(e1)
        unique_entities.add(e2)
        triples_data.append(
            (entity2id[e1], int(relation), entity2id[e2]))
        if not directed:
                # Connecting source and tail entity
            rows.append(entity2id[e1])
            cols.append(entity2id[e2])
            if is_unweigted:
                data.append(1)
            else:
                data.append(int(relation))

        # Connecting tail and source entity
        rows.append(entity2id[e2])
        cols.append(entity2id[e1])
        if is_unweigted:
            data.append(1)
        else:
            data.append(int(relation))

    print("number of unique_entities ->", len(unique_entities))
    return triples_data, (rows, cols, data), list(unique_entities)


def build_data(path='/home/devsdb/MovieGraphs/ych_result/MG_DATA/', is_unweigted=False, directed=True, relation_num = 9):
    entity2id, entity2situation = read_entity_from_id(path + 'entity2id.txt')

    situation2id = read_situations(path + "statistic.txt")
    # Adjacency matrix only required for training phase
    # Currenlty creating as unweighted, undirected
    train_triples, train_adjacency_mat, unique_entities_train = load_data(os.path.join(
        path, 'train.txt'), entity2id, is_unweigted, directed)
    test_triples, test_adjacency_mat, unique_entities_test = load_data(os.path.join(
        path, 'test.txt'), entity2id, is_unweigted, directed)

    # 相当于得到一个逆向映射
    id2entity = {v: k for k, v in entity2id.items()}
    left_entity, right_entity = {}, {}

    with open(os.path.join(path, 'train.txt')) as f:
        lines = f.readlines()

    for line in lines:
        e1, relation, e2 = parse_line(line)
        relation = int(relation)

        # 将relationId作为set的key，entityId作为subKey
        # Count number of occurences for each (e1, relation)
        if relation not in left_entity:
            left_entity[relation] = {}
        if entity2id[e1] not in left_entity[relation]:
            left_entity[relation][entity2id[e1]] = 0
        left_entity[relation][entity2id[e1]] += 1

        # Count number of occurences for each (relation, e2)
        if relation not in right_entity:
            right_entity[relation] = {}
        if entity2id[e2] not in right_entity[relation]:
            right_entity[relation][entity2id[e2]] = 0
        right_entity[relation][entity2id[e2]] += 1

    # 对于每个relation，计算 与之联系的实体出现的总数/实体类别数
    left_entity_avg = {}
    for i in range(relation_num):
        left_entity_avg[i] = sum(
            left_entity[i].values()) * 1.0 / len(left_entity[i])

    right_entity_avg = {}
    for i in range(relation_num):
        right_entity_avg[i] = sum(
            right_entity[i].values()) * 1.0 / len(right_entity[i])

    headTailSelector = {}
    for i in range(relation_num):
        headTailSelector[i] = 1000 * right_entity_avg[i] / \
            (right_entity_avg[i] + left_entity_avg[i])

    return (train_triples, train_adjacency_mat), (test_triples, test_adjacency_mat), \
        entity2id, headTailSelector, unique_entities_train, entity2situation, situation2id
