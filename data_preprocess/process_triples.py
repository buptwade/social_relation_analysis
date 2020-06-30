import os
import sys
import numpy as np
import json
import datetime
from random import sample

'''
author：yan
description: 读取所有实体的特征。并生成编号。

'''

def collect(video_dir, triple_dir, result_dir):
    print('start collecting!')

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    total_entity2id = {}
    entity_feature = []
    total_triples = []
    id_idx = 0
    for video in os.listdir(video_dir):
        print("[{}] video:{} start processing".format(datetime.datetime.now(), video))

        video_triple_dir = os.path.join(triple_dir, video)
        entity2id = []
        with open(os.path.join(video_triple_dir, "entity2id.txt"), 'r') as e:
            for line in e:
                line = line.strip().split(";")
                entity2id.append((line[0], line[1]))

        features = []
        with open(os.path.join(video_triple_dir, "features.txt"), 'r') as f:
            for line in f:
                features.append([float(val) for val in line.strip().split()])
        
        for i, entity in enumerate(entity2id):
            if str(i+1) != entity[1]:
                print("error!!!!!")
                continue

            total_entity2id[id_idx] = entity[0]
            id_idx += 1
            entity_feature.append(features[i])

        print("video:{} had processed.".format(video))
    
    f1 = open(os.path.join(result_dir, "entity2id.txt"), 'w')
    for key, value in total_entity2id.items():
        string = value + ";" + str(key) + "\n"
        f1.write(string)
    f1.close()

    f2 = open(os.path.join(result_dir, "features.txt"), 'w')
    for i, feature in enumerate(entity_feature):
        # 降为1维
        feature = str(feature)
        feature = feature.replace("[",'').replace("]",'')
        feature = list(eval(feature))
        # 处理为字符串
        feature = [str(i) + ' ' for i in feature]
        string = "".join(feature) + "\n"
        f2.write(string)
    f2.close()