import os
import sys
import numpy as np
import json
import datetime
import random
from random import sample

def divide_for_prediction(video_dir, triple_dir, result_dir):
    print('start divide for prediction!')

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    train_triples = []
    test_triples = []
    val_triples = []
    for video in os.listdir(video_dir):
        print("[{}] video:{} start processing".format(datetime.datetime.now(), video))

        video_triple_dir = os.path.join(triple_dir, video)

        triples = []
        with open(os.path.join(video_triple_dir, "new-triple.txt"), 'r') as nt:
            for line in nt:
                line = line.strip().split(";")
                triples.append((line[0], line[1], line[2], line[3]))

        scene_triple_dict = {}
        for i, triple in enumerate(triples):
            name1, name2, relation_id, desc = triple
            if desc == 'equal':
                continue 
            scene = name1.split("-")[1]
            name1 = name1.split("-")[3]
            name2 = name2.split("-")[3]
            if scene not in scene_triple_dict.keys():
                scene_triple_dict[scene] = [(name1, name2, relation_id)]
            else:
                scene_triple_dict[scene].append((name1, name2, relation_id))
        
        test_triple_dict = {}
        for scene, tps in scene_triple_dict.items():
            tps = list(set(tps))
            unique_list = []
            for triple in tps:
                name1, name2, r = triple
                if triple not in unique_list and (name2, name1, r) not in unique_list:
                    unique_list.append(triple)
        
            if len(unique_list) < 3:
                continue
            
            num_sample = len(unique_list) // 5 + 1
            sample_list = sample(unique_list, num_sample)
            test_triple_dict[scene] = sample_list
            
        sample_triples = []
        for i, triple in enumerate(triples):
            name1, name2, relation_id, desc = triple
            scene = name1.split("-")[1]
            n1 = name1.split("-")[3]
            n2 = name2.split("-")[3]
            if scene in test_triple_dict.keys():
                if (n1, n2, relation_id) in test_triple_dict[scene] or (n2, n1, relation_id) in test_triple_dict[scene]:
                    sample_triples.append(triple)
        
        print("video:{}, samples:{}, all:{}".format(video, len(sample_triples), len(triples)))


        # # 不关心equal三元组
        # valid_triples = list(filter(lambda x: x[3] != 'equal', triples))
        # sample_num = len(valid_triples) // 10
        # sample_triples = sample(valid_triples, sample_num)

        # 差集
        triples = list(set(triples).difference(set(sample_triples)))

        train_triples.extend(triples)
        test_triples.extend(sample_triples)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    f1 = open(os.path.join(result_dir, "train.txt"), 'w')
    for triple in train_triples:
        name1, name2, relation_id, desc = triple
        string = name1 + ";" + name2 + ";" + str(relation_id) + ";" + desc + "\n"
        f1.write(string)
    f1.close()

    f2 = open(os.path.join(result_dir, "test.txt"), 'w')
    for triple in test_triples:
        name1, name2, relation_id, desc = triple
        string = name1 + ";" + name2 + ";" + str(relation_id) + ";" + desc + "\n"
        f2.write(string)
    f2.close()



def divide_for_recognition(video_dir, triple_dir, result_dir):
    print('start divide for recognition!')

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    train_triples = []
    test_triples = []
    val_triples = []
    for video in os.listdir(video_dir):
        print("[{}] video:{} start processing".format(datetime.datetime.now(), video))

        video_triple_dir = os.path.join(triple_dir, video)

        triples = []
        with open(os.path.join(video_triple_dir, "new-triple.txt"), 'r') as nt:
            for line in nt:
                line = line.strip().split(";")
                triples.append((line[0], line[1], line[2], line[3]))

        scene_triple_dict = {}
        for i, triple in enumerate(triples):
            name1, name2, relation_id, desc = triple
            if desc == 'equal':
                continue 
            scene = name1.split("-")[1]
            name1 = name1.split("-")[3]
            name2 = name2.split("-")[3]
            if scene not in scene_triple_dict.keys():
                scene_triple_dict[scene] = [(name1, name2, relation_id)]
            else:
                scene_triple_dict[scene].append((name1, name2, relation_id))
        
        test_triple_dict = {}
        for scene, tps in scene_triple_dict.items():
            tps = list(set(tps))
            unique_list = []
            for triple in tps:
                name1, name2, r = triple
                if triple not in unique_list and (name2, name1, r) not in unique_list:
                    unique_list.append(triple)
        
                scene_id = int(scene)
                if scene_id % 7 == 0:
                    test_triple_dict[scene] = unique_list
            
        sample_triples = []
        for i, triple in enumerate(triples):
            name1, name2, relation_id, desc = triple
            scene = name1.split("-")[1]
            n1 = name1.split("-")[3]
            n2 = name2.split("-")[3]
            if scene in test_triple_dict.keys():
                if (n1, n2, relation_id) in test_triple_dict[scene] or (n2, n1, relation_id) in test_triple_dict[scene]:
                    sample_triples.append(triple)
        
        print("video:{}, samples:{}, all:{}".format(video, len(sample_triples), len(triples)))

        # 差集
        triples = list(set(triples).difference(set(sample_triples)))

        train_triples.extend(triples)
        test_triples.extend(sample_triples)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    f1 = open(os.path.join(result_dir, "train.txt"), 'w')
    for triple in train_triples:
        name1, name2, relation_id, desc = triple
        string = name1 + ";" + name2 + ";" + str(relation_id) + ";" + desc + "\n"
        f1.write(string)
    f1.close()

    f2 = open(os.path.join(result_dir, "test.txt"), 'w')
    for triple in test_triples:
        name1, name2, relation_id, desc = triple
        string = name1 + ";" + name2 + ";" + str(relation_id) + ";" + desc + "\n"
        f2.write(string)
    f2.close()
