import os
import sys
import numpy as np
import json
import datetime
from random import sample

'''
得到每个人物对应的 situations
'''

def add_situation(situation_dir, video_dir, data_dir):
    print('start add situation!')

    entity_file = os.path.join(data_dir, "entity2id.txt")
    entity2id = {}
    with open(entity_file, 'r') as f:
        for line in f:
            if len(line.strip().split(";")) > 1:
                entity, entity_id = line.strip().split(";")[0].strip(), line.strip().split(";")[1].strip()
                entity2id[entity] = entity_id

    scene_dict = {}
    for video in os.listdir(video_dir):
        print("[{}] video:{} start processing".format(datetime.datetime.now(), video))
        scene_file = os.path.join(situation_dir, video + ".txt")

        with open(scene_file, "r") as f:
            scene_str = f.read()

        scene_dict[video] = {}
        for scene in scene_str.split("scene-"):
            if len(scene) == 0:
                continue

            scene_split = scene.split("\n")
            for i, s in enumerate(scene_split):
                if i == 0:
                    scene_NO = s[0:3]
                    continue
                if len(s) == 0:
                    continue
                
                scene_name = s.strip()
                scene_dict[video][scene_NO] = scene_name

    result_file_path = os.path.join(data_dir, "entity2id.txt")

    scene_statistic_dict = {}
    result_file = open(result_file_path, "w")
    for entity, id in entity2id.items():
        entity_split = entity.split("-")
        video = entity_split[0]
        scene_id = entity_split[1]
        if scene_id in scene_dict[video].keys():
            scene_name = scene_dict[video][scene_id]
            if scene_name not in scene_statistic_dict.keys():
                scene_statistic_dict[scene_name] = 1
            else:
                scene_statistic_dict[scene_name] += 1
        else:
            print("warn!!, {} has not situations".format(entity))
            scene_name = 'other'
            if 'other' not in scene_statistic_dict.keys():
                scene_statistic_dict['other'] = 1
            else:
                scene_statistic_dict['other'] += 1
        s = entity + ";" + id + ";" + scene_name + "\n"

        result_file.write(s)
    result_file.close()

    scene_statistic_dict = sorted(scene_statistic_dict.items(), key = lambda kv:(int(kv[1]), kv[0]), reverse = True)
    statistic_file = open(os.path.join(data_dir, "statistic.txt"), 'w')
    for i, v in enumerate(scene_statistic_dict):
        key, val = v
        s = key + ":" + str(val) + "\n"
        statistic_file.write(s)
    statistic_file.close()
    print('***** Succeed. *****')            
    