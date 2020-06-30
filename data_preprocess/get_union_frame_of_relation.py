import os
import sys
import numpy as np
import json
import datetime

'''
author：yan
description: 取出每个视频中 两两人物存在关系的帧的并集和交集

每一个视频最终对应两个json， 分别为并集和交集
    key = video-scene-name1-name2-relation_label 
    value = frames_list
'''

# 根据sceneId匹配全称
def get_scene_full_name(scene_id, frame_label_list):
    for name in frame_label_list:
        if "scene-" + scene_id in name:
            return name

# 给定一个场景的label字典和两个人名，查并集和交集
def get_union_frames(name1, name2, scene_label_dirt):
    union_frames = []
    intersection_frames = []

    for key in scene_label_dirt:
        # 从key中取frameId
        frame_id = key.split('-')[2]

        # 每一个frame里可能包含多个人，取出来是list
        message_dirt_list = scene_label_dirt[key]

        # 用于取交集
        names = []

        for message_dirt in message_dirt_list:
            names.append(message_dirt["name"])

            # 取并集
            if message_dirt["name"] == name1 or message_dirt["name"] == name2: 
                # 去重
                if frame_id not in union_frames:
                    union_frames.append(frame_id)
        
        if name1 in names and name2 in names:
            intersection_frames.append(frame_id)
    
    union_frames.sort(key = lambda d:int(d))
    intersection_frames.sort(key = lambda d:int(d))
    return union_frames, intersection_frames

def get_union_frames_for_each_relation(video_dir, relation_label_dir, frame_label_dir, result_dir):
    print('start generate union frames.')

    start_time = datetime.datetime.now()
    for video in os.listdir(video_dir):
        print("[{}] video:{} start processing".format(datetime.datetime.now(), video))

        # 处理视频名 
        relation_label_path = os.path.join(relation_label_dir, video + '.txt')
        with open(relation_label_path, "r") as f:
            relation_str = f.read()
        video_triples = {}
        
        # 输出文件名
        result_file1 = os.path.join(result_dir, video + '-Union.json')
        result_file2 = os.path.join(result_dir, video + '-Intersection.json')
        if os.path.exists(result_file1) or os.path.exists(result_file2):
            continue
        union_result_dict = {}
        intersection_result_dict = {}

        # 处理关系三元组，得到video_triple字典。key = sceneId， value = list[(name, name, relation_label)]
        for scene in relation_str.split("scene-"):
            if len(scene) == 0:
                continue

            relations = scene.split("\n")
            for i, relation in enumerate(relations):
                if i == 0:
                    scene_NO = relation[0:3]
                    continue
                if len(relation) == 0:
                    continue
                clips = relation.split(";")
                if scene_NO in video_triples.keys():
                    video_triples[scene_NO].append((clips[0],clips[1],clips[2]))
                else:
                    video_triples[scene_NO] = [(clips[0],clips[1],clips[2])]
        
        # 得到该视频所有场景的 frame_label list
        video_frame_label_dir = os.path.join(frame_label_dir, video)
        frame_label_list = os.listdir(video_frame_label_dir)
        
        for key in video_triples:
            # 场景匹配
            scene_label_name = get_scene_full_name(key, frame_label_list)
            if not scene_label_name:
                print("[{}][WARN] Not scene label name for key:{}".format(datetime.datetime.now(), key))
                continue

            # load 该场景的标签json
            with open(os.path.join(video_frame_label_dir, scene_label_name), 'r') as fr:
                scene_label_dirt = json.load(fr)

            # 有些场景没有对应的json
            if not scene_label_dirt:
                print("[{}][WARN] {} not exist!".format(datetime.datetime.now(), scene_label_name))
                continue
            
            for triple in video_triples[key]:
                name1, name2, relation_label = triple
                union_frames, intersection_frames = get_union_frames(name1, name2, scene_label_dirt)

                # 这里只存不为空的list
                result_key = video + '-' + key + '-' + name1 + '-' + name2 + '-' + relation_label
                if union_frames:
                    union_result_dict[result_key] = union_frames
                if intersection_frames:
                    intersection_result_dict[result_key] = intersection_frames

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        with open(result_file1, 'w') as rf:
            json.dump(union_result_dict, rf)
        with open(result_file2, 'w') as rf:
            json.dump(intersection_result_dict, rf)
    
    end_time = datetime.datetime.now()
    print("Spend time:{}".format(end_time - start_time))
    print('***** Succeed. *****')
