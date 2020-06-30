import os
import sys
import numpy as np
import json
import datetime
from random import sample

'''
author：yan
description: 将每个场景中的三元组进行实体选取，并生成新三元组 + 时序三元组 + equal三元组
'''

# 根据sceneId匹配全称
def get_scene_full_name(scene_id, frame_label_list):
    for name in frame_label_list:
        if "scene-" + scene_id in name:
            return name

def get_scene_frame_label(scene_id, frame_label_list, video_frame_label_dir):
    # 场景匹配
    scene_label_name = get_scene_full_name(scene_id, frame_label_list)
    if not scene_label_name:
        print("[{}][WARN] Not scene label name for key:{}".format(datetime.datetime.now(), key))
        return null

    # load 该场景的标签json
    with open(os.path.join(video_frame_label_dir, scene_label_name), 'r') as fr:
        frames_label_dirt = json.load(fr)

    return frames_label_dirt

def get_feature_dict(video, name, frame_mes_list):
    # hack
    if video == "tt0109831":
        name = name.replace("_", " - ")
        
    for feature_dict in frame_mes_list:
        if feature_dict["name"] == name:
            return feature_dict
    return null

# 主要是为了筛选出已经提好特征的三元组，防止三元组中的实体无对应特征
def check_triples_feature(triples, time_based_triples, video_feature_dict):
    # 先处理正常三元组
    good_triples = []
    for triple in triples:
        is_good = check_triple(triple, video_feature_dict)
        if is_good:
            good_triples.append(triple)

    good_time_based_triples = []
    for time_based_triple in time_based_triples:
        is_good = check_triple(time_based_triple, video_feature_dict)
        if is_good:
            good_time_based_triples.append(time_based_triple)
    
    return good_triples, good_time_based_triples

def check_triple(triple, video_feature_dict):
    full_name1, full_name2 = triple[0], triple[1]
    person1_feature_dict = check_entity_feature(full_name1, video_feature_dict)
    if not person1_feature_dict:
        return False

    person2_feature_dict = check_entity_feature(full_name2, video_feature_dict)
    if not person2_feature_dict:
        return False
    
    return True

def check_entity_feature(full_name, video_feature_dict):
    full_name = full_name.split("-")
    frame_idx = full_name[0] + "-" + full_name[1] + "-" + full_name[2]
    person_name = full_name[3]
    # hack
    if full_name[0] == 'tt0109831':
        person_name = person_name.replace("_"," - ")

    try:
        frame_features = video_feature_dict[frame_idx]
    except Exception as e:
        print(e)
        return None
    person_feature_dict = list(filter(lambda x : x["name"] == person_name, frame_features))
    return person_feature_dict

def get_person_feature(full_name, video_feature_dict):
    full_name = full_name.split("-")
    frame_idx = full_name[0] + "-" + full_name[1] + "-" + full_name[2]
    person_name = full_name[3]
    if full_name[0] == 'tt0109831':
        person_name = person_name.replace("_"," - ")

    frame_features = video_feature_dict[frame_idx]
    person_feature_dict = list(filter(lambda x : x["name"] == person_name, frame_features))
    return person_feature_dict[0]["feature"]


def generate(frame_label_dir, video_dir, union_dir, feature_dir, result_dir, time_gen_gap, syn_gen_max, time_gen_max):
    print('start generate new triples!')
    total_triple_num = 0
    total_equal_num = 0
    total_entity_num = 0

    # 用于统计
    video_triple_count_dirt = {}
    video_equal_count_dirt = {}
    video_entity_count_dirt = {}
    start_time = datetime.datetime.now()
    for video in os.listdir(video_dir):
        print("[{}] video:{} start processing".format(datetime.datetime.now(), video))

        # 得到该视频所有场景的 frame_label list
        video_frame_label_dir = os.path.join(frame_label_dir, video)
        frame_label_list = os.listdir(video_frame_label_dir)

        # 读取提到的特征
        feature_file = os.path.join(feature_dir, video + '.json')
        with open(feature_file, "r") as rf:
            features_dict = json.load(rf)

        # 设置日志
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        m_path = os.path.join(result_dir, video)  
        if not os.path.exists(m_path):
            os.makedirs(m_path)   

        log_path = os.path.join(result_dir, video)
        log = open(os.path.join(log_path, "statistics.log"), "w")

        # 读取并集
        data_file = os.path.join(union_dir, video + '-Union.json')
        with open(data_file, "r") as fl:
            union_dirt = json.load(fl)
        
        # 记录新生成的三元组
        new_gen_triples = []

        # key = scene_id， value = dir<name, full_name> 便于生成equal关系
        name_dict = {}

        # 实体2id的映射
        name2entityId = {}

        # 实体的feature
        entity_features = []

        # 实体2坐标的映射
        name2bboxes = {}
        entityId = 1
        for key, union_list in union_dirt.items():
            # 特殊处理。。
            if video == "tt0109831":              
                key = key.replace(" - ", "_") 

            key_split = key.split("-")
            # key 的结构为 scene-sceneId-name1-name2-relationId
            scene_id = key_split[1]
            name1, name2 = key_split[2], key_split[3]
            relation_id = key_split[4]
           
            frames_label_dirt = get_scene_frame_label(scene_id, frame_label_list, video_frame_label_dir)
            if not frames_label_dirt:
                continue
           
            # 用于追踪时序关系
            last_frame_names = []
            last_frame_mes_list = []
            last_frame_id = -1

            # 用于暂存所有的实体、三元组。最后再进行抽样
            name2bboxes_temp = {}
            triples = []
            time_based_triples = []

            # 顺序遍历两人出现帧的并集
            for i, frame_id in enumerate(union_list):
                frames_key = video + "-" + scene_id + "-" + frame_id
                frame_mes_list = frames_label_dirt[frames_key]
                names = [feature_dict.get('name') for feature_dict in frame_mes_list]
                names = list(set(names))

                # trick
                if video == "tt0109831": 
                    for i in range(len(names)):
                        names[i] = names[i].replace(" - ", "_")

                # 判断是否能生成常规三元组
                if name1 in names and name2 in names:                 
                    name2bboxes_temp[video + '-' + scene_id + '-' + frame_id + '-' + name1] \
                        = get_feature_dict(video, name1, frame_mes_list)    
                    name2bboxes_temp[video + '-' + scene_id + '-' + frame_id + '-' + name2] \
                        = get_feature_dict(video, name2, frame_mes_list)

                    triple = (video + '-' + scene_id + '-' + frame_id + '-' + name1 ,
                            video + '-' + scene_id + '-' + frame_id + '-' + name2, relation_id, "ordinary")
                    triples.append(triple)
                
                # 判断是否能生成时序三元组, 暂时先考虑单向试试，如果得到的比较少就改成双向
                if len(last_frame_names) > 0:
                    # 顺序前后帧距离小于阈值才视为有关系
                    if name1 in last_frame_names and name2 in names and (int(frame_id) - int(last_frame_id)) < time_gen_gap:
                        full_name1 = video + '-' + scene_id + '-' + last_frame_id + '-' + name1
                        full_name2 = video + '-' + scene_id + '-' + frame_id + '-' + name2
                        if full_name1 not in name2bboxes_temp.keys():
                            name2bboxes_temp[full_name1] = get_feature_dict(video, name1, last_frame_mes_list)

                        name2bboxes_temp[full_name2] = get_feature_dict(video, name2, frame_mes_list)

                        triple = (full_name1 ,full_name2, relation_id, "time-based")
                        time_based_triples.append(triple)

                # 记录上一顺序帧
                last_frame_id = frame_id
                last_frame_names = names
                last_frame_mes_list = frame_mes_list

            log.write("scene:{} has {} new_triples\n".format(scene_id, len(triples)))
            log.write("scene:{} has {} new time_based_triples\n".format(scene_id, len(time_based_triples)))
            log.write("scene:{} has {} entity\n".format(scene_id, len(name2bboxes_temp)))
            log.write("---------------------------------------\n")

            triples, time_based_triples = check_triples_feature(triples, time_based_triples, features_dict)

            # 随机抽样
            if len(triples) > syn_gen_max:
                triples = sample(triples, syn_gen_max)
            new_gen_triples.extend(triples)

            if len(time_based_triples) > time_gen_max:
                time_based_triples = sample(time_based_triples, time_gen_max)
            new_gen_triples.extend(time_based_triples)
    
            for triple in (triples + time_based_triples):
                full_name1, full_name2, relation_id, _ = triple
                if full_name1 not in name2entityId.keys():
                    name2entityId[full_name1] = entityId
                    entityId += 1
                    name2bboxes[full_name1] = name2bboxes_temp[full_name1]
                    person_feature = get_person_feature(full_name1, features_dict)
                    entity_features.append(person_feature)

                # 这里存着的可能有重复，使用时需要先去重
                if scene_id not in name_dict.keys():
                    name_dict[scene_id] = {}
                    name_dict[scene_id][name1] = [full_name1]
                else:
                    if name1 not in name_dict[scene_id].keys():
                        name_dict[scene_id][name1] = [full_name1]
                    else:
                        name_dict[scene_id][name1].append(full_name1)

                if full_name2 not in name2entityId.keys():
                    name2entityId[full_name2] = entityId
                    entityId += 1
                    name2bboxes[full_name2] = name2bboxes_temp[full_name2]
                    person_feature = get_person_feature(full_name2, features_dict)
                    entity_features.append(person_feature)

                if name2 not in name_dict[scene_id].keys():
                    name_dict[scene_id][name2] = [full_name2]
                else:
                    name_dict[scene_id][name2].append(full_name2)

        # 生成equal关系
        equal_triple = []
        for scene_id, full_name_dir in name_dict.items():
            for names in full_name_dir.values():
                #  去重
                names = list(set(names))
                names.sort(key = lambda i: int(i.split("-")[2]))

                for i, name in enumerate(names):
                    if i == 0:
                        continue
                    last_name = names[i-1]
                    triple = (last_name, name, 8, "equal")
                    equal_triple.append(triple)

        new_gen_triples.extend(equal_triple)
        log.write("---------final result----------\n")
        log.write("video {} has {} equal triples\n".format(video, len(equal_triple)))
        log.write("video {} has {} new_triples\n".format(video, len(new_gen_triples)))
        #log.write("equal relation ratio:{}\n".format(float(len(equal_triple)/len(new_gen_triples))))   
        log.write("video {} has {} entities\n".format(video, len(name2entityId)))
        log.close()

        total_equal_num += len(equal_triple)
        total_triple_num += len(new_gen_triples)
        total_entity_num += len(name2entityId)
        video_equal_count_dirt[video] = len(equal_triple)
        video_triple_count_dirt[video] = len(new_gen_triples)
        video_entity_count_dirt[video] = len(name2entityId)

        # 将结果写入文件   
        result_file1 = os.path.join(m_path, "new-triple.txt")
        result_file2 = os.path.join(m_path, "entity2id.txt")
        result_file3 = os.path.join(m_path, "bboxes.json")
        result_file4 = os.path.join(m_path, "features.txt")

        f1 = open(result_file1, "w")
        for triple in new_gen_triples:
            name1, name2, relation_id, desc = triple
            string = name1 + ";" + name2 + ";" + str(relation_id) + ";" + desc + "\n"
            f1.write(string)
        f1.close()

        f2 = open(result_file2, "w")
        for key, value in name2entityId.items():
            string = key + ";" + str(value) + "\n"
            f2.write(string)
        f2.close()

        with open(result_file3, "w") as f3:
            json.dump(name2bboxes, f3)

        f4 = open(result_file4, "w") 
        for i, feature in enumerate(entity_features):
            # 降维
            feature = str(feature)
            feature = feature.replace("[",'').replace("]",'')
            feature = list(eval(feature))
            # 处理为字符串
            feature = [str(i) + ' ' for i in feature]
            string = "".join(feature) + "\n"
            f4.write(string)
        f4.close()

    statistics_path = os.path.join(result_dir, "statistics.txt")
    s_file = open(statistics_path, "w")
    s_file.write("total equal number:{}\n".format(total_equal_num))
    for video in os.listdir(video_dir):
        s_file.write("video {} has {} equal triples\n".format(video, video_equal_count_dirt[video]))
    s_file.write("------------------------------\n")

    s_file.write("total triples number:{}\n".format(total_triple_num))
    for video in os.listdir(video_dir):
        s_file.write("video {} has {} new_triples\n".format(video, video_triple_count_dirt[video]))
    s_file.write("------------------------------\n")

    s_file.write("total entities number:{}\n".format(total_entity_num))
    for video in os.listdir(video_dir):
        s_file.write("video {} has {} entities\n".format(video, video_entity_count_dirt[video]))

    end_time = datetime.datetime.now()
    print("Spend time:{}".format(end_time - start_time))
    print('***** Succeed. *****')

