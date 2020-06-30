import numpy as np
import argparse

from get_union_frame_of_relation import get_union_frames_for_each_relation
from generate_new_triple import generate
from process_triples import collect
from divide_complete import divide_for_prediction, divide_for_recognition
from get_entity_with_situation import add_situation

'''
处理数据
'''
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("-video_dir", "--video_dir", default = "/home/devsdb/MovieGraphs/all_frames/")
    args.add_argument("-r_label_dir", "--relation_label_dir", default = "/home/devsdb/MovieGraphs/relation_label_final")
    args.add_argument("-f_label_dir", "--frame_label_dir", default = "/home/devsdb/MovieGraphs/frames_label")
    args.add_argument("-union_frame_dir", "--union_frame_dir", default = "/home/devsdb/MovieGraphs/ych_result/tmp/Movie_frame_in_pairs_relation")
    args.add_argument("-feature_dir", "--feature_dir", default = "/home/devsdb/MovieGraphs/feature_Arcface")
    args.add_argument("-triples_dir", "--triples_dir", default = "/home/devsdb/MovieGraphs/ych_result/tmp/New_Triple_ArcFace", help = "生成的三元组的存放目录")
    args.add_argument("-data", "--data", default = "/home/devsdb/MovieGraphs/ych_result/tmp/MG_DATA", help = "处理好的数据的目录")
    args.add_argument("-situation_dir", "--situation_dir", default = "/home/devsdb/MovieGraphs/situations")
    args.add_argument("-time_gap", "--time_gen_gap", type = int, default = 60)
    args.add_argument("-syn_max", "--syn_triples_gen_max", type = int, default = 8)
    args.add_argument("-time_max", "--time_triples_gen_max", type = int, default = 4)
    args.add_argument("-prediction", "--prediction", type = bool, default = True)

    args = args.parse_args()

    # 先提取帧并集
    get_union_frames_for_each_relation(args.video_dir, args.relation_label_dir, args.frame_label_dir, args.union_frame_dir)

    # 生成三元组
    generate(args.frame_label_dir, args.video_dir, args.union_frame_dir, args.feature_dir, args.triples_dir, 
        args.time_gen_gap, args.syn_triples_gen_max, args.time_triples_gen_max)

    # 汇集各个视频片段中筛选的实体和特征到一个文件
    collect(args.video_dir, args.triples_dir, args.data)

    # 划分三元组为train/test
    if args.prediction:
        divide_for_prediction(args.video_dir, args.triples_dir, args.data)
    else:
        divide_for_recognition(args.video_dir, args.triples_dir, args.data)

    # 添加每个实体对应的情景信息
    add_situation(args.situation_dir, args.video_dir, args.data)
    