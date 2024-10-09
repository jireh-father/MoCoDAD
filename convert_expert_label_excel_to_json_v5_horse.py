import argparse
import sys

import pandas as pd
import os
import json
import random
import math
import numpy as np
import shutil


def filter_positive_df(df, args):
    if args.dont_use_ambiguous_label:
        return df[(df.vet_label_lameness >= args.score_threshold) & (df.vet_label_lameness % 1 == 0)]
    else:
        return df[(df.vet_label_lameness >= args.score_threshold)]


def filter_negative_df(df, args):
    if args.train_score_threshold is not None:
        if args.dont_use_ambiguous_label:
            return df[(df.vet_label_lameness <= args.train_score_threshold) | (df.vet_label_lameness % 1 != 0)]
        else:
            return df[(df.vet_label_lameness <= args.train_score_threshold)]
    else:
        if args.dont_use_ambiguous_label:
            return df[(df.vet_label_lameness < args.score_threshold) | (df.vet_label_lameness % 1 != 0)]
        else:
            return df[(df.vet_label_lameness < args.score_threshold)]


def is_positive_sample(label, args):
    if args.dont_use_ambiguous_label:
        # if label % 1 > 0:
        #     print("over 1", label % 1, label >= args.score_threshold and label % 1 == 0)
        return label >= args.score_threshold and label % 1 == 0
    else:
        return label >= args.score_threshold


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.label_path)

    print("ori df.shape", df.shape)

    if args.horse_types:
        horse_types = args.horse_types.split(",")
        df = df[df['purpose'].isin(horse_types)]
        print("horse_types filtering df.shape", df.shape)

    if args.directions:
        directions = args.directions.split(",")
        df = df[df['direction'].isin(directions)]
        print("directions filtering df.shape", df.shape)

    if args.gait_types:
        gait_types = args.gait_types.split(",")
        df = df[df['gait'].isin(gait_types)]
        print("gait_types filtering df.shape", df.shape)

    df['lameness_final'] = df['lameness_final'].apply(lambda x: int(x))
    df['num_labels'] = df['num_labels'].apply(lambda x: int(x))
    df = df[((df.lameness_final == 1) & (df.num_labels >= args.pos_num_labels_threshold)) | (
            (df.lameness_final == 0) & (df.num_labels >= args.neg_num_labels_threshold))]
    print("score threshold df.shape", df.shape)

    df['frame_count'] = df['frame_count'].apply(lambda x: int(x))
    if args.min_frames:
        df = df[df['frame_count'] >= args.min_frames]
        print("min_frames df.shape", df.shape)

    if args.remove_missing_frame_samples:
        df = df[df['num_missing_frames'] == 0]
        print("remove_missing_frame_samples df.shape", df.shape)

    if args.remove_multi_video_samples:
        df = df[df['has_multiple_videos'] == False]
        print("remove_multi_video_samples df.shape", df.shape)

    df = df.drop_duplicates(subset=['keypoint_path'])
    print("drop_duplicates df.shape", df.shape)

    # check if path exists
    if args.check_exist_keypoint_path:
        no_exist_keypoint_paths = []
        for keypoint_path, mp4_filename in df[['keypoint_path', 'mp4_filename']].values:
            if args.change_kp_path:
                if "/auto/" in keypoint_path:
                    # real_keypoint_path = keypoint_path.replace("/auto/", f"/infer/Video/{os.path.splitext(mp4_filename)[0]}/")
                    real_keypoint_path = keypoint_path.replace("/auto/", f"/{os.path.splitext(mp4_filename)[0]}/")
                else:
                    real_keypoint_path = keypoint_path.replace("/label/", f"/{os.path.splitext(mp4_filename)[0]}/")
            else:
                real_keypoint_path = keypoint_path
            server_keypoint_path = os.path.join(args.server_path, real_keypoint_path)
            if not os.path.exists(server_keypoint_path):
                no_exist_keypoint_paths.append(keypoint_path)
        if no_exist_keypoint_paths:
            df = df[~df['keypoint_path'].isin(no_exist_keypoint_paths)]
            print("no exist kp file filtering df.shape", df.shape)

    horse_ids = df['horse_id'].unique()
    drop_horse_ids = []
    for horse_id in horse_ids:
        label_df = df[(df['horse_id'] == horse_id)]
        if len(label_df) == 1:
            continue
        if label_df[
            'lameness_final'].nunique() > 1:  # or label_df['sum_lameness'].nunique() > 1:# or label_df['num_labels'].nunique() > 1:
            drop_horse_ids.append(horse_id)
    if drop_horse_ids:
        df = df[~df['horse_id'].isin(drop_horse_ids)]
        print("wrong label horse ids, df.shape", df.shape)

    print("total samples", len(df))
    print("total horses", len(df['horse_id'].unique()))
    print("total abnormal samples", len(df[df.lameness_final == 1]))
    print("total normal samples", len(df[df.lameness_final == 0]))
    print("total abnormal horses", len(df[df.lameness_final == 1]['horse_id'].unique()))
    print("total normal horses", len(df[df.lameness_final == 0]['horse_id'].unique()))

    num_val = int(len(df[df.lameness_final == 1]) * args.pos_val_ratio)
    if args.max_train_samples and args.max_train_samples + num_val < len(df[df.lameness_final == 0]):
        df = pd.concat([df[df.lameness_final == 0].sample(n=args.max_train_samples + num_val, random_state=args.seed),
                        df[df.lameness_final == 1]])
        print("max_train_samples df.shape", df.shape)

    if len(df) < 1:
        print("no data")
        sys.exit()

    if args.output_filtered_csv_path:
        df.to_csv(args.output_filtered_csv_path, index=False)

    # shuffle df
    df = df.sample(frac=1).reset_index(drop=True)

    horse_ids = df['horse_id'].unique()

    cur_num_pos_val = 0
    cur_num_neg_val = 0

    result = []
    no_exist_horse_list = []

    for horse_id in horse_ids:
        label_df = df[(df['horse_id'] == horse_id)]

        lameness_int = int(label_df['lameness_final'].iloc[0])
        is_val = False
        if lameness_int > 0:
            if cur_num_pos_val < num_val:
                cur_num_pos_val += len(label_df)
                is_val = True
        else:
            if cur_num_neg_val < num_val:
                cur_num_neg_val += len(label_df)
                is_val = True

        if args.all_val:
            is_val = True
        if args.all_neg_train and lameness_int == 0:
            is_val = False
        # iter label_df
        for row_index, sub_df in label_df.iterrows():
            label_item = {
                "keypoints": {"path_and_direction": []},
                # "symptoms": set(),
                "isVal": is_val
            }
            row = sub_df
            keypoint_path = row['keypoint_path'].strip()

            if args.change_kp_path:
                # keypoint_path = keypoint_path.replace("/auto/", f"/infer/VIDEO/{os.path.splitext(row['mp4_filename'])[0]}/")
                if "/auto/" in keypoint_path:
                    keypoint_path = keypoint_path.replace("/auto/", f"/{os.path.splitext(row['mp4_filename'])[0]}/")
                else:
                    keypoint_path = keypoint_path.replace("/label/", f"/{os.path.splitext(row['mp4_filename'])[0]}/")

            if args.kp_file_name:
                keypoint_path = f"{os.path.dirname(keypoint_path)}/{args.kp_file_name}"

            if args.copy_keypoint_from_server:
                keypoint_dir = os.path.join(args.keypoint_dir, os.path.dirname(keypoint_path))
                os.makedirs(keypoint_dir, exist_ok=True)
                # skip if exists
                if not os.path.exists(os.path.join(keypoint_dir, os.path.basename(keypoint_path))):
                    if not os.path.exists(os.path.join(args.server_path, keypoint_path)) and args.skip_no_exist_kp_file:
                        no_exist_horse_list.append(keypoint_path)
                        continue
                    shutil.copy(os.path.join(args.server_path, keypoint_path), keypoint_dir)

            direction = row['direction'].strip()
            kp_item = {
                "keypoint_full_path": keypoint_path,
                "direction": direction
            }
            label_item["keypoints"]["path_and_direction"].append(kp_item)
            if lameness_int > 0:
                sick_legs = None
                for pos_idx in range(2, 5):
                    if not pd.isnull(row[f"sick_leg{pos_idx}"]):
                        sick_legs = row[f"sick_leg{pos_idx}"]
                        break
                if sick_legs:
                    sick_legs = sick_legs.replace("RH", "RB").replace("LH", "LB")
                    positions = sick_legs.split(",")
                else:
                    positions = ["None"]
            else:
                positions = ["None"]
            label_item["leg_position"] = positions

            label_item["lameness"] = lameness_int > 0

            label_item['horse_id'] = row['horse_id'].strip()

            result.append(label_item)

    json.dump(result, open(args.output_path, "w+", encoding="utf-8"), ensure_ascii=False)
    print("no_exist_horse_list")
    print(json.dumps(no_exist_horse_list))
    if no_exist_horse_list:
        json.dump(no_exist_horse_list, open(args.output_path.replace(".json", "_no_exist_horse_list.json"), "w+"),
                  ensure_ascii=False)
    print("len no_exist_horse_list", len(no_exist_horse_list))
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 라벨 csv 파일 경로
    parser.add_argument('--label_path', type=str,
                        # default='E:\dataset\\afp\horse\\20240607/merged_with_score_with_missing_frames_20240710.csv')
                        default='./merged_with_score_with_missing_frames_20240710.csv')
    # keypoint저장된 서버의 루트 폴더 경로
    parser.add_argument('--server_path', type=str, default='Y:\RnD\Project\horse\\NK')

    # json파일 생성 경로
    parser.add_argument('--output_path', type=str, default='./horse_20240607_test.json')

    # 테스트용
    parser.add_argument('--output_filtered_csv_path', type=str, default=None)  # './horse_20240607_test.csv')

    # 키포인트 csv파일이 경로에 존재하는지 체크
    parser.add_argument('--check_exist_keypoint_path', action='store_true', default=False)

    parser.add_argument('--seed', type=int, default=10)

    # keypoint csv파일들이 저장된 루트 경로
    parser.add_argument('--keypoint_dir', type=str, default='E:\dataset\\afp\horse_kp_20240710')

    # 키포인트 파일이 없을경우 서버에서 카피해올지 여부
    parser.add_argument('--copy_keypoint_from_server', action='store_true', default=False)

    # max_train_samples
    parser.add_argument('--max_train_samples', type=int, default=300)

    # abnormal 데이터의 val 비율. 1.0이면 전체 abnormal 데이터를 val로 사용. anomaly detection에서는 1.0으로 설정
    parser.add_argument('--pos_val_ratio', type=float, default=1.0)

    # positive 샘플의 라벨러 숫자 threshold
    parser.add_argument('--pos_num_labels_threshold', type=float, default=2)
    # negative 샘플의 라벨러 숫자 threshold
    parser.add_argument('--neg_num_labels_threshold', type=float, default=2)

    # min_frames. 이것보다 적으면 제거
    parser.add_argument('--min_frames', type=int, default=30)  # 30)

    # horse_type, ride or race, 현재는 전부 사용중
    parser.add_argument('--horse_types', type=str, default=None)  # 'ride,race'

    # gait_type, walk or trot
    parser.add_argument('--gait_types', type=str, default='trot')

    # 걷는 방향. BtoF,FtoB,LtoR,RtoL
    parser.add_argument('--directions', type=str, default='LtoR,RtoL')  # 'LtoR,RtoL')  # BtoF,FtoB  LtoR,RtoL

    # all_val, 모든 데이터 강제로 val으로 세팅, 테스트용
    parser.add_argument('--all_val', action='store_true', default=False)

    # all_neg_train, 모든 normal 데이터 강제로 train으로 세팅, 테스트용
    parser.add_argument('--all_neg_train', action='store_true', default=False)

    # remove_missing_frame_samples, 빠진 frame이 있는 동영상 제외
    parser.add_argument('--remove_missing_frame_samples', action='store_true', default=False)

    # remove_multi_video_samples, 하나의 키포인트 파일에 여러개 동영상의 프레임이 섞여 있는 샘플 제거
    parser.add_argument('--remove_multi_video_samples', action='store_true', default=False)

    # 키포인트 파일 없어도 서버에서 카피하지 말고 skip할지 여부
    parser.add_argument('--skip_no_exist_kp_file', action='store_true', default=False)

    # change_kp_path, 키포인트 경로나 파일명이 달라질 경우, 예외처리를 위한..
    parser.add_argument('--change_kp_path', action='store_true', default=False)
    
    # 키포인트 경로나 파일명이 달라질 경우 예외처리를 위한
    parser.add_argument('--kp_file_name', type=str, default=None)  # coords.csv

    main(parser.parse_args())
