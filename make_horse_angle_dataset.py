import argparse
import copy
import glob
import os
import json
import sys

import pandas as pd
import numpy as np
import math

KEYPOINT_COLS = ['bodyparts', 'Unnamed: 1', 'Unnamed: 2', 'Nostril_x', 'Nostril_y', 'Eye_x',
                 'Eye_y', 'Poll_x', 'Poll_y', 'Withers_x', 'Withers_y', 'LowestBack_x',
                 'LowestBack_y', 'T16L1_x', 'T16L1_y', 'T_sacrale_x', 'T_sacrale_y',
                 'Tail_Root_x', 'Tail_Root_y', 'T_ischiadicum_x', 'T_ischiadicum_y', 'Tub_x',
                 'Tub_y', 'Spina_scapulae_x', 'Spina_scapulae_y', 'ElbowJoint_L_x',
                 'ElbowJoint_L_y', 'ElbowJoint_R_x', 'ElbowJoint_R_y', 'Carpuse_L_x',
                 'Carpuse_L_y', 'Carpuse_R_x', 'Carpuse_R_y', 'Fetlock_L_x', 'Fetlock_L_y',
                 'Fetlock_R_x', 'Fetlock_R_y', 'Front_Heel_L_x', 'Front_Heel_L_y',
                 'Front_Heel_R_x', 'Front_Heel_R_y', 'Front_Toe_L_x', 'Front_Toe_L_y',
                 'Front_Toe_R_x', 'Front_Toe_R_y', 'Abdomen_x', 'Abdomen_y', 'T_Coxae_x',
                 'T_Coxae_y', 'Coxofemoral_x', 'Coxofemoral_y', 'Stifle_Joint_L_x',
                 'Stifle_Joint_L_y', 'Stifle_Joint_R_x', 'Stifle_Joint_R_y',
                 'Rear_Tarsus_L_x', 'Rear_Tarsus_L_y', 'Rear_Tarsus_R_x', 'Rear_Tarsus_R_y',
                 'Rear_Fetlock_L_x', 'Rear_Fetlock_L_y', 'Rear_Fetlock_R_x',
                 'Rear_Fetlock_R_y', 'Rear_Heel_L_x', 'Rear_Heel_L_y', 'Rear_Heel_R_x',
                 'Rear_Heel_R_y', 'Rear_Toe_L_x', 'Rear_Toe_L_y', 'Rear_Toe_R_x',
                 'Rear_Toe_R_y']

KEYPOINT_COLS_WITH_SCORE = ['bodyparts', 'Unnamed: 1', 'Unnamed: 2', 'Nostril_x', 'Nostril_y', 'Nostril_score',
                            'Eye_x', 'Eye_y', 'Eye_score', 'Poll_x', 'Poll_y', 'Poll_score', 'Withers_x', 'Withers_y',
                            'Withers_score', 'LowestBack_x', 'LowestBack_y', 'LowestBack_score', 'T16L1_x', 'T16L1_y',
                            'T16L1_score', 'T_sacrale_x', 'T_sacrale_y', 'T_sacrale_score', 'Tail_Root_x',
                            'Tail_Root_y',
                            'Tail_Root_score', 'T_ischiadicum_x', 'T_ischiadicum_y', 'T_ischiadicum_score', 'Tub_x',
                            'Tub_y', 'Tub_score', 'Spina_scapulae_x', 'Spina_scapulae_y', 'Spina_scapulae_score',
                            'ElbowJoint_L_x', 'ElbowJoint_L_y', 'ElbowJoint_L_score', 'ElbowJoint_R_x',
                            'ElbowJoint_R_y',
                            'ElbowJoint_R_score', 'Carpuse_L_x', 'Carpuse_L_y', 'Carpuse_L_score', 'Carpuse_R_x',
                            'Carpuse_R_y', 'Carpuse_R_score', 'Fetlock_L_x', 'Fetlock_L_y', 'Fetlock_L_score',
                            'Fetlock_R_x', 'Fetlock_R_y', 'Fetlock_R_score', 'Front_Heel_L_x', 'Front_Heel_L_y',
                            'Front_Heel_L_score', 'Front_Heel_R_x', 'Front_Heel_R_y', 'Front_Heel_R_score',
                            'Front_Toe_L_x', 'Front_Toe_L_y', 'Front_Toe_L_score', 'Front_Toe_R_x', 'Front_Toe_R_y',
                            'Front_Toe_R_score', 'Abdomen_x', 'Abdomen_y', 'Abdomen_score', 'T_Coxae_x', 'T_Coxae_y',
                            'T_Coxae_score', 'Coxofemoral_x', 'Coxofemoral_y', 'Coxofemoral_score',
                            'Stifle_Joint_L_x', 'Stifle_Joint_L_y', 'Stifle_Joint_L_score', 'Stifle_Joint_R_x',
                            'Stifle_Joint_R_y', 'Stifle_Joint_R_score', 'Rear_Tarsus_L_x', 'Rear_Tarsus_L_y',
                            'Rear_Tarsus_L_score', 'Rear_Tarsus_R_x', 'Rear_Tarsus_R_y', 'Rear_Tarsus_R_score',
                            'Rear_Fetlock_L_x', 'Rear_Fetlock_L_y', 'Rear_Fetlock_L_score', 'Rear_Fetlock_R_x',
                            'Rear_Fetlock_R_y', 'Rear_Fetlock_R_score', 'Rear_Heel_L_x', 'Rear_Heel_L_y',
                            'Rear_Heel_L_score', 'Rear_Heel_R_x', 'Rear_Heel_R_y', 'Rear_Heel_R_score',
                            'Rear_Toe_L_x', 'Rear_Toe_L_y', 'Rear_Toe_L_score', 'Rear_Toe_R_x', 'Rear_Toe_R_y',
                            'Rear_Toe_R_score']

BACK_KP_COLS = ['bodyparts', 'Unnamed: 1', 'Unnamed: 2', 'Tail_root_x', 'Tail_root_y', 'T_Coxae_L_x',
                'T_Coxae_L_y', 'T_Coxae_R_x', 'T_Coxae_R_y', 'Stifile_Joint_L_x', 'Stifile_Joint_L_y',
                'Stifile_Joint_R_x', 'Stifile_Joint_R_y', 'T_ischiadicum_L_x', 'T_ischiadicum_L_y',
                'T_ischiadicum_R_x', 'T_ischiadicum_R_y', 'Hock_L_x', 'Hock_L_y', 'Hock_R_x', 'Hock_R_y',
                'Fetlock_Rear_L_x', 'Fetlock_Rear_L_y', 'Fetlock_Rear_R_x', 'Fetlock_Rear_R_y',
                'Hoof_Rear_L_x', 'Hoof_Rear_L_y', 'Hoof_Rear_R_x', 'Hoof_Rear_R_y', ]
FRONT_KP_COLS = ['bodyparts', 'Unnamed: 1', 'Unnamed: 2', 'Forehead_x', 'Forehead_y', 'Nasal_bridge_x',
                 'Nasal_bridge_y', 'Muzzle_x', 'Muzzle_y', 'Elbow_L_x', 'Elbow_L_y', 'Elbow_R_x', 'Elbow_R_y',
                 'Shoulder_C_x', 'Shoulder_C_y', 'Shoulder_L_x', 'Shoulder_L_y', 'Shoulder_R_x', 'Shoulder_R_y',
                 'Carpus_Front_L_x', 'Carpus_Front_L_y', 'Carpus_Front_R_x', 'Carpus_Front_R_y',
                 'Fetlock_Front_L_x', 'Fetlock_Front_L_y', 'Fetlock_Front_R_x', 'Fetlock_Front_R_y',
                 'Hoof_Front_L_x', 'Hoof_Front_L_y', 'Hoof_Front_R_x', 'Hoof_Front_R_y', ]

# Withers,Throat,R_F_Elbow,R_F_Knee,R_F_Paw,L_F_Elbow,L_F_Knee,L_F_Paw,TailBase,R_B_Elbow,R_B_Knee,R_B_Paw,L_B_Elbow,L_B_Knee,L_B_Paw
TARGET_KP_COL_DICT = {
    "baseline": [
        [
            'Front_Toe_R', 'Front_Heel_R', 'Fetlock_R', 'Carpuse_R', 'ElbowJoint_R', 'Tub', 'Spina_scapulae'
        ],
        [
            'Front_Toe_L', 'Front_Heel_L', 'Fetlock_L', 'Carpuse_L', 'ElbowJoint_L', 'Tub', 'Spina_scapulae'
        ],
        [
            'Rear_Toe_R', 'Rear_Heel_R', 'Rear_Fetlock_R', 'Rear_Tarsus_R', 'Stifle_Joint_R', 'Coxofemoral', 'T_Coxae'
        ],
        [
            'Rear_Toe_L', 'Rear_Heel_L', 'Rear_Fetlock_L', 'Rear_Tarsus_L', 'Stifle_Joint_L', 'Coxofemoral', 'T_Coxae'
        ],
        [
            'Nostril', 'Eye', 'Poll', 'Withers', 'LowestBack', 'T16L1', 'T_sacrale', 'Tail_Root', 'T_ischiadicum',
        ],
        [
            'Coxofemoral', 'T_Coxae', 'LowestBack', 'Withers'
        ],
        [
            'T_Coxae', 'LowestBack', 'T16L1'
        ],
        [
            'Tub', 'Spina_scapulae', 'Withers', 'LowestBack'
        ],
        [
            'Spina_scapulae', 'Withers', 'LowestBack', 'Poll'
        ],
        [
            'Abdomen', 'LowestBack', 'Withers'
        ],
        [
            'Abdomen', 'LowestBack', 'T16L1'
        ],
        [
            'Abdomen', 'LowestBack', 'T_Coxae'
        ]
    ],
    "no_head_and_tail": [
        [
            'Front_Toe_R', 'Front_Heel_R', 'Fetlock_R', 'Carpuse_R', 'ElbowJoint_R', 'Tub', 'Spina_scapulae'
        ],
        [
            'Front_Toe_L', 'Front_Heel_L', 'Fetlock_L', 'Carpuse_L', 'ElbowJoint_L', 'Tub', 'Spina_scapulae'
        ],
        [
            'Rear_Toe_R', 'Rear_Heel_R', 'Rear_Fetlock_R', 'Rear_Tarsus_R', 'Stifle_Joint_R', 'Coxofemoral', 'T_Coxae'
        ],
        [
            'Rear_Toe_L', 'Rear_Heel_L', 'Rear_Fetlock_L', 'Rear_Tarsus_L', 'Stifle_Joint_L', 'Coxofemoral', 'T_Coxae'
        ],
        [
            'Withers', 'LowestBack', 'T16L1', 'T_sacrale', 'Tail_Root',
        ],
        [
            'Coxofemoral', 'T_Coxae', 'LowestBack', 'Withers'
        ],
        [
            'T_Coxae', 'LowestBack', 'T16L1'
        ],
        [
            'Tub', 'Spina_scapulae', 'Withers', 'LowestBack'
        ],
        [
            'Spina_scapulae', 'Withers', 'LowestBack', 'Poll'
        ],
        [
            'Abdomen', 'LowestBack', 'Withers'
        ],
        [
            'Abdomen', 'LowestBack', 'T16L1'
        ],
        [
            'Abdomen', 'LowestBack', 'T_Coxae'
        ]
    ],
    "only_legs": [
        [
            'Front_Toe_R', 'Front_Heel_R', 'Fetlock_R', 'Carpuse_R', 'ElbowJoint_R', 'Tub', 'Spina_scapulae'
        ],
        [
            'Front_Toe_L', 'Front_Heel_L', 'Fetlock_L', 'Carpuse_L', 'ElbowJoint_L', 'Tub', 'Spina_scapulae'
        ],
        [
            'Rear_Toe_R', 'Rear_Heel_R', 'Rear_Fetlock_R', 'Rear_Tarsus_R', 'Stifle_Joint_R', 'Coxofemoral', 'T_Coxae'
        ],
        [
            'Rear_Toe_L', 'Rear_Heel_L', 'Rear_Fetlock_L', 'Rear_Tarsus_L', 'Stifle_Joint_L', 'Coxofemoral', 'T_Coxae'
        ],
    ],
    "only_foots": [
        [
            'Front_Toe_R', 'Front_Heel_R', 'Fetlock_R'
        ],
        [
            'Front_Toe_L', 'Front_Heel_L', 'Fetlock_L'
        ],
        [
            'Rear_Toe_R', 'Rear_Heel_R', 'Rear_Fetlock_R'
        ],
        [
            'Rear_Toe_L', 'Rear_Heel_L', 'Rear_Fetlock_L'
        ],
    ],
    "only_foots_and_fetlocks": [
        [
            'Front_Toe_R', 'Front_Heel_R', 'Fetlock_R', 'Carpuse_R'
        ],
        [
            'Front_Toe_L', 'Front_Heel_L', 'Fetlock_L', 'Carpuse_L'
        ],
        [
            'Rear_Toe_R', 'Rear_Heel_R', 'Rear_Fetlock_R', 'Rear_Tarsus_R'
        ],
        [
            'Rear_Toe_L', 'Rear_Heel_L', 'Rear_Fetlock_L', 'Rear_Tarsus_L'
        ],
    ],
    "only_foots_and_fetlocks_and_carpuse": [
        [
            'Front_Toe_R', 'Front_Heel_R', 'Fetlock_R', 'Carpuse_R', 'ElbowJoint_R'
        ],
        [
            'Front_Toe_L', 'Front_Heel_L', 'Fetlock_L', 'Carpuse_L', 'ElbowJoint_L'
        ],
        [
            'Rear_Toe_R', 'Rear_Heel_R', 'Rear_Fetlock_R', 'Rear_Tarsus_R', 'Stifle_Joint_R'
        ],
        [
            'Rear_Toe_L', 'Rear_Heel_L', 'Rear_Fetlock_L', 'Rear_Tarsus_L', 'Stifle_Joint_L'
        ],
    ]
}


# 두 벡터의 내적을 계산하는 함수


def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


# 벡터의 크기를 계산하는 함수
def magnitude(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2)


# 두 벡터 사이의 각도를 구하는 함수 (라디안)
def angle_between(v1, v2):
    dot_prod = dot_product(v1, v2)
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)

    # magnitude 값이 0인 경우는 벡터의 크기가 0이므로 각도를 계산할 수 없으니 예외 처리
    if mag1 == 0 or mag2 == 0:
        return float('nan')

    # 내적 결과를 -1과 1 사이로 제한하여 math.acos()에서 발생하는 에러 방지
    cos_angle = dot_prod / (mag1 * mag2)
    cos_angle = max(-1, min(1, cos_angle))  # 클램핑 처리

    return math.acos(cos_angle)


# 각도를 도(degree)로 변환하는 함수
def rad_to_deg(rad):
    return math.degrees(rad)


# 세 점의 사이 각을 구하는 함수
def angle_between_points(row, first_key, second_key, third_key):
    p1 = (row[f'{first_key}_x'], row[f'{first_key}_y'])
    p2 = (row[f'{second_key}_x'], row[f'{second_key}_y'])
    p3 = (row[f'{third_key}_x'], row[f'{third_key}_y'])

    # 벡터 p1->p2와 p1->p3을 구함
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p1[0], p3[1] - p1[1])

    # 두 벡터 사이의 각도를 라디안으로 구함
    angle_rad = angle_between(v1, v2)

    # 각도를 도(degree)로 변환
    return rad_to_deg(angle_rad)


def read_csv(csv_file, all_keys, all_x_axis_keys, target_skeleton_key_sets, window_length, direction, num_div=None,
             num_thr_div=0.1, frame_stride=None, max_frames=None, use_random_frame_range=False,
             reset_index=False, skip_not_continuous_sample=False, use_score_col=False, sort_max_frames=False):
    try:
        df = pd.read_csv(csv_file, skiprows=lambda x: x in [2], header=1, encoding='CP949')
    except:
        df = pd.read_csv(csv_file, skiprows=lambda x: x in [2], header=1, encoding='utf-8')
    # print(df.index.values)

    if len(df) < window_length:
        return False, False

    len_df = len(df)


    if direction == 'side':
        cols = KEYPOINT_COLS
        if use_score_col:
            cols = KEYPOINT_COLS_WITH_SCORE
    elif direction == 'front':
        cols = FRONT_KP_COLS
    elif direction == 'back':
        cols = BACK_KP_COLS
    else:
        raise Exception("invalid direction", direction)

    if len(df.columns) != len(cols):
        raise Exception("invalid df keys", df.columns, csv_file)
    df.columns = cols

    df = df.dropna(subset=all_keys, how='any')
    # print(df.index.values)
    if df.index.max() - df.index.min() + 1 != len(df):
        if skip_not_continuous_sample:
            print("skip. index is not continuous", csv_file)
            return False, False
    if len(df) < window_length:
        return False, False

    # reset index
    if reset_index:
        df.reset_index(drop=True, inplace=True)

    df['index_col'] = df.index + 1

    df = df[['index_col'] + all_keys]

    if frame_stride and frame_stride > 1:
        df = df[df['index_col'] % frame_stride == 0]

    if num_div:
        if use_random_frame_range and max_frames:
            if len(df) > max_frames:
                start_idx = np.random.randint(0, len(df) - max_frames)
                df = df.iloc[start_idx:start_idx + max_frames]
                # print("random frame range", len(df))
                # print(df.index)
        else:
            min_x = df[all_x_axis_keys].min().min()
            max_x = df[all_x_axis_keys].max().max()
            width = max_x - min_x
            div_width = width / num_div
            thr_width = div_width * num_thr_div
            left_thr = min_x + thr_width
            right_thr = max_x - thr_width

            # if any keypoint is out of the left_thr or right_thr, remove the sample(row)
            df = df[
                (df[all_x_axis_keys] > left_thr).all(axis=1) & (df[all_x_axis_keys] < right_thr).all(axis=1)]
            # print("num_div", len(df))
            # print(df.index)

            if max_frames and len(df) > max_frames:
                center_x = (min_x + max_x) / 2
                # find index of frame closest to center_x
                # closest_idx = (df[all_x_axis_keys[0]] - center_x).abs().idxmin()
                # print("closest_idx", closest_idx)
                # print(center_x)
                # remain max_frames rows that frames closest to center_x by first key in x_axis_keys.
                df = df.iloc[(df[all_x_axis_keys[0]] - center_x).abs().argsort()[:max_frames]]
                # print("max_frames", len(df))
                # print(df.index)
                if sort_max_frames:
                    # sort by index_col
                    df.sort_values(by='index_col', inplace=True)

    else:
        if max_frames and len(df) > max_frames:
            df = df.iloc[:max_frames]

    # 좌표를 연결된 3개의 관절들의 사이 각도로 변환
    for key_set in target_skeleton_key_sets:
        for start_idx in range(len(key_set) - 2):
            first_key = key_set[start_idx]
            second_key = key_set[start_idx + 1]
            third_key = key_set[start_idx + 2]
            df[f"{first_key}_{second_key}_{third_key}_angle"] = df.apply(angle_between_points, axis=1,
                                                                         first_key=first_key,
                                                                         second_key=second_key,
                                                                         third_key=third_key)

            first_key = key_set[start_idx + 2]
            second_key = key_set[start_idx]
            third_key = key_set[start_idx + 1]
            df[f"{first_key}_{second_key}_{third_key}_angle_2"] = df.apply(angle_between_points, axis=1,
                                                                           first_key=first_key,
                                                                           second_key=second_key,
                                                                           third_key=third_key)

    # drop df cols of all_keys
    df = df.drop(columns=all_keys)
    # print("drop")
    # print(df.index)

    # 각도 컬럼에 180초과 값이 있는지 확인
    for col in df.columns:
        if "angle" in col:
            if (df[col] > 180).any():
                print(f"angle over 180 in {col}")
                return False, False

    return df, len(df)


def get_key_data(target_keypoint_name):
    target_skeleton_key_sets = TARGET_KP_COL_DICT[target_keypoint_name]
    y_axs_key_sets = []
    x_axs_key_sets = []
    all_key_sets = set()
    all_x_axis_keys = []
    all_y_axis_keys = []
    keypoint_names = []
    for key_set in target_skeleton_key_sets:
        tmp_key_set = copy.deepcopy(key_set)
        keypoint_names += tmp_key_set[1:-1]
        y_axs_key_sets.append([f"{k}_y" for k in key_set])
        x_axs_key_sets.append([f"{k}_x" for k in key_set])
        all_key_sets.update(y_axs_key_sets[-1])
        all_key_sets.update(x_axs_key_sets[-1])
        all_x_axis_keys.extend(x_axs_key_sets[-1])
        all_y_axis_keys.extend(y_axs_key_sets[-1])
    all_keys = list(all_key_sets)
    return all_keys, all_x_axis_keys, target_skeleton_key_sets, keypoint_names


def main(args):
    # index번호는 1부터
    labels = json.load(open(args.label_file, encoding="utf-8"))
    all_keys, all_x_axis_keys, target_skeleton_key_sets, keypoint_names = get_key_data(args.target_keypoint_name)

    moco_fname_to_csv_fname_dict = {}

    label_key = "lameness"
    for sample_idx, sample in enumerate(labels):
        # print(f"processing {sample_idx}th sample")
        label = sample[label_key]

        is_val = 'isVal' in sample and sample['isVal']

        if not is_val and label:
            print("skip positive train sample")
            continue

        if is_val:
            kp_output_dir = os.path.join(args.output_dir, "validating", "trajectories")
            label_output_dir = os.path.join(args.output_dir, "validating", "test_frame_mask")
            os.makedirs(label_output_dir, exist_ok=True)
            if args.save_test:
                test_kp_output_dir = os.path.join(args.output_dir, "testing", "trajectories")
                test_output_dir = os.path.join(args.output_dir, "testing", "test_frame_mask")
                os.makedirs(label_output_dir, exist_ok=True)
        else:
            kp_output_dir = os.path.join(args.output_dir, "training", "trajectories")

        # if sample_label:
        #     if 'leg_position' not in sample:
        #         print("no leg position data in sample")
        #         continue
        #
        #     if not use_unknown_leg_position and "None" in sample['leg_position']:
        #         print("skip unknown leg position")
        #         continue
        kp_sample_prefix = "0" if label else "1"

        for csv_idx, path_and_dir in enumerate(sample["keypoints"]["path_and_direction"]):
            csv_path = path_and_dir["keypoint_full_path"]

            if args.use_old_keypoint:
                csv_path = csv_path.replace("/auto/", "/").replace("LABEL_DATA_FINAL", "LABEL_DATA2/*")
                csv_files = glob.glob(os.path.join(args.keypoint_dir, csv_path))
                if len(csv_files) == 0:
                    print("no csv file", csv_path)
                    continue
                if len(csv_files) > 1:
                    print("multiple csv files", csv_files)
                    sys.exit(1)

                csv_file = csv_files[0]

            else:
                csv_file = os.path.join(args.keypoint_dir, path_and_dir["keypoint_full_path"])

            if args.kp_file_name:
                csv_file = os.path.join(os.path.dirname(csv_file), args.kp_file_name)

            df, len_df = read_csv(csv_file, all_keys, all_x_axis_keys, target_skeleton_key_sets, args.window_length,
                          args.direction, args.num_div, args.num_thr_div, args.frame_stride, args.max_frames,
                          args.use_random_frame_range, args.reset_index, args.skip_not_continuous_sample,
                          args.use_score_col, args.sort_max_frames)

            # three digit number using sample index
            sample_idx_str = f"{sample_idx:04d}"
            kp_sample_output_path = os.path.join(kp_output_dir, f"{kp_sample_prefix}{sample_idx_str}-0{csv_idx + 101}",
                                                 "00001.csv")
            # print(kp_sample_output_path)
            os.makedirs(os.path.dirname(kp_sample_output_path), exist_ok=True)

            # save df to csv without header'
            df.to_csv(kp_sample_output_path, index=False, header=False)

            if args.save_test and is_val:
                test_kp_sample_output_path = os.path.join(test_kp_output_dir,
                                                          f"{kp_sample_prefix}{sample_idx_str}-0{csv_idx + 101}",
                                                          "00001.csv")
                os.makedirs(os.path.dirname(test_kp_sample_output_path), exist_ok=True)
                df.to_csv(test_kp_sample_output_path, index=False, header=False)


            if is_val:
                label_output_path = os.path.join(label_output_dir,
                                                 f"{kp_sample_prefix}{sample_idx_str}_0{csv_idx + 101}.npy")
                os.makedirs(os.path.dirname(label_output_path), exist_ok=True)
                if label:
                    # label_np = np.ones(len_df - args.window_length + 1, dtype=np.int8)
                    label_np = np.ones(len(df), dtype=np.int8)
                    # set last 5 elements to 0
                    start_idx = args.window_length - 1
                    label_np[-start_idx:] = 0
                else:
                    label_np = np.zeros(len(df) - args.window_length + 1, dtype=np.int8)
                # label_np = np.zeros(len(df), dtype=np.int8)
                np.save(label_output_path, label_np)
                if args.save_test:
                    test_label_output_path = os.path.join(test_output_dir,
                                                          f"{kp_sample_prefix}{sample_idx_str}_0{csv_idx + 101}.npy")
                    os.makedirs(os.path.dirname(test_label_output_path), exist_ok=True)
                    np.save(test_label_output_path, label_np)
                    moco_fname = os.path.splitext(os.path.basename(test_label_output_path))[0]
                    moco_fname_to_csv_fname_dict[moco_fname] = csv_path

    if moco_fname_to_csv_fname_dict:
        with open(os.path.join(args.output_dir, "moco_fname_to_csv_fname_dict.json"), "w") as f:
            json.dump(moco_fname_to_csv_fname_dict, f, ensure_ascii=False, indent=4)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose_AD_Experiment')
    parser.add_argument('--label_file', type=str,
                        default='./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_1.json')
    parser.add_argument('--keypoint_dir', type=str, default='E:\dataset\\afp\horse_kp_20240710')
    parser.add_argument('--target_keypoint_name', type=str,
                        default='baseline')
    # keypoint_threshold
    parser.add_argument('--keypoint_threshold', type=float, default=None)  # 0.6)
    # window_length
    parser.add_argument('--window_length', type=int, default=4)
    # output_dir
    parser.add_argument('--output_dir', type=str, default='E:\dataset\\afp\horse_mocodad_test4')

    # reset_index
    parser.add_argument('--reset_index', action='store_true', default=False)
    # save test
    parser.add_argument('--save_test', action='store_true', default=False)

    parser.add_argument('--use_old_keypoint', action='store_true', default=False)

    # direction
    parser.add_argument('--direction', type=str, default='side')  # side, front, back
    # kp_file_name
    parser.add_argument('--kp_file_name', type=str, default=None)
    # parser.add_argument('--kp_file_name', type=str, default='coords.csv')#coords.csv
    # num_div
    parser.add_argument('--num_div', type=int, default=None)  # 6
    # num_thr_div
    parser.add_argument('--num_thr_div', type=int, default=1)  # 1

    parser.add_argument('--max_frames', type=int, default=None)  # 75

    # frame_stride
    parser.add_argument('--frame_stride', type=int, default=None)

    # use_score_col
    parser.add_argument('--use_score_col', action='store_true', default=False)
    parser.add_argument('--skip_not_continuous_sample', action='store_true', default=False)

    # use_random_frame_range
    parser.add_argument('--use_random_frame_range', action='store_true', default=False)
    parser.add_argument('--sort_max_frames', action='store_true', default=False)
    # parser.add_argument('--flip_rtol_to_ltor', action='store_true', default=False)

    main(parser.parse_args())
