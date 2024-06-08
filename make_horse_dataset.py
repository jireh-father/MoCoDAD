import argparse
import glob
import os
import json
import sys

import pandas as pd
import numpy as np

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
        'Stifle_Joint_R', 'Front_Heel_L', 'Front_Heel_R', 'Spina_scapulae', 'T16L1', 'Abdomen', 'LowestBack',
        'T_sacrale', 'Nostril', 'T_ischiadicum', 'Rear_Fetlock_L', 'Rear_Tarsus_L', 'Front_Toe_L', 'Rear_Toe_L', 'Eye',
        'Stifle_Joint_L', 'T_Coxae', 'Rear_Toe_R', 'ElbowJoint_R', 'Tub', 'Coxofemoral', 'ElbowJoint_L',
        'Rear_Fetlock_R', 'Withers', 'Carpuse_R', 'Rear_Heel_R', 'Fetlock_L', 'Front_Toe_R', 'Rear_Tarsus_R', 'Poll',
        'Rear_Heel_L', 'Carpuse_L', 'Fetlock_R', 'Tail_Root'
    ],
    "only_foot": [  # foot is heel and toe
        'Front_Heel_L', 'Front_Heel_R', 'Front_Toe_L', 'Front_Toe_R', 'Rear_Heel_L', 'Rear_Heel_R', 'Rear_Toe_L',
        'Rear_Toe_R',
    ],
    "only_heel": [
        'Front_Heel_L', 'Front_Heel_R', 'Rear_Heel_L', 'Rear_Heel_R',
    ],
    "only_toe": [
        'Front_Toe_L', 'Front_Toe_R', 'Rear_Toe_L', 'Rear_Toe_R',
    ],
    "only_leg": [
        'Stifle_Joint_R', 'Stifle_Joint_L', 'Rear_Tarsus_L', 'Rear_Tarsus_R', 'Rear_Fetlock_L', 'Rear_Fetlock_R',
        'Rear_Heel_L', 'Rear_Heel_R', 'Rear_Toe_L', 'Rear_Toe_R',
        'ElbowJoint_R', 'ElbowJoint_L', 'Carpuse_R', 'Carpuse_L', 'Fetlock_L', 'Fetlock_R',
        'Front_Heel_L', 'Front_Heel_R', 'Front_Toe_L', 'Front_Toe_R',
    ],
    'front_baseline': [
        'Forehead', 'Nasal_bridge', 'Muzzle', 'Elbow_L', 'Elbow_R', 'Shoulder_C', 'Shoulder_L', 'Shoulder_R',
        'Carpus_Front_L', 'Carpus_Front_R', 'Fetlock_Front_L', 'Fetlock_Front_R', 'Hoof_Front_L', 'Hoof_Front_R',
    ],
    'back_baseline': [
        'Tail_root', 'T_Coxae_L', 'T_Coxae_R', 'Stifile_Joint_L', 'Stifile_Joint_R', 'T_ischiadicum_L',
        'T_ischiadicum_R', 'Hock_L', 'Hock_R', 'Fetlock_Rear_L', 'Fetlock_Rear_R', 'Hoof_Rear_L', 'Hoof_Rear_R',
    ],

}


def main(args):
    # index번호는 1부터
    labels = json.load(open(args.label_file, encoding="utf-8"))
    use_keys = TARGET_KP_COL_DICT[args.target_keypoint_name if args.direction == 'side' else f"{args.direction}_{args.target_keypoint_name}"]
    # score_keys = [f"{k}_score" for k in use_keys]
    y_axis_keys = [f"{k}_y" for k in use_keys]
    x_axis_keys = [f"{k}_x" for k in use_keys]
    keypoint_root = args.keypoint_dir

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
                csv_files = glob.glob(os.path.join(keypoint_root, csv_path))
                if len(csv_files) == 0:
                    print("no csv file", csv_path)
                    continue
                if len(csv_files) > 1:
                    print("multiple csv files", csv_files)
                    sys.exit(1)

                csv_file = csv_files[0]

            else:
                csv_file = os.path.join(keypoint_root, path_and_dir["keypoint_full_path"])

            try:
                df = pd.read_csv(csv_file, skiprows=lambda x: x in [2], header=1, encoding='CP949')
            except:
                df = pd.read_csv(csv_file, skiprows=lambda x: x in [2], header=1, encoding='utf-8')

            if len(df) < args.window_length:
                continue

            len_df = len(df)

            if args.direction == 'side':
                cols = KEYPOINT_COLS
            elif args.direction == 'front':
                cols = FRONT_KP_COLS
            elif args.direction == 'back':
                cols = BACK_KP_COLS
            else:
                raise Exception("invalid direction", args.direction)

            if len(df.columns) != len(cols):
                raise Exception("invalid df keys", df.columns, csv_file)
            df.columns = cols

            # if args.keypoint_threshold and args.keypoint_threshold > 0:
            #     # drop rows by self.keypoint_threshold and score_keys
            #     for k in score_keys:
            #         df = df[df[k] > args.keypoint_threshold]

            # drop na rows even if one of the keypoints is na
            df = df.dropna(subset=x_axis_keys + y_axis_keys, how='any')
            if len(df) < args.window_length:
                continue

            # reset index
            if args.reset_index:
                df.reset_index(drop=True, inplace=True)
            df['index_col'] = df.index + 1

            df = df[['index_col'] + x_axis_keys + y_axis_keys]
            # three digit number using sample index
            sample_idx_str = f"{sample_idx:04d}"
            kp_sample_output_path = os.path.join(kp_output_dir, f"{kp_sample_prefix}{sample_idx_str}-0{csv_idx + 101}",
                                                 "00001.csv")
            print(kp_sample_output_path)
            os.makedirs(os.path.dirname(kp_sample_output_path), exist_ok=True)

            # save df to csv without header
            df.to_csv(kp_sample_output_path, index=False, header=False)

            if args.save_test and is_val:
                test_kp_sample_output_path = os.path.join(test_kp_output_dir,
                                                          f"{kp_sample_prefix}{sample_idx_str}-0{csv_idx + 101}",
                                                          "00001.csv")
                os.makedirs(os.path.dirname(test_kp_sample_output_path), exist_ok=True)
                df.to_csv(test_kp_sample_output_path, index=False, header=False)

            if args.reset_index:
                len_df = len(df)

            if is_val:
                label_output_path = os.path.join(label_output_dir,
                                                 f"{kp_sample_prefix}{sample_idx_str}_0{csv_idx + 101}.npy")
                os.makedirs(os.path.dirname(label_output_path), exist_ok=True)
                if label:
                    # label_np = np.ones(len_df - args.window_length + 1, dtype=np.int8)
                    label_np = np.ones(len_df, dtype=np.int8)
                    # set last 5 elements to 0
                    label_np[-5:] = 0
                else:
                    # label_np = np.zeros(len_df - args.window_length + 1, dtype=np.int8)
                    label_np = np.zeros(len_df, dtype=np.int8)
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
    parser.add_argument('--label_file', type=str, default='./expert_label_20231031_balance_pos_neg.json')
    parser.add_argument('--keypoint_dir', type=str, default='E:\dataset\\afp\pei_20231031')
    parser.add_argument('--target_keypoint_name', type=str,
                        default='baseline')
    # keypoint_threshold
    parser.add_argument('--keypoint_threshold', type=float, default=None)  # 0.6)
    # window_length
    parser.add_argument('--window_length', type=int, default=6)
    # output_dir
    parser.add_argument('--output_dir', type=str, default='E:\dataset\\afp\dog_mocodad')

    # reset_index
    parser.add_argument('--reset_index', action='store_true', default=False)
    # save test
    parser.add_argument('--save_test', action='store_true', default=False)

    parser.add_argument('--use_old_keypoint', action='store_true', default=False)

    # direction
    parser.add_argument('--direction', type=str, default='side') # side, front, back

    main(parser.parse_args())
