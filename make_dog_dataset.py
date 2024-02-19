import argparse
import os
import json
import pandas as pd
import numpy as np

KEYPOINT_COLS = ['filename', 'L_Eye_x', 'L_Eye_y', 'L_Eye_score', 'R_Eye_x', 'R_Eye_y', 'R_Eye_score',
                 'L_EarBase_x',
                 'L_EarBase_y', 'L_EarBase_score',
                 'R_EarBase_x',
                 'R_EarBase_y', 'R_EarBase_score', 'Nose_x', 'Nose_y', 'Nose_score', 'Throat_x', 'Throat_y',
                 'Throat_score', 'TailBase_x', 'TailBase_y', 'TailBase_score',
                 'Withers_x',
                 'Withers_y', 'Withers_score', 'L_F_Elbow_x', 'L_F_Elbow_y', 'L_F_Elbow_score', 'R_F_Elbow_x',
                 'R_F_Elbow_y', 'R_F_Elbow_score', 'L_B_Elbow_x',
                 'L_B_Elbow_y', 'L_B_Elbow_score',
                 'R_B_Elbow_x', 'R_B_Elbow_y', 'R_B_Elbow_score', 'L_F_Knee_x', 'L_F_Knee_y',
                 'L_F_Knee_score', 'R_F_Knee_x', 'R_F_Knee_y', 'R_F_Knee_score',
                 'L_B_Knee_x',
                 'L_B_Knee_y', 'L_B_Knee_score', 'R_B_Knee_x', 'R_B_Knee_y', 'R_B_Knee_score', 'L_F_Paw_x',
                 'L_F_Paw_y',
                 'L_F_Paw_score', 'R_F_Paw_x',
                 'R_F_Paw_y', 'R_F_Paw_score',
                 'L_B_Paw_x', 'L_B_Paw_y', 'L_B_Paw_score', 'R_B_Paw_x', 'R_B_Paw_y', 'R_B_Paw_score',
                 'Body_Middle_x', 'Body_Middle_y', 'Body_Middle_score']

# Withers,Throat,R_F_Elbow,R_F_Knee,R_F_Paw,L_F_Elbow,L_F_Knee,L_F_Paw,TailBase,R_B_Elbow,R_B_Knee,R_B_Paw,L_B_Elbow,L_B_Knee,L_B_Paw
TARGET_KP_COL_DICT = {
    "baseline": [
        'Nose', 'Withers', 'Throat',
        'R_F_Elbow', 'R_F_Knee', 'R_F_Paw',
        'L_F_Elbow', 'L_F_Knee', 'L_F_Paw',
        'Body_Middle', 'TailBase',
        'R_B_Elbow', 'R_B_Knee', 'R_B_Paw',
        'L_B_Elbow', 'L_B_Knee', 'L_B_Paw'
    ]
}


def main(args):
    # index번호는 1부터
    labels = json.load(open(args.label_file, encoding="utf-8"))
    use_keys = TARGET_KP_COL_DICT[args.target_keypoint_name]
    score_keys = [f"{k}_score" for k in use_keys]
    y_axis_keys = [f"{k}_y" for k in use_keys]
    x_axis_keys = [f"{k}_x" for k in use_keys]
    keypoint_root = args.keypoint_dir

    label_key = "lameness"
    for sample_idx, sample in enumerate(labels):
        print(f"processing {sample_idx}th sample")
        label = sample[label_key]

        is_val = 'isVal' in sample and sample['isVal']

        if not is_val and label:
            print("skip positive train sample")
            continue

        if is_val:
            kp_output_dir = os.path.join(args.output_dir, "validating", "trajectories")
            label_output_dir = os.path.join(args.output_dir, "validating", "test_frame_mask")
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
            csv_file = os.path.join(keypoint_root, path_and_dir["keypoint_full_path"])

            try:
                df = pd.read_csv(csv_file, skiprows=lambda x: x in [2], header=1, encoding='CP949')
            except:
                df = pd.read_csv(csv_file, skiprows=lambda x: x in [2], header=1, encoding='utf-8')

            if len(df) < args.window_length:
                continue

            len_df = len(df)

            if len(df.columns) != len(KEYPOINT_COLS):
                raise Exception("invalid df keys", df.columns, csv_file)
            df.columns = KEYPOINT_COLS

            if args.keypoint_threshold and args.keypoint_threshold > 0:
                # drop rows by self.keypoint_threshold and score_keys
                for k in score_keys:
                    df = df[df[k] > args.keypoint_threshold]

            if len(df) < args.window_length:
                continue

            # reset index
            if args.reset_index:
                df.reset_index(drop=True, inplace=True)
            df['index_col'] = df.index + 1

            df = df[['index_col'] + x_axis_keys + y_axis_keys]
            # three digit number using sample index
            sample_idx_str = f"{sample_idx:03d}"
            kp_sample_output_path = os.path.join(kp_output_dir, f"{kp_sample_prefix}{sample_idx_str}-0{csv_idx + 101}",
                                                 "00001.csv")
            os.makedirs(os.path.dirname(kp_sample_output_path), exist_ok=True)

            # save df to csv without header
            df.to_csv(kp_sample_output_path, index=False, header=False)

            if args.reset_index:
                len_df = len(df)

            if is_val:
                label_output_path = os.path.join(label_output_dir,
                                                 f"{kp_sample_prefix}{sample_idx_str}_0{csv_idx + 101}.npy")
                os.makedirs(os.path.dirname(label_output_path), exist_ok=True)
                if label:
                    # label_np = np.ones(len_df - args.window_length + 1, dtype=np.int8)
                    label_np = np.ones(len_df, dtype=np.int8)
                else:
                    # label_np = np.zeros(len_df - args.window_length + 1, dtype=np.int8)
                    label_np = np.zeros(len_df, dtype=np.int8)
                np.save(label_output_path, label_np)

    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose_AD_Experiment')
    parser.add_argument('--label_file', type=str, default='./expert_label_20231031_balance_pos_neg.json')
    parser.add_argument('--keypoint_dir', type=str, default='E:\dataset\\afp\pei_20231031')
    parser.add_argument('--target_keypoint_name', type=str,
                        default='baseline')
    # keypoint_threshold
    parser.add_argument('--keypoint_threshold', type=float, default=0.6)
    # window_length
    parser.add_argument('--window_length', type=int, default=6)
    # output_dir
    parser.add_argument('--output_dir', type=str, default='E:\dataset\\afp\dog_mocodad')

    # reset_index
    parser.add_argument('--reset_index', action='store_true', default=False)

    main(parser.parse_args())
