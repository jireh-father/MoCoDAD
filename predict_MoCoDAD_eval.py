import argparse
import json
import os
import time
import traceback
import uuid

import torch
import yaml
from models.mocodad import MoCoDAD
from utils.dataset import get_test_dataset_and_loader
import random
import numpy as np
import make_horse_dataset
import make_horse_angle_dataset
from mocodad import Mocodad


def processing_data(data_array):
    loss = data_array[0].cpu().numpy()
    pred = data_array[1].cpu().numpy()
    input = data_array[2].cpu().numpy()
    transform_idx = data_array[3].cpu().numpy()
    metadata = data_array[4].cpu().numpy()
    actual_frames = data_array[5].cpu().numpy()

    return loss, pred, input, transform_idx, metadata, actual_frames


def filter_vectors_by_cond(vecs, cond):
    return [filter_by_cond(vec, cond) for vec in vecs]


def filter_by_cond(vec, cond):
    return vec[cond]


def compute_var_matrix(pos, frames_pos, n_frames):
    pose = np.zeros(shape=(pos.shape[0], n_frames))

    for n in range(pose.shape[0]):
        pose[n, frames_pos[n] - 1] = pos[n]

    return pose


def main(config, tmp_dir, data_json, keypoint_dir):
    mocodad = Mocodad(config, tmp_dir)
    dataset = json.load(open(data_json, encoding="utf-8"))

    num_samples = 0
    num_true = 0
    losses = []
    pos_losses = []
    neg_losses = []
    for sample_idx, sample in enumerate(dataset):
        # print(f"processing {sample_idx}th sample")
        label = sample['lameness']

        is_val = 'isVal' in sample and sample['isVal']

        if not is_val:
            continue

        for csv_idx, path_and_dir in enumerate(sample["keypoints"]["path_and_direction"]):
            csv_file = os.path.join(keypoint_dir, path_and_dir["keypoint_full_path"])
            try:
                pred, position = mocodad.inference(csv_file)
                if pred == label:
                    num_true += 1
                num_samples += 1
            except Exception as e:
                traceback.print_exc()

    print(f"accuracy: {num_true / num_samples}")
    print("num_samples", num_samples)
    print("num_true", num_true)

    # print stat losses
    losses = np.array(losses)
    print("mean loss", np.mean(losses))
    print("std loss", np.std(losses))
    print("max loss", np.max(losses))
    print("min loss", np.min(losses))

    pos_losses = np.array(pos_losses)
    print("mean pos loss", np.mean(pos_losses))
    print("std pos loss", np.std(pos_losses))
    print("max pos loss", np.max(pos_losses))
    print("min pos loss", np.min(pos_losses))

    neg_losses = np.array(neg_losses)
    print("mean neg loss", np.mean(neg_losses))
    print("std neg loss", np.std(neg_losses))
    print("max neg loss", np.max(neg_losses))
    print("min neg loss", np.min(neg_losses))


if __name__ == '__main__':
    # Parse command line arguments and load config file
    parser = argparse.ArgumentParser(description='MoCoDAD')
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--tmp_dir', type=str, default="./tmp")
    parser.add_argument('--data_json', type=str,
                        default='./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_9.json')
    parser.add_argument('--keypoint_dir', type=str,
                        default='./horse_kp_20240710')
    global_args = parser.parse_args()

    main(global_args.config, global_args.tmp_dir, global_args.data_json, global_args.keypoint_dir)
