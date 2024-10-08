import argparse
import os
import time
import traceback
import uuid

import torch
import pytorch_lightning as pl
import yaml
from models.mocodad import MoCoDAD
from utils.dataset import get_test_dataset_and_loader
from utils.model_utils import processing_data
import random
import numpy as np
import make_horse_dataset
import make_horse_angle_dataset


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


def main(args, tmp_dir):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(tmp_dir, exist_ok=True)
    # Initialize the model
    model = MoCoDAD(args)
    checkpoint = torch.load(args.load_ckpt, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to('cuda')

    print('Loading data and creating loaders.....')
    ckpt_path = args.load_ckpt

    if args.use_angle:
        all_keys, all_x_axis_keys, target_skeleton_key_sets = make_horse_angle_dataset.get_key_data(
            args.target_keypoint_name)
        df, len_df = make_horse_angle_dataset.read_csv(args.test_path, all_keys, all_x_axis_keys, target_skeleton_key_sets,
                                               window_length=args.seg_len, direction=args.camera_direction,
                                               max_frames=args.max_frames,
                                               num_div=args.num_div,
                                               use_random_frame_range=args.use_random_frame_range,
                                               skip_not_continuous_sample=args.skip_not_continuous_sample,
                                               sort_max_frames=args.sort_max_frames)
    else:
        x_axis_keys, y_axis_keys = make_horse_dataset.get_axis_keys(args.camera_direction, args.target_keypoint_name)
        df = make_horse_dataset.read_csv(args.test_path, x_axis_keys, y_axis_keys, window_length=args.seg_len,
                                         direction='side',
                                         max_frames=args.max_frames, num_div=args.num_div,
                                         use_random_frame_range=args.use_random_frame_range,
                                         skip_not_continuous_sample=args.skip_not_continuous_sample,
                                         sort_max_frames=args.sort_max_frames)

    kp_path = os.path.join(tmp_dir, str(uuid.uuid4()) + '.csv')

    try:
        df.to_csv(kp_path, index=False, header=False)
        dataset, loader = get_test_dataset_and_loader(args, kp_path)

        start = time.time()
        for batch in loader:
            with torch.no_grad():
                batch = [b.to('cuda') for b in batch]
                out = model.forward(batch)
                break

        out = processing_data(out)

        loss = out[0]

        trans = out[3]
        trans_losses = []
        for transformation in range(args.num_transform):
            cond_transform = (trans == transformation)
            trans_loss, = filter_vectors_by_cond([loss], cond_transform)

            loss_matrix = compute_var_matrix(trans_loss, out[5], len_df)
            # loss_matrix = [num_windows, num_frames]
            trans_losses.append(np.nanmax(loss_matrix, axis=0))

        trans_losses = np.stack(trans_losses, axis=0)
        trans_losses = np.mean(trans_losses, axis=0)
        loss = np.mean(trans_losses)

        # loss = np.mean(loss, axis=0)
        if args.pred_threshold <= loss:
            print("positive sample")
        else:
            print("negative sample")
        print("loss", loss)

        prediction = out[1]
        pred_window = prediction.shape[2]
        gt_data = out[2][:, :, -pred_window:, :]
        diff = np.abs(prediction - gt_data)
        diff = np.mean(diff, axis=(0, 1, 2))
        print(time.time() - start)
        print("max diff index", diff.argmax(), np.max(diff))
        print("min diff index", diff.argmin(), np.min(diff))
        print("exec time", time.time() - start)

    except Exception as e:
        traceback.print_exc()
    finally:
        os.remove(kp_path)


if __name__ == '__main__':
    # Parse command line arguments and load config file
    parser = argparse.ArgumentParser(description='MoCoDAD')
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--tmp_dir', type=str, default="./tmp")
    args = parser.parse_args()
    tmp_dir = args.tmp_dir
    args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    args = argparse.Namespace(**args)
    main(args, tmp_dir)
