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


class Mocodad:
    def __init__(self, config_path, tmp_dir):
        args = yaml.load(open(config_path), Loader=yaml.FullLoader)
        args = argparse.Namespace(**args)
        self.args = args

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        os.makedirs(tmp_dir, exist_ok=True)
        self.tmp_dir = tmp_dir

        model = MoCoDAD(args)
        checkpoint = torch.load(args.load_ckpt, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model.to('cuda')
        self.model = model

        if args.use_angle:
            all_keys, all_x_axis_keys, target_skeleton_key_sets, keypoint_names = make_horse_angle_dataset.get_key_data(
                args.target_keypoint_name)
            self.all_keys = all_keys
            self.all_x_axis_keys = all_x_axis_keys
            self.target_skeleton_key_sets = target_skeleton_key_sets
        else:
            x_axis_keys, y_axis_keys, keypoint_names = make_horse_dataset.get_axis_keys(args.camera_direction,
                                                                        args.target_keypoint_name)
            self.x_axis_keys = x_axis_keys
            self.y_axis_keys = y_axis_keys
        self.keypoint_names = keypoint_names

    def inference(self, keypoint_csv_path):
        if self.args.use_angle:
            df, len_df = make_horse_angle_dataset.read_csv(keypoint_csv_path, self.all_keys, self.all_x_axis_keys,
                                                           self.target_skeleton_key_sets,
                                                           window_length=self.args.seg_len,
                                                           direction=self.args.camera_direction,
                                                           max_frames=self.args.max_frames,
                                                           num_div=self.args.num_div,
                                                           use_random_frame_range=self.args.use_random_frame_range,
                                                           skip_not_continuous_sample=self.args.skip_not_continuous_sample,
                                                           sort_max_frames=self.args.sort_max_frames)
        else:
            df, len_df = make_horse_dataset.read_csv(keypoint_csv_path, self.x_axis_keys, self.y_axis_keys,
                                                     window_length=self.args.seg_len,
                                                     direction='side',
                                                     max_frames=self.args.max_frames, num_div=self.args.num_div,
                                                     use_random_frame_range=self.args.use_random_frame_range,
                                                     skip_not_continuous_sample=self.args.skip_not_continuous_sample,
                                                     sort_max_frames=self.args.sort_max_frames)

        kp_path = os.path.join(self.tmp_dir, str(uuid.uuid4()) + '.csv')

        try:
            df.to_csv(kp_path, index=False, header=False)
            dataset, loader = get_test_dataset_and_loader(self.args, kp_path)

            for batch in loader:
                with torch.no_grad():
                    batch = [b.to('cuda') for b in batch]
                    out = self.model.forward(batch)
                    break

            out = processing_data(out)

            loss = out[0]

            trans = out[3]
            trans_losses = []
            for transformation in range(self.args.num_transform):
                cond_transform = (trans == transformation)
                trans_loss, = filter_vectors_by_cond([loss], cond_transform)
                print(len(trans_loss), out[5].shape, len_df)
                loss_matrix = compute_var_matrix(trans_loss, out[5], len_df)
                # loss_matrix = [num_windows, num_frames]
                trans_losses.append(np.nanmax(loss_matrix, axis=0))

            trans_losses = np.stack(trans_losses, axis=0)
            trans_losses = np.mean(trans_losses, axis=0)
            loss = np.mean(trans_losses)

            if self.args.pred_threshold <= loss:
                result = True
                prediction = out[1]
                pred_window = prediction.shape[2]
                gt_data = out[2][:, :, -pred_window:, :]
                diff = np.abs(prediction - gt_data)
                diff = np.mean(diff, axis=(0, 1, 2))
                position = self.keypoint_names[diff.argmax()]
            else:
                result = False
                position = None

            return result, position
        except Exception as e:
            traceback.print_exc()
            raise e
        finally:
            os.remove(kp_path)


def main(config, tmp_dir, keypoint_csv_path):
    mocodad = Mocodad(config, tmp_dir)

    result, position = mocodad.inference(keypoint_csv_path)
    print(result, position)


if __name__ == '__main__':
    # Parse command line arguments and load config file
    parser = argparse.ArgumentParser(description='MoCoDAD')
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--tmp_dir', type=str, default="./tmp")
    parser.add_argument('--keypoint_csv_path', type=str,
                        default="./horse_kp_20240710/2000343/trot/left/2000343_part3_2_c_20230915_104056_Left_001/CollectedData_teamDLC.csv")
    global_args = parser.parse_args()
    main(global_args.config, global_args.tmp_dir, global_args.keypoint_csv_path)
