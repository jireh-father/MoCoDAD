import argparse
import glob
import json
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


def main(args, tmp_dir, data_json):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(tmp_dir, exist_ok=True)
    # Initialize the model
    model = MoCoDAD(args)
    # Initialize trainer and test
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                         max_epochs=1, logger=False)

    print('Loading data and creating loaders.....')
    ckpt_path = args.load_ckpt
    dataset = json.load(open(data_json, encoding="utf-8"))

    if args.use_angle:
        all_keys, all_x_axis_keys, target_skeleton_key_sets = make_horse_angle_dataset.get_key_data(
            args.target_keypoint_name)
    else:
        x_axis_keys, y_axis_keys = make_horse_dataset.get_axis_keys(args.camera_direction, args.target_keypoint_name)

    for sample_idx, sample in enumerate(dataset):
        # print(f"processing {sample_idx}th sample")
        label = sample['lameness']

        is_val = 'isVal' in sample and sample['isVal']

        if not is_val:
            print("skip positive train sample")
            continue

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

            if args.use_angle:
                df = make_horse_angle_dataset.read_csv(csv_file, all_keys, all_x_axis_keys, target_skeleton_key_sets,
                                                       window_length=args.seg_len, direction=args.camera_direction,
                                                       max_frames=args.max_frames,
                                                       num_div=args.num_div,
                                                       use_random_frame_range=args.use_random_frame_range,
                                                       skip_not_continuous_sample=args.skip_not_continuous_sample,
                                                       sort_max_frames=args.sort_max_frames)
            else:
                df = make_horse_dataset.read_csv(csv_file, x_axis_keys, y_axis_keys, window_length=args.seg_len,
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
                out = trainer.predict(model, dataloaders=loader, ckpt_path=ckpt_path, return_predictions=True)

                unpacked_result = processing_data(out)

                loss = np.mean(unpacked_result[0], axis=0)
                if args.pred_threshold <= loss:
                    print("positive sample")
                else:
                    print("negative sample")
                print("loss", loss)

                prediction = unpacked_result[1]
                pred_window = prediction.shape[2]
                gt_data = unpacked_result[2][:, :, -pred_window:, :]
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
    parser.add_argument('--data_json', type=str,
                        default='./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_1.json')
    args = parser.parse_args()
    tmp_dir = args.tmp_dir
    data_json = args.data_json
    args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    args = argparse.Namespace(**args)
    main(args, tmp_dir, data_json)
