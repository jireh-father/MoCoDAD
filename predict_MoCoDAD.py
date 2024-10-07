import argparse
import os
import time
import traceback
import uuid

import torch
import pytorch_lightning as pl
import yaml
from models.mocodad import MoCoDAD
from utils.argparser import init_args
from utils.dataset import get_test_dataset_and_loader
from utils.model_utils import processing_data
import random
import numpy as np
import make_horse_dataset
import make_horse_angle_dataset

# Parse command line arguments and load config file
parser = argparse.ArgumentParser(description='MoCoDAD')
parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('--tmp_dir', type=str, default="./tmp")
args = parser.parse_args()
tmp_dir = args.tmp_dir
args = yaml.load(open(args.config), Loader=yaml.FullLoader)
args = argparse.Namespace(**args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

os.makedirs(tmp_dir, exist_ok=True)
# Initialize the model
model = MoCoDAD(args)

print('Loading data and creating loaders.....')
ckpt_path = args.load_ckpt

if args.use_angle:
    all_keys, all_x_axis_keys, target_skeleton_key_sets = make_horse_angle_dataset.get_key_data(
        args.target_keypoint_name)
    df = make_horse_angle_dataset.read_csv(args.test_path, all_keys, all_x_axis_keys, target_skeleton_key_sets,
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

    # Initialize trainer and test
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                         # default_root_dir=args.ckpt_dir,
                         max_epochs=1, logger=False)
    start = time.time()
    out = trainer.predict(model, dataloaders=loader, ckpt_path=ckpt_path, return_predictions=True)

    print("out length", len(out[0]))
    for i in range(len(out[0])):
        print(f"{i}th out shape", out[0][i].shape)
    # when return is pose
    # out length 5
    # 0th out shape torch.Size([360, 2, 1, 8])
    # 1th out shape torch.Size([360, 2, 4, 8])
    # 2th out shape torch.Size([360])
    # 3th out shape torch.Size([360, 4])
    # 4th out shape torch.Size([360, 4]

    # when return is pose and loss
    # out length 6
    # 0th out shape torch.Size([360])
    # 1th out shape torch.Size([360, 2, 1, 8])
    # 2th out shape torch.Size([360, 2, 4, 8])
    # 3th out shape torch.Size([360])
    # 4th out shape torch.Size([360, 4])
    # 5th out shape torch.Size([360, 4])
    # 360 = 72frames * 5(num transform)
    # [frames & transforms, keypoint axis, window, num of keypoint]

    unpacked_result = processing_data(out)

    loss = np.mean(unpacked_result[0], axis=0)
    if args.pred_threshold >= loss:
        print("positive sample")
    else:
        print("negative sample")
    print("loss", loss)
    # thr, 0.002
    print("unpacked_result length", len(unpacked_result))
    prediction = unpacked_result[1]
    print("prediction shape", prediction.shape)
    pred_window = prediction.shape[2]
    gt_data = unpacked_result[2][:, :, -pred_window:, :]
    print(prediction.shape)
    print(gt_data.shape)
    # np abs
    diff = np.abs(prediction - gt_data)
    diff = np.mean(diff, axis=(0, 1, 2))
    print(time.time() - start)
    print("diff", diff.shape)
    print("max diff index", diff.argmax(), np.max(diff))
    print("min diff index", diff.argmin(), np.min(diff))

except Exception as e:
    traceback.print_exc()
# finally:
    # os.remove(kp_path)
