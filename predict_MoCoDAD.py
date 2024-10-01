import argparse
import os

import torch
import pytorch_lightning as pl
import yaml
from models.mocodad import MoCoDAD
from utils.argparser import init_args
from utils.dataset import get_test_dataset_and_loader
from utils.model_utils import processing_data

# Parse command line arguments and load config file
parser = argparse.ArgumentParser(description='MoCoDAD')
parser.add_argument('-c', '--config', type=str, required=True)
args = parser.parse_args()
args = yaml.load(open(args.config), Loader=yaml.FullLoader)
args = argparse.Namespace(**args)

# Initialize the model
model = MoCoDAD(args)

print('Loading data and creating loaders.....')
ckpt_path = args.load_ckpt
dataset, loader = get_test_dataset_and_loader(args)

# Initialize trainer and test
trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                     # default_root_dir=args.ckpt_dir,
                     max_epochs=1, logger=False)
out = trainer.predict(model, dataloaders=loader, ckpt_path=ckpt_path, return_predictions=True)
unpacked_result = processing_data(out)
prediction = unpacked_result[0]
pred_window = prediction.shape[2]
gt_data = unpacked_result[1][:,:,-pred_window:, :]
print(prediction.shape)
print(gt_data.shape)


# file_names = ['prediction', 'gt_data', 'trans', 'metadata', 'frames']
# for i in range(len(unpacked_result)):
#     print(file_names[i])
#     print(unpacked_result[i].shape)
