import argparse
import os
import time
import torch
import pytorch_lightning as pl
import yaml
from models.mocodad import MoCoDAD
from utils.argparser import init_args
from utils.dataset import get_test_dataset_and_loader
from utils.model_utils import processing_data
import random
import numpy as np

# Parse command line arguments and load config file
parser = argparse.ArgumentParser(description='MoCoDAD')
parser.add_argument('-c', '--config', type=str, required=True)
args = parser.parse_args()
args = yaml.load(open(args.config), Loader=yaml.FullLoader)
args = argparse.Namespace(**args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# Initialize the model
model = MoCoDAD(args)

print('Loading data and creating loaders.....')
ckpt_path = args.load_ckpt
dataset, loader = get_test_dataset_and_loader(args)

# Initialize trainer and test
trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                     # default_root_dir=args.ckpt_dir,
                     max_epochs=1, logger=False)
start = time.time()
out = trainer.predict(model, dataloaders=loader, ckpt_path=ckpt_path, return_predictions=True)
unpacked_result = processing_data(out)
prediction = unpacked_result[0]
pred_window = prediction.shape[2]
gt_data = unpacked_result[1][:, :, -pred_window:, :]
print(prediction.shape)
print(gt_data.shape)
# np abs
diff = np.abs(prediction - gt_data)
diff = np.mean(diff, axis=(0, 1, 2))
print(time.time() - start)
print("diff", diff.shape)
print("max diff index", diff.argmax(), np.max(diff))
print("min diff index", diff.argmin(), np.min(diff))
