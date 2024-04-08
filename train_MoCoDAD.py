import argparse
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from models.mocodad import MoCoDAD
from models.mocodad_latent import MoCoDADlatent
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from utils.argparser import init_args
from utils.dataset import get_dataset_and_loader
from utils.ema import EMACallback
from utils import slack

if __name__ == '__main__':

    # Parse command line arguments and load config file
    parser = argparse.ArgumentParser(description='Pose_AD_Experiment')
    parser.add_argument('-c', '--config', type=str, required=True,
                        default='/your_default_config_file_path')
    parser.add_argument('--slack_webhook_url', type=str, default=None)

    args = parser.parse_args()
    slack_webhook_url = args.slack_webhook_url
    config_path = args.config
    args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    args = argparse.Namespace(**args)
    args = init_args(args)
    # Save config file to ckpt_dir
    os.system(f'cp {config_path} {os.path.join(args.ckpt_dir, "config.yaml")}')

    # Set seeds    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    pl.seed_everything(args.seed)

    # Set callbacks and logger
    if (hasattr(args, 'diffusion_on_latent') and args.stage == 'pretrain'):
        monitored_metric = 'pretrain_rec_loss'
        metric_mode = 'min'
    elif args.validation:
        monitored_metric = 'AUC'
        metric_mode = 'max'
    else:
        monitored_metric = 'loss_noise'
        metric_mode = 'min'
    callbacks = [ModelCheckpoint(dirpath=args.ckpt_dir, save_top_k=2,
                                 monitor=monitored_metric,
                                 mode=metric_mode)]

    callbacks += [EMACallback()] if args.use_ema else []

    if args.use_wandb:
        callbacks += [LearningRateMonitor(logging_interval='step')]
        wandb_logger = WandbLogger(project=args.project_name, group=args.group_name, entity=args.wandb_entity,
                                   name=args.dir_name, config=vars(args), log_model='all')
    else:
        wandb_logger = False

    # Get dataset and loaders
    _, train_loader, _, val_loader = get_dataset_and_loader(args, split=args.split, validation=args.validation)

    # Initialize model and trainer
    model = MoCoDADlatent(args) if hasattr(args, 'diffusion_on_latent') else MoCoDAD(args)

    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, default_root_dir=args.ckpt_dir,
                         max_epochs=args.n_epochs,
                         logger=wandb_logger, callbacks=callbacks, strategy=DDPStrategy(find_unused_parameters=False),
                         log_every_n_steps=20, num_sanity_val_steps=0, deterministic=True)

    # Train the model    
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("best metrics")
    clip_fname_pred_map = model.best_metrics['clip_fname_pred_map']
    del model.best_metrics['clip_fname_pred_map']
    print(model.best_metrics)
    # save clip_fname_pred_map
    np.save(os.path.join(args.ckpt_dir, 'clip_fname_pred_map.npy'), clip_fname_pred_map)

    if slack_webhook_url:
        keys = ['clip_auc', 'auc', 'f1', 'recall', 'precision', 'best_thr', 'ori_clip_auc', 'ori_auc']

        slack.send_info_to_slack(
            f"Mocodad Trained. {args.exp_dir}.\n{', '.join(keys)}\n{' '.join([str(round(model.best_metrics[k] * 100, 2)) for k in keys])}\nconfusion matrix: {model.best_metrics['confusion_matrix']}",
            slack_webhook_url)
