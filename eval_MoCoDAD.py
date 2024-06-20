import argparse
import os

import pytorch_lightning as pl
import yaml
from models.mocodad import MoCoDAD
from models.mocodad_latent import MoCoDADlatent
from utils.argparser import init_args
from utils.dataset import get_dataset_and_loader
from utils import slack
import pickle


if __name__== '__main__':
    
    # Parse command line arguments and load config file
    parser = argparse.ArgumentParser(description='MoCoDAD')
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--slack_webhook_url', type=str, default=None)
    args = parser.parse_args()
    slack_webhook_url = args.slack_webhook_url
    args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    args = argparse.Namespace(**args)
    args = init_args(args)


    # Initialize the model
    model = MoCoDADlatent(args) if hasattr(args, 'diffusion_on_latent') else MoCoDAD(args)
    
    if args.load_tensors:
        # Load tensors and test
        model.test_on_saved_tensors(split_name=args.split)
    else:
        # Load test data
        print('Loading data and creating loaders.....')
        ckpt_path = os.path.join(args.ckpt_dir, args.load_ckpt)
        dataset, loader, _, _ = get_dataset_and_loader(args, split=args.split)
        
        # Initialize trainer and test
        trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices[:1],
                             default_root_dir=args.ckpt_dir, max_epochs=1, logger=False)
        out = trainer.test(model, dataloaders=loader, ckpt_path=ckpt_path)

        clip_fname_pred_map = model.best_metrics['clip_fname_pred_map']
        pickle.dump(clip_fname_pred_map, open(os.path.join(args.ckpt_dir, 'eval_clip_fname_pred_map.pkl'), 'wb+'))
        # np.save(os.path.join(args.ckpt_dir, 'clip_fname_pred_map.npy'), clip_fname_pred_map)
        del model.best_metrics['clip_fname_pred_map']
        pickle.dump(model.best_metrics, open(os.path.join(args.ckpt_dir, 'eval_best_metrics.pkl'), 'wb+'))
        print(model.best_metrics)

    if slack_webhook_url:
        keys = ['clip_auc', 'auc', 'f1', 'recall', 'precision', 'best_thr', 'ori_clip_auc', 'ori_auc']

        slack.send_info_to_slack(
            f"Mocodad Trained. {args.dir_name}.\n{', '.join(keys)}\n{' '.join([str(round(model.best_metrics[k] * 100, 2)) for k in keys])}\nconfusion matrix: {model.best_metrics['confusion_matrix']}",
            slack_webhook_url)