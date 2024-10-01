import argparse
import copy
import os
from math import prod
from typing import Dict, List, Tuple
import traceback
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stsae.stsae import STSAE, STSE
from models.stsae.stsae_unet import STSAE_Unet, STSE_Unet
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, f1_score, recall_score, precision_score, \
    accuracy_score, confusion_matrix
from torch.optim import Adam
from tqdm import tqdm
from utils.diffusion_utils import Diffusion
from utils.eval_utils import (compute_var_matrix, filter_vectors_by_cond,
                              get_avenue_mask, get_hr_ubnormal_mask, pad_scores, score_process)
from utils.model_utils import processing_data


class MoCoDAD(pl.LightningModule):
    losses = {'l1': nn.L1Loss, 'smooth_l1': nn.SmoothL1Loss, 'mse': nn.MSELoss}
    conditioning_strategies = {'cat': 'concat', 'concat': 'concat',
                               'add2layers': 'inject', 'inject': 'inject',
                               'inbetween_imp': 'inbetween_imp', 'interleave': 'inbetween_imp',
                               'random_indices': 'random_imp', 'random_imp': 'random_imp',
                               'no_condition': 'no_condition', 'none': 'no_condition'}

    def __init__(self, args: argparse.Namespace) -> None:
        """
        This class implements MoCoDAD model.
        
        Args:
            args (argparse.Namespace): arguments containing the hyperparameters of the model
        """

        super(MoCoDAD, self).__init__()

        # Log the hyperparameters of the model
        self.save_hyperparameters(args)

        # Set the internal variables of the model
        # Data parameters
        self.n_frames = args.seg_len
        self.num_coords = args.num_coords
        self.remove_last_remain_frame = args.remove_last_remain_frame if hasattr(args,
                                                                                 'remove_last_remain_frame') else False
        self.use_original_anomaly_score = args.use_original_anomaly_score if hasattr(args,
                                                                                     'use_original_anomaly_score') else False
        self.n_joints = self._infer_number_of_joint(args)

        # Model parameters
        # Main network
        self.embedding_dim = args.embedding_dim
        self.dropout = args.dropout
        self.conditioning_strategy = self.conditioning_strategies[args.conditioning_strategy]
        # Conditioning network
        self.conditioning_indices = args.conditioning_indices
        self.n_frames_condition, self.n_frames_corrupt, self.input_n_frames = self._set_conditioning_strategy()
        self.conditioning_architecture = args.conditioning_architecture if self.conditioning_strategy == 'inject' else None
        self.cond_h_dim = args.h_dim
        self.cond_latent_dim = args.latent_dim
        self.cond_channels = args.channels
        self.cond_dropout = args.dropout

        # Training and inference parameters
        self.learning_rate = args.opt_lr
        self.loss_fn = self.losses[args.loss_fn](reduction='none')
        self.rec_weight = args.rec_weight  # weight of the reconstruction loss
        self.noise_steps = args.noise_steps
        self.aggregation_strategy = args.aggregation_strategy
        self.n_generated_samples = args.n_generated_samples
        self.model_return_value = args.model_return_value
        if hasattr(args, 'gt_path'):
            self.gt_path = args.gt_path
        self.split = args.split
        if hasattr(args, 'use_hr'):
            self.use_hr = args.use_hr
        if hasattr(args, 'ckpt_dir'):
            self.ckpt_dir = args.ckpt_dir
        self.save_tensors = args.save_tensors
        self.num_transforms = args.num_transform
        self.anomaly_score_pad_size = args.pad_size
        self.anomaly_score_filter_kernel_size = args.filter_kernel_size
        self.anomaly_score_frames_shift = args.frames_shift
        if hasattr(args, 'dataset_choice'):
            self.dataset_name = args.dataset_choice

        # Set the noise scheduler for the diffusion process
        self._set_diffusion_variables()

        # Build the model
        self.build_model()

        self.best_clip_auc = 0.0
        self.best_metrics = None

        self.args = args

    def build_model(self) -> None:
        """
        Build the model according to the specified hyperparameters.
        If the conditioning strategy is 'inject', the conditioning network is built and the available architectures are:
        AutoEncoder (AE), Encoder (E), Encoder-UNet (E_unet). For the other conditioning strategies, the conditioning network is set to `None`.

        Raises:
            NotImplementedError: if the conditioning architecture is not implemented
        """

        if self.conditioning_strategy == 'inject':
            if self.conditioning_architecture == 'AE':
                condition_encoder = STSAE(c_in=self.num_coords, h_dim=self.cond_h_dim,
                                          latent_dim=self.cond_latent_dim, n_frames=self.n_frames_condition,
                                          dropout=self.cond_dropout, n_joints=self.n_joints,
                                          layer_channels=self.cond_channels, device=self.device)
            elif self.conditioning_architecture == 'E':
                condition_encoder = STSE(c_in=self.num_coords, h_dim=self.cond_h_dim,
                                         latent_dim=self.cond_latent_dim, n_frames=self.n_frames_condition,
                                         dropout=self.cond_dropout, n_joints=self.n_joints,
                                         layer_channels=self.cond_channels, device=self.device)
            elif self.conditioning_architecture == 'E_unet':
                condition_encoder = STSE_Unet(c_in=self.num_coords, embedding_dim=None,
                                              latent_dim=self.cond_latent_dim, n_frames=self.n_frames_condition,
                                              n_joints=self.n_joints, dropout=self.cond_dropout,
                                              device=self.device, set_out_layer=True)
            else:
                raise NotImplementedError(
                    f'Conditioning architecture {self.conditioning_architecture} not implemented.')
        else:
            condition_encoder = None

        model = STSAE_Unet(c_in=self.num_coords, embedding_dim=self.embedding_dim,
                           n_frames=self.input_n_frames, dropout=self.dropout,
                           n_joints=self.n_joints, device=self.device,
                           inject_condition=(self.conditioning_strategy == 'inject'))

        self.condition_encoder, self.model = condition_encoder, model

    def forward(self, input_data: List[torch.Tensor], aggr_strategy: str = None, return_: str = None) -> List[
        torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            input_data (List[torch.Tensor]): list containing the following tensors:
                                             - tensor_data: tensor of shape (B, C, T, V) containing the input sequences
                                             - transformation_idx
                                             - metadata
                                             - actual_frames
            aggr_strategy (str, optional): aggregation strategy to use. If not specified as a function parameter, the aggregation strategy 
                                           specified in the model hyperparameters is used. Defaults to None. 
            return_ (str, optional): return value of the model: 
                                     - only the selected poses according to the aggregation strategy ('pose')
                                     - only the loss of the selected poses ('loss')
                                     - both ('all'). 
                                     If not specified as a function parameter, the return value specified in the model hyperparameters is used. Defaults to None.

        Returns:
            List[torch.Tensor]: [predicted poses and the loss, tensor_data, transformation_idx, metadata, actual_frames]
        """

        # Unpack data: tensor_data is the input data, meta_out is a list of metadata
        tensor_data, meta_out = self._unpack_data(input_data)
        B = tensor_data.shape[0]
        # Select frames to condition on and to corrupt according to the conditioning strategy
        condition_data, corrupt_data, idxs = self._select_frames(tensor_data)
        # Encode the condition data
        condition_embedding, _ = self._encode_condition(condition_data)

        generated_xs = []
        for _ in range(self.n_generated_samples):
            # Generate gaussian noise of the same shape as the corrupt_data
            x = torch.randn_like(corrupt_data, device=self.device)
            for i in reversed(range(1, self.noise_steps)):
                # Set the time step
                t = torch.full(size=(B,), fill_value=i, dtype=torch.long, device=self.device)
                # Prepare the input data for the conditioning strategies 'concat', 'random_imp' and 'inbetween_imp'
                input_data = self._prepare_input_data(condition_data, x, idxs[1])
                # Predict the noise
                predicted_noise = self._unet_forward(input_data, t=t, condition_data=condition_embedding,
                                                     corrupt_idxs=idxs[1])
                # Get the alpha and beta values and expand them to the shape of the predicted noise
                alpha = self._alpha[t][:, None, None, None]
                alpha_hat = self._alpha_hat[t][:, None, None, None]
                beta = self._beta[t][:, None, None, None]
                # Generate gaussian noise of the same shape as the predicted noise
                noise = torch.randn_like(x, device=self.device) if i > 1 else torch.zeros_like(x, device=self.device)
                # Recover the predicted sequence
                x = (1 / torch.sqrt(alpha)) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
            # Append the generated sequence to the list of generated sequences
            generated_xs.append(x)

        print("generated_xs.shape", len(generated_xs), generated_xs[0].shape)
        selected_x, loss_of_selected_x = self._aggregation_strategy(generated_xs, corrupt_data, aggr_strategy)

        return self._pack_out_data(selected_x, loss_of_selected_x, [tensor_data] + meta_out, return_=return_)

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.float32:
        """
        Training step of the model.

        Args:
            batch (List[torch.Tensor]): list containing the following tensors:
                                        - tensor_data: tensor of shape (B, C, T, V) containing the input sequences
                                        - transformation_idx
                                        - metadata
                                        - actual_frames
            batch_idx (int): index of the batch

        Returns:
            torch.float32: loss of the model
        """

        # Unpack data: tensor_data is the input data, meta_out is a list of metadata
        tensor_data, _ = self._unpack_data(batch)
        # print("tensor_data", tensor_data)
        # Select frames to condition on and to corrupt according to the conditioning strategy
        print("tensor_data.shape", tensor_data.shape)
        condition_data, corrupt_data, idxs = self._select_frames(tensor_data)
        print("condition_data.shape", condition_data.shape)
        print("corrupt_data.shape", corrupt_data.shape)
        print("idxs", idxs)
        # Encode the condition data
        condition_embedding, rec_cond_data = self._encode_condition(condition_data)
        # Sample the time steps and corrupt the data
        t = self.noise_scheduler.sample_timesteps(corrupt_data.shape[0]).to(self.device)
        x_t, noise = self.noise_scheduler.noise_graph(corrupt_data, t)
        # Prepare the input data for the conditioning strategies 'concat', 'random_imp' and 'inbetween_imp'
        x_t = self._prepare_input_data(condition_data, x_t, idxs[1])
        # Predict the noise
        predicted_noise = self._unet_forward(x_t, t=t, condition_data=condition_embedding, corrupt_idxs=idxs[1])
        # Compute the loss
        loss_noise = torch.mean(self.loss_fn(predicted_noise, noise))
        # ori_shape = predicted_noise.shape
        # predicted_noise_vectors = predicted_noise.reshape(ori_shape[0], ori_shape[1] * ori_shape[2] * ori_shape[3])
        # noise_vectors = noise.reshape(ori_shape[0], ori_shape[1] * ori_shape[2] * ori_shape[3])
        #
        # cosine_loss = F.cosine_embedding_loss(predicted_noise_vectors, noise_vectors, torch.Tensor([1]).to(self.device), reduction="mean")
        # cent_loss = F.cross_entropy(F.normalize(predicted_noise_vectors), noise_vectors, reduction="mean")
        # loss_noise = cosine_loss + 0.1 * cent_loss

        # print("loss_noise", loss_noise.shape, loss_noise, loss_noise.dtype)
        # loss_noise = F.cosine_similarity(predicted_noise, noise, dim=1)
        # loss_noise = torch.mean(loss_noise)
        # print("loss_noise",loss_noise.shape, loss_noise, loss_noise.dtype)
        self.log('loss_noise', loss_noise)

        if self.conditioning_architecture == 'AE':
            loss_rec_cond = F.mse_loss(rec_cond_data, condition_data)
            print("diff", rec_cond_data.shape, condition_data.shape)
            # diff abs rec_cond_data and condition_data
            # diff = torch.abs(rec_cond_data - condition_data)
            # print("diff", diff.shape)
            # 1024, 2, 3, 12
            # diff has 4 axis, reduce mean on axis 1, 2, 3
            # diff = torch.mean(diff, dim=(0,1,2))
            # print("diff", diff.shape)
            # print index of max value of diff
            # print("max diff index", torch.argmax(diff), torch.max(diff))
            # print("min diff index", torch.argmin(diff), torch.min(diff))

            # ori_shape = rec_cond_data.shape
            # rec_cond_data_vectors = rec_cond_data.reshape(ori_shape[0], ori_shape[1] * ori_shape[2] * ori_shape[3])
            # condition_data_vectors = condition_data.reshape(ori_shape[0], ori_shape[1] * ori_shape[2] * ori_shape[3])
            # cosine_loss = F.cosine_embedding_loss(rec_cond_data_vectors, condition_data_vectors, torch.Tensor([1]).to(self.device), reduction="mean")
            # cent_loss = F.cross_entropy(F.normalize(rec_cond_data_vectors), condition_data_vectors, reduction="mean")
            # loss_rec_cond = cosine_loss + 0.1 * cent_loss
            # print("loss_rec_cond", loss_rec_cond.shape, loss_rec_cond)
            # loss_rec_cond = F.cosine_similarity(rec_cond_data, condition_data, dim=1)
            # loss_rec_cond = torch.mean(loss_rec_cond)
            # print("loss_rec_cond", loss_rec_cond.shape)
            loss = loss_noise + loss_rec_cond * self.rec_weight
            self.log("loss_recons", loss_rec_cond)
        else:
            loss = loss_noise

        return loss

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """
        Test step of the model. It saves the output of the model and the input data as 
        List[torch.Tensor]: [predicted poses and the loss, tensor_data, transformation_idx, metadata, actual_frames]

        Args:
            batch (List[torch.Tensor]): list containing the following tensors:
                                        - tensor_data: tensor of shape (B, C, T, V) containing the input sequences
                                        - transformation_idx
                                        - metadata
                                        - actual_frames
            batch_idx (int): index of the batch
        """
        # print('test step batch: ', batch[0].shape)
        output = self.forward(batch)
        # print("test step output shape: ", output.shape)
        self._test_output_list.append(output)
        return

    def on_test_epoch_start(self) -> None:
        """
        Called when the test epoch begins.
        """

        super().on_test_epoch_start()
        self._test_output_list = []
        return

    def on_test_epoch_end(self) -> float:
        """
        Test epoch end of the model.

        Returns:
            float: test auc score
        """

        out, gt_data, trans, meta, frames = processing_data(self._test_output_list)
        print("out.shape", out.shape)
        print("mean out: ", np.mean(out))
        del self._test_output_list
        if self.save_tensors:
            tensors = {'prediction': out, 'gt_data': gt_data,
                       'trans': trans, 'metadata': meta, 'frames': frames}
            self._save_tensors(tensors, split_name=self.split, aggr_strategy=self.aggregation_strategy,
                               n_gen=self.n_generated_samples)
        metrics = self.post_processing(out, gt_data, trans, meta, frames)
        clip_auc, auc, best_thr, ori_clip_auc, ori_auc, f1, recall, precision, accuracy, cf_matrix, clip_fname_pred_map = metrics
        self.log('AUC', clip_auc)

        self.best_clip_auc = clip_auc
        self.best_metrics = {
            'clip_auc': clip_auc, 'auc': auc, 'best_thr': best_thr, 'ori_clip_auc': ori_clip_auc,
            'ori_auc': ori_auc, 'f1': f1, 'recall': recall, 'precision': precision, 'accuracy': accuracy,
            'confusion_matrix': cf_matrix,

        }
        print(self.best_metrics)
        self.best_metrics['clip_fname_pred_map'] = clip_fname_pred_map
        return clip_auc

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """
        Validation step of the model. It saves the output of the model and the input data as 
        List[torch.Tensor]: [predicted poses and the loss, tensor_data, transformation_idx, metadata, actual_frames]

        Args:
            batch (List[torch.Tensor]): list containing the following tensors:
                                        - tensor_data: tensor of shape (B, C, T, V) containing the input sequences
                                        - transformation_idx
                                        - metadata
                                        - actual_frames
            batch_idx (int): index of the batch
        """
        output = self.forward(batch)
        print("val output length: ", len(output))
        print("output shape: ", output[0].shape)
        print("output shape: ", output[1].shape)
        print("output shape: ", output[2].shape)
        print("output shape: ", output[3].shape)
        print("output shape: ", output[4].shape)
        self._validation_output_list.append(output)
        return

    def on_validation_epoch_start(self) -> None:
        """
        Called when the test epoch begins.
        """

        super().on_validation_epoch_start()
        self._validation_output_list = []
        return

    def on_validation_epoch_end(self) -> float:
        """
        Validation epoch end of the model.

        Returns:
            float: validation auc score
        """

        out, gt_data, trans, meta, frames = processing_data(self._validation_output_list)

        del self._validation_output_list
        if self.save_tensors:
            tensors = {'prediction': out, 'gt_data': gt_data,
                       'trans': trans, 'metadata': meta, 'frames': frames}
            self._save_tensors(tensors, split_name=self.split, aggr_strategy=self.aggregation_strategy,
                               n_gen=self.n_generated_samples)
        metrics = self.post_processing(out, gt_data, trans, meta, frames)
        clip_auc, auc, best_thr, ori_clip_auc, ori_auc, f1, recall, precision, accuracy, cf_matrix, clip_fname_pred_map = metrics
        self.log('AUC', clip_auc, sync_dist=True)
        print(f'AUC score: {clip_auc:.6f}')

        if self.use_original_anomaly_score:
            if self.best_clip_auc < ori_clip_auc:
                self.best_clip_auc = ori_clip_auc
                self.best_metrics = {
                    'clip_auc': clip_auc, 'auc': auc, 'best_thr': best_thr, 'ori_clip_auc': ori_clip_auc,
                    'ori_auc': ori_auc, 'f1': f1, 'recall': recall, 'precision': precision, 'accuracy': accuracy,
                    'confusion_matrix': cf_matrix,
                    'clip_fname_pred_map': clip_fname_pred_map
                }
            return ori_clip_auc
        else:
            if self.best_clip_auc < clip_auc:
                self.best_clip_auc = clip_auc
                self.best_metrics = {
                    'clip_auc': clip_auc, 'auc': auc, 'best_thr': best_thr, 'ori_clip_auc': ori_clip_auc,
                    'ori_auc': ori_auc, 'f1': f1, 'recall': recall, 'precision': precision, 'accuracy': accuracy,
                    'confusion_matrix': cf_matrix,
                    'clip_fname_pred_map': clip_fname_pred_map
                }
            return clip_auc

    def configure_optimizers(self) -> Dict:
        """
        Configure the optimizers and the learning rate schedulers.

        Returns:
            Dict: dictionary containing the optimizers, the learning rate schedulers and the metric to monitor
        """

        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1, verbose=False)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'AUC'}

    def post_processing(self, out: np.ndarray, gt_data: np.ndarray, trans: np.ndarray, meta: np.ndarray,
                        frames: np.ndarray) -> float:
        """
        Post processing of the model.

        Args:
            out (np.ndarray): output of the model
            gt_data (np.ndarray): ground truth data
            trans (np.ndarray): transformation index
            meta (np.ndarray): metadata
            frames (np.ndarray): frame indexes of the data

        Returns:
            float: auc score
        """
        all_gts = [file_name for file_name in os.listdir(self.gt_path) if file_name.endswith('.npy')]
        all_gts = sorted(all_gts)
        scene_clips = [(int(fn.split('_')[0]), int(fn.split('_')[1].split('.')[0])) for fn in all_gts]
        hr_ubnormal_masked_clips = get_hr_ubnormal_mask(self.split) if (
                self.use_hr and (self.dataset_name == 'UBnormal')) else {}
        hr_avenue_masked_clips = get_avenue_mask() if self.dataset_name == 'HR-Avenue' else {}

        num_transform = self.num_transforms
        model_scores_transf = {}
        dataset_gt_transf = {}
        model_scores_transf_orig = {}

        model_scores_transf_each_clip = {}
        dataset_gt_transf_each_clip = {}
        model_scores_transf_orig_each_clip = {}

        pos_score_stds = []
        neg_score_stds = []

        pos_score_outlier_ratios = []
        neg_score_outlier_ratios = []
        pos_score_outliers = []
        neg_score_outliers = []

        pos_mean_orig_scores = []
        neg_mean_orig_scores = []

        pos_mean_scores = []
        neg_mean_scores = []

        clip_pred_frames = []

        for transformation in tqdm(range(num_transform)):
            # iterating over each transformation T

            dataset_gt = []
            model_scores = []
            model_scores_orig = []
            dataset_gt_each_clip = []
            model_scores_each_clip = []
            model_scores_orig_each_clip = []

            cond_transform = (trans == transformation)

            out_transform, gt_data_transform, meta_transform, frames_transform = filter_vectors_by_cond(
                [out, gt_data, meta, frames], cond_transform)

            for idx in range(len(all_gts)):
                # iterating over each clip C with transformation T

                scene_idx, clip_idx = scene_clips[idx]
                is_pos = scene_idx < 1000
                gt = np.load(os.path.join(self.gt_path, all_gts[idx]))
                n_frames = gt.shape[0]

                cond_scene_clip = (meta_transform[:, 0] == scene_idx) & (meta_transform[:, 1] == clip_idx)
                out_scene_clip, gt_scene_clip, meta_scene_clip, frames_scene_clip = filter_vectors_by_cond(
                    [out_transform, gt_data_transform,
                     meta_transform, frames_transform],
                    cond_scene_clip)
                # person ids
                figs_ids = sorted(list(set(meta_scene_clip[:, 2])))
                error_per_person = []

                for fig in figs_ids:
                    # iterating over each actor A in each clip C with transformation T

                    cond_fig = (meta_scene_clip[:, 2] == fig)
                    out_fig, _, frames_fig = filter_vectors_by_cond([out_scene_clip, gt_scene_clip, frames_scene_clip],
                                                                    cond_fig)
                    loss_matrix = compute_var_matrix(out_fig, frames_fig, n_frames)
                    fig_reconstruction_loss = np.nanmax(loss_matrix, axis=0)

                    if self.anomaly_score_pad_size != -1:
                        fig_reconstruction_loss = pad_scores(fig_reconstruction_loss, gt, self.anomaly_score_pad_size)

                    error_per_person.append(fig_reconstruction_loss)

                clip_score = np.stack(error_per_person, axis=0)

                clip_score_orig = np.stack(error_per_person, axis=0)
                clip_score_orig = np.mean(clip_score_orig, axis=0)

                model_scores_orig.append(clip_score_orig)
                model_scores_orig_each_clip.append(np.mean(clip_score_orig))

                clip_score_log = np.log1p(clip_score)
                clip_score = np.mean(clip_score, axis=0) + (
                        np.amax(clip_score_log, axis=0) - np.amin(clip_score_log, axis=0))

                # removing the non-HR frames for UBnormal dataset
                if (scene_idx, clip_idx) in hr_ubnormal_masked_clips:
                    clip_score = clip_score[hr_ubnormal_masked_clips[(scene_idx, clip_idx)]]
                    gt = gt[hr_ubnormal_masked_clips[(scene_idx, clip_idx)]]

                # removing the non-HR frames for Avenue dataset
                if clip_idx in hr_avenue_masked_clips:
                    clip_score = clip_score[np.array(hr_avenue_masked_clips[clip_idx]) == 1]
                    gt = gt[np.array(hr_avenue_masked_clips[clip_idx]) == 1]

                clip_score = score_process(clip_score, self.anomaly_score_frames_shift,
                                           self.anomaly_score_filter_kernel_size)
                model_scores.append(clip_score)
                dataset_gt.append(gt)

                std_clip_score = (clip_score - np.mean(clip_score)) / np.std(clip_score)

                # calculating the z-score for clip_score
                mean = np.mean(clip_score)
                std = np.std(clip_score)
                #
                # threshold = 3#2~3
                # clip_score_outliers = []
                # for x in clip_score:
                #     z_score = (x - mean) / std
                #     if abs(z_score) > threshold:
                #         clip_score_outliers.append(x)

                # detect outliers using iqr for clip_score
                q1 = np.percentile(clip_score, 35)
                q3 = np.percentile(clip_score, 65)
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                clip_score_outliers = []
                for x in clip_score:
                    if x < lower_bound or x > upper_bound:
                        clip_score_outliers.append(x)

                if is_pos:
                    pos_score_stds.append(std)
                    pos_score_outlier_ratios.append(len(clip_score_outliers) / len(clip_score))
                    pos_score_outliers.append(len(clip_score_outliers))
                    pos_mean_scores.append(np.mean(clip_score))
                    pos_mean_orig_scores.append(np.mean(clip_score_orig))
                else:
                    neg_score_stds.append(std)
                    neg_score_outlier_ratios.append(len(clip_score_outliers) / len(clip_score))
                    neg_score_outliers.append(len(clip_score_outliers))
                    neg_mean_scores.append(np.mean(clip_score))
                    neg_mean_orig_scores.append(np.mean(clip_score_orig))

                # append average clip score for each transformation
                if self.remove_last_remain_frame:
                    each_clip_score = clip_score[:len(clip_score) - self.n_frames + 1]
                else:
                    each_clip_score = clip_score
                model_scores_each_clip.append(np.mean(each_clip_score))
                dataset_gt_each_clip.append(np.mean(gt))
                if transformation == 0:
                    clip_pred_frames.append((all_gts[idx].split(".")[0], len(clip_score)))

            model_scores = np.concatenate(model_scores, axis=0)
            dataset_gt = np.concatenate(dataset_gt, axis=0)
            model_scores_orig = np.concatenate(model_scores_orig, axis=0)

            model_scores_transf_each_clip[transformation] = model_scores_each_clip
            dataset_gt_transf_each_clip[transformation] = dataset_gt_each_clip
            model_scores_transf_orig_each_clip[transformation] = model_scores_orig_each_clip

            model_scores_transf[transformation] = model_scores
            dataset_gt_transf[transformation] = dataset_gt
            model_scores_transf_orig[transformation] = model_scores_orig

        # aggregating the anomaly scores for all transformations
        pds = np.mean(np.stack(list(model_scores_transf.values()), 0), 0)
        gt = dataset_gt_transf[0]
        pds_orig = np.mean(np.stack(list(model_scores_transf_orig.values()), 0), 0)
        print("gt", gt.shape)
        print("pds", pds.shape)
        pds_each_clip = np.mean(np.stack(list(model_scores_transf_each_clip.values()), 0), 0)
        pds_orig_each_clip = np.mean(np.stack(list(model_scores_transf_orig_each_clip.values()), 0), 0)
        gt_each_clip = dataset_gt_transf_each_clip[0]
        # print("gt_each_clip", gt_each_clip.shape)
        # print("pds_each_clip", pds_each_clip.shape)

        print("len_pos_score_stds", len(pos_score_stds))
        print("len_neg_score_stds", len(neg_score_stds))
        print("len_pos_score_outlier_ratios", len(pos_score_outlier_ratios))
        print("len_neg_score_outlier_ratios", len(neg_score_outlier_ratios))
        print("len_pos_score_outliers", len(pos_score_outliers))
        print("len_neg_score_outliers", len(neg_score_outliers))
        print("len_pos_mean_orig_scores", len(pos_mean_orig_scores))
        print("len_neg_mean_orig_scores", len(neg_mean_orig_scores))
        print("len_pos_mean_scores", len(pos_mean_scores))
        print("len_neg_mean_scores", len(neg_mean_scores))

        print(f"pos_score_stds: {np.mean(pos_score_stds)}")
        print(f"neg_score_stds: {np.mean(neg_score_stds)}")
        print(f"pos_score_outlier_ratios: {np.mean(pos_score_outlier_ratios)}")
        print(f"neg_score_outlier_ratios: {np.mean(neg_score_outlier_ratios)}")
        print(f"num pos_score_outliers: {np.mean(pos_score_outliers)}")
        print(f"num neg_score_outliers: {np.mean(neg_score_outliers)}")
        print(f"pos_mean_orig_scores: {np.mean(pos_mean_orig_scores)}")
        print(f"neg_mean_orig_scores: {np.mean(neg_mean_orig_scores)}")
        print(f"pos_mean_scores: {np.mean(pos_mean_scores)}")
        print(f"neg_mean_scores: {np.mean(neg_mean_scores)}")

        # convert gt_each_clip's type to int using round
        gt_each_clip = np.round(gt_each_clip).astype(int)

        try:
            clip_auc = roc_auc_score(gt_each_clip, pds_each_clip)
            print(f'Clip AUC: {clip_auc:.6f}')


        except:
            traceback.print_exc()
            print("Error in computing clip AUC")

        try:
            ori_score_auc = roc_auc_score(gt, pds_orig)
            print(f'Original Score AUC: {ori_score_auc:.6f}')
        except:
            traceback.print_exc()
            print("Error in computing ori AUC")

        try:
            clip_ori_score_auc = roc_auc_score(gt_each_clip, pds_orig_each_clip)
            print(f'Clip Original Score AUC: {clip_ori_score_auc:.6f}')
        except:
            traceback.print_exc()
            print("Error in computing clip ori AUC")

        auc = roc_auc_score(gt, pds)
        print(f'AUC: {auc:.4f}')

        if self.use_original_anomaly_score:
            fpr, tpr, thresholds = roc_curve(gt_each_clip, pds_orig_each_clip)
        else:
            fpr, tpr, thresholds = roc_curve(gt_each_clip, pds_each_clip)
        J = tpr - fpr
        ix = np.argmax(J)
        best_thr = thresholds[ix]

        print('Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f' % (
            best_thr, tpr[ix], 1 - fpr[ix], J[ix]))
        if self.use_original_anomaly_score:
            y_prob_pred = (pds_orig_each_clip >= best_thr).astype(bool)
        else:
            y_prob_pred = (pds_each_clip >= best_thr).astype(bool)

        print(classification_report(gt_each_clip, y_prob_pred, target_names=['normal', 'abnormal']))
        print(f'F1 Score: {f1_score(gt_each_clip, y_prob_pred)}')
        print(f'Recall Score: {recall_score(gt_each_clip, y_prob_pred)}')
        print(f'Precision Score: {precision_score(gt_each_clip, y_prob_pred)}')
        print(f'Accuracy Score: {accuracy_score(gt_each_clip, y_prob_pred)}')

        print(f'Confusion Matrix: {confusion_matrix(gt_each_clip, y_prob_pred)}')

        # if self.args.slack_webhook_url:
        #     slack.send_info_to_slack(
        #         f"Clip AUC, Frame AUC, Clip Ori Score AUC, Frame Ori Score AUC\n{clip_auc * 100:.2f} {auc * 100:.2f} {clip_ori_score_auc * 100:.2f} {ori_score_auc * 100:.2f}",
        #         self.args.slack_webhook_url)

        # gt (85,)
        # clip_score (85,)
        # dataset_gt (8405,)
        # model_scores (8405,)
        # dataset_gt_transf (8405,)
        # model_scores_transf (8405,)
        # gt(8405, )
        # pds(8405, )

        # computing the AUC

        # pds
        # best thr 기반으로 postive, negative로 나눠서 저장
        #
        clip_fname_pred_map = {}
        print("len pds_each_clip: ", len(pds_each_clip), "len clip_pred_frames: ", len(clip_pred_frames),
              "len ori pds_each_clip: ", len(pds_orig_each_clip))
        total_num_frames = 0
        print("len pds", len(pds), "len ori pds", len(pds_orig))
        tmp_pds = copy.deepcopy(pds)
        tmp_pds_orig = copy.deepcopy(pds_orig)
        for i, (fname, num_frames) in enumerate(clip_pred_frames):
            # convert pds to list
            total_num_frames += num_frames
            sample_anomaly_scores = tmp_pds[:num_frames]
            ori_sample_anomaly_scores = tmp_pds_orig[:num_frames]
            # print(fname, "num_frames: ", num_frames)
            # print(i * num_frames, (i + 1) * num_frames, len(pds))
            # print("sample_anomaly_scores: ", len(sample_anomaly_scores))
            # print("ori sample_anomaly_scores: ", len(ori_sample_anomaly_scores))
            # sample_anomaly_scores = sample_anomaly_scores[:len(sample_anomaly_scores) - self.n_frames + 1]
            clip_fname_pred_map[fname] = {
                "sample_anomaly_scores": sample_anomaly_scores,
                "ori_sample_anomaly_scores": ori_sample_anomaly_scores,
                "clip_gt": bool(gt_each_clip[i]),
                "clip_pred": y_prob_pred[i],
                "clip_gt_desc": "abnormal" if gt_each_clip[i] else "normal",
                "clip_pred_desc": "abnormal" if y_prob_pred[i] else "normal",
                "pred_result": "correct" if gt_each_clip[i] == y_prob_pred[i] else "wrong",
                "clip_anomaly_score": pds_each_clip[i],
                "ori_clip_anomaly_score": pds_orig_each_clip[i],
                "threshold": best_thr,
            }
            # print("mean sample_anomaly_scores: ", np.mean(sample_anomaly_scores), "mean ori sample_anomaly_scores: ",
            #       np.mean(ori_sample_anomaly_scores),
            #       "clip_anomaly_score", pds_each_clip[i], "ori_clip_anomaly_score", pds_orig_each_clip[i])
            # check TFPN
            TFPN = None
            if gt_each_clip[i] == y_prob_pred[i] and gt_each_clip[i] == 1:
                TFPN = "TP"
            elif gt_each_clip[i] != y_prob_pred[i] and gt_each_clip[i] == 1:
                TFPN = "FP"
            elif gt_each_clip[i] == y_prob_pred[i] and gt_each_clip[i] == 0:
                TFPN = "TN"
            elif gt_each_clip[i] != y_prob_pred[i] and gt_each_clip[i] == 0:
                TFPN = "FN"
            clip_fname_pred_map[fname]["TFPN"] = TFPN
            tmp_pds = tmp_pds[num_frames:]
            tmp_pds_orig = tmp_pds_orig[num_frames:]
        print("total_num_frames: ", total_num_frames, "len pds: ", len(pds), "len ori pds: ", len(pds_orig))
        return clip_auc, auc, best_thr, clip_ori_score_auc, ori_score_auc, f1_score(gt_each_clip,
                                                                                    y_prob_pred), recall_score(
            gt_each_clip, y_prob_pred), precision_score(gt_each_clip, y_prob_pred), accuracy_score(gt_each_clip,
                                                                                                   y_prob_pred), confusion_matrix(
            gt_each_clip, y_prob_pred), clip_fname_pred_map

    def test_on_saved_tensors(self, split_name: str) -> float:
        """
        Skip the prediction step and test the model on the saved tensors.

        Args:
            split_name (str): split name (val, test)

        Returns:
            float: auc score
        """

        tensors = self._load_tensors(split_name, self.aggregation_strategy, self.n_generated_samples)
        auc_score = self.post_processing(tensors['prediction'], tensors['gt_data'], tensors['trans'],
                                         tensors['metadata'], tensors['frames'])
        print(f'AUC score: {auc_score:.6f}')
        return auc_score

    ## Helper functions

    def _aggregation_strategy(self, generated_xs: List[torch.Tensor], input_sequence: torch.Tensor,
                              aggr_strategy: str) -> Tuple[torch.Tensor]:
        """
        Aggregates the generated samples and returns the selected one and its reconstruction error.
        Strategies:
            - all: returns all the generated samples
            - random: returns a random sample
            - best: returns the sample with the lowest reconstruction loss
            - worst: returns the sample with the highest reconstruction loss
            - mean: returns the mean of the losses of the generated samples
            - median: returns the median of the losses of the generated samples
            - mean_pose: returns the mean of the generated samples
            - median_pose: returns the median of the generated samples

        Args:
            generated_xs (List[torch.Tensor]): list of generated samples
            input_sequence (torch.Tensor): ground truth sequence
            aggr_strategy (str): aggregation strategy

        Raises:
            ValueError: if the aggregation strategy is not valid

        Returns:
            Tuple[torch.Tensor]: selected sample and its reconstruction error
        """

        aggr_strategy = self.aggregation_strategy if aggr_strategy is None else aggr_strategy
        if aggr_strategy == 'random':
            return generated_xs[np.random.randint(len(generated_xs))]

        B, repr_shape = input_sequence.shape[0], input_sequence.shape[1:]
        compute_loss = lambda x: torch.mean(self.loss_fn(x, input_sequence).reshape(-1, prod(repr_shape)), dim=-1)
        # def compute_loss(x, input_sequence):
        #     x = x.reshape(-1, prod(repr_shape))
        #     input_sequence = input_sequence.reshape(-1, prod(repr_shape))
        #     cosine_loss = F.cosine_embedding_loss(x, input_sequence,
        #                                           torch.Tensor([1]).to(self.device), reduce=False)
        #     cent_loss = F.cross_entropy(F.normalize(x), input_sequence, reduce=False)
        #     loss_noise = cosine_loss + 0.1 * cent_loss
        #     return loss_noise
        # losses = [compute_loss(x, input_sequence) for x in generated_xs]
        losses = [compute_loss(x) for x in generated_xs]

        if aggr_strategy == 'all':
            dims_idxs = list(range(2, len(repr_shape) + 2))
            dims_idxs = [1, 0] + dims_idxs
            selected_x = torch.stack(generated_xs).permute(*dims_idxs)
            loss_of_selected_x = torch.stack(losses).permute(1, 0)
        elif aggr_strategy == 'mean':
            selected_x = None
            loss_of_selected_x = torch.mean(torch.stack(losses), dim=0)
        elif aggr_strategy == 'mean_pose':
            selected_x = torch.mean(torch.stack(generated_xs), dim=0)
            loss_of_selected_x = compute_loss(selected_x)
        elif aggr_strategy == 'median':
            loss_of_selected_x, _ = torch.median(torch.stack(losses), dim=0)
            selected_x = None
        elif aggr_strategy == 'median_pose':
            selected_x, _ = torch.median(torch.stack(generated_xs), dim=0)
            loss_of_selected_x = compute_loss(selected_x)
        elif aggr_strategy == 'best' or aggr_strategy == 'worst':
            strategy = (lambda x, y: x < y) if aggr_strategy == 'best' else (lambda x, y: x > y)
            loss_of_selected_x = torch.full((B,), fill_value=(1e10 if aggr_strategy == 'best' else -1.),
                                            device=self.device)
            selected_x = torch.zeros((B, *repr_shape)).to(self.device)

            for i in range(len(generated_xs)):
                mask = strategy(losses[i], loss_of_selected_x)
                loss_of_selected_x[mask] = losses[i][mask]
                selected_x[mask] = generated_xs[i][mask]
        elif 'quantile' in aggr_strategy:
            q = float(aggr_strategy.split(':')[-1])
            loss_of_selected_x = torch.quantile(torch.stack(losses), q, dim=0)
            selected_x = None
        else:
            raise ValueError(f'Unknown aggregation strategy {aggr_strategy}')

        return selected_x, loss_of_selected_x

    def _cut_array_from_indices(self, x: torch.Tensor, indices: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Cuts the input array according to the indices. This function is called when the conditioning strategy is 'random imputation'.

        Args:
            x (torch.Tensor): input sequence
            indices (torch.Tensor): indices of the conditioning frames

        Returns:
            Tuple[torch.Tensor]: conditioning frames, non-conditioning frames, conditioning indices, non-conditioning indices
        """

        mask = indices < self.conditioning_indices
        idxs_sequence = torch.arange(self.n_frames, device=self.device)[None, :].expand(mask.shape[0], -1)
        indices_ = idxs_sequence[mask].reshape(-1, self.conditioning_indices)
        not_indices = idxs_sequence[~mask].reshape(-1, self.n_frames - self.conditioning_indices)
        mask = mask[:, None, None, :].expand(-1, self.num_coords, self.n_joints, -1)
        container = x[mask].reshape(-1, self.num_coords, self.n_joints, self.conditioning_indices)
        not_container = x[~mask].reshape(-1, self.num_coords, self.n_joints, self.n_frames - self.conditioning_indices)

        return container, not_container, indices_, not_indices

    def _encode_condition(self, condition_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the conditioning data if the conditioning strategy is 'inject', returns None otherwise.

        Args:
            condition_data (torch.Tensor): conditioning data

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: encoded conditioning data, reconstructed conditioning data (if AE is used as condition encoder)
        """

        if self.condition_encoder is None:
            return None, None

        return self.condition_encoder(condition_data, t=None)

    def _infer_number_of_joint(self, args: argparse.Namespace) -> int:
        """
        Infer the number of joints based on the dataset parameters.

        Args:
            args (argparse.Namespace): arguments containing the hyperparameters of the model

        Returns:
            int: number of joints
        """
        if args.custom_num_joints is not None:
            return args.custom_num_joints
        else:
            if args.headless:
                joints_to_consider = 14
            elif args.kp18_format:
                joints_to_consider = 18
            else:
                joints_to_consider = 17
        return joints_to_consider

    def _load_tensors(self, split_name: str, aggr_strategy: str, n_gen: int) -> Dict[str, torch.Tensor]:
        """
        Loads the tensors from the experiment directory.

        Args:
            split_name (str): name of the split (train, val, test)
            aggr_strategy (str): aggregation strategy
            n_gen (int): number of generated samples

        Returns:
            Dict[str, torch.Tensor]: dictionary containing the tensors. The keys are inferred from the file names.
        """

        name = 'saved_tensors_{}_{}_{}'.format(split_name, aggr_strategy, n_gen)
        path = os.path.join(self.ckpt_dir, name)
        tensor_files = os.listdir(path)
        tensors = {}
        for t_file in tensor_files:
            t_name = t_file.split('.')[0]
            tensors[t_name] = torch.load(os.path.join(path, t_file))
        return tensors

    def _pack_out_data(self, selected_x: torch.Tensor, loss_of_selected_x: torch.Tensor,
                       additional_out: List[torch.Tensor], return_: str) -> List[torch.Tensor]:
        """
        Packs the output data according to the return_ argument.

        Args:
            selected_x (torch.Tensor): generated samples selected among the others according to the aggregation strategy
            loss_of_selected_x (torch.Tensor): loss of the selected samples
            additional_out (List[torch.Tensor]): additional output data (ground truth, meta-data, etc.)
            return_ (str): return strategy. Can be 'pose', 'loss', 'all'

        Raises:
            ValueError: if return_ is None and self.model_return_value is None

        Returns:
            List[torch.Tensor]: output data
        """

        if return_ is None:
            if self.model_return_value is None:
                raise ValueError('Either return_ or self.model_return_value must be set')
            else:
                return_ = self.model_return_value

        if return_ == 'pose':
            out = [selected_x]
        elif return_ == 'loss':
            out = [loss_of_selected_x]
        elif return_ == 'all':
            out = [loss_of_selected_x, selected_x]

        return out + additional_out

    def _pos_encoding(self, t: torch.Tensor, channels: int) -> torch.Tensor:
        """
        Encodes the time information of the input sequence.

        Args:
            t (torch.Tensor): time steps
            channels (int): dimension of the encoding

        Returns:
            torch.Tensor: encoded time steps
        """

        return self.model.pos_encoding(t, channels)

    def _prepare_input_data(self, condition_data: torch.Tensor, corrupt_data: torch.Tensor,
                            corrupt_idxs: torch.Tensor) -> torch.Tensor:
        """
        Prepares the input data for the conditioning strategies 'concat', 'random_imp' and 'inbetween_imp'.

        Args:
            condition_data (torch.Tensor): condition data of shape (B, C, number of conditioning frames, V)
            corrupt_data (torch.Tensor): corrupt data of shape (B, C, number of corrupt frames, V)
            corrupt_idxs (torch.Tensor): indexes of the corrupt frames of shape (B, number of corrupt frames) if the conditioning strategy is 'random_imp' 
                                         or (number of conditioning frames,) if the conditioning strategy is 'inbetween_imp'

        Returns:
            torch.Tensor: input data
        """

        if self.conditioning_strategy == 'concat':
            x = torch.cat((condition_data, corrupt_data), dim=2)
        elif self.conditioning_strategy == 'inject':
            x = corrupt_data
        elif self.conditioning_strategy in ['random_imp', 'inbetween_imp']:
            # indexing is performed on CPU because it fails on GPU 
            x = torch.empty((corrupt_data.shape[0], self.num_coords, self.n_frames, self.n_joints),
                            dtype=corrupt_data.dtype, device=self.device)
            corrupt_mask = torch.zeros(x.shape, dtype=bool, device='cpu')
            if self.conditioning_strategy == 'random_imp':
                corrupt_idxs = corrupt_idxs.flatten().to('cpu')
                batch_idxs = torch.arange(corrupt_data.shape[0], device='cpu').repeat_interleave(self.n_frames_corrupt)
                corrupt_mask[batch_idxs, :, corrupt_idxs, :] = True
            else:
                corrupt_mask.index_fill_(2, corrupt_idxs.to('cpu'), True)
            corrupt_mask = corrupt_mask.to(self.device)
            x[~corrupt_mask], x[corrupt_mask] = condition_data.flatten(), corrupt_data.flatten()
        else:
            x = corrupt_data
        return x

    def _save_tensors(self, tensors: Dict[str, torch.Tensor], split_name: str, aggr_strategy: str, n_gen: int) -> None:
        """
        Saves the tensors in the experiment directory.

        Args:
            tensors (Dict[str, torch.Tensor]): tensors to save
            split_name (str): name of the split (val, test)
            aggr_strategy (str): aggregation strategy
            n_gen (int): number of generated samples
        """

        name = 'saved_tensors_{}_{}_{}'.format(split_name, aggr_strategy, n_gen)
        path = os.path.join(self.ckpt_dir, name)
        if not os.path.exists(path):
            os.mkdir(path)
        for t_name, tensor in tensors.items():
            torch.save(tensor, os.path.join(path, t_name + '.pt'))

    def _select_frames(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Selects the conditioning frames according to the conditioning strategy.

        Args:
            data (torch.Tensor): input sequence

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]: conditioning frames, non-conditioning frames, indexes
        """

        if self.conditioning_strategy == 'random_imp':
            # Randomly select the indices of the conditioning frames and cut the input sequence accordingly
            indices = torch.tensor([torch.randperm(self.n_frames).tolist() for _ in range(data.shape[0])],
                                   device=self.device)
            cond_data, corrupt_data, cond_idxs, corrupt_idxs = self._cut_array_from_indices(data.permute((0, 1, 3, 2)),
                                                                                            indices)
            cond_data = cond_data.permute((0, 1, 3, 2))
            corrupt_data = corrupt_data.permute((0, 1, 3, 2))

        elif self.conditioning_strategy == 'no_condition':
            # The input to the model is the whole sequence
            cond_data, cond_idxs = None, None
            corrupt_data, corrupt_idxs = data, torch.arange(self.n_frames, device=self.device)

        elif isinstance(self.conditioning_indices, int):
            if self.conditioning_strategy == 'inbetween_imp':
                # Take conditioning frames with step equal to `conditioning indices` starting from the first frame
                cond_idxs = torch.arange(start=0, end=self.n_frames, step=self.conditioning_indices, device=self.device)
                corrupt_idxs = torch.tensor([i for i in range(self.n_frames) if i not in cond_idxs], device=self.device)
            else:
                # Use the integer in `conditioning indices` to split the input sequence in two parts
                cond_idxs = torch.arange(start=0, end=self.n_frames // self.conditioning_indices, device=self.device)
                corrupt_idxs = torch.arange(start=self.n_frames // self.conditioning_indices, end=self.n_frames,
                                            device=self.device)
            cond_data = torch.index_select(data, 2, cond_idxs)
            corrupt_data = torch.index_select(data, 2, corrupt_idxs)

        else:
            # Take the indices explicitly specified in `conditioning indices`
            cond_idxs = torch.tensor(self.conditioning_indices, device=self.device)
            corrupt_idxs = torch.tensor([i for i in range(self.n_frames) if i not in self.conditioning_indices],
                                        device=self.device)
            cond_data = torch.index_select(data, 2, cond_idxs)
            corrupt_data = torch.index_select(data, 2, corrupt_idxs)

        return cond_data, corrupt_data, [cond_idxs, corrupt_idxs]

    def _set_conditioning_strategy(self) -> Tuple[int]:
        """
        Sets the conditioning strategy.

        Raises:
            NotImplementedError: if the conditioning strategy is not implemented

        Returns:
            Tuple[int]: number of conditioning frames, number of non-conditioning frames (input to the model)
        """

        input_n_frames = self.n_frames

        if self.conditioning_strategy == 'no_condition':
            n_frames_cond = 0

        elif self.conditioning_strategy == 'random_imp':
            assert isinstance(self.conditioning_indices, int), \
                'Random imputation requires an integer number of frames to condition on, not a list of indices'
            n_frames_cond = self.conditioning_indices

        elif self.conditioning_strategy == 'inbetween_imp':
            if isinstance(self.conditioning_indices, int):
                n_frames_cond = self.n_frames // self.conditioning_indices
            else:
                n_frames_cond = len(self.conditioning_indices)

        elif self.conditioning_strategy in ['concat', 'inject']:
            if isinstance(self.conditioning_indices, int):
                n_frames_cond = self.n_frames // self.conditioning_indices
            else:
                assert self.conditioning_indices == list(
                    range(min(self.conditioning_indices), max(self.conditioning_indices) + 1)), \
                    'Conditioning indices must be a list of consecutive integers'
                assert (min(self.conditioning_indices) == 0) or (max(self.conditioning_indices) == (self.n_frames - 1)), \
                    'Conditioning indices must start from 0 or end at the last frame'
                n_frames_cond = len(self.conditioning_indices)

            input_n_frames = (
                    self.n_frames - n_frames_cond) if self.conditioning_strategy == 'inject' else input_n_frames

        else:
            raise NotImplementedError(f'Conditioning strategy {self.conditioning_strategy} not implemented')

        n_frames_to_corrupt = self.n_frames - n_frames_cond
        return n_frames_cond, n_frames_to_corrupt, input_n_frames

    def _set_diffusion_variables(self) -> None:
        """
        Sets the diffusion variables.
        """

        self.noise_scheduler = Diffusion(noise_steps=self.noise_steps, n_joints=self.n_joints,
                                         device=self.device, time=self.n_frames)
        self._beta_ = self.noise_scheduler.schedule_noise()
        self._alpha_ = (1. - self._beta_)
        self._alpha_hat_ = torch.cumprod(self._alpha_, dim=0)

    def _unet_forward(self, input_data: torch.Tensor, t: torch.Tensor = None, condition_data: torch.Tensor = None,
                      *, corrupt_idxs: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the UNet model.

        Args:
            input_data (torch.Tensor): input tensor of shape (batch_size, n_coords, n_frames, n_joints)
            t (torch.Tensor, optional): encoded time tensor. Defaults to None.
            condition_data (torch.Tensor, optional): conditioning embedding of shape (batch_size, latent_dim). Defaults to None.
            corrupt_idxs (torch.Tensor): indices of the frames for which the model should predict the noise

        Returns:
            torch.Tensor: predicted noise of shape (batch_size, n_coords, n_frames, n_joints)
        """

        prediction, _ = self.model(input_data, t, condition_data=condition_data)

        if self.conditioning_strategy not in ['inject', 'no_condition']:
            # indexing is performed on CPU because it fails on GPU 
            corrupt_mask = torch.zeros(prediction.shape, dtype=bool, device='cpu')
            if self.conditioning_strategy == 'random_imp':
                batch_idxs = torch.arange(prediction.shape[0], device='cpu').repeat_interleave(self.n_frames_corrupt)
                corrupt_idxs = corrupt_idxs.flatten().to('cpu')
                corrupt_mask[batch_idxs, :, corrupt_idxs, :] = True
            else:
                corrupt_mask.index_fill_(2, corrupt_idxs.to('cpu'), True)
            corrupt_mask = corrupt_mask.to(self.device)
            prediction = prediction[corrupt_mask].reshape(-1, self.num_coords, self.n_frames_corrupt, self.n_joints)

        return prediction

    def _unpack_data(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Unpacks the data.

        Args:
            x (torch.Tensor): list containing the input data, the transformation index, the metadata and the actual frames.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: input data, list containing the transformation index, the metadata and the actual frames.
        """
        tensor_data = x[0].to(self.device)
        transformation_idx = x[1]
        metadata = x[2]
        actual_frames = x[3]
        meta_out = [transformation_idx, metadata, actual_frames]
        return tensor_data, meta_out

    @property
    def _beta(self) -> torch.Tensor:
        return self._beta_.to(self.device)

    @property
    def _alpha(self) -> torch.Tensor:
        return self._alpha_.to(self.device)

    @property
    def _alpha_hat(self) -> torch.Tensor:
        return self._alpha_hat_.to(self.device)
