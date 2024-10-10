# MoCoDAD for Animals
## Setup
### Environment
```sh
conda env create -f environment.yaml
conda activate mocodad
```

## Inference
### [20241010] model files
- walk model ckpt
  - deploy/20241010/walk_best_epoch=8-step=792.ckpt
- walk config file
  - deploy/20241010/walk_inference_config.yaml
- walk scaler file
  - deploy/20241010/walk_local_robust.pkl
- trot model ckpt
  - deploy/20241010/trot_best_epoch=8-step=792.ckpt
- trot config file
  - deploy/20241010/trot_inference_config.yaml
- trot scaler file
  - deploy/20241010/trot_local_robust.pkl

### Inference in python code
```python
import mocodad
config = "./deploy/20241010/walk_inference_config.yaml"
tmp_dir = "./tmp"
keypoint_csv_path = "./deploy/20241010/walk_test_keypoint.csv"
mocodad = mocodad.Mocodad(config, tmp_dir)

result, position = mocodad.inference(keypoint_csv_path)
```

### Inference in command line
```sh
# walk
python mocodad.py --config ./deploy/20241010/walk_inference_config.yaml --keypoint_csv_path ./deploy/20241010/walk_test_keypoint.csv

# trot
python mocodad.py --config ./deploy/20241010/trot_inference_config.yaml --keypoint_csv_path ./deploy/20241010/walk_test_keypoint.csv
````

### Evaluation in command line
```sh
# you need to prepare the keypoint directory and data json file
# request the keypoint directory to the owner
# the sample data json file is in ./deploy/20241010/horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_9.json 
python predict_MoCoDAD_eval.py --config ./deploy/20241010/walk_inference_config.yaml --keypoint_dir ./horse_kp_20240710 --data_json ./deploy/20241010/horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_9.json
```

### Added parameters in the config file for inference
- 학습용 설정 파일을 그대로 복사한 후 아래 내용만 상황에 맞게 변경하면 됨.
```yaml
split: 'test' # do not change
seed: 999 # same seed used for training

## Paths
scaler_path: './deploy/20241010/walk_local_robust.pickle' # path to the fitted scaler
load_ckpt: 'deploy/20241010/walk_best_epoch=8-step=792.ckpt' # name of the checkpoint to load at inference time
pred_threshold: 0.0386 # threshold for the anomaly score

# dataset parameters
# 학습 데이터셋 생성했던 파라미터와 동일하게 세팅해야 함.
camera_direction: side # side, front, back
target_keypoint_name: only_foots_and_fetlocks # make_horse_dataset.py 혹은 make_horse_angle_dataset.py의 TARGET_KP_COL_DICT 딕셔너리의 키값 사용
max_frames: 75
num_div: 5 # x좌표 기준 가운데 프레임만 사용할 경우, 값이 5일 경우 5등분에서 양쪽 2개 제외 가운데 3개 범위에 해당되는 프레임만 사용 
num_thr_div: 1 # 고정
sort_max_frames: false # 실험중
skip_not_continuous_sample: false # 실험중
use_random_frame_range: false # 실험중
reset_index: false # 실험중
use_num_last_frames: false # 실험중
```

## Training

### Dataset
#### Make Common Dataset Format
- 라벨링 완료된 csv 파일 기반으로 Common Dataset json파일 생성
  - 샘플 라벨링 파일: ./merged_with_score_with_missing_frames_20240710.csv
```sh
# 10 fold cross validation
# 모든 파라미터는 파일 내부에 설명있음
python convert_expert_label_excel_to_json_v5_horse.py --output_path ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_1.json --seed 1 --pos_num_labels_threshold 4 --neg_num_labels_threshold 3 --gait_types walk --directions LtoR,RtoL --remove_missing_frame_samples --remove_multi_video_samples --copy_keypoint_from_server --change_kp_path
python convert_expert_label_excel_to_json_v5_horse.py --output_path ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_2.json --seed 2 --pos_num_labels_threshold 4 --neg_num_labels_threshold 3 --gait_types walk --directions LtoR,RtoL --remove_missing_frame_samples --remove_multi_video_samples --copy_keypoint_from_server --change_kp_path
python convert_expert_label_excel_to_json_v5_horse.py --output_path ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_3.json --seed 3 --pos_num_labels_threshold 4 --neg_num_labels_threshold 3 --gait_types walk --directions LtoR,RtoL --remove_missing_frame_samples --remove_multi_video_samples --copy_keypoint_from_server --change_kp_path
python convert_expert_label_excel_to_json_v5_horse.py --output_path ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_4.json --seed 4 --pos_num_labels_threshold 4 --neg_num_labels_threshold 3 --gait_types walk --directions LtoR,RtoL --remove_missing_frame_samples --remove_multi_video_samples --copy_keypoint_from_server --change_kp_path
python convert_expert_label_excel_to_json_v5_horse.py --output_path ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_5.json --seed 5 --pos_num_labels_threshold 4 --neg_num_labels_threshold 3 --gait_types walk --directions LtoR,RtoL --remove_missing_frame_samples --remove_multi_video_samples --copy_keypoint_from_server --change_kp_path
python convert_expert_label_excel_to_json_v5_horse.py --output_path ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_6.json --seed 6 --pos_num_labels_threshold 4 --neg_num_labels_threshold 3 --gait_types walk --directions LtoR,RtoL --remove_missing_frame_samples --remove_multi_video_samples --copy_keypoint_from_server --change_kp_path
python convert_expert_label_excel_to_json_v5_horse.py --output_path ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_7.json --seed 7 --pos_num_labels_threshold 4 --neg_num_labels_threshold 3 --gait_types walk --directions LtoR,RtoL --remove_missing_frame_samples --remove_multi_video_samples --copy_keypoint_from_server --change_kp_path
python convert_expert_label_excel_to_json_v5_horse.py --output_path ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_8.json --seed 8 --pos_num_labels_threshold 4 --neg_num_labels_threshold 3 --gait_types walk --directions LtoR,RtoL --remove_missing_frame_samples --remove_multi_video_samples --copy_keypoint_from_server --change_kp_path
python convert_expert_label_excel_to_json_v5_horse.py --output_path ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_9.json --seed 9 --pos_num_labels_threshold 4 --neg_num_labels_threshold 3 --gait_types walk --directions LtoR,RtoL --remove_missing_frame_samples --remove_multi_video_samples --copy_keypoint_from_server --change_kp_path
python convert_expert_label_excel_to_json_v5_horse.py --output_path ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_10.json --seed 10 --pos_num_labels_threshold 4 --neg_num_labels_threshold 3 --gait_types walk --directions LtoR,RtoL --remove_missing_frame_samples --remove_multi_video_samples --copy_keypoint_from_server --change_kp_path
```

#### Make MoCoDAD Dataset Format
- 바로 위 Common Dataset를 사용해서 MoCoDAD Dataset을 생성

##### 1. 좌표 사용
```sh
# 10 fold cross validation
# horse_kp_20240710 데이터는 요청해주세요.
# 모든 파라미터는 파일 내부에 설명있음
python make_horse_dataset.py --keypoint_dir ./horse_kp_20240710/ --target_keypoint_name baseline --window_length 4 --save_test --output_dir ./data/horse/baseline_data_240710_win4_center5_1_maaxf_75_walk_side_pos_thr4_neg_thr3_rem_mis_cv10/cv1 --label_file ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_1.json --num_div 5 --num_thr_div 1 --max_frames 75
python make_horse_dataset.py --keypoint_dir ./horse_kp_20240710/ --target_keypoint_name baseline --window_length 4 --save_test --output_dir ./data/horse/baseline_data_240710_win4_center5_1_maaxf_75_walk_side_pos_thr4_neg_thr3_rem_mis_cv10/cv2 --label_file ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_2.json --num_div 5 --num_thr_div 1 --max_frames 75
python make_horse_dataset.py --keypoint_dir ./horse_kp_20240710/ --target_keypoint_name baseline --window_length 4 --save_test --output_dir ./data/horse/baseline_data_240710_win4_center5_1_maaxf_75_walk_side_pos_thr4_neg_thr3_rem_mis_cv10/cv3 --label_file ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_3.json --num_div 5 --num_thr_div 1 --max_frames 75
python make_horse_dataset.py --keypoint_dir ./horse_kp_20240710/ --target_keypoint_name baseline --window_length 4 --save_test --output_dir ./data/horse/baseline_data_240710_win4_center5_1_maaxf_75_walk_side_pos_thr4_neg_thr3_rem_mis_cv10/cv4 --label_file ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_4.json --num_div 5 --num_thr_div 1 --max_frames 75
python make_horse_dataset.py --keypoint_dir ./horse_kp_20240710/ --target_keypoint_name baseline --window_length 4 --save_test --output_dir ./data/horse/baseline_data_240710_win4_center5_1_maaxf_75_walk_side_pos_thr4_neg_thr3_rem_mis_cv10/cv5 --label_file ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_5.json --num_div 5 --num_thr_div 1 --max_frames 75
python make_horse_dataset.py --keypoint_dir ./horse_kp_20240710/ --target_keypoint_name baseline --window_length 4 --save_test --output_dir ./data/horse/baseline_data_240710_win4_center5_1_maaxf_75_walk_side_pos_thr4_neg_thr3_rem_mis_cv10/cv6 --label_file ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_6.json --num_div 5 --num_thr_div 1 --max_frames 75
python make_horse_dataset.py --keypoint_dir ./horse_kp_20240710/ --target_keypoint_name baseline --window_length 4 --save_test --output_dir ./data/horse/baseline_data_240710_win4_center5_1_maaxf_75_walk_side_pos_thr4_neg_thr3_rem_mis_cv10/cv7 --label_file ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_7.json --num_div 5 --num_thr_div 1 --max_frames 75
python make_horse_dataset.py --keypoint_dir ./horse_kp_20240710/ --target_keypoint_name baseline --window_length 4 --save_test --output_dir ./data/horse/baseline_data_240710_win4_center5_1_maaxf_75_walk_side_pos_thr4_neg_thr3_rem_mis_cv10/cv8 --label_file ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_8.json --num_div 5 --num_thr_div 1 --max_frames 75
python make_horse_dataset.py --keypoint_dir ./horse_kp_20240710/ --target_keypoint_name baseline --window_length 4 --save_test --output_dir ./data/horse/baseline_data_240710_win4_center5_1_maaxf_75_walk_side_pos_thr4_neg_thr3_rem_mis_cv10/cv9 --label_file ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_9.json --num_div 5 --num_thr_div 1 --max_frames 75
python make_horse_dataset.py --keypoint_dir ./horse_kp_20240710/ --target_keypoint_name baseline --window_length 4 --save_test --output_dir ./data/horse/baseline_data_240710_win4_center5_1_maaxf_75_walk_side_pos_thr4_neg_thr3_rem_mis_cv10/cv10 --label_file ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_10.json --num_div 5 --num_thr_div 1 --max_frames 75

````
##### 2. 각도 사용
```sh
# 10 fold cross validation
# horse_kp_20240710 데이터는 요청해주세요.
# 모든 파라미터는 파일 내부에 설명있음
python make_horse_angle_dataset.py --keypoint_dir ./horse_kp_20240710/ --target_keypoint_name baseline --window_length 4 --save_test --output_dir ./data/horse/baseline_data_240710_win4_center5_1_maxf_75_walk_side_pos_thr4_neg_thr3_rem_mis_cv10_angle/cv1 --label_file ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_1.json --num_div 5 --num_thr_div 1 --max_frames 75
python make_horse_angle_dataset.py --keypoint_dir ./horse_kp_20240710/ --target_keypoint_name baseline --window_length 4 --save_test --output_dir ./data/horse/baseline_data_240710_win4_center5_1_maxf_75_walk_side_pos_thr4_neg_thr3_rem_mis_cv10_angle/cv2 --label_file ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_2.json --num_div 5 --num_thr_div 1 --max_frames 75
python make_horse_angle_dataset.py --keypoint_dir ./horse_kp_20240710/ --target_keypoint_name baseline --window_length 4 --save_test --output_dir ./data/horse/baseline_data_240710_win4_center5_1_maxf_75_walk_side_pos_thr4_neg_thr3_rem_mis_cv10_angle/cv3 --label_file ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_3.json --num_div 5 --num_thr_div 1 --max_frames 75
python make_horse_angle_dataset.py --keypoint_dir ./horse_kp_20240710/ --target_keypoint_name baseline --window_length 4 --save_test --output_dir ./data/horse/baseline_data_240710_win4_center5_1_maxf_75_walk_side_pos_thr4_neg_thr3_rem_mis_cv10_angle/cv4 --label_file ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_4.json --num_div 5 --num_thr_div 1 --max_frames 75
python make_horse_angle_dataset.py --keypoint_dir ./horse_kp_20240710/ --target_keypoint_name baseline --window_length 4 --save_test --output_dir ./data/horse/baseline_data_240710_win4_center5_1_maxf_75_walk_side_pos_thr4_neg_thr3_rem_mis_cv10_angle/cv5 --label_file ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_5.json --num_div 5 --num_thr_div 1 --max_frames 75
python make_horse_angle_dataset.py --keypoint_dir ./horse_kp_20240710/ --target_keypoint_name baseline --window_length 4 --save_test --output_dir ./data/horse/baseline_data_240710_win4_center5_1_maxf_75_walk_side_pos_thr4_neg_thr3_rem_mis_cv10_angle/cv6 --label_file ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_6.json --num_div 5 --num_thr_div 1 --max_frames 75
python make_horse_angle_dataset.py --keypoint_dir ./horse_kp_20240710/ --target_keypoint_name baseline --window_length 4 --save_test --output_dir ./data/horse/baseline_data_240710_win4_center5_1_maxf_75_walk_side_pos_thr4_neg_thr3_rem_mis_cv10_angle/cv7 --label_file ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_7.json --num_div 5 --num_thr_div 1 --max_frames 75
python make_horse_angle_dataset.py --keypoint_dir ./horse_kp_20240710/ --target_keypoint_name baseline --window_length 4 --save_test --output_dir ./data/horse/baseline_data_240710_win4_center5_1_maxf_75_walk_side_pos_thr4_neg_thr3_rem_mis_cv10_angle/cv8 --label_file ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_8.json --num_div 5 --num_thr_div 1 --max_frames 75
python make_horse_angle_dataset.py --keypoint_dir ./horse_kp_20240710/ --target_keypoint_name baseline --window_length 4 --save_test --output_dir ./data/horse/baseline_data_240710_win4_center5_1_maxf_75_walk_side_pos_thr4_neg_thr3_rem_mis_cv10_angle/cv9 --label_file ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_9.json --num_div 5 --num_thr_div 1 --max_frames 75
python make_horse_angle_dataset.py --keypoint_dir ./horse_kp_20240710/ --target_keypoint_name baseline --window_length 4 --save_test --output_dir ./data/horse/baseline_data_240710_win4_center5_1_maxf_75_walk_side_pos_thr4_neg_thr3_rem_mis_cv10_angle/cv10 --label_file ./horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_10.json --num_div 5 --num_thr_div 1 --max_frames 75
````

### Create Training Config File
- 예제 파일: ./config/DogLeg/horse_baseline_win4_center5_1_maxf_75_data_240710_walk_side_pos_thr4_neg_thr3_rem_old_mis_cv10.yaml
- 예제 설정 파일을 복사하여 파라미터 변경후 사용
```yaml
## General settings
split: 'train' # data split; choices ['train', 'test']
debug: false # if true, load only a few data samples
seed: 999
validation: true # use validation; only for UBnormal
use_hr: false # for validation and test on UBnormal

## Computational resources
accelerator: 'gpu'
devices: [0] # indices of cuda devices to use

## Paths
dir_name: 'train_experiment' # name of the directory of the current experiment
data_dir: './data/horse/baseline_data_240710_win4_center5_1_maxf_75_walk_side_pos_thr4_neg_thr3_rem_old_mis_cv10/' # path to the data
exp_dir: './baseline_data_240710_win4_center5_1_maxf_75_walk_side_pos_thr4_neg_thr3_rem_old_mis_cv10' # path to the directory that will contain the current experiment directory
test_path: './data/horse/baseline_data_240710_win4_center5_1_maxf_75_walk_side_pos_thr4_neg_thr3_rem_old_mis_cv10/validating/test_frame_mask' # path to the test data
load_ckpt: '' # name of the checkpoint to load at inference time
create_experiment_dir: true

## WANDB configuration
use_wandb: false
project_name: "project_name"
wandb_entity: "entity_name"
group_name: "group_name"
use_ema: false

##############################


### Model's configuration

## U-Net's configuration
embedding_dim: 16 # dimension of the embedding of the UNet
dropout: 0. # probability of dropout
conditioning_strategy: 'inject' # choices ['inject' (add2layers), 'concat' (cat), 'inbetween_imp' (interleave), 'random_imp' (random_indices), 'no_condition' (none)]

## Conditioning network's configuration
conditioning_architecture: 'AE' # choices ['AE', 'E', 'E_unet']
conditioning_indices: [0,1,2] # 윈도우안에서 컨디션 역할을 할 프레임의 인덱스 번호
h_dim: 32 # dimension of the bottleneck at the end of the encoder of the conditioning network
latent_dim: 16 # dimension of the latent space of the conditioning encoder
channels: [32,16,32] # channels for the encoder (ignored if conditioning_architecture=E_unet)

##############################


### Training's configuration

## Diffusion's configuration
noise_steps: 10 # how many diffusion steps to perform

### Optimizer and scheduler's configuration
n_epochs: 100
opt_lr: 0.001

## Losses' configuration
loss_fn: 'smooth_l1' # loss function; choices ['mse', 'l1', 'smooth_l1']
rec_weight: 0.01 # weight of the reconstruction loss

##############################


### Inference's configuration
n_generated_samples: 5 # number of samples to generate
model_return_value: 'loss' # choices ['loss', 'poses', 'all']; if 'loss', the model will return the loss; 
                           # if 'poses', the model will return the generated poses; 
                           # if 'all', the model will return both the loss and the generated poses
aggregation_strategy: 'best' # choices ['best', 'mean', 'median', 'random']; if 'best', the best sample will be selected; 
                             # if 'mean', the mean of loss of the samples will be selected; 
                             # if 'median', the median of the loss of the samples will be selected; 
                             # if 'random', a random sample will be selected;
                             # if 'mean_poses', the mean of the generated poses will be selected;
                             # if 'median_poses', the median of the generated poses will be selected;
                             # if 'all', all the generated poses will be selected
filter_kernel_size: 30 # size of the kernel to use for smoothing the anomaly score of each clip
frames_shift: 6 # it compensates the shift of the anomaly score due to the sliding window;
                 # in conjuction with pad_size and filter_kernel_size, it strongly depends on the dataset
save_tensors: true # if true, save the generated tensors for faster inference
load_tensors: false # if true, load the generated tensors for faster inference

##############################


### Dataset's configuration

## Important parameters
dataset_choice: 'UBnormal'
seg_len: 4 # length of the window (cond+noised)
vid_res: [10800,7200] # 그대로 사용하면 됨
batch_size: 1024
pad_size: -1 # size of the padding 

## Other parameters
headless: false # remove the keypoints of the head
hip_center: false # center the keypoints on the hip
kp18_format: false # use the 18 keypoints format
normalization_strategy: 'robust' # use 'none' to avoid normalization, 'robust' otherwise
num_coords: 2 # number of coordinates to use
num_transform: 5 # number of transformations to apply
num_workers: 2
seg_stride: 1 # stride of the window
custom_num_joints: 34 # number of joints to use
seg_th: 0 # 무시
start_offset: 0 # 사용할 프레임의 시작 위치
symm_range: true # 무시
use_fitted_scaler: false # 무시
remove_last_remain_frame: false # 테스트용
use_original_anomaly_score: true # 테스트용
use_angle: false # 각도 사용여부
use_angle_norm: false # 테스트용
```

### Train MoCoDAD
```sh
# --data_root: mocodad용 데이터셋의 루트 경로 지정
python -u train_MoCoDAD_cross_validation.py \
--config ./config/DogLeg/horse_baseline_win4_center5_1_maxf_75_data_240710_walk_side_pos_thr4_neg_thr3_rem_old_mis_cv10.yaml \
--data_root ./data/horse/baseline_data_240710_win4_center5_1_maxf_75_walk_side_pos_thr4_neg_thr3_rem_old_mis_cv10
```

--- 
# VVV Original MOCODAD README.md

# Multimodal Motion Conditioned Diffusion Model for Skeleton-based Video Anomaly Detection
_Alessandro Flaborea*, Luca Collorone*, Guido D'Amely*, Stefano D'Arrigo*, Bardh Prenkaj, Fabio Galasso_

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multimodal-motion-conditioned-diffusion-model/video-anomaly-detection-on-hr-avenue)](https://paperswithcode.com/sota/video-anomaly-detection-on-hr-avenue?p=multimodal-motion-conditioned-diffusion-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multimodal-motion-conditioned-diffusion-model/video-anomaly-detection-on-hr-shanghaitech)](https://paperswithcode.com/sota/video-anomaly-detection-on-hr-shanghaitech?p=multimodal-motion-conditioned-diffusion-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multimodal-motion-conditioned-diffusion-model/video-anomaly-detection-on-hr-ubnormal)](https://paperswithcode.com/sota/video-anomaly-detection-on-hr-ubnormal?p=multimodal-motion-conditioned-diffusion-model)

<p align="center">
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/pytorch-lightning-blue.svg?logo=PyTorch%20Lightning"></a>
    <a href="https://wandb.ai/site"><img alt="Logging: wandb" src="https://img.shields.io/badge/logging-wandb-yellow"></a>

</p>


The official PyTorch implementation of the IEEE/CVF International Conference on Computer Vision (ICCV) '23 paper [**Multimodal Motion Conditioned Diffusion Model for Skeleton-based Video Anomaly Detection**](https://openaccess.thecvf.com/content/ICCV2023/html/Flaborea_Multimodal_Motion_Conditioned_Diffusion_Model_for_Skeleton-based_Video_Anomaly_Detection_ICCV_2023_paper.html).

<!-- Visit our [**webpage**](https://www.pinlab.org/coskad) for more details. -->


<div align="center">
<a href="https://www.youtube.com/watch?v=IuDzVez--9U">
  <img src="https://markdown-videos-api.jorgenkh.no/url?url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DIuDzVez--9U" alt="mocodad" title="mocodad"  width="560" height="315"/>
</a>
</div>


## Content
```
.
├── assets
│   ├── mocodad.jpg
├── config
│   ├── Avenue
│   │   ├── mocodad_test.yaml
│   │   └── mocodad_train.yaml
│   ├── STC
│   │   ├── mocodad_test.yaml
│   │   └── mocodad_train.yaml
│   └── UBnormal
|       ├── mocodad-latent_train.yaml
│       ├── mocodad-latent_train.yaml
│       ├── mocodad_test.yaml
│       └── mocodad_train.yaml
├── environment.yaml
├── eval_MoCoDAD.py
├── models
│   ├── common
│   │   └── components.py
│   ├── gcae
│   │   └── stsgcn.py
│   ├── mocodad_latent.py
│   ├── mocodad.py
│   └── stsae
│       ├── stsae.py
│       └── stsae_unet.py
├── predict_MoCoDAD.py
├── README.md
├── train_MoCoDAD.py
└── utils
    ├── argparser.py
    ├── data.py
    ├── dataset.py
    ├── dataset_utils.py
    ├── diffusion_utils.py
    ├── ema.py
    ├── eval_utils.py
    ├── get_robust_data.py
    ├── __init__.py
    ├── model_utils.py
    ├── preprocessing.py
    └── tools.py
    
```
![teaser](assets/mocodad.jpg) 

## Setup
### Environment
```sh
conda env create -f environment.yaml
conda activate mocodad
```

### Datasets
You can download the extracted poses for the datasets HR-Avenue, HR-ShanghaiTech and HR-UBnormal from the [GDRive](https://drive.google.com/drive/folders/1aUDiyi2FCc6nKTNuhMvpGG_zLZzMMc83?usp=drive_link).

Place the extracted folder in a `./data` folder and change the configs accordingly.


### **Training** 

To train MoCoDAD, you can select the different type of conditioning of the model. The default parameters achieve the best results reported in the paper 

In each config file you can choose the conditioning strategy and change the diffusion process parameters:

1. conditioning_strategy
    -  'inject': Inject condition information into the model. The indices to be used as conditioning can be set using the 'conditioning_indices' parameter. Enabled by default. 
    - 'concat': concat conditioning and noised data to be passed to the model. The indices to be used as conditioning can be set using the 'conditioning_indices' parameter.
    - 'inbetween_imp': Uses the list of indices of the 'conditioning_indices' parameter to select the indices to be used as conditioning.
    - 'random_imp': 'conditioning_indices' must be int and it is used as the number of random indices that will be selected 
    - 'no_condition': if enabled, no motion condition is passed to the model

2. Diffusion Process
    -  noise_steps: how many diffusion steps have to be performed

Update the args 'data_dir', 'test_path', 'dataset_path_to_robust' with the path where you stored the datasets.  To better track your experiments, change 'dir_name' and the wandb parameters.

To train MoCoDAD:
```sh
python train_MoCoDAD.py --config config/[Avenue/UBnormal/STC]/{config_name}.yaml
```


### Once trained, you can run the **Evaluation**

The training config is saved the associated experiment directory (`/args.exp_dir/args.dataset_choice/args.dir_name`). 
To evaluate the model on the test set, you need to change the following parameters in the config:

- split: 'Test'
- validation: 'False'
- load_ckpt: 'name_of_ckpt'

Test MoCoDAD
```sh
python eval_MoCoDAD.py --config /args.exp_dir/args.dataset_choice/args.dir_name/config.yaml
```
additional flag you can use:
- use_hr: False -> just for test. Use the entire version of the dataset or the Human-Related one.

### **Pretrained Models**

The checkpoints for the pretrained models on the three datasets can be found [HERE](https://drive.google.com/drive/folders/1KoxjwArqcIGQVBsxrlHcNJw9wtwJ7jQx?usp=drive_link).
To evaluate them follow the following steps:
1. Download the checkpoints
2. Add them to the corresponding folder `/checkpoints/[Avenue/UBnormal/STC]/pretrained_model`
3. Copy the config file /config/[Avenue/UBnormal/STC]/mocodad_test.yaml in the correct checkpoint folder
4. Update the 'load_ckpt' field with the downloaded ckpt
5. run 
    ```sh
    python eval_MoCoDAD.py --config `/checkpoints/[Avenue/UBnormal/STC]/pretrained_model/mocodad_test.yaml]
    ```

## Citation
```
@InProceedings{Flaborea_2023_ICCV,
    author    = {Flaborea, Alessandro and Collorone, Luca and di Melendugno, Guido Maria D'Amely and D'Arrigo, Stefano and Prenkaj, Bardh and Galasso, Fabio},
    title     = {Multimodal Motion Conditioned Diffusion Model for Skeleton-based Video Anomaly Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {10318-10329}
}
```
