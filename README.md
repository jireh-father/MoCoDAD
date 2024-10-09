# MoCoDAD for Animals
## Setup
### Environment
```sh
conda env create -f environment.yaml
conda activate mocodad
```

## Inference
### [20241010] model files
- model ckpt
  - deploy/20241010/epoch=8-step=792.ckpt
- config file
  - deploy/20241010/inference_config.yaml
- scaler file
  - deploy/20241010/local_robust.pkl

### Inference in python code
```python
import mocodad
config = "./deploy/20241010/inference_config.yaml"
tmp_dir = "./tmp"
keypoint_csv_path = "./deploy/20241010/test_keypoint.csv"
mocodad = mocodad.Mocodad(config, tmp_dir)

result, position = mocodad.inference(keypoint_csv_path)
```

### Inference in command line
```sh
python mocodad.py --config ./deploy/20241010/inference_config.yaml --keypoint_csv_path ./deploy/20241010/test_keypoint.csv
````

### Evaluation in command line
```sh
# you need to prepare the keypoint directory and data json file
# request the keypoint directory to the owner
# the sample data json file is in ./deploy/20241010/horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_9.json 
python predict_MoCoDAD_eval.py --config ./deploy/20241010/inference_config.yaml --keypoint_dir ./horse_kp_20240710 --data_json ./deploy/20241010/horse_20240710_walk_side_pos_thr_4_neg_thr_3_rem_mis_seed_9.json
```

### Inference config file
```yaml
split: 'test' # do not change
debug: false # do not change
seed: 999 # same seed used for training

## Computational resources
accelerator: 'gpu'
devices: [0] # indices of cuda devices to use

## Paths
scaler_path: './deploy/20241010/local_robust.pickle' # path to the fitted scaler
load_ckpt: 'deploy/20241010/epoch=8-step=792.ckpt' # name of the checkpoint to load at inference time
create_experiment_dir: true

### Model's configuration

## U-Net's configuration
embedding_dim: 16 # dimension of the embedding of the UNet
dropout: 0. # probability of dropout
conditioning_strategy: 'inject' # choices ['inject' (add2layers), 'concat' (cat), 'inbetween_imp' (interleave), 'random_imp' (random_indices), 'no_condition' (none)]

## Conditioning network's configuration
conditioning_architecture: 'AE' # choices ['AE', 'E', 'E_unet']
conditioning_indices: [0,1,2] # If conditioning_strategy=random_imp, it must be int and it is used as the number of random indices that will be selected
                              # If an int is given and conditioning_strategy=[inject|concat], n_frames//conditioning_indices will be used as the number of conditioning indices
                              # If an int is given and conditioning_strategy=inbetween_imp, it will be used as the step of the conditioning indices, starting from 0
                              # If a list is given, it will be used as the conditioning indices 
h_dim: 32 # dimension of the bottleneck at the end of the encoder of the conditioning network
latent_dim: 16 # dimension of the latent space of the conditioning encoder
channels: [32,16,32] # channels for the encoder (ignored if conditioning_architecture=E_unet)

##############################


### Training's configuration

## Diffusion's configuration
noise_steps: 10 # how many diffusion steps to perform

### Optimizer and scheduler's configuration
opt_lr: 0.001

## Losses' configuration
loss_fn: 'smooth_l1' # loss function; choices ['mse', 'l1', 'smooth_l1']
rec_weight: 0.01 # weight of the reconstruction loss

##############################


### Inference's configuration
n_generated_samples: 5 # number of samples to generate
model_return_value: 'all' # choices ['loss', 'poses', 'all']; if 'loss', the model will return the loss;
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
seg_len: 4 # length of the window (cond+noised)
vid_res: [10800,7200]
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
seg_stride: 1
custom_num_joints: 8
seg_th: 0
start_offset: 0
symm_range: true
use_fitted_scaler: false
remove_last_remain_frame: false
use_original_anomaly_score: true
use_angle: true
use_angle_norm: false
camera_direction: side
target_keypoint_name: only_foots_and_fetlocks
max_frames: 75
num_div: 5
num_thr_div: 1
sort_max_frames: false
skip_not_continuous_sample: false
use_random_frame_range: false
pred_threshold: 0.0386
reset_index: false
use_num_last_frames: true
```

## Training

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
