# Generally processing photometric images of galaxies by planting human-in-the-loop upon a large vision model

## Mingxiang Fu, Yu Song, Jiameng Lv 
***
> Abstract : The exponential growth of astronomical datasets provides an unprecedented
opportunity for humans to gain insight into the Universe. However, effec-
tively analyzing this vast amount of data poses a significant challenge. As-
tronomers are turning to deep learning techniques to address this, but the
methods are limited by their specific training sets. Hence, to avoid dupli-
cate workloads, we built a framework for general analysis of astronomical
vision tasks based on a large vision model for downstream tasks, including
morphological classification, image restoration, object detection, parameter
extraction, and more. Considering the low signal-to-noise ratio in astro-
nomical datasets and the imbalanced distribution of different astronomical
objects, we have incorporated a Human-in-the-loop module into our large vi-
sion model, which leverages human knowledge to enhance the reliability and
interpretability of processing astronomical data interactively. The proposed
framework exhibits notable few-shot learning capabilities and versatile adapt-
ability to all the abovementioned tasks in data from large-scale sky surveys,
such as the DESI legacy imaging surveys. Furthermore, multimodal data
can be integrated similarly, which opens up possibilities for conducting joint
analyses with datasets spanning diverse domains in this era of multi-message
astronomy



## Quick Run  
You can download the relevant code files directly from [github](https://github.com/Songyu1026/LVM).  

You can also download the pre-trained weight size is about 400 MB, you can download it on the [website](https://pan.baidu.com/s/1Q8G8gMTzJc7Q2NfRULr60Q?pwd=LVMC ) 
password: LVMC

Using our pre-trained weights, we can extract the features of astronomical images, and finally get the features of 768 dimensions of the data. You can do anything with this data, or continue to fine-tune the network. The following is how to use:
```
Here is an example command:
```
python demo.py --input_dir './demo_samples/' --result_dir './demo_results' --weights './pretrained_model/denoising_model.pth'
```
To test the pre-trained models of denoising on your arbitrary resolution images, run
```
python demo_any_resolution.py --input_dir images_folder_path --stride shifted_window_stride --result_dir save_images_here --weights path_to_models
```
SUNset could only handle the fixed size input which the resolution in training phase same as the mostly transformer-based methods because of the attention masks are fixed. If we want to denoise the arbitrary resolution input, the shifted-window method will be applied to avoid border effect. The code of `demo_any_resolution.py` is supported to fix the problem.

## Train  
To train the restoration models of Denoising. You should check the following components:  
- `training.yaml`:  

  ```
    # Training configuration
    GPU: [0,1,2,3] 

    VERBOSE: False

    SWINUNET:
      IMG_SIZE: 256
      PATCH_SIZE: 4
      WIN_SIZE: 8
      EMB_DIM: 96
      DEPTH_EN: [8, 8, 8, 8]
      HEAD_NUM: [8, 8, 8, 8]
      MLP_RATIO: 4.0
      QKV_BIAS: True
      QK_SCALE: 8
      DROP_RATE: 0.
      ATTN_DROP_RATE: 0.
      DROP_PATH_RATE: 0.1
      APE: False
      PATCH_NORM: True
      USE_CHECKPOINTS: False
      FINAL_UPSAMPLE: 'Dual up-sample'

    MODEL:
      MODE: 'Denoising'

    # Optimization arguments.
    OPTIM:
      BATCH: 4
      EPOCHS: 500
      # EPOCH_DECAY: [10]
      LR_INITIAL: 2e-4
      LR_MIN: 1e-6
      # BETA1: 0.9

    TRAINING:
      VAL_AFTER_EVERY: 1
      RESUME: False
      TRAIN_PS: 256
      VAL_PS: 256
      TRAIN_DIR: './datasets/Denoising_DIV2K/train'       # path to training data
      VAL_DIR: './datasets/Denoising_DIV2K/test' # path to validation data
      SAVE_DIR: './checkpoints'           # path to save models and images
  ```
- Dataset:  
  The preparation of dataset in more detail, see [datasets/README.md](datasets/README.md).  
  
- Train:  
  If the above path and data are all correctly setting, just simply run:  
  ```
  python train.py
  ```  
## Result  

<img src = "https://i.imgur.com/golsiWN.png" width="800">  

## Visual Comparison  

<img src = "https://i.imgur.com/UeeOO0M.png" width="800">  

<img src = "https://i.imgur.com/YavgU0r.png" width="800">  



## Citation  
If you use SUNet, please consider citing:  
```
@inproceedings{fan2022sunet,
  title={SUNet: swin transformer UNet for image denoising},
  author={Fan, Chi-Mao and Liu, Tsung-Jung and Liu, Kuan-Hsien},
  booktitle={2022 IEEE International Symposium on Circuits and Systems (ISCAS)},
  pages={2333--2337},
  year={2022},
  organization={IEEE}
}
```

![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FFanChiMao%2FSUNet&label=visitors&countColor=%232ccce4&style=plastic)  

