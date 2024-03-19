# [Generally processing photometric images of galaxies by planting human-in-the-loop upon a large vision model](https://github.com/Songyu1026/LVM)

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

## Network Architecture  
![image](https://github.com/Songyu1026/LVM/blob/main/Structure%20of%20the%20LVM.png)
## Instructions  
You can download the relevant code files directly from [**github**](https://github.com/Songyu1026/LVM).  

You can also download the pre-trained weight on the [**website**](https://pan.baidu.com/s/1Q8G8gMTzJc7Q2NfRULr60Q?pwd=LVMC )  which is about 400 MB.

**The password: LVMC**

Using our pre-trained weights, you can extract the features of astronomical images, and finally get the features of 768 dimensions of the data. You can do anything with this data, or continue to fine-tune the network. The following is how to use:
```
To test the pre-trained models of LVM on your arbitrary resolution images, here is an example command:
```
--input_dir './data/FITS' 

--result_dir './demo_results' 

--weights './checkpoints/best_checkpoints.pth'

python demo.py 
```
LVM could only handle the fixed size input which the resolution in training phase same as the mostly transformer-based methods because of the attention masks are fixed. 
```

####  Training 
To train the LVM. You should check the following components:  
- `training.yaml`:  
  ```
    # Training configuration
    GPU: [0,1]
    
    VERBOSE: False
    
    SWINUNET:
      IMG_SIZE: 128
      PATCH_SIZE: 4
      WIN_SIZE: 8
      EMB_DIM: 96
      DEPTH_EN: [8, 8, 8, 8]
      HEAD_NUM: [8, 8, 8, 8]
      MLP_RATIO: 4.0
      QKV_BIAS: True
      QK_SCALE: 8
      DROP_RATE: 0.
      ATTN_DROP_RATE: 0.1
      DROP_PATH_RATE: 0.1
      APE: False
      PATCH_NORM: True
      USE_CHECKPOINTS: False
      FINAL_UPSAMPLE: 'Dual up-sample'


    # Optimization arguments.
    OPTIM:
      BATCH: 128
      EPOCHS: 50000
      # EPOCH_DECAY: [10]
      LR_INITIAL: 2e-3
      LR_MIN: 1e-6
      # BETA1: 0.9
    
    TRAINING:
      VAL_AFTER_EVERY: 10
      SAVE_LOG: 10
      TRAIN_PS: 128
      VAL_PS: 128
      TRAIN_DIR: "/nfsdata/share/ssl_legacy/ssl_legacyA/file/south-train/"
    
    TESTING:
      VAL_AFTER_EVERY: 10
      VAL_PS: 128
      DATA_TYPE: "H5"    #FITS
      VAL_DIR: "./data/FITS"
  ```
- Dataset:  
  The format of data input to the model is fits (Flexible Image Transport System). The data selected for training of this algorithm is a single target image with edge detection. You can add required data for training according to your own requirements as long as you control the size of the image to meet network requirements.
  
- Train:  
  If the above path and data are all correctly setting, just simply run:  
  ```
  python train.py
  ```  
- Pretraining Result  
Result formatting: 

#### Downstream tasks 
- the usage of the code

- result


## Citation  
If you use LVM, please consider citing:  
```
@{
}
```



