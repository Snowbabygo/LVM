import os
import cv2
import torch
import yaml
from torchvision import transforms
from PIL import Image
import PIL

import matplotlib.pyplot as plt
from utils import network_parameters
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time
import utils
import numpy as np
import random
from data_RGB import get_training_data, get_validation_data
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model.SUNet import SUNet_model
from model.SUNet import Classifier
from torch import distributed as dist
from datetime import datetime
import torch.nn.functional as Fun
import pandas as pd 
#from Tools import Klein,RP2,Tours,Ring,Mobius,Elastic_transform
torch.multiprocessing.set_sharing_strategy('file_system')


#--------------------------------  GPU  ----------------------------------#

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = "./checkpoints/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

#---------------------------  加载训练好的权重  --------------------------#

model_path = "./checkpoints/best_checkpoints.pth"
## Set Seeds
torch.backends.cudnn.benchmark = False
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


#------------------  Load yaml configuration file  -----------------------#
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Test = opt['TESTING']
OPT = opt['OPTIM']
val_dir = Test['VAL_DIR']


model_restored = SUNet_model(opt)
pretrained_dict = torch.load(model_path, map_location="cpu")
model_restored.load_state_dict(pretrained_dict["model"], strict=False)
model_restored.to(device)
p_number = network_parameters(model_restored)

#------------------------------ Optimizer --------------------------------#
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=0.01)

warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(OPT['LR_MIN']))

#--------------------------------- Loss -----------------------------------#
mse_loss = nn.L1Loss()

## DataLoaders
print('==> Loading datasets')
val_dataset = get_validation_data(val_dir, {'patch_size': Test['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=10, shuffle=True, num_workers=0,
                        drop_last=False)

# Show the training configuration
print(f'''==> Testing details:
------------------------------------------------------------------
    Val patches size:   {str(Test['VAL_PS']) + 'x' + str(Test['VAL_PS'])}
    Model parameters:   {p_number}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}''')
print('------------------------------------------------------------------')

test_data = []

model_restored.eval()
for i, data_info in enumerate(tqdm(val_loader), 0):

    # Forward propagation
    input_, file_name= data_info[0].to(device), data_info[1],

    restored, F = model_restored(input_)
    loss = mse_loss(restored, input_)
    if len(file_name)<10:
        pass
    else:
        for ba in range(10):
            
            Feature = F[ba].unsqueeze(0)
            Feature = Feature.transpose(2, 1)
            Feature = Fun.avg_pool1d(Feature, kernel_size=4, stride=4)
            HIDDEN_DIM = 768
            Z = Feature.flatten().tolist()
            testdata_output = {}
            # print(index_label)
            testdata_output["file_name"] = file_name[ba]

            testdata_output["features"] = Z

            test_data.append(testdata_output)
#
df = pd.DataFrame(test_data)
for j in tqdm(range(HIDDEN_DIM)):

    df["Z"+str(j+1)] = df['features'].apply(lambda Z: Z[j])
df = df.drop(columns=['features'])
df.to_csv('./feature_result.csv', index=False)


end_time = datetime.now()
