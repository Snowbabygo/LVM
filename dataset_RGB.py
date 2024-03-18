import os
import h5py
from torch.utils.data import Dataset
from skimage import transform
import torch
from PIL import Image
import dask.array as da
import pandas as pd
from astropy.io import fits
import torchvision.transforms.functional as TF
import random
import numpy as np
from skimage.transform import resize
from Tools import dr2_rgb, DESI_find_Contour


class DataLoaderTrain(Dataset):
    def __init__(self, data_dir, img_options=None, propress_fuction = None, batch_size=1000):
        super(DataLoaderTrain, self).__init__()

        self.data_dir = data_dir
        self.img_options = img_options
        self.ps = self.img_options['patch_size']
        self.data = []
        self.filenames = []

        for filename in os.listdir(data_dir):
            if not filename.endswith('.h5'):
                continue
            self.filenames.append(os.path.join(data_dir, filename))

        with h5py.File(self.filenames[0], 'r') as file:
            self.data = file['images'][:3500000:100]
            self.data_index = range(3500000)[:3500000:100]

        self.sizex = self.data.shape[0]
        self.propress_fuction = propress_fuction

        print("Train_count:{}".format(self.sizex))

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):

        ps = self.ps

        inp_img = self.data[index].transpose(1, 2, 0)    # 3,152,152 --> 152,152,3
        index_label = self.data_index[index]

        # 数据预处理算法 ！！！！！！！！！！！！
        if self.propress_fuction == None:
            inp_img = inp_img.transpose(2,  1,0)  # 152,152,3 -->  3,152,152
            
        channel, w, h = np.shape(inp_img)

        if w > ps or h > ps:
            if w > h:
                inp_img = transform.resize(inp_img.transpose(1, 2, 0), (ps, ps), anti_aliasing=True)
            else:
                inp_img = transform.resize(inp_img.transpose(1, 2, 0), (ps, ps), anti_aliasing=True)
        else:
            padw = ps - w if w < ps else 0
            padh = ps - h if h < ps else 0
        # Reflect Pad in case image is smaller than patch_size
            if padw != 0 or padh != 0:
                inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
        # print(np.shape(inp_img))
        inp_img = TF.to_tensor(inp_img).float()

        aug = random.randint(0, 8)

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)

        elif aug == 2:
            inp_img = inp_img.flip(2)

        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))

        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)

        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)

        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))

        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))

        return inp_img,index_label


class DataLoaderVal(Dataset):
    def __init__(self, data_dir, img_options=None, propress_fuction = None, batch_size=1000):
        super(DataLoaderVal, self).__init__()

        self.data_dir = data_dir
        self.img_options = img_options
        self.ps = self.img_options['patch_size']

        self.data = []
        self.filenames = []
        self.data_index = []

        for filename in os.listdir(data_dir):
            if filename.endswith('.fits'):
                self.filenames.append(os.path.join(data_dir,filename))
                self.name.append(filename)
                self.data_index.append(filename)

        for filename in self.filenames:
            print(filename)
            hdu = fits.open(filename)
            img = hdu[0].data
            img = np.array(img, dtype=np.float32)
            img = transform.resize(img, (3, 128, 128), anti_aliasing=True)
            self.data.append(img)

        self.sizex = len(self.data)
        self.propress_fuction = propress_fuction

        print("Test_count:{}".format(self.sizex))

    def __len__(self):

        return self.sizex

    def __getitem__(self, index):

        ps = self.ps
        inp_img = self.data[index]
        index_label = self.data_index[index]
        print(self.data_index)

        x0, x1, y0, y1 = DESI_find_Contour(np.transpose(inp_img, (2, 1, 0)))
        inp_img = inp_img[:, x0:x1, y0:y1]
        print(np.shape(inp_img))

        # 数据预处理算法 ！！！！！！！！！！！！
        if self.propress_fuction != None:
            inp_img = inp_img.transpose(1, 2, 0)
            inp_img = self.propress_fuction(inp_img).transpose(2, 1, 0)  # 152,152,3 -->  3,152,152

        inp_img = transform.resize(inp_img.transpose(1, 2, 0), (ps, ps), anti_aliasing=True)
        inp_img = TF.to_tensor(inp_img).float()

        aug = random.randint(0, 8)

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)

        elif aug == 2:
            inp_img = inp_img.flip(2)

        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))

        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)

        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)

        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))

        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))

        return inp_img, index_label










