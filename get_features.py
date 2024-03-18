import torch
import torchvision.transforms.functional as TF
from PIL import Image
import os
import tqdm
import h5py
import dask.array as da
from skimage import img_as_ubyte
from collections import OrderedDict
from torchinfo import summary
from skimage import transform
from sklearn.cluster import KMeans
from natsort import natsorted
from glob import glob
import cv2
import pandas as pd
import argparse
from model.SUNet import SUNet_model
from model.SUNet import Classifier
import yaml
import numpy as np
import torch.nn.functional as Fun
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.decomposition import PCA
import seaborn as sns
from skimage.transform import resize
import umap
from sklearn.decomposition import PCA
from Tools import dr2_rgb,DESI_find_Contour

with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)

Train = opt['TRAINING']
parser = argparse.ArgumentParser(description='Demo Image Restoration')
parser.add_argument('--input_dir', default="/share2/fumingxiang/BGS_cut_v1/raw/", type=str,
                    help='Input images')
parser.add_argument('--window_size', default=8, type=int, help='window size')
parser.add_argument('--result_dir', default='./result', type=str, help='Directory for results')

args = parser.parse_args()

def MSE(img1,img2):
    mse = np.mean( (img1 - img2) ** 2 )
    return mse

inp_dir = args.input_dir
out_dir = args.result_dir

os.makedirs(out_dir, exist_ok=True)

files = natsorted(glob(os.path.join(inp_dir, '*.jpg'))
                  + glob(os.path.join(inp_dir, '*.JPG'))
                  + glob(os.path.join(inp_dir, '*.png'))
                  + glob(os.path.join(inp_dir, '*.h5'))
                  + glob(os.path.join(inp_dir, '*.PNG')))

if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

# Load corresponding model architecture and weights
model = SUNet_model(opt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model.cuda()

model_path = "/share/songyu/Abnormal_galaxy/result/Elastic/best_old.pth"
pretrained_dict = torch.load(model_path, map_location="cpu")

model.load_state_dict(pretrained_dict["model"], strict=False)
model.to(device)
model.eval()

print('processing images......')

def get_feature_umap(num, label):
    data_list = []
    id_list = []
    indx_list = []
    y_ids = []
    y_path = []
    filenames = []
    y_feature_vectors = []
    ra_list = []
    dec_list = []
    ra_select = []
    dec_select = []
    loss_list= []
    # gaia_csv_path = "./GAIA_dr2--BGS-all-h5.csv"
    # gaia_csv = pd.read_csv(gaia_csv_path)

    for filename in os.listdir(inp_dir):
        if not filename.endswith('.h5'):
            continue
        filenames.append(os.path.join(inp_dir,filename))
        filenames = [per for per in filenames if "_" not in per.split("/")[-1] and "big" in per]
        # filenames = [per for per in filenames if "_" in per.split("/")[-1] and "failure" not in per]
        print(filenames)

    for filename in filenames:
        print(filename)
        images = h5py.File(filename, 'r')['image'][::num]
        data_list.extend(h5py.File(filename, 'r')['image'][::num])
        ra_list.extend(h5py.File(filename, 'r')['ra'][::num])
        dec_list.extend(h5py.File(filename, 'r')['dec'][::num])
        indices = range(0, len(images) * num, num)  # 添加当前文件的索引列表
        combined = [f"{filename}_{i}" for i in indices]
        indx_list.extend(combined)

    for data_id, data, h5_path, ra, dec in tqdm.tqdm(zip(range(len(indx_list)), data_list, indx_list, ra_list, dec_list)):
        # data = np.transpose(data, (0, 2, 3, 1))
        # if dec in gaia_csv['dec'].values: 
        #     pass
        # else:     
        x0,x1,y0,y1 = DESI_find_Contour(np.transpose(data,(2,1,0)))
        img_jpg = data[:,x0:x1,y0:y1]
        # resized_img = resize(img_jpg, (3, 128, 128))
        data = transform.resize(img_jpg, (1, 3, 128, 128), anti_aliasing=True)
        data_tensor = torch.tensor(data)
        with torch.no_grad():
            data_tensor = data_tensor.to(device)
            re, F = model(data_tensor)
            loss =  MSE(data_tensor.cpu().detach().numpy(),re.cpu().detach().numpy())
            # print(loss)
            # if loss<0.09:
            F = F.transpose(2, 1)
            F = Fun.avg_pool1d(F, kernel_size=4, stride=4)
            HIDDEN_DIM = 768
            Z = F.flatten().tolist()
            y_feature_vectors.append(Z)
            
            # data_id = str(data_id)                           
            y_ids.append(data_id)
            y_path.append(h5_path)
            ra_select.append(ra)
            dec_select.append(dec)
            loss_list.append(loss)


    print(len(y_ids), len(indx_list),len(y_feature_vectors))
    df = pd.DataFrame({'galaxy_id': y_ids,'h5':y_path, 'ra':ra_select, 'dec':dec_select, "loss":loss_list, 'features': y_feature_vectors})
    for j in tqdm.tqdm(range(HIDDEN_DIM)):
        df["Z"+str(j+1)] = df['features'].apply(lambda Z: Z[j])
    
    # df_standardized = df.iloc[:, 4:].values - np.mean(df.iloc[:, 4:].values, axis=0)
    # print(np.shape(y_feature_vectors), np.shape(df.iloc[:, 4:].values))
    # cov_matrix = np.cov(df.iloc[:, 4:].values, rowvar=False)

    # # 计算特征值和特征向量
    # eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # # 对特征值进行排序
    # sorted_indices = np.argsort(eigenvalues)[::-1]
    # sorted_eigenvalues = eigenvalues[sorted_indices]
    # top_100_sorted_eigenvalues = sorted_eigenvalues[:]

    # # 绘制Scree Plot
    # plt.plot(np.arange(1, len(top_100_sorted_eigenvalues) + 1), top_100_sorted_eigenvalues, marker='o')
    # plt.title('Scree Plot (Top 100)')
    # plt.xlabel('Principal Component Number')
    # plt.ylabel('Eigenvalue')
    # plt.show()
    # plt.savefig("./pca_plot.jpg")

    pca = PCA(10) 
    df = pd.DataFrame({'galaxy_id': y_ids,'h5':y_path, 'ra':ra_select, 'dec':dec_select,"loss":loss_list, 'features': y_feature_vectors})
    X_reduced = pca.fit_transform(y_feature_vectors)
    # new_df = pd.DataFrame(X_reduced, columns=['P1', 'P2'])
    new_df = pd.DataFrame(X_reduced, columns=['P1', 'P2','P3', 'P4','P5', 'P6','P7', 'P8','P9', 'P10',])
    df = pd.concat([df, new_df], axis=1)

    df = df.drop(columns=['features'])
    df.to_csv('./feature_BGS_big_%s.csv'%label, index=False)
    return df


def plt_image(components, method):
    nx, ny = 50, 50
    inds_use = np.arange(components.shape[0])
    z_emb = components
    print(np.shape(z_emb))
    iseed = 13579

    from scipy.stats import binned_statistic_2d
    from PIL import Image
    z_emb = z_emb.iloc[:, 4:].values
    xmin = z_emb[:, 0].min()
    xmax = z_emb[:, 0].max()
    ymin = z_emb[:, 1].min()
    ymax = z_emb[:, 1].max()

    binx = np.linspace(xmin, xmax, nx + 1)
    biny = np.linspace(ymin, ymax, ny + 1)

    ret = binned_statistic_2d(z_emb[:, 0], z_emb[:, 1], z_emb[:, 1], 'count', bins=[binx, biny], expand_binnumbers=True)
    z_emb_bins = ret.binnumber.T

    inds_used = []
    inds_lin = np.arange(z_emb.shape[0])
    plotq = []
    # First get all indexes that will be used
    for ix in range(nx):
        for iy in range(ny):
            dm = (z_emb_bins[:, 0] == ix) & (z_emb_bins[:, 1] == iy)
            inds = inds_lin[dm]

            np.random.seed(ix * nx + iy + iseed)
            if len(inds) > 0:
                ind_plt = np.random.choice(inds)
                inds_used.append(inds_use[ind_plt])
                plotq.append(ix + iy * nx)
    plt.figure(figsize=(200, 200))


    for index, i in enumerate(inds_used):
 
    # 获得id_value对应的索引
        galaxy_index = components["h5"].values[i]
        path_parts = galaxy_index.split('/')  # Split the image path using '_'
        h5_filename_index = path_parts[-1].split('_')

        h5_path = '/'.join(path_parts[:-1]) + '/' + h5_filename_index[0] + '_' + h5_filename_index[1]  # "/share2/fumingxiang/BGS_cut_v1/raw/3_big.h5"
        h5_file = h5py.File(h5_path, 'r')

        image_index = int(h5_filename_index[2])

        image_value = h5_file['image'][image_index]

        x0,x1,y0,y1 = DESI_find_Contour(np.transpose(image_value,(2,1,0)))
        img_jpg = image_value[:,x0:x1,y0:y1]
        resized_img = resize(img_jpg, (3, 128, 128))
        # image_value = (image_value - np.min(image_value)) / (np.max(image_value) - np.min(image_value))
        # image_value = image_value / 6
        img_jpg = dr2_rgb(resized_img, ['g', 'r', 'z'])[::-1]

        # img_jpg = np.clip(img_jpg * 255, 0, 255).astype(np.uint8)
        # img_jpg = (img_jpg - np.min(img_jpg)) / (np.max(img_jpg) - np.min(img_jpg))
        plt.subplot(nx, ny, plotq[index])
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img_jpg)

    plt.savefig('PLT_feature_{}_cut.png'.format(method))


label = "cut_nogaia"
num = 1
get_feature_umap(num, label)
# df_1 = pd.read_csv("./feature_BGS_cut.csv")
# plt_image(df_1)
















