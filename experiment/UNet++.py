#pip -q install gdown
#pip -q install rawpy
#pip install numpy pandas matplotlib opencv-python imageio rawpy scikit-image tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

# 确保 python2 和 python3 兼容
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import imageio
import rawpy
import sys
import os
import gc
import time
import random
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from IPython import display
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as SSIM

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import models
from torch.optim import Adam

torch.backends.cudnn.deterministic = True

#-------------------------------------------------------------------------------------------------!!!
device = torch.device("cuda:2")


# Check https://github.com/mv-lab/AISP/utils.py for more utils for RAW image manipulation.

def load_img(filename, debug=False, norm=True, resize=None):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if norm:
        img = img / 255.
        img = img.astype(np.float32)
    if debug:
        print(img.shape, img.dtype, img.min(), img.max())

    if resize:
        img = cv2.resize(img, (resize[0], resize[1]), interpolation=cv2.INTER_AREA)

    return img


def save_rgb(img, filename):
    if np.max(img) <= 1:
        img = img * 255

    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(filename, img)


def load_raw(raw, max_val=2 ** 10):
    raw = np.load(raw) / max_val
    return raw.astype(np.float32)


def demosaic(raw):
    """Simple demosaicing to visualize RAW images
    Inputs:
     - raw: (h,w,4) RAW RGGB image normalized [0..1] as float32
    Returns:
     - Simple Avg. Green Demosaiced RAW image with shape (h*2, w*2, 3)
    """

    assert raw.shape[-1] == 4
    shape = raw.shape

    red = raw[:, :, 0]
    green_red = raw[:, :, 1]
    green_blue = raw[:, :, 2]
    blue = raw[:, :, 3]
    avg_green = (green_red + green_blue) / 2
    image = np.stack((red, avg_green, blue), axis=-1)
    image = cv2.resize(image, (shape[1] * 2, shape[0] * 2))
    return image


def mosaic(rgb):
    """Extracts RGGB Bayer planes from an RGB image."""

    assert rgb.shape[-1] == 3
    shape = rgb.shape

    red = rgb[0::2, 0::2, 0]
    green_red = rgb[0::2, 1::2, 1]
    green_blue = rgb[1::2, 0::2, 1]
    blue = rgb[1::2, 1::2, 2]

    image = np.stack((red, green_red, green_blue, blue), axis=-1)
    return image


def gamma_compression(image):
    """Converts from linear to gamma space."""
    return np.maximum(image, 1e-8) ** (1.0 / 2.2)


def tonemap(image):
    """Simple S-curved global tonemap"""
    return (3 * (image ** 2)) - (2 * (image ** 3))


def postprocess_raw(raw):
    """Simple post-processing to visualize demosaic RAW imgaes
    Input:  (h,w,3) RAW image normalized
    Output: (h,w,3) post-processed RAW image
    """
    raw = gamma_compression(raw)
    raw = tonemap(raw)
    raw = np.clip(raw, 0, 1)
    return raw


def plot_pair(rgb, raw, t1='RGB', t2='RAW', axis='off'):
    fig = plt.figure(figsize=(12, 6), dpi=80)
    plt.subplot(1, 2, 1)
    plt.title(t1)
    plt.axis(axis)
    plt.imshow(rgb)

    plt.subplot(1, 2, 2)
    plt.title(t2)
    plt.axis(axis)
    plt.imshow(raw)
    plt.show()


########## METRICS

def PSNR(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    if (mse == 0):
        return np.inf

    max_pixel = np.max(y_true)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def get_filenames(PATH):
    train_raws = sorted(glob(PATH + '/train/*.npy'))
    train_rgbs = sorted(glob(PATH + '/train/*.png'))
    valid_rgbs = sorted(glob(PATH + '/val/*'))
    assert len(train_raws) == len(train_rgbs)
    print (f'Training samples: {len(train_raws)} \t Validation samples: {len(valid_rgbs)}')
    return train_raws, train_rgbs, valid_rgbs

PATH    = f'data'


#-------------------------------------------------------------------------------------------------!!!
BATCH_TRAIN = 4
BATCH_TEST  = 1

# if DEBUG use only 250 datasamples, if not all the dataset
DEBUG = False

train_raws, train_rgbs, valid_rgbs = get_filenames(PATH)


class LoadData(Dataset):

    def __init__(self, root, rgb_files, raw_files=None, debug=False, test=None):

        self.root = root
        self.test = test
        self.rgbs = sorted(rgb_files)
        if self.test:
            self.raws = None
        else:
            self.raws = sorted(raw_files)

        self.debug = debug
        if self.debug:
            self.rgbs = self.rgbs[:100]
            self.raws = self.raws[:100]

    def __len__(self):
        return len(self.rgbs)

    def __getitem__(self, idx):

        rgb = load_img(self.rgbs[idx], norm=True)
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1)))

        if self.test:
            return rgb, self.rgbs[idx]
        else:
            raw = load_raw(self.raws[idx])
            raw = torch.from_numpy(raw.transpose((2, 0, 1)))
            return rgb, raw

train_dataset = LoadData(root=PATH, rgb_files=train_rgbs,raw_files=train_raws, debug=DEBUG, test=False)
train_loader  = DataLoader(dataset=train_dataset, batch_size=BATCH_TRAIN, shuffle=True, num_workers=0,
                          pin_memory=True, drop_last=True)

test_dataset = LoadData(root=PATH, rgb_files=valid_rgbs, test=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=BATCH_TEST, shuffle=False, num_workers=0,
                         pin_memory=True, drop_last=False)

print (f'Train Dataloader BS={BATCH_TRAIN} N={len(train_loader)} / Test/Val Dataloader BS={BATCH_TEST} N={len(test_loader)}')





#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, enc_chs, dec_chs, out_ch, out_sz):
        super(UNet, self).__init__()
        self.enc_chs = enc_chs
        self.dec_chs = dec_chs
        self.out_ch = out_ch
        self.out_sz = out_sz

        # Encoder
        self.encoder = nn.ModuleList([
            DoubleConv(enc_chs[i], enc_chs[i+1]) for i in range(len(enc_chs) - 1)
        ])

        # Decoder
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(dec_chs[i], dec_chs[i+1], kernel_size=2, stride=2) for i in range(len(dec_chs) - 1)
        ])
        self.decoder = nn.ModuleList([
            DoubleConv(dec_chs[i] + enc_chs[-(i+1)], dec_chs[i+1]) for i in range(len(dec_chs) - 1)
        ])

        # Final layer
        self.final_conv = nn.Conv2d(dec_chs[-1], out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
            x = F.max_pool2d(x, 2)

        # Decoder
        for i, (upconv, dec) in enumerate(zip(self.upconvs, self.decoder)):
            x = upconv(x)
            x = torch.cat((x, enc_outs[-(i+1)]), dim=1)
            x = dec(x)

        # Final layer
        x = self.final_conv(x)
        return x

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------


class CFG:
    encoder = (3, 64, 128, 256)
    decoder = (256, 128, 64)
    out_ch = 4
    out_sz = (512, 512)
    lr = 5e-5
    lr_decay = 2e-6
    epochs = 100
    loss = nn.MSELoss()
    name = 'unet-rev-isp.pt'
    out_dir = './out2'
    save_freq = 1



model = UNet(enc_chs=CFG.encoder, dec_chs=CFG.decoder, out_ch=CFG.out_ch, out_sz=CFG.out_sz)
# 加载权重文件
# pretrained_weights_path = "./137.pt"
# model = torch.load(pretrained_weights_path, map_location=device, weights_only=False)
model = model.to(device)

opt   = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.lr_decay)

criterion = CFG.loss
metrics   = defaultdict(list)

min_loss = 1
min_loss_epoch = -1
for epoch in range(CFG.epochs):

    torch.cuda.empty_cache()
    start_time = time.time()
    train_loss = []

    model.train()

    for rgb_batch, raw_batch in tqdm(train_loader):
        opt.zero_grad()

        rgb_batch = rgb_batch.to(device)
        raw_batch = raw_batch.to(device)

        recon_raw = model(rgb_batch)

        loss = criterion(raw_batch, recon_raw)
        loss.backward()
        opt.step()

        train_loss.append(loss.item())

    now_loss = np.mean(train_loss)
    metrics['train_loss'].append(now_loss)
    if now_loss < min_loss:
        min_loss = now_loss
        min_loss_epoch = len(metrics['train_loss'])

    # --------------metrics['train_loss'] 里有 loss 数据--------

    print(f"Epoch {epoch + 1} of {CFG.epochs} took {time.time() - start_time:.3f}s 当前loss：{now_loss}\n")

    if ((epoch + 1) % CFG.save_freq == 0):
        torch.save(model, os.path.join(CFG.out_dir, f"{len(metrics['train_loss'])}.pt"))

torch.save(model.state_dict(), os.path.join(CFG.out_dir, CFG.name))
print("最小的 loss 权重为：", min_loss_epoch, ".pt")


flag = input("是否查看loss曲线？(y/n)")

if flag == 'y' or flag == 'Y':
    # 查看 loss
    min_loss = 1
    min_loss_epoch = -1
    for i, v in enumerate(metrics['train_loss']):
        print(f"第{i + 1}代:  ", v)
        if v < min_loss:
            min_loss = v
            min_loss_epoch = i

    print(f"最小值为：{min_loss} ，它属于第 {min_loss_epoch + 1} 代")
else:
    print("结束")


