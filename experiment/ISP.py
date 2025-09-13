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
device = torch.device("cuda:1")


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


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 获取输入特征图的平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 将平均值和最大值拼接
        x = torch.cat([avg_out, max_out], dim=1)
        # 通过卷积层生成空间注意力图
        x = self.conv1(x)
        # 使用Sigmoid函数将注意力图的值限制在[0, 1]范围内
        return self.sigmoid(x)

class PositionAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(PositionAttention, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels // reduction_ratio, kernel_size=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels // reduction_ratio, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.conv1(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.conv2(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.cam = ChannelAttention(out_ch)
        #self.pam = PositionAttention(out_ch)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.cam(x)
        #x = self.pam(x)
        return x

class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

class Decoder(nn.Module):
    def __init__(self, chs=(256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = encoder_features[i]
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

class UNet(nn.Module):
    def __init__(self, enc_chs=(3, 64, 128, 256), dec_chs=(256, 128, 64), out_ch=4, out_sz=(512, 512)):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], out_ch, 1)
        self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        out = F.interpolate(out, self.out_sz)
        out = torch.clamp(out, min=0., max=1.)
        return out

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
    out_dir = './out'
    save_freq = 1



model = UNet(enc_chs=CFG.encoder, dec_chs=CFG.decoder, out_ch=CFG.out_ch, out_sz=CFG.out_sz)
# 加载权重文件
pretrained_weights_path = "./out/200.pt"
model = torch.load(pretrained_weights_path, map_location=device, weights_only=False)
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


