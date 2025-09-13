import cv2
import torch
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
 

PATH = f'.'
def get_filenames(path):
    test = glob(path + '*.png')
    return test


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


########## VISUALIZATION

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
    fig = plt.figure(figsize=(16, 8), dpi=100)
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
