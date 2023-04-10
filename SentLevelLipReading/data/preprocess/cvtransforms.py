# coding: utf-8
import random
import cv2
import numpy as np


def center_crop(img, outsize):
    '''
    首先截取图片中心180x180区域，然后缩放至img_size x img_size
    :param img:
    :param outsize:
    :return:
    '''
    hidden_size = 180
    h, w = img.shape
    if h < hidden_size or w < hidden_size:
        print('Image clip warnning! image size={}'.format(img.shape))
    else:
        h = (h - hidden_size) // 2
        w = (w - hidden_size) // 2
        img = img[h: h + hidden_size, w: w + hidden_size]
    img_clip = cv2.resize(img, (outsize, outsize), interpolation=cv2.INTER_CUBIC)
    # img_clip -= np.mean(img_clip)
    # img_clip /= np.std(img_clip)
    return img_clip


def cut_out(img, n_holes=1, length=8):
    '''
    Randomly mask out one or more patches from an image.
        img (Tensor): Tensor image of size (C, H, W).
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    Returns:
        Tensor: Image with n_holes of dimension length x length cut out of it.
    '''
    h = img.shape[1]
    w = img.shape[2]
    mask = np.ones((h, w), np.float32)
    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
    img = img * mask
    return img



def HorizontalFlip(batch_img, p=0.5):
    # (T, H, W, C)
    if random.random() > p:
        batch_img = batch_img[:, :, ::-1, ...]
    return batch_img


def ColorNormalize(batch_img):
    batch_img = batch_img / 255.0
    return batch_img


def AddGaussianNoise(img, p=0.1):
    G_Noiseimg = img.copy()
    w = img.shape[1]
    h = img.shape[0]
    G_NoiseNum = int(p * img.shape[0] * img.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        if len(img.shape) == 2 or img.shape[-1] == 1:
            G_Noiseimg[temp_x][temp_y] = np.random.randn(1)[0]
        else:
            channel = np.random.randint(len(img.shape))
            G_Noiseimg[temp_x][temp_y][channel] = np.random.randn(1)[0]
    return G_Noiseimg


def BilateralFilter(img, p=0.2):
    if random.random() > p:
        return img
    # 引入双边滤波去噪
    img = cv2.bilateralFilter(src=img, d=0, sigmaColor=random.randint(15, 30), sigmaSpace=15)
    # 归一化，转换数据类型 并限定上下界限的大小必须为fixed_side
    # img = img.astype(np.float32)
    # # 标准化处理
    # img -= np.mean(img)  # 减去均值
    # img /= np.std(img)  # 除以标准差
    return img