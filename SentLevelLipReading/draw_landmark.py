import time

import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import cv2


def load_lms(lm_dir):
    lm_paths = glob.glob(os.path.join(lm_dir, 'grid_test_lm*.txt'))
    x, y, z = [], [], []
    for path in lm_paths:   # 75
        xyz = np.loadtxt(path, dtype=float)   # 68 x 3
        # 消除平移T的影响
        xyz -= np.mean(xyz, axis=0)
        # 消除缩放因子s的影响
        xyz /= np.std(xyz)

        # x.append(xyz[:, 0])
        # y.append(xyz[:, 1])
        # z.append(xyz[:, 2])
        # x.append(xyz[-20:, 0])
        # y.append(xyz[-20:, 1])
        # z.append(xyz[-20:, 2])
        # x.append(xyz[48:59, 0])
        # y.append(xyz[48:59, 1])
        # z.append(xyz[48:59, 2])
        # x.append(xyz[60:, 0])
        # y.append(xyz[60:, 1])
        # z.append(xyz[60:, 2])
        # x.append(abs(xyz[51, 0] - xyz[57, 0]))
        # y.append(abs(xyz[51, 1] - xyz[57, 1]))
        # z.append(abs(xyz[51, 2] - xyz[57, 2]))
        x.append((xyz[52, 0] - xyz[56, 0])**2 + (xyz[52, 1] - xyz[56, 1])**2)
        y.append(abs(xyz[52, 1] - xyz[56, 1]))
        z.append(abs(xyz[52, 2] - xyz[56, 2]))
    return np.asarray(x), np.asarray(y), np.asarray(z)


def draw_landmarks(lm_dir):
    x, y, z = load_lms(lm_dir)
    print(x.shape)
    # plt.plot(range(len(x)), x)
    plt.plot(range(1, 75), abs(x[1:] - x[:-1]))
    plt.xlabel('#Frame', fontsize=10)
    plt.ylabel('#Landmark', fontsize=10)
    plt.xlim(1, 75)
    plt.xticks(range(1, 76), [str(i) for i in range(1, 76)], fontsize=3.5)
    plt.show()


def landmark_center_crop(img_path, lm_path, length=32):
    img = cv2.imread(img_path)
    lm = np.loadtxt(lm_path)
    sub_crops = []
    for x, y, _ in lm[-20:]:
        left_x, left_y = int(x-length/2), int(y-length/2)
        lm_crop = img[left_y: left_y+length, left_x: left_x+length] / 255.
        sub_crops.append(lm_crop)

    for sc in sub_crops:
        cv2.imshow('Landmark Crop', sc)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_patches(img_path, patch_size=50):
    # 长宽整除问题：1、四舍五入，用0填充对齐；2、图像先缩放
    img = cv2.imread(img_path)   # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)
    h, w = img.shape[0], img.shape[1]
    # patch_num = (h * w) // (path_size**2)
    m = h // patch_size
    n = w // patch_size
    img = cv2.resize(img, (n*patch_size, m*patch_size), cv2.INTER_LINEAR)  # (w, h)
    print(img.shape)
    patches = []
    for i in range(m):
        for j in range(n):
            patch = img[i*patch_size: (i+1)*patch_size, j*patch_size: (j+1)*patch_size]
            patches.append(patch)
            print(patch.shape)
            plt.subplot(m, n, i * n + j + 1)
            plt.imshow(patch)    # RGB
            plt.axis('off')
            plt.title('p' + str(i * n + j + 1))
    plt.show()


def transform_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    # 消除平移T的影响
    points1 -= c1
    points2 -= c2
    # 消除缩放因子s的影响
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    # 旋转矩阵
    U, S, Vt = np.linalg.svd(points1.T @ points2)   # svd(M)
    R = U @ Vt.T
    # R = U @ Vt
    # 构建仿射变换矩阵
    s = s2 / s1
    sR = s * R
    c1 = c1.reshape(3, 1)
    c2 = c2.reshape(3, 1)
    T = c2 - sR @ c1
    trans_mat = np.hstack([sR, T])
    return trans_mat
    # return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)),
    #                   np.matrix([0., 0., 1.])])


def transform_landmarks():
    lms1 = np.loadtxt('landmarks/grid_test_lm1.txt')
    lms2 = np.loadtxt('landmarks/grid_test_lm2.txt')
    M = transform_from_points(lms1, lms2)
    print(M)

# draw_landmarks('landmarks')
# transform_landmarks()
# landmark_center_crop('landmarks/grid_test_lm1.jpg', 'landmarks/grid_test_lm1.txt')
img_patches('landmarks/grid_test_lm1.jpg')