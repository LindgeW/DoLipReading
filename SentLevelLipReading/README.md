### Sentence-level LipReading

Architecture: *3DCNN + Transformer + CTC/Attention + BeamSearch*

Dataset: [GRID](https://spandh.dcs.shef.ac.uk/gridcorpus/)

#### Training 

```
python train.py --cuda 0 --phase train --batch_size 32 --num_workers 2 [--weights checkpoints/grid/best.ep10.pt] --enc_layers 6 --dec_layers 6 --tgt_vocab_size 30 --grad_clip 1 --enc_lr 5e-4 --dec_lr 5e-4 --align_root /content/grid/align_txt --video_root /content/grid/lip
```


#### Testing

```
python train.py --cuda 0 --phase test --batch_size 32 --load checkpoints/grid/best.ep10.pt --enc_layers 6 --dec_layers 6 --tgt_vocab_size 30 --align_root /content/grid/align_txt --video_root /content/grid/lip
```


#### Landmark Detection
```
import cv2
import face_alignment
import numpy as np
import os
import math

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')

def get_position(desired_size, padding=0.25):
    # Average positions of face points 17-67  (reference shape)
    x = [0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
         0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
         0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
         0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
         0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
         0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
         0.553364, 0.490127, 0.42689]  # mean_face_shape_x

    y = [0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
         0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
         0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
         0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
         0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
         0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
         0.784792, 0.824182, 0.831803, 0.824182]  # mean_face_shape_y

    x, y = np.array(x), np.array(y)
    x = (x + padding) / (2 * padding + 1) * desired_size
    y = (y + padding) / (2 * padding + 1) * desired_size
    return np.matrix(list(zip(x, y)))

'''
def transformation_from_points(points1, points2):
    # 注：points1和points2是np.matrix类型，*相当于矩阵乘，即np.array的dot()和@
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    # 计算R的核心步骤：M=BA^T，SVD(M)=UΣV^T，R=UV^T
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    trans_mat = np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])   # 3 x 3
    return trans_mat
'''

def transformation_from_points(points1, points2):
    # points1：需要对齐的人脸关键点；points2：对齐的模板人脸(平均脸关键点)
    '''0 - 先确定是float数据类型 '''
    points1 = np.copy(points1).astype(np.float64)
    points2 = np.copy(points2).astype(np.float64)
    '''1 - 消除平移的影响 '''
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    '''2 - 消除缩放的影响 '''
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    '''3 - 计算矩阵M=BA^T；对矩阵M进行SVD分解；计算得到R '''
    # ||RA-B||; M=BA^T
    A = points1.T  # 2xN
    B = points2.T  # 2xN
    M = np.dot(B, A.T)
    U, S, Vt = np.linalg.svd(M)
    R = np.dot(U, Vt)
    '''4 - 构建仿射变换矩阵 '''
    s = s2 / s1
    sR = s * R
    c1 = c1.reshape(2, 1)
    c2 = c2.reshape(2, 1)
    T = c2 - np.dot(sR, c1)   # 模板人脸的中心位置减去需要对齐的中心位置（经过旋转和缩放之后）
    trans_mat = np.hstack([sR, T])  # 2x3
    return trans_mat

def get_landmarks(img):
    preds = fa.get_landmarks_from_image(img)  # list
    if preds is None:
        return None
    lms = preds[0].tolist()  # 68
    return np.matrix(lms)

def align_img(img_dir, save_dir, desired_size=256):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    mean_shape = get_position(desired_size)  # 模板脸(平均脸)
    files = list(os.listdir(img_dir))
    files = [file for file in files if (file.find('.jpg') != -1)]
    shapes = []
    imgs = []

    for file in files:
        I = cv2.imread(os.path.join(img_dir, file))
        shape = get_landmarks(I)
        if shape is None:
            print(file)
            continue
        imgs.append(I)
        shapes.append(shape[17:])

    for i, img in enumerate(imgs):
        M = transformation_from_points(shapes[i], mean_shape)
        img = cv2.warpAffine(img,
                             M[:2],   # [R|t]  2 x 3 仿射变换矩阵
                             (desired_size, desired_size),  # (cols, rows)
                             borderMode=cv2.BORDER_TRANSPARENT)
        cv2.imwrite(os.path.join(save_dir, files[i]), img)
    print('Done!!')

align_img('src_faces', 'tgt_faces')
```


#### Resources
+ https://github.com/arxrean/LipRead-seq2seq
+ https://github.com/bentrevett/pytorch-seq2seq
+ https://github.com/budzianowski/PyTorch-Beam-Search-Decoding
+ https://github.com/liuxubo717/V-ACT/blob/main/tools/beam.py
+ https://github.com/haantran96/wavetransformer/blob/main/modules/beam.py
+ https://github.com/prajwalkr/vtp/blob/master/search.py
