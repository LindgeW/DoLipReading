import os
import cv2
import numpy as np
import glob
import face_alignment
from skimage import io
import json

# 批量处理视频帧的面部关键点


def get_imgs_from_video(video, ext='jpg', RGB=False):
    frames = []
    if os.path.isdir(video):
        frames = sorted(glob.glob(os.path.join(video, '*.{}'.format(ext))),
                        key=lambda x: int(x.split(os.path.sep)[-1].split('.')[0]))
        frames = [cv2.imread(f) for f in frames]
    else:
        cap = cv2.VideoCapture(video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度  640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度  480
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率 30fps
        frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_num / fps   # sec
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  # 视频的编码  22
        # fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
        print(width, height, fps, frame_num, duration, fourcc)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

    frames = np.array(frames)
    if RGB:
        return frames[..., ::-1]
    else:
        return frames


def get_landmarks(path, save_dir='landmarks'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    frames = get_imgs_from_video(path)
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='sfd')
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='blazeface')
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device="cpu")
    for i, frame in enumerate(frames):
        preds = fa.get_landmarks_from_image(frame)   # list
        points_list = preds[0].tolist()   # 68
        assert len(points_list) == 68
        for one_point in points_list:
            x, y, _ = one_point
            cv2.circle(
                frame, (int(x), int(y)), 2, (0, 255, 255), thickness=-1, lineType=cv2.FILLED
            )
        lm_img = os.path.join(save_dir, f'{path.split(".")[0]}_lm{i + 1}.jpg')
        cv2.imwrite(lm_img, frame)

        lm_pts = os.path.join(save_dir, f'{path.split(".")[0]}_lm{i+1}' + '.txt')
        # with open(lm_pts, "w") as f:
        #     f.write(json.dumps(points_list))
        np.savetxt(lm_pts, points_list, fmt='%.10e')
    print('Done')


def get_batch_landmarks(path):
    import torch
    frames = get_imgs_from_video(path)
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='sfd')
    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='blazeface')
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device="cpu")
    batch = np.stack(frames)   # (B, H, W, C)
    print(batch.shape)
    batch = batch.transpose(0, 3, 1, 2)   # (B, C, H, W)
    batch = torch.Tensor(batch)
    preds = fa.get_landmarks_from_batch(batch)
    print(len(preds), len(preds[0]))
    # lm_pts = f'{path.split(".")[0]}_lm{i+1}' + '.json'
    # with open(lm_pts, "w") as f:
    #     f.write(json.dumps(points_list))
    print('Done')


get_landmarks('grid_test.mpg')
# get_batch_landmarks('grid_test.mpg')