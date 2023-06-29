
import os
import cv2
import numpy as np
import glob
import numba
import face_alignment
from threading import Thread
#from multiprocessing import Process
import time


#os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='sfd')
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='blazeface')
#fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda')   # 默认sfd


def get_imgs_from_video(video, ext='jpg', RGB=False):
    frames = []
    if os.path.isdir(video):
        frames = sorted(glob.glob(os.path.join(video, '*.{}'.format(ext))),
                        key=lambda x: int(x.split(os.path.sep)[-1].split('.')[0]))
        frames = [cv2.imread(f) for f in frames]
    else:
        cap = cv2.VideoCapture(video)
        # cap.set(cv2.CAP_PROP_FPS, 25)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度  640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度  480
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率 30fps
        frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_num / fps   # sec
        print(width, height, fps, frame_num, duration)
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



def get_lms_from_video(file, lm_dir, face_dir, fa):
    if not os.path.exists(lm_dir):
        os.makedirs(lm_dir)
        os.makedirs(face_dir)
    #elif len(os.listdir(lm_dir)) < 75:
    #    pass
    #elif len(os.listdir(face_dir)) < 75:
    #    pass
    else:
        return

    #fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda:'+str(gpu_id))   # 默认sfd
    frames = get_imgs_from_video(file)
    for i, frame in enumerate(frames):
        preds = fa.get_landmarks_from_image(frame)   # list
        if preds is None:
            print(f'Bad File: {file}')
            continue
        points_list = preds[0].tolist()   # 68
        if len(points_list) == 68:
            np.savetxt(os.path.join(lm_dir, f'{i+1}' + '.txt'), points_list)
            cv2.imwrite(os.path.join(face_dir, f'{i+1}' + '.jpg'), frame)



vid_root = r'./video-high'
lm_root = r'./landmark'
face_root = r'./face'


def get_landmarks():
    def video_process(vdir, fa):
        vids = os.listdir(os.path.join(vid_root, vdir))
        for vid_name in vids:  # bbaf2n, bbaf3s, ...
            t1 = time.time()
            lm_path = os.path.join(lm_root, vdir, os.path.splitext(vid_name)[0])
            face_path = os.path.join(face_root, vdir, os.path.splitext(vid_name)[0])
            get_lms_from_video(os.path.join(vid_root, vdir, vid_name), lm_path, face_path, fa)
            t2 = time.time()
            print(f'{vdir}-{vid_name} takes {t2 - t1} secs.')

    if not os.path.exists(lm_root):
        os.mkdir(lm_root)
    if not os.path.exists(face_root):
        os.mkdir(face_root)

    records = []
    #gids = list(range(10))
    #gids.remove(2)
    gids = [1]
    #vid_dirs = os.listdir(vid_root)
    vid_dirs = ['s7']
    # 每个gpu分配一个face-alignment
    fas = [face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda:'+str(gpu_id)) for gpu_id in gids]  # 默认sfd
    for i, vid_dir in enumerate(vid_dirs):  # s1, s2, s3 ...
        th = Thread(target=video_process, args=(vid_dir, fas[i%len(gids)], ))
        # th = Process(target=video_process, args=(vid_dir,))
        th.start()  # 启动线程运行
        records.append(th)

    for th in records:
        th.join()

    print('Done!!!')


get_landmarks()
