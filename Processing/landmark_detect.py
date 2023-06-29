
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



def run(files, fa):
    for img_name, save_name in files:
        if os.path.exists(save_name) and os.path.getsize(save_name) > 0:
            continue
        t1 = time.time()
        frame = cv2.imread(img_name)
        preds = fa.get_landmarks_from_image(frame)   # list
        if preds is None:
            print(f'Bad File: {img_name}')
            continue
        points_list = preds[0].tolist()   # 68
        if len(points_list) == 68:
            np.savetxt(save_name, points_list, fmt='%d')
            t2 = time.time()
            print(f'time cost: {t2 - t1}s')



face_root = r'./faces'

def get_landmarks():
    assert os.path.exists(face_root)
    #faces = glob.glob(os.path.join(face_root, '*', '*.jpg'))
    faces = []
    for root, dirs, files in os.walk(face_root):
        for f in files:
            if f.endswith('.jpg'):
                faces.append(os.path.join(root, f))
    
    #data = [(name, name.replace('.jpg', '.txt')) for name in faces]
    data = [(name, name.replace('.jpg', '.xy')) for name in faces]
    print(len(faces))

    records = []
    #gids = list(range(10))
    #gids.remove(2)
    gids = [0, 2, 3, 4, 5, 6]
    bs = len(data) // len(gids)
    # 每个gpu分配一个face-alignment
    #fas = [face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda:'+str(gpu_id)) for gpu_id in gids]  # 默认sfd
    fas = [face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:'+str(gpu_id)) for gpu_id in gids]  # 默认sfd
    for i, fa in enumerate(fas):  # 平分数据，分给每个gpu处理
        if i == len(gids) - 1:
            bs = len(data)
        th = Thread(target=run, args=(data[:bs], fa, ))
        # th = Process(target=video_process, args=(vid_dir,))
        data = data[bs:]
        th.start()  
        records.append(th)

    for th in records:
        th.join()

    print('Done!!!')


get_landmarks()
