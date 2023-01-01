import glob
import os

from torch.utils.data import Dataset
from turbojpeg import TurboJPEG, TJPF_GRAY

from .cvtransforms import *

# jpeg图像的解码和编码
jpeg = TurboJPEG('C:\libjpeg-turbo-gcc64\\bin\libturbojpeg.dll')


class LRW1000_Dataset(Dataset):
    def __init__(self, phase, args):
        self.phase = phase
        print('Now is in ' + phase)
        self.args = args
        self.data = []
        if self.phase == 'train':
            self.index_root = 'E:\speech-interaction-task1\\preliminary_training_set\\train_40_pkl_jpeg'
        elif self.phase in ['val', 'test']:
            self.index_root = 'E:\speech-interaction-task1\\preliminary_training_set\\val_40_pkl_jpeg'
        else:   # inference
            self.index_root = 'E:\speech-interaction-task1\\preliminary_test_set\\test_pkl_jpeg'

        self.data = glob.glob(os.path.join(self.index_root, '*.pkl'))
        with open('res_name.txt', 'w', encoding='utf-8') as fw:
            for x in self.data:
                fp, fn = os.path.split(x)
                fn, ext = os.path.splitext(fn)
                fw.write(fn + '\n')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pkl = torch.load(self.data[idx])
        video = pkl.get('video')
        # decodes JPEG memory buffer to numpy array
        video = [jpeg.decode(img, pixel_format=TJPF_GRAY) for img in video]
        if self.phase == 'inference':
            flipped_video = [cv2.flip(img, 1)[:, :, None] for img in video]  # 水平翻转
        video = np.stack(video, 0)   # (t, h, w, c),  c=1
        video = video[:, :, :, 0]   # (t, h, w)
        if self.phase == 'train':
            video = RandomCrop(video, (88, 88))
            video = HorizontalFlip(video)
        else:
            video = CenterCrop(video, (88, 88))   # 以中心截取88x88的图像

        if self.phase == 'inference':
            flipped_video = np.stack(flipped_video, 0)  # (t, h, w, c),  c=1
            flipped_video = flipped_video[:, :, :, 0]  # (t, h, w)
            flipped_video = CenterCrop(flipped_video, (88, 88))   # 以中心截取88x88的图像
            pkl['flipped_video'] = torch.FloatTensor(flipped_video)[:, None, ...] / 255.0   # (t, 1, h, w)

        pkl['video'] = torch.FloatTensor(video)[:, None, ...] / 255.0   # (t, 1, h, w)
        return pkl
