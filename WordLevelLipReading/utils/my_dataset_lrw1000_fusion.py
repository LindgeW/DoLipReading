import glob
import os
from torch.utils.data import Dataset
from .cvtransforms import *
import librosa
import numpy as np


class LRW1000_Dataset(Dataset):
    def __init__(self, phase, args):
        assert phase in ['train', 'val', 'test', 'inference']
        self.phase = phase
        print('Now is in ' + phase)
        self.args = args
        if self.phase == 'train':
            self.data_root = 'E:\speech-interaction-task1\\preliminary_training_set'
        elif self.phase in ['val', 'test']:
            self.data_root = 'E:\speech-interaction-task1\\preliminary_training_set'
        else:   # inference
            self.data_root = 'E:\speech-interaction-task1\\final_competition'

        print(self.data_root)
        self.padding = 40
        self.mel_max_len = 100
        if self.phase == 'inference':
            self.data = []
            index_file = os.path.join(self.data_root, 'meta.txt')
            with open(index_file, 'r', encoding='utf-8') as fin:
                for x in fin:
                    if x.strip() != '':
                        self.data.append(x.strip())
        else:
            index_file = os.path.join(self.data_root, 'meta.txt')
            lines = []
            # 01b7a9520196fad67d4670fdf85cf59d,bf55fdb6d3adc33121f6c988934b35dd,扫描,sao miao,1.08,1.32
            with open(index_file, 'r', encoding='utf-8') as f:
                lines.extend([line.strip().split(',') for line in f])
            pinyins = sorted(np.unique([line[3] for line in lines]))
            all_data = [(line[0], line[1], int(float(line[4]) * 25) + 1, int(float(line[5]) * 25) + 1, pinyins.index(line[3]))
                for line in lines]
            # max_len = max([data[3] - data[2] for data in all_data])
            filter_data = list(filter(lambda data: 5 < data[3] - data[2] <= self.padding, all_data))
            print(len(filter_data))
            np.random.seed(1234)
            rand_ids = np.random.permutation(len(filter_data))
            if self.phase == 'train':
                self.data = [filter_data[i] for i in rand_ids[:-10000]]
            else:
                self.data = [filter_data[i] for i in rand_ids[-10000:]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        res = {}
        if self.phase == 'inference':
            video_path = os.path.join(self.data_root, 'img', self.data[idx])
            video = self.load_images(video_path)
            flipped_video = [cv2.flip(img, 1) for img in video]    # 水平翻转
            audio_path = os.path.join(self.data_root, 'wav', self.data[idx]+'.wav')
        else:
            (video_path, audio_path, op, ed, label) = self.data[idx]
            audio_path = os.path.join(self.data_root, 'wav', audio_path+'.wav')
            res['label'] = int(label)
            video = self.load_images(os.path.join(self.data_root, 'img', video_path), op, ed)

        audio = self.load_audio(audio_path)
        res['audio'] = torch.FloatTensor(audio)

        if self.phase == 'train':
            video = RandomCrop(video, (88, 88))
            video = HorizontalFlip(video)
        else:
            video = CenterCrop(video, (88, 88))   # 以中心截取88x88的图像

        if self.phase == 'inference':
            flipped_video = np.stack(flipped_video, 0)  # (t, h, w)
            flipped_video = CenterCrop(flipped_video, (88, 88))   # 以中心截取88x88的图像
            res['flipped_video'] = torch.FloatTensor(flipped_video)[:, None, ...] / 255.0   # (t, c, h, w)  c=1

        res['video'] = torch.FloatTensor(video)[:, None, ...] / 255.0   # (t, c, h, w)  c=1
        return res

    def load_images(self, path, op=None, ed=None):
        if op is None and ed is None:
            # img_paths = [int(i.split('.')[0]) for i in os.listdir(path) if i.split('.')[0].isdigit()]
            # files = [os.path.join(path, '{}.jpg'.format(i)) for i in sorted(img_paths)]
            files = [os.path.join(path, f) for f in sorted(os.listdir(path))]
        else:
            # length = (ed - op + 1)
            center = (op + ed) / 2
            op = int(center - self.padding // 2)
            ed = int(op + self.padding)
            files = [os.path.join(path, '{}.jpg'.format(i)) for i in range(op, ed)]

        files = filter(lambda path: os.path.exists(path), files)
        files = [cv2.imread(file, 0) for file in files]       # (h, w)  grayscale
        files = [cv2.resize(file, (96, 96)) for file in files]
        files = np.stack(files, 0)
        t = files.shape[0]
        video = np.zeros((self.padding, 96, 96)).astype(files.dtype)  # (t, h, w)
        video[:t, ...] = files.copy()
        return video

    def load_audio(self, path):
        sr = 16000
        n_fft = 512  # fft窗口大小(默认等于win_len)，n_fft=hop_len+overlapping，一般为2^n
        frame_len = 0.025    # s
        frame_shift = 0.01  # s
        pre_emphasis = 0.97
        n_mels = 80   # mel通道数 (特征维度)
        win_len = int(sr * frame_len)    # 400 (25ms)
        hop_len = int(sr * frame_shift)  # 160 (10ms)
        if librosa.get_duration(filename=path) == 0:
            print('empty audio!')
            return np.zeros((self.mel_max_len, n_mels), dtype=np.float32)

        y, sr = librosa.load(path, sr=sr)
        # pre-emphasis
        y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
        # stft
        linear = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_len, win_length=win_len)
        # magnitude spectrogram
        mag = np.abs(linear)  # (1+n_fft//2, T)
        # mel spectrogram
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)  # (n_mels, 1+n_fft//2)
        mel = np.dot(mel_basis, mag)  # (n_mels, T)
        # to decibel
        mel = 20 * np.log10(np.maximum(1e-5, mel))
        mel = mel.T.astype(np.float32)  # (T, n_mels)
        audio = np.zeros((self.mel_max_len, n_mels)).astype(mel.dtype)  # (max_T, n_mels)
        L = min(self.mel_max_len, mel.shape[0])
        audio[:L] = mel.copy()[:L]
        return audio