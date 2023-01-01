import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from turbojpeg import TurboJPEG

jpeg = TurboJPEG('C:\libjpeg-turbo-gcc64\\bin\libturbojpeg.dll')


class LRW1000_Dataset(Dataset):
    def __init__(self, index_file, target_dir):
        self.data = []
        self.target_dir = target_dir
        # test phase
        # self.data_root = 'E:\speech-interaction-task1\\test_set\img'
        # self.data = os.listdir(self.data_root)

        # training phase
        self.data_root = 'E:\speech-interaction-task1\\training_set\img'
        self.padding = 40
        lines = []
        # 01b7a9520196fad67d4670fdf85cf59d,bf55fdb6d3adc33121f6c988934b35dd,扫描,sao miao,1.08,1.32
        with open(index_file, 'r', encoding='utf-8') as f:
            lines.extend([line.strip().split(',') for line in f])
        pinyins = sorted(np.unique([line[3] for line in lines]))
        self.data = [(line[0], int(float(line[4]) * 25) + 1, int(float(line[5]) * 25) + 1, pinyins.index(line[3])) for
                     line in lines]
        # max_len = max([data[2] - data[1] for data in self.data])
        self.data = list(filter(lambda data: data[2] - data[1] <= self.padding, self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r = self.load_video(self.data[idx])
        return r

    def load_video(self, item):
        # load video into a tensor
        (path, op, ed, label) = item
        inputs, border, length = self.load_images(os.path.join(self.data_root, path), op, ed)
        result = {'video': inputs, 'label': int(label), 'duration': border.astype(bool), 'length': length}
        savename = os.path.join(self.target_dir, f'{path}_{op}_{ed}.pkl')
        torch.save(result, savename)
        return True

    def load_images(self, path, op, ed):
        center = (op + ed) / 2
        length = (ed - op + 1)
        op = int(center - self.padding // 2)
        ed = int(op + self.padding)
        left_border = max(int(center - length / 2 - op), 0)
        right_border = min(int(center + length / 2 - op), self.padding)
        files = [os.path.join(path, '{}.jpg'.format(i)) for i in range(op, ed)]
        files = filter(lambda path: os.path.exists(path), files)
        files = [cv2.imread(file) for file in files]
        files = [cv2.resize(file, (96, 96)) for file in files]
        files = np.stack(files, 0)
        t = files.shape[0]
        tensor = np.zeros((self.padding, 96, 96, 3)).astype(files.dtype)
        border = np.zeros(self.padding)
        tensor[:t, ...] = files.copy()
        border[left_border: right_border] = 1.0
        tensor = [jpeg.encode(tensor[_]) for _ in range(self.padding)]
        return tensor, border, t


if __name__ == '__main__':
    # target_dir = f'E:\speech-interaction-task1\\test_set\\test_pkl_jpeg'
    index_file = f'E:\speech-interaction-task1\\training_set\\meta.txt'
    target_dir = f'E:\speech-interaction-task1\\training_set\\train_40_pkl_jpeg'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    loader = DataLoader(LRW1000_Dataset(index_file, target_dir),
                        batch_size=96,
                        num_workers=0,
                        shuffle=False,
                        drop_last=False)

    import time
    tic = time.time()
    for i, batch in enumerate(loader):
        toc = time.time()
        eta = ((toc - tic) / (i + 1) * (len(loader) - i)) / 3600.0
        print(f'eta:{eta:.5f}')
    print('Done')
