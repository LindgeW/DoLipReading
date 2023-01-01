import torch
import torch.nn as nn
from torch.cuda.amp import autocast  # 混合精度
from .video_cnn import VideoCNN


class VideoModel(nn.Module):
    def __init__(self, args, dropout=0.5):
        super(VideoModel, self).__init__()
        self.args = args
        self.video_cnn = VideoCNN(se=self.args.se)
        if self.args.border:
            in_dim = 512 + 1  # 在前端特征上拼接词边界信息
        else:
            in_dim = 512
        self.gru = nn.GRU(in_dim, 1024, num_layers=3, batch_first=True, bidirectional=True, dropout=0.2)
        self.v_cls = nn.Linear(1024 * 2, self.args.n_class)
        self.drop = dropout

    def forward(self, v, border=None):
        self.gru.flatten_parameters()
        if self.training:
            with autocast():
                f_v = self.video_cnn(v)
                f_v = nn.functional.dropout(f_v, p=self.drop, training=self.training)
            f_v = f_v.float()
        else:
            f_v = self.video_cnn(v)
            f_v = nn.functional.dropout(f_v, p=self.drop, training=self.training)

        if self.args.border:
            border = border[:, :, None]
            h, _ = self.gru(torch.cat([f_v, border], -1))
        else:
            h, _ = self.gru(f_v)

        h = nn.functional.dropout(h, p=self.drop, training=self.training)
        y_v = self.v_cls(h).mean(1)   # avg-pooling over time-step
        # y_v = self.v_cls(h)
        return y_v