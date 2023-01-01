import torch
import torch.nn as nn
from torch.cuda.amp import autocast  # 混合精度
from .video_cnn import VideoCNN
from .audio_cnn import AudioCNN


class AVSRModel(nn.Module):
    def __init__(self, args, dropout=0.5):
        super(AVSRModel, self).__init__()
        self.args = args
        self.drop = dropout
        self.video_cnn = VideoCNN(se=self.args.se)
        self.audio_cnn = AudioCNN()

        in_dim = 512
        self.gru = nn.GRU(in_dim, 512, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        # self.v_cls = nn.Linear(512 * 2, self.args.n_class)
        # self.a_cls = nn.Linear(512 * 2, self.args.n_class)
        self.cls = nn.Linear(2 * 512 * 2, self.args.n_class)

    def forward(self, v, a):
        self.gru.flatten_parameters()
        if self.training:
            with autocast():
                f_v = self.video_cnn(v)
                f_v = nn.functional.dropout(f_v, p=self.drop, training=self.training)
            f_v = f_v.float()
        else:
            f_v = self.video_cnn(v)
        # h_v, _ = self.gru(f_v)
        # h_v = self.v_cls(h_v).mean(1)   # avg-pooling over time-step
        #
        # h_a = self.audio_cnn(a)
        # h_a = self.a_cls(h_a).mean(1)
        # y_v = (h_v + h_a) / 2.

        h_v, _ = self.gru(f_v)
        h_a = self.audio_cnn(a)
        h = torch.cat((h_v.mean(1), h_a.mean(1)), dim=-1).contiguous()
        y_v = self.cls(h)
        return y_v