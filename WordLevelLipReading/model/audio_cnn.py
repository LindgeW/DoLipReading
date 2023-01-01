import torch
from torch import nn
import torch.nn.functional as F


def mish(x):
    return x * torch.tanh(F.softplus(x))


class PreNet(nn.Module):
    """
    前置变换层，对mel特征变换
    """
    def __init__(self):
        super(PreNet, self).__init__()
        self.layer_0 = nn.Linear(80, 1024)
        self.layer_1 = nn.Linear(1024, 512)

    def forward(self, x):       # [bn, n_frames, n_mel_chs]
        x = self.layer_0(x)
        x = mish(x)
        x = F.dropout(x, p=0.3, training=self.training)
        return self.layer_1(x)  # [bn, n_frames, trans_layer_out_dim]


class ConvNorm(torch.nn.Module):
    """
    对conv1d的简单封装，主要是权重初始化
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation)
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain('linear'))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.prenet = PreNet()
        conv_layers = []
        ksize = 5
        for _ in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(in_channels=512,
                         out_channels=512,
                         kernel_size=ksize,
                         padding=int(ksize // 2)),
                nn.BatchNorm1d(512))
            conv_layers.append(conv_layer)
        self.convolution1ds = nn.ModuleList(conv_layers)
        self.gru = nn.GRU(input_size=512, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)

    def forward(self, x):
        self.gru.flatten_parameters()
        x = self.prenet(x)   # [bn, n_frames, chs]
        x = x.transpose(1, 2).contiguous()  # [bn, chs, n_frames]
        for conv in self.convolution1ds:
            # [bn, chs, n_frames]
            if self.training:
                x = F.dropout(mish(conv(x)), p=0.3, training=self.training)
            else:
                x = mish(conv(x))
        x = x.transpose(1, 2).contiguous()   # [bn, n_frames, chs]
        out, hn = self.gru(x)
        return out