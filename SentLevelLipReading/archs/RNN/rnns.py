import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# class GRU(nn.Module):
#     def __init__(self, input_size,
#                  hidden_size,
#                  num_layers=1,
#                  dropout=0.2,
#                  batch_first=True,
#                  bidirectional=False):
#         super(GRU, self).__init__()
#
#         dropout = 0.0 if num_layers == 1 else dropout
#         self.gru = nn.GRU(input_size=input_size,
#                           hidden_size=hidden_size,
#                           num_layers=num_layers,
#                           dropout=dropout,
#                           batch_first=batch_first,
#                           bidirectional=bidirectional)
#
#     def forward(self, x, seq_lens=None, hx=None):
#         '''
#         :param x: (bz, seq_len, embed_dim)
#         :param seq_lens: (bz, seq_len)
#         :param hx: 初始隐层
#         :return:
#         '''
#         if seq_lens is not None:
#             sort_lens, sort_idxs = torch.sort(seq_lens, dim=0, descending=True)
#             pack_embed = pack_padded_sequence(x[sort_idxs], lengths=sort_lens, batch_first=True)
#             pack_enc_out, hx = self.gru(pack_embed, hx)
#             enc_out, _ = pad_packed_sequence(pack_enc_out, batch_first=True)
#             _, unsort_idxs = torch.sort(sort_idxs, dim=0, descending=False)
#             enc_out = enc_out[unsort_idxs]
#             hx = hx[:, unsort_idxs]
#         else:
#             enc_out, hx = self.gru(x, hx)
#         return enc_out, hx


# class GRU(nn.Module):
#     def __init__(self,
#                  input_size,
#                  hidden_size,
#                  num_layers=2,
#                  bidirectional=False,
#                  dropout=0.):
#         super().__init__()
#         self.bidirectional = bidirectional
#         # bi_hidden_size = 2 * uni_hidden_size
#         if bidirectional:
#             assert hidden_size % 2 == 0
#             hidden_size = hidden_size // 2
#         else:
#             hidden_size = hidden_size
#
#         self.RNN = nn.GRU(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             bidirectional=bidirectional,
#             dropout=(0 if num_layers == 1 else dropout),
#             batch_first=True)
#
#     def forward(self, inputs, lengths):
#         """
#         :param self:
#         :param inputs:[batch_size, seq_len, input_size]
#         :param lengths:[batch_size]
#         :return:outputs: [batch_size, seq_len, hidden_size]
#                 final_state: [num_layers, batch_size, hidden_size]
#         """
#         inputs = pack_padded_sequence(inputs, lengths, enforce_sorted=False, batch_first=True)
#         outputs, hn = self.RNN(inputs)
#         outputs, _ = pad_packed_sequence(outputs, batch_first=True)
#         # final_state=[2*num_layers, batch_size, hidden_size]
#         if self.bidirectional:
#             final_state_forward = hn[0::2, :, :]
#             final_state_backward = hn[1::2, :, :]
#             hn = torch.cat([final_state_forward, final_state_backward], dim=2)
#         return outputs, hn


class GRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0.2,
                 batch_first=True,
                 bidirectional=False):
        super(GRU, self).__init__()

        dropout = 0.0 if num_layers == 1 else dropout
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          batch_first=batch_first,
                          bidirectional=bidirectional)
        self.init_param()

    def init_param(self):
        # nn.init.orthogonal_(self.gru.weight_ih_l0)
        # nn.init.orthogonal_(self.gru.weight_hh_l0)
        # nn.init.zeros_(self.gru.bias_ih_l0)
        # nn.init.zeros_(self.gru.bias_hh_l0)
        for param in self.gru.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

    def forward(self, x, hx=None, non_pad_mask=None):
        '''
        :param x: (bz, seq_len, embed_dim)
        :param non_pad_mask: (bz, seq_len)
        :return:
        '''
        self.gru.flatten_parameters()  # speed up
        if non_pad_mask is None:
            enc_out, hx = self.gru(x, hx)
        else:
            seq_lens = non_pad_mask.sum(dim=1).cpu()
            sort_lens, sort_idxs = torch.sort(seq_lens, dim=0, descending=True)
            pack_embed = pack_padded_sequence(x[sort_idxs], lengths=sort_lens, batch_first=True)
            pack_enc_out, hx = self.gru(pack_embed, hx)
            enc_out, _ = pad_packed_sequence(pack_enc_out, batch_first=True)
            _, unsort_idxs = torch.sort(sort_idxs, dim=0, descending=False)
            enc_out = enc_out[unsort_idxs]
            hx = hx[:, unsort_idxs]
        return enc_out, hx


class LSTM(nn.Module):
    def __init__(self, input_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0.2,
                 bidirectional=True,
                 batch_first=True):
        super(LSTM, self).__init__()

        dropout = 0.0 if num_layers == 1 else dropout
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=batch_first,
                            bidirectional=bidirectional)
        self.init_param()

    def init_param(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x, hx=None, non_pad_mask=None):
        '''
        :param x: (bz, seq_len, embed_dim)
        :param non_pad_mask: (bz, seq_len)
        :return:
        '''
        bs = x.shape[0]
        if non_pad_mask is None:
            enc_out, hx = self.lstm(x, hx)
        else:
            seq_lens = non_pad_mask.data.sum(dim=1)
            sort_lens, sort_idxs = torch.sort(seq_lens, dim=0, descending=True)  # 降序
            pack_embed = pack_padded_sequence(x[sort_idxs], lengths=sort_lens, batch_first=True)
            pack_enc_out, (hn, cn) = self.lstm(pack_embed, hx)
            enc_out, _ = pad_packed_sequence(pack_enc_out, batch_first=True)
            _, unsort_idxs = torch.sort(sort_idxs, dim=0, descending=False)
            enc_out = enc_out[unsort_idxs]
            hx = hn[:, unsort_idxs], cn[:, unsort_idxs]

        last_hn = hx[0].transpose(0, 1).reshape(bs, -1)  # for batch-first
        return enc_out, last_hn
