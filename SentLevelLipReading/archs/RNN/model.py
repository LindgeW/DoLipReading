import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.cuda.amp import autocast
import random
from typing import Union, List
from .rnns import GRU
from .attn import AttentiveAttention, MLPAttention


class TextEncoder(nn.Module):
    def __init__(self, num_embeddings, emb_dim, hidden_dim, drop_rate=0.):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, emb_dim)
        self.gru = GRU(emb_dim, hidden_dim, bidirectional=True)
        self.drop_rate = drop_rate

    def forward(self, src, src_mask=None):
        embedded = F.dropout(self.embedding(src), p=self.drop_rate, training=self.training)
        enc_outputs, _ = self.gru(embedded, non_pad_mask=src_mask)
        return enc_outputs


class VisualEncoder(nn.Module):
    def __init__(self, in_channel=1, init_dim=64, out_dim=256, num_layers=1, drop_rate=0.2):
        super(VisualEncoder, self).__init__()
        self.drop_rate = drop_rate
        self.num_layers = num_layers
        # 3D-CNN
        self.stcnn = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(in_channel, init_dim, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3),
                       bias=False)),
            ('norm', nn.BatchNorm3d(init_dim)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))]))

        # omitting ResNet layers

        self.gru = GRU(32768, out_dim, num_layers=num_layers, bidirectional=True, batch_first=True)

    def forward(self, x):   # (b, t, c, h, w)
        mask = torch.abs(torch.sum(x, dim=(-1, -2, -3))) > 0  # (b, t)
        x = x.transpose(1, 2).contiguous()  # (b, c, t, h, w)
        cnn = self.stcnn(x)  # (N, Cout, Dout, Hout, Wout)
        cnn = cnn.permute(0, 2, 1, 3, 4).contiguous()  # (N, Dout, Cout, Hout, Wout)
        batch, seq, channel, high, width = cnn.size()
        cnn = cnn.reshape(batch, seq, -1)  # (B, N, D)
        out, hn = self.gru(cnn, non_pad_mask=mask)  # hn: (n_layer * n_direct, B, D)
        hn = hn.reshape(self.num_layers, hn.size(1), -1)
        return out, hn, mask


class TextDecoder(nn.Module):
    def __init__(self, vocab_size, dec_embed_size, hidden_size, num_layers=1, drop_rate=0.2):
        super(TextDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dec_embed_size)
        self.attn_layer = MLPAttention(hidden_size)
        self.gru = GRU(dec_embed_size + hidden_size, hidden_size, num_layers=num_layers, dropout=drop_rate, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, cur_input, state, enc_states, enc_mask=None):
        """
         cur_input shape: (batch, )
         state shape: (num_layers*num_directs, batch, num_hiddens)
         """
        embedded = self.embedding(cur_input)
        ctx_vec = self.attn_layer(state[-1], enc_states, enc_mask)
        inp_and_ctx = torch.cat((embedded, ctx_vec), dim=1).contiguous()  # (B, D)
        output, state = self.gru(inp_and_ctx.unsqueeze(1), state)  # (B, 1, D)
        output = self.fc_out(output.squeeze(dim=1))  # (B, V)
        # output = self.fc_out(torch.cat((output.squeeze(dim=1), inp_and_ctx), dim=-1))  # (B, V)
        return output, state


# class TextDecoder(nn.Module):
#     def __init__(self, vocab_size, dec_embed_size, hidden_size, attn_size, num_layers=1, drop_rate=0.):
#         super(TextDecoder, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, dec_embed_size)
#         self.attn_model = self.attn_layer(2 * hidden_size, attn_size)
#         # 输⼊包含attention输出的c和实际输⼊
#         self.gru = GRU(dec_embed_size + hidden_size, hidden_size, num_layers, dropout=drop_rate, batch_first=True)
#         self.fc_out = nn.Linear(hidden_size, vocab_size)
#
#     def attn_layer(self, input_size, attn_size):
#         return nn.Sequential(nn.Linear(input_size, attn_size, bias=False),
#                              nn.Tanh(),
#                              nn.Linear(attn_size, 1, bias=False))
#
#     def attn_forward(self, enc_states, dec_state, src_mask=None):
#         """
#         model:函数attention_model返回的模型
#         enc_states: 编码端的输出，shape是(batch_size, seq_len, hidden_dim)
#         dec_state: 解码端一个时间步的输出，shape是(batch_size, hidden_dim)
#         """
#         # (batch_size, 1, hidden_dim) -> (batch_size, seq_len, hidden_dim)
#         enc_dec_states = torch.cat((enc_states, dec_state.unsqueeze(dim=1).expand_as(enc_states)), dim=2)  # (batch_size, seq_len, 2*hidden_dim)
#         e = self.attn_model(enc_dec_states).squeeze(-1)  # (batch_size, seq_len, 1) -> (batch_size, seq_len)
#         if src_mask is not None:
#             e = e.masked_fill(src_mask == 0, -1e9)
#         alpha = F.softmax(e, dim=1)
#         return (alpha.unsqueeze(-1) * enc_states).sum(dim=1)  # context vector  (batch_size, hidden_dim)
#
#     def forward(self, cur_input, state, enc_states, enc_mask=None):
#         """
#          cur_input shape: (batch, )
#          state shape: (num_layers, batch, num_hiddens)
#          """
#         embedded = self.embedding(cur_input)
#         ctx = self.attn_forward(enc_states, state[-1], enc_mask)
#         input_and_ctx = torch.cat((embedded, ctx), dim=1)  # (batch_size, 2*embed_size)
#         output, state = self.gru(input_and_ctx.unsqueeze(1), state)  # (batch_size, 1, 2*embed_size)
#         output = self.fc_out(output.squeeze(dim=1))  # (batch_size, vocab_size)
#         # output = self.fc_out(torch.cat((output.squeeze(dim=1), input_and_ctx), dim=-1))  # (batch_size, vocab_size)
#         return output, state


class Seq2Seq(nn.Module):
    def __init__(self, opt):
        super(Seq2Seq, self).__init__()
        self.encoder = VisualEncoder(in_channel=opt.in_channel,
                                     num_layers=opt.enc_layers,
                                     out_dim=opt.hidden_dim,
                                     drop_rate=opt.dropout)

        self.decoder = TextDecoder(vocab_size=opt.tgt_vocab_size,
                                   dec_embed_size=opt.hidden_dim,
                                   hidden_size=2*opt.hidden_dim,
                                   num_layers=opt.dec_layers,
                                   drop_rate=opt.dropout)
        self.opt = opt

    def forward(self, src, tgt=None):
        if tgt is None:
            tgt_len = self.opt.max_dec_len
        else:
            tgt_len = tgt.size(1)

        enc_outputs, enc_state, enc_mask = self.encoder(src)
        dec_state = enc_state
        dec_outputs = []
        dec_input = tgt[:, 0]
        for t in range(1, tgt_len):
            dec_output_t, dec_state = self.decoder(dec_input, dec_state, enc_outputs, enc_mask)
            dec_outputs.append(dec_output_t.unsqueeze(1))   # (B, V) -> (B, 1, V)
            if random.random() < self.opt.teacher_forcing_ratio:
                dec_input = tgt[:, t]
            else:
                dec_input = dec_output_t.argmax(dim=-1)

        dec_outs = torch.cat(dec_outputs, dim=1).contiguous()   # (B, N, V)
        loss = F.cross_entropy(dec_outs.transpose(-1, -2), tgt[:, 1:].long(), ignore_index=self.opt.tgt_pad_idx)
        return loss, dec_outs

    def greedy_decoding(self, src_inp, bos_id, eos_id, pad_id=0):
        tgt_len = self.opt.max_dec_len
        bs = src_inp.size(0)
        dec_preds = []
        dec_input = torch.tensor([bos_id] * bs).to(src_inp.device)
        with torch.no_grad():
            enc_outputs, enc_state, enc_mask = self.encoder(src_inp)
            dec_state = enc_state
            for t in range(tgt_len):
                dec_output_t, dec_state = self.decoder(dec_input, dec_state, enc_outputs, enc_mask)
                pred = dec_output_t.argmax(dim=-1)  # (B, )
                if pred.cpu().tolist() == [eos_id] * bs or pred.cpu().tolist() == [pad_id] * bs:
                    break
                dec_preds.append(pred.unsqueeze(1))
                dec_input = pred

        dec_pred = torch.cat(dec_preds, dim=1)  # (B, N)
        return dec_pred.detach().cpu().numpy()

