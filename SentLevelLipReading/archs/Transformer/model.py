import torch
import torch.nn as nn
import math
from collections import OrderedDict
from batch_beam_search import beam_decode


class PositionEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Embedding(num_embeddings, embedding_dim)
        torch.nn.init.xavier_normal_(self.weight.weight)

    def forward(self, x):
        embeddings = self.weight(x)
        return embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)   # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)   # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # PE(pos, 2i)
        pe[:, 1::2] = torch.cos(position * div_term)  # PE(pos, 2i+1)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, L):
        return self.pe[:, :L].detach()


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, bias=True, dropout=0.1):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.fc_q = nn.Linear(hid_dim, hid_dim, bias)
        self.fc_k = nn.Linear(hid_dim, hid_dim, bias)
        self.fc_v = nn.Linear(hid_dim, hid_dim, bias)
        self.fc_o = nn.Linear(hid_dim, hid_dim, bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.shape[0]
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        
        Q, K, V = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        # energy = torch.einsum('bnqh,bnkh->bnqk', Q, K) * scale
        energy = torch.matmul(Q, K.transpose(-1, -2).contiguous()) * self.scale
        # energy = [batch size, n heads, query len, key len]

        if mask is not None:   # [batch size, 1, 1, key len]
            energy = energy.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(energy, dim=-1)
        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)
        # x = [batch size, n heads, query len, head dim]

        x = x.transpose(1, 2).contiguous().reshape(bs, -1, self.hid_dim)
        # x = [batch size, query len, n heads, head dim] -> [batch size, query len, hid dim]

        x = self.fc_o(x)
        # x = [batch size, query len, hid dim]
        return x, attention


class FeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, ffn_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, ffn_dim)
        self.fc_2 = nn.Linear(ffn_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))
        # x = [batch size, seq len, ffn dim]

        x = self.fc_2(x)
        # x = [batch size, seq len, hid dim]
        return x


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 ffn_dim,
                 dropout):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.ff_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.feedforward = FeedforwardLayer(hid_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]
        _src, _ = self.self_attention(src, src, src, src_mask)

        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]

        _src = self.feedforward(src)

        src = self.ff_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]
        return src


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 ffn_dim,
                 dropout):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.ff_layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.feedforward = FeedforwardLayer(hid_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_src, tgt_mask, src_mask):
        # tgt = [batch size, tgt len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # tgt_mask = [batch size, 1, tgt len, tgt len]
        # src_mask = [batch size, 1, 1, src len]

        _tgt, _ = self.self_attention(tgt, tgt, tgt, tgt_mask)     # Masked Multi-Head Self-Attention
        tgt = self.self_attn_layer_norm(tgt + self.dropout(_tgt))
        # tgt = [batch size, tgt len, hid dim]

        _tgt, attention = self.encoder_attention(tgt, enc_src, enc_src, src_mask)
        tgt = self.enc_attn_layer_norm(tgt + self.dropout(_tgt))
        # tgt = [batch size, tgt len, hid dim]

        _tgt = self.feedforward(tgt)
        tgt = self.ff_layer_norm(tgt + self.dropout(_tgt))
        # tgt = [batch size, tgt len, hid dim]
        # attention = [batch size, n heads, tgt len, src len]
        return tgt, attention


class VisualFrontEnd(nn.Module):
    def __init__(self, in_channel=3, hidden_dim=64, out_channel=512, drop_rate=0.3):
        super(VisualFrontEnd, self).__init__()
        # 3DCNN + ResNet18 + GlobalPooling
        self.stcnn = nn.Sequential(OrderedDict([
            # grayscale for 1, rgb for 3
            ('conv', nn.Conv3d(in_channel, hidden_dim, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3),
                       bias=False)),
            ('norm', nn.BatchNorm3d(hidden_dim)),
            ('relu', nn.ReLU(inplace=True)),
            ('dropout', nn.Dropout3d(drop_rate)),   # (B, C, T, H, W)，对每一个通道维度C按概率赋值为0
            ('pool', nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))]))

        # self.fc_out = nn.Linear(32768, out_channel)
        self.fc_out = nn.Linear(hidden_dim, out_channel)

    def forward(self, x):   # (B, C, T, H, W)
        cnn = self.stcnn(x)  # (B, D, T, H, W)
        cnn = cnn.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, D, H, W)
        batch, seq, channel, height, width = cnn.size()
        cnn = cnn.reshape(batch, seq, -1)  # (B, T, D)
        return self.fc_out(cnn)   # or using global pooling
        # pooled = nn.functional.adaptive_avg_pool2d(cnn.reshape(-1, channel, height, width), output_size=1)
        # return self.fc_out(pooled.reshape(batch, seq, -1))


class Encoder(nn.Module):
    def __init__(self,
                 in_channel,
                 hid_dim,
                 n_layers,
                 n_heads,
                 ffn_dim,
                 dropout,
                 max_length=100):
        super().__init__()
        self.hid_dim = hid_dim

        self.visual_front = VisualFrontEnd(in_channel=in_channel, out_channel=hid_dim)

        self.pos_embedding = PositionalEncoding(hid_dim, max_length)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  ffn_dim,
                                                  dropout)
                                     for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    # def make_src_mask(self, src):  # src = [batch size, src len]
    #     src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    #     # src_mask = [batch size, 1, 1, src len]
    #     return src_mask

    def forward(self, feat):  # (b, t, c, h, w)
        src_mask = torch.abs(torch.sum(feat, dim=(-1, -2, -3))) > 0   # (b, t)
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)  # (b, 1, 1, t)

        feat = feat.transpose(1, 2).contiguous()  # (b, c, t, h, w)
        feat = self.visual_front(feat)
        bs, src_len = feat.shape[0], feat.shape[1]
        pos_embed = self.pos_embedding(src_len)
        src = self.dropout(feat + pos_embed)
        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)
        # src = [batch size, src len, hid dim]
        return src, src_mask


class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 ffn_dim,
                 dropout,
                 pad_idx=0,
                 max_length=100):
        super().__init__()
        self.hid_dim = hid_dim
        self.tgt_pad_idx = pad_idx  # pad token index

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = PositionalEncoding(hid_dim, max_length)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  ffn_dim,
                                                  dropout)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def make_tgt_mask(self, tgt):  # [batch size, tgt len]
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        # tgt_pad_mask = [batch size, 1, 1, tgt len]
        tgt_len = tgt.shape[1]
        # 下三角(包括对角线)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        # tgt_sub_mask = [tgt len, tgt len]
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        # tgt_mask = [batch size, 1, tgt len, tgt len]
        return tgt_mask

    def forward(self, tgt, enc_src, src_mask):
        # tgt = [batch size, tgt len]
        # enc_src = [batch size, src len, hid dim]
        # tgt_mask = [batch size, 1, tgt len, tgt len]
        # src_mask = [batch size, 1, 1, src len]
        bs, tgt_len = tgt.shape[0], tgt.shape[1]

        tgt_mask = self.make_tgt_mask(tgt)  # [batch size, 1, tgt len, tgt len]

        pos_embed = self.pos_embedding(tgt_len)
        tok_embed = self.tok_embedding(tgt) * self.hid_dim**0.5
        tgt = self.dropout(tok_embed + pos_embed)
        # tgt = [batch size, tgt len, hid dim]

        for layer in self.layers:
            tgt, attention = layer(tgt, enc_src, tgt_mask, src_mask)
        # tgt = [batch size, tgt len, hid dim]
        # attention = [batch size, n heads, tgt len, src len]

        output = self.fc_out(tgt)
        # output = [batch size, tgt len, output dim]
        return output, attention


class Seq2Seq(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.encoder = Encoder(opt.in_channel,
                               opt.hidden_dim,
                               opt.enc_layers,
                               opt.head_num,
                               opt.enc_ffn_dim,
                               opt.drop_attn)
        
        self.decoder = Decoder(opt.tgt_vocab_size,
                               opt.hidden_dim,
                               opt.dec_layers,
                               opt.head_num,
                               opt.dec_ffn_dim,
                               opt.drop_attn)

    def forward(self, src, tgt, src_lens=None, tgt_lens=None):  # src - video,  tgt - text
        enc_src, src_mask = self.encoder(src)  # [batch size, src len, hid dim]
        output, attention = self.decoder(tgt[:, :-1], enc_src, src_mask)
        # output = [batch size, tgt len, output dim]
        # attention = [batch size, n heads, tgt len, src len]
        loss = nn.functional.cross_entropy(output.transpose(-1, -2).contiguous(), tgt[:, 1:].long(),
                                           ignore_index=self.opt.tgt_pad_idx)
        if src_lens is not None and tgt_lens is not None:
            log_probs = output.transpose(0, 1).log_softmax(dim=-1)  # (T, B, C)
            ctc_loss = nn.functional.ctc_loss(log_probs, tgt[:, 1:], src_lens.reshape(-1), tgt_lens.reshape(-1),
                                              blank=0)    # blank is identical to PAD
            # flattened_targets = tgt[:, 1:].masked_select(tgt[:, 1:] > 0)
            # ctc_loss = nn.functional.ctc_loss(log_probs, flattened_targets, src_lens.reshape(-1), tgt_lens.reshape(-1),
            #                                   blank=0)    # blank is identical to PAD
            # loss += ctc_loss
            loss = 0.8 * loss + 0.2 * ctc_loss
        return loss, output

    def greedy_decoding(self, src_inp, bos_id, eos_id, pad_id=0):  # (bs, src len)
        tgt_len = self.opt.max_dec_len
        bs = src_inp.shape[0]
        tgt_align = torch.tensor([[bos_id]] * bs).to(src_inp.device)  # (bs, 1)
        with torch.no_grad():
            enc_src, src_mask = self.encoder(src_inp)
            for t in range(tgt_len):
                output, attn = self.decoder(tgt_align, enc_src, src_mask)
                pred = output.argmax(dim=-1)[:, -1]  # (bs, tgt_len) -> (bs, )
                if pred.cpu().tolist() == [eos_id] * bs or pred.cpu().tolist() == [pad_id] * bs:
                    break
                tgt_align = torch.cat((tgt_align, pred.unsqueeze(-1)), dim=1).contiguous()

        dec_pred = tgt_align[:, 1:].detach().cpu().numpy()  # (bs, tgt_len)
        return dec_pred

    # def greedy_decoding(self, src_inp, bos_id, eos_id, pad_id=0):  # (bs, src len)
    #     tgt_len = self.opt.max_dec_len
    #     bs = src_inp.shape[0]
    #     res = torch.zeros((bs, tgt_len)).to(src_inp.device)
    #     tgt_align = torch.zeros((bs, tgt_len), dtype=torch.long).to(src_inp.device)
    #     tgt_align[:, 0] = bos_id
    #     with torch.no_grad():
    #         enc_src, src_mask = self.encoder(src_inp)
    #         for t in range(tgt_len):
    #             output, attn = self.decoder(tgt_align, enc_src, src_mask)
    #             pred = output.argmax(dim=-1)
    #             if pred[:, t].cpu().tolist() == [eos_id] * bs or pred[:, t].cpu().tolist() == [pad_id] * bs:
    #                 break
    #             res[:, t] = pred[:, t]
    #             if t < tgt_len - 1:
    #                 tgt_align[:, t + 1] = pred[:, t]
    #
    #     dec_pred = res.detach().cpu().numpy()  # (bs, tgt_len)
    #     return dec_pred

    def topk_decoding(self, src_inp, bos_id, eos_id, pad_id=0):  # (bs, src len)
        def step_decoding(next_token_logits, k=3, T=0.7):
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float("Inf")
            probs = nn.functional.softmax(next_token_logits / T, dim=-1)
            # multinominal方法可以根据给定权重对数组进行多次采样，返回采样后的元素下标
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            return next_token

        tgt_len = self.opt.max_dec_len
        bs = src_inp.shape[0]
        tgt_align = torch.tensor([[bos_id]] * bs).to(src_inp.device)  # (bs, 1)
        with torch.no_grad():
            enc_src, src_mask = self.encoder(src_inp)
            for t in range(tgt_len):
                output, attn = self.decoder(tgt_align, enc_src, src_mask)
                pred = step_decoding(output[:, -1])
                if pred.cpu().tolist() == [eos_id] * bs or pred.cpu().tolist() == [pad_id] * bs:
                    break
                tgt_align = torch.cat((tgt_align, pred.unsqueeze(1)), dim=1).contiguous()
        dec_pred = tgt_align[:, 1:].detach().cpu().numpy()  # (bs, tgt_len)
        return dec_pred

    def beam_search_decoding(self, src_inp, bos_id, eos_id, pad_id=0):
        with torch.no_grad():
            enc_src, src_mask = self.encoder(src_inp)
            res = beam_decode(self.decoder, enc_src, src_mask, bos_id, eos_id, beam_width=5)
        return res.detach().cpu().numpy()