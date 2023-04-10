import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Embedding(num_embeddings, embedding_dim)
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        embeddings = self.weight(x)
        return embeddings


class MultiHeadAttention(nn.Module):
    def __init__(self, opt, d_model=512, head_num=8, bias=True):
        super(MultiHeadAttention, self).__init__()
        self.opt = opt
        self.d_model = d_model
        self.head_num = head_num
        self.d_head = d_model // head_num
        self.bias = bias
        self.linear_q = nn.Linear(d_model, d_model, bias)
        self.linear_k = nn.Linear(d_model, d_model, bias)
        self.linear_v = nn.Linear(d_model, d_model, bias)
        self.linear_o = nn.Linear(d_model, d_model, bias)
        self.layer_norm = nn.LayerNorm(512)
        self.drop = nn.Dropout(self.opt.dropout_attention)

    def forward(self, q, k, v, query_mask=None, key_mask=None, causality=False):
        # q: (B, L_dec, D)
        # k=v: (B, L_enc, D)
        # L_dec = L_enc
        bs = q.size(0)
        _q, _k, _v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        _q = _q.reshape(bs, -1, self.head_num, self.d_head).transpose(1, 2)  # (B, NH, L, HD)
        _k = _k.reshape(bs, -1, self.head_num, self.d_head).transpose(1, 2)  # (B, NH, L, HD)
        _v = _v.reshape(bs, -1, self.head_num, self.d_head).transpose(1, 2)  # (B, NH, L, HD)

        # (B, L_enc) -> (B, NH, L_enc) -> (B, NH, 1, L_enc)
        key_mask = key_mask.unsqueeze(1).expand(-1, self.head_num, -1).unsqueeze(2)
        # key_mask = key_mask.expand(-1, -1, _q.size(1), -1)   # (B, NH, L_dec, L_enc)
        # (B, L_dec) -> (B, NH, L_dec) -> (B, NH, L_dec, 1)
        query_mask = query_mask.unsqueeze(1).expand(-1, self.head_num, -1).unsqueeze(-1)
        # query_mask = query_mask.expand(-1, -1, , -1, k.size(1))   # (B, NH, L_dec, L_enc)

        y = self.ScaledDotProductAttention(_q, _k, _v, key_mask, query_mask, causality)
        y = y.transpose(1, 2).contiguous().reshape(bs, -1, self.d_model)
        y = self.linear_o(y) + q
        y = self.layer_norm(y)
        return y

    def ScaledDotProductAttention(self, query, key, value, key_mask=None, query_mask=None, causality=False):
        # query: from decoder    (B, NH, L_dec, HD)
        # key: from encoder      (B, NH, L_enc, HD)
        # value: from encoder    (B, NH, L_enc, HD)
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / (dk ** 0.5)   # (B, NH, L_dec, L_enc)
        if key_mask is not None:
            scores = scores.masked_fill(key_mask == 0, -1e9)

        if causality:
            diag_vals = torch.ones_like(scores[0, :, :])
            diag_vals[np.triu_indices(diag_vals.size(-1), 1)] = 0
            diag_vals = diag_vals.unsqueeze(0).repeat(scores.size(0), 1, 1)
            scores = scores.masked_fill(diag_vals == 0, -1e9)

        attn_weight = F.softmax(scores, dim=-1)   # (B, NH, L_dec, L_enc)
        if query_mask is not None:
            attn_weight = attn_weight * query_mask

        attn_weight = self.drop(attn_weight)
        return attn_weight.matmul(value)   # (B, NH, L_dec, HD)


class MultiAtten(nn.Module):
    def __init__(self, opt, causality=False):
        super(MultiAtten, self).__init__()
        self.opt = opt
        self.multiAtten = MultiHeadAttention(opt)
        self.causality = causality

    def forward(self, feat, dec=None, query_mask=None, key_mask=None):
        if dec is None:
            key_mask = torch.zeros_like(query_mask)
            key_mask[query_mask == 1] = 1
            feat = self.multiAtten(
                feat, feat, feat, query_mask=query_mask, key_mask=key_mask, causality=self.causality)
        else:
            feat = self.multiAtten(
                feat, dec, dec, query_mask=query_mask, key_mask=key_mask, causality=self.causality)
        return feat


class MultiForward(nn.Module):
    def __init__(self, opt):
        super(MultiForward, self).__init__()
        self.opt = opt
        self.feed_f = nn.Sequential(
            nn.Conv1d(512, 2048, 1),
            nn.ReLU(True),
            # nn.BatchNorm1d(num_features=2048),
            nn.Conv1d(2048, 512, 1),
            nn.ReLU(True),
            # nn.BatchNorm1d(num_features=512),
        )
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, feat):
        feat2 = self.feed_f(feat.transpose(1, 2)).transpose(1, 2)
        feat3 = feat + feat2
        feat3 = self.layer_norm(feat3)
        return feat3


class EncodeAttention(nn.Module):
    def __init__(self, opt):
        super(EncodeAttention, self).__init__()
        self.opt = opt
        self.position_embed = PositionEmbedding(num_embeddings=opt.min_frame, embedding_dim=512)
        self.atten_list = nn.ModuleList()
        for _ in range(self.opt.block_num):
            self.atten_list.append(MultiAtten(opt, causality=False))
            self.atten_list.append(MultiForward(opt))

    def forward(self, feat):
        pos_embed = torch.arange(0, feat.size(1), 1, dtype=torch.long, device=feat.device).unsqueeze(0).repeat(feat.size(0), 1)
        query_mask = torch.abs(feat.sum(-1))
        query_mask[query_mask > 0] = 1
        feat = feat + self.position_embed(pos_embed)
        for i in range(len(self.atten_list)):
            if i % 2 == 0:
                feat = self.atten_list[i](feat, query_mask=query_mask)
            else:
                feat = self.atten_list[i](feat)
        return feat, query_mask


class DecodeAttention(nn.Module):
    def __init__(self, opt):
        super(DecodeAttention, self).__init__()
        self.opt = opt
        self.decode_embed = PositionEmbedding(num_embeddings=opt.out_channel + 1, embedding_dim=512)
        self.position_embed = PositionEmbedding(num_embeddings=opt.min_frame, embedding_dim=512)
        self.atten_list = nn.ModuleList()
        for _ in range(self.opt.block_num):
            self.atten_list.append(MultiAtten(opt, causality=True))
            self.atten_list.append(MultiAtten(opt, causality=False))
            self.atten_list.append(MultiForward(opt))
        self.fc = nn.Linear(512, opt.out_channel)
        self.drop = nn.Dropout(self.opt.dropout_embed)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, feat, align, align_gt, enc_query_mask, pad_index=0):
        query_masks_dec = torch.abs(align)
        query_masks_dec[query_masks_dec > 0] = 1
        align_embed = self.decode_embed(align.long())
        align_pos_embed = torch.arange(0, align_embed.size(1), 1).unsqueeze(0).repeat(feat.size(0), 1).long()
        if self.opt.gpu:
            align_pos_embed = align_pos_embed.cuda()
        align_embed = align_embed + self.position_embed(align_pos_embed)
        align_embed = self.drop(align_embed)
        for i in range(len(self.atten_list)):
            if i % 3 == 1:
                align_embed = self.atten_list[i](align_embed, feat, query_mask=query_masks_dec.float(),
                                                 key_mask=enc_query_mask)
            elif i % 3 == 0:
                align_embed = self.atten_list[i](align_embed, query_mask=query_masks_dec.float())
            else:
                align_embed = self.atten_list[i](align_embed)

        align_embed = self.fc(align_embed)

        is_target = align_gt != pad_index
        # 标签平滑
        smoothed_one_hot = one_hot(self.opt, align_gt.reshape(-1), self.opt.out_channel)
        smoothed_one_hot = smoothed_one_hot * \
                           (1 - self.opt.loss_smooth) + (1 - smoothed_one_hot) * \
                           self.opt.loss_smooth / (self.opt.out_channel - 1)
        align_embed = align_embed.reshape(-1, align_embed.size(-1))
        log_prb = F.log_softmax(align_embed, dim=1)
        loss = -(smoothed_one_hot * log_prb).sum(dim=1)
        final_loss = loss.mean()
        mean_loss = loss.sum() / torch.sum(is_target)
        return final_loss, mean_loss, align_embed

    def forward_infer(self, feat, align, enc_query_mask, pad_index=0):
        query_masks_dec = torch.abs(align)
        query_masks_dec[query_masks_dec > 0] = 1
        align_embed = self.decode_embed(align.type(torch.cuda.LongTensor))
        align_pos_embed = torch.arange(0, align_embed.size(1), 1).unsqueeze(
            0).repeat(feat.size(0), 1).long().cuda()
        align_embed = align_embed + self.position_embed(align_pos_embed)
        for i in range(len(self.atten_list)):
            if i % 3 == 1:
                align_embed = self.atten_list[i](align_embed, feat, query_mask=query_masks_dec.float(),
                                                 key_mask=enc_query_mask)
            elif i % 3 == 0:
                align_embed = self.atten_list[i](align_embed, query_mask=query_masks_dec.float())
            else:
                align_embed = self.atten_list[i](align_embed)

        b = align_embed.size(0)
        align_embed = align_embed.view(-1, align_embed.size(-1))
        align_embed = self.fc(align_embed)
        align_embed = align_embed.view(b, -1, align_embed.size(-1))
        return align_embed


def one_hot(opt, indices, depth):
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    if opt.gpu:
        encoded_indicies = encoded_indicies.cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)
    return encoded_indicies