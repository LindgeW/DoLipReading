import torch
import torch.nn as nn


class AttentiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentiveAttention, self).__init__()
        self.W = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim, bias=False),
                      nn.Tanh(),
                      nn.Linear(hidden_dim, 1, bias=False))

    # 在decode过程中根据上一个step的hidden计算对encoder结果的attention
    def forward(self, q, v, mask=None):
        '''
        :param q: (b, h)   decoder state
        :param v: (b, t, h)  encoder states
        :param mask: (b, t) encoder mask
        :return:
        '''
        qs = q.unsqueeze(1).expand_as(v)
        score = self.W(torch.cat((qs, v), dim=-1).contiguous())   # (b, t, 1)
        if mask is not None:
            score = score.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        ws = torch.softmax(score, dim=1)
        # ws = score / score.sum(1, keepdim=True)
        attn_out = (ws * v).sum(dim=1)
        return attn_out


class MLPAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(MLPAttention, self).__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, q, v, mask=None):
        '''
        :param q: (b, h)   decoder state
        :param v: (b, t, h)  encoder states
        :param mask: (b, t) encoder mask
        :return:
        '''
        qs = q.unsqueeze(1)
        score = self.V(torch.tanh(self.W1(qs) + self.W2(v)))   # (b, t, 1)
        if mask is not None:
            score = score.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        ws = torch.softmax(score, dim=1)
        # ws = score / score.sum(1, keepdim=True)
        attn_out = (ws * v).sum(dim=1)
        return attn_out

