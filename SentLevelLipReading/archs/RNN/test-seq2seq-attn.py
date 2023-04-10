import collections
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import sys
sys.path.append("../../..")

PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 将⼀个序列中所有的词记录在all_tokens中以便之后构造词典，
# 然后在该序列后⾯添加PAD直到序列⻓度变为max_seq_len，然后将序列保存在all_seqs中
def process_one_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
    all_tokens.extend(seq_tokens)
    seq_tokens += [EOS] + [PAD] * (max_seq_len - len(seq_tokens) - 1)
    all_seqs.append(seq_tokens)


# 使⽤所有的词来构造词典并将所有序列中的词变换为词索引后构造Tensor
def build_data(all_tokens, all_seqs):
    vocab = [PAD, BOS, EOS] + list(collections.Counter(all_tokens))
    indices = [[vocab.index(w) for w in seq] for seq in all_seqs]
    return vocab, torch.tensor(indices)


def read_data(max_seq_len):
    in_tokens, out_tokens, in_seqs, out_seqs = [], [], [], []
    with open('../../data/fr-en-small.txt') as f:
        lines = f.readlines()

    for line in lines:
        in_seq, out_seq = line.rstrip().split('\t')
        in_seq_tokens, out_seq_tokens = in_seq.split(' '), out_seq.split(' ')
        if max(len(in_seq_tokens), len(out_seq_tokens)) > max_seq_len - 1:
            continue  # 如果加上EOS后⻓于max_seq_len，则忽略掉此样本
        process_one_seq(in_seq_tokens, in_tokens, in_seqs, max_seq_len)
        process_one_seq(out_seq_tokens, out_tokens, out_seqs, max_seq_len)

    in_vocab, in_data = build_data(in_tokens, in_seqs)
    out_vocab, out_data = build_data(out_tokens, out_seqs)
    return in_vocab, out_vocab, Data.TensorDataset(in_data, out_data)


max_seq_len = 7
in_vocab, out_vocab, dataset = read_data(max_seq_len)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, drop_prob=0.):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 输入维度 (seq_len, batch_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=drop_prob)

    def forward(self, inputs, state):
        # (batch, seq_len)
        embedding = self.embedding(inputs.long()).permute(1, 0, 2)
        # (seq_len, batch, input_size)
        return self.rnn(embedding, state)

    def begin_state(self):
        return None  # init hidden state = 0


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, attention_size, drop_prob=0.):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = self.attention_model(2 * num_hiddens, attention_size)
        # 输⼊包含attention输出的c和实际输⼊, 所以尺⼨是2*embed_size
        self.rnn = nn.GRU(2 * embed_size, num_hiddens, num_layers, dropout=drop_prob)
        self.out = nn.Linear(num_hiddens, vocab_size)

    def attention_model(self, input_size, attention_size):
        return nn.Sequential(nn.Linear(input_size, attention_size, bias=False),
                             nn.Tanh(),
                             nn.Linear(attention_size, 1, bias=False))

    def attention_forward(self, model, enc_states, dec_state):
        """
        model:函数attention_model返回的模型
        enc_states: 编码端的输出，shape是(seq_len, batch_size, hidden_dim)
        dec_state: 解码端一个时间步的输出，shape是(batch_size, hidden_dim)
        """
        dec_states = dec_state.unsqueeze(dim=0).expand_as(enc_states)
        enc_and_dec_states = torch.cat((enc_states, dec_states), dim=2)
        e = model(enc_and_dec_states)  # (seq_len, batch_size, 1)
        alpha = F.softmax(e, dim=0)
        return (alpha * enc_states).sum(dim=0)  # context vector

    def forward(self, cur_input, state, enc_states):
        """
         cur_input shape: (batch, )
         state shape: (num_layers, batch, num_hiddens)
         """
        c = self.attention_forward(self.attention, enc_states, state[-1])
        input_and_c = torch.cat((self.embedding(cur_input), c), dim=1)  # (batch_size, 2*embed_size)
        output, state = self.rnn(input_and_c.unsqueeze(0), state)  # (1, batch_size, 2*embed_size)
        output = self.out(output).squeeze(dim=0)  # (batch_size, vocab_size)
        return output, state

    def begin_state(self, enc_state):
        # 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态
        return enc_state


def batch_loss(encoder, decoder, X, Y, loss):
    batch_size = X.shape[0]
    enc_state = encoder.begin_state()
    enc_outputs, enc_state = encoder(X, enc_state)
    dec_state = decoder.begin_state(enc_state)
    dec_input = torch.tensor([out_vocab.index(BOS)] * batch_size)
    # ⽤掩码变量mask来忽略掉标签为填充项PAD的损失
    mask, num_not_pad_tokens = torch.ones(batch_size, ), 0
    l = torch.tensor([0.0])
    for y in Y.permute(1, 0):  # Y shape: (batch, seq_len)
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        l = l + (mask * loss(dec_output, y)).sum()
        dec_input = y  # teacher forcing
        num_not_pad_tokens += mask.sum().item()
        # 将PAD对应位置的掩码设成0
        mask = mask * (y != out_vocab.index(PAD)).float()
    return l / num_not_pad_tokens


def train(encoder, decoder, dataset, lr, batch_size, num_epochs):
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
    for epoch in range(num_epochs):
        l_sum = 0.0
        for X, Y in data_iter:
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            l = batch_loss(encoder, decoder, X, Y, loss)
            l.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            l_sum += l.item()
        if (epoch + 1) % 10 == 0:
            print("epoch %d, loss %.3f" % (epoch + 1, l_sum / len(data_iter)))


embed_size, num_hiddens, num_layers = 64, 64, 2
attention_size, drop_prob, lr, batch_size, num_epochs = 10, 0.5, 0.01, 2, 50
encoder = Encoder(len(in_vocab), embed_size, num_hiddens, num_layers, drop_prob)
decoder = Decoder(len(out_vocab), embed_size, num_hiddens, num_layers, attention_size, drop_prob)
train(encoder, decoder, dataset, lr, batch_size, num_epochs)