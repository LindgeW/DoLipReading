import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv(nn.Module):
    def __init__(self, out_channel):
        super(TemporalConv, self).__init__()
        self.inputDim = 512
        self.backend_conv1 = nn.Sequential(
            nn.Conv1d(self.inputDim, 2 * self.inputDim, kernel_size=5, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(2 * self.inputDim),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(2 * self.inputDim, 4 * self.inputDim, kernel_size=5, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(4 * self.inputDim),
            nn.ReLU(True))

        self.backend_conv2 = nn.Sequential(
            nn.Linear(4 * self.inputDim, self.inputDim),
            nn.BatchNorm1d(self.inputDim),
            nn.ReLU(True),
            nn.Linear(self.inputDim, out_channel))

        # initialize
        self._initialize_weights()

    def forward(self, feat):   # (B, N, D)
        feat = feat.transpose(1, 2).contiguous()
        feat = self.backend_conv1(feat)
        feat = torch.mean(feat, dim=2)
        feat = self.backend_conv2(feat)
        return feat

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**0.5)
                # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class TextConvEncoder(nn.Module):
    def __init__(self,
                 num_embeddings,
                 emb_dim,
                 hid_dim,
                 n_layers,
                 kernel_size=5,
                 dropout=0.,
                 max_length=100):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        self.tok_embedding = nn.Embedding(num_embeddings, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2 * hid_dim,
                                              kernel_size=kernel_size,
                                              padding=(kernel_size - 1) // 2)
                                    for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_inp):  # (bs, src_len)
        scale = 1. / 2 ** 0.5
        bs, src_len = src_inp.size(0), src_inp.size(1)
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(bs, 1).to(src_inp.device)
        tok_embedded = self.tok_embedding(src_inp)
        pos_embedded = self.pos_embedding(pos)
        embedded = self.dropout(tok_embedded + pos_embedded)
        conv_input = self.emb2hid(embedded)  # (bs, src_len, hidden_dim)
        conv_input = conv_input.transpose(1, 2).contiguous()  # (bs, hidden_dim, src_len)
        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))  # (bs, 2*hidden_dim, src_len)
            conved = F.glu(conved, dim=1)   # (bs, hidden_dim, src_len)
            conved = (conved + conv_input) * scale
            conv_input = conved

        conved = self.hid2emb(conved.transpose(1, 2).contiguous())  # (bs, src_len, embed_dim)
        combined = (conved + embedded) * scale
        return conved, combined


class TextConvDecoder(nn.Module):
    def __init__(self,
                 output_dim,
                 emb_dim,
                 hid_dim,
                 n_layers,
                 kernel_size,
                 dropout,
                 trg_pad_idx,
                 max_length=100):
        super().__init__()
        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.hid_dim = hid_dim

        self.tok_embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)
        self.fc_out = nn.Linear(emb_dim, output_dim)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2 * hid_dim,
                                              kernel_size=kernel_size)
                                    for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        # embedded = (bs, trg_len, embed_dim)
        # conved = (bs, hidden_dim, trg_len)
        # encoder_conved = encoder_combined = (bs, src_len, embed_dim)
        scale = 1. / 2 ** 0.5
        conved_emb = self.attn_hid2emb(conved.transpose(1, 2))  # (bs, trg_len, embed_dim)
        combined = (conved_emb + embedded) * scale
        energy = torch.matmul(combined, encoder_conved.transpose(1, 2))  # (bs, trg_len, src_len)
        attention = F.softmax(energy, dim=2)   # (bs, trg_len, src_len)
        attended_encoding = torch.matmul(attention, encoder_combined)  # (bs, trg_len, embed_dim)
        attended_encoding = self.attn_emb2hid(attended_encoding)  # (bs, trg_len, hidden_dim)
        attended_combined = (conved + attended_encoding.transpose(1, 2)) * scale  # (bs, hidden_dim, trg_len)
        return attention, attended_combined

    def forward(self, trg, encoder_conved, encoder_combined):
        # trg = (bs, trg_len)
        # encoder_conved = encoder_combined = (bs, src_len, embed_dim)
        scale = 1. / 2 ** 0.5
        bs, trg_len = trg.size(0), trg.size(1)
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(bs, 1).to(trg.device)
        tok_embedded = self.tok_embedding(trg)
        pos_embedded = self.pos_embedding(pos)
        embedded = self.dropout(tok_embedded + pos_embedded)  # (bs, trg_len, embed_dim)
        conv_input = self.emb2hid(embedded)  # (bs, trg_len, hidden_dim)
        conv_input = conv_input.transpose(1, 2)  # (bs, hidden_dim, trg_len)
        for i, conv in enumerate(self.convs):
            conv_input = self.dropout(conv_input)
            padding = torch.zeros(bs, self.hid_dim, self.kernel_size - 1).fill_(self.trg_pad_idx).to(trg.device)
            padded_conv_input = torch.cat((padding, conv_input), dim=2)  # (bs, hidden_dim, trg_len + kernel_size - 1)
            conved = conv(padded_conv_input)  # (bs, 2*hidden_dim, trg_len)
            conved = F.glu(conved, dim=1)  # (bs, hidden_dim, trg_len)
            attention, conved = self.calculate_attention(embedded,
                                                         conved,
                                                         encoder_conved,
                                                         encoder_combined)  # (bs, trg_len, src_len) (bs, hidden_dim, trg_len)
            conved = (conved + conv_input) * scale  # (bs, hidden_dim, trg_len)
            conv_input = conved

        conved = self.hid2emb(conved.transpose(1, 2))  # (bs, trg_len, embed_dim)
        output = self.fc_out(self.dropout(conved))  # (bs, trg_len, output_dim)
        return output, attention


class Seq2Seq(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = TextConvEncoder(num_embeddings=config.src_vocab_size,
                                       emb_dim=config.embed_dim,
                                       hid_dim=config.hidden_dim,
                                       n_layers=config.enc_layers,
                                       kernel_size=config.enc_kernel_size,
                                       dropout=config.dropout)

        self.decoder = TextConvDecoder(output_dim=config.trg_vocab_size,
                                       emb_dim=config.embed_dim,
                                       hid_dim=config.hidden_dim,
                                       n_layers=config.dec_layers,
                                       kernel_size=config.dec_kernel_size,
                                       dropout=config.dropout,
                                       trg_pad_idx=config.trg_pad_idx)

    def forward(self, src_inp, trg):
        # src = (bs, src_len)
        # trg = (bs, trg_len - 1] (<eos> token sliced off the end)
        encoder_conved, encoder_combined = self.encoder(src_inp)  # (bs, src_len, embed_dim) (bs, src_len, embed_dim)
        # calculate predictions of next words
        # output is a batch of predictions for each word in the trg sentence
        # attention a batch of attention scores across the src sentence for each word in the trg sentence
        output, attention = self.decoder(trg, encoder_conved, encoder_combined)  # (bs, trg_len-1, output_dim)  (bs, trg_len-1, src_len)
        return output, attention

    def greedy_decoding(self, src_inp, bos_id, eos_id, pad_id=0):
        trg_len = self.config.max_dec_steps
        bs = src_inp.size(0)
        trg_tensor = torch.tensor([[bos_id]] * bs).to(src_inp.device)  # (bs, 1)
        with torch.no_grad():
            encoder_conved, encoder_combined = self.encoder(src_inp)
            for t in range(trg_len):
                output, attn = self.decoder(trg_tensor, encoder_conved, encoder_combined)
                pred = output.argmax(dim=-1)[:, -1]   # (bs, trg_len) -> (bs, )
                if pred.cpu().tolist() == [eos_id] * bs or pred.cpu().tolist() == [pad_id] * bs:
                    break
                trg_tensor = torch.cat((trg_tensor, pred.unsqueeze(-1)), dim=1).contiguous()

        dec_pred = trg_tensor.detach().cpu().numpy()  # (bs, trg_len)
        return dec_pred
