"""
https://github.com/liuxubo717/V-ACT/blob/main/tools/beam.py
Adapted from https://github.com/budzianowski/PyTorch-Beam-Search-Decoding,
and https://github.com/haantran96/wavetransformer/blob/main/modules/beam.py
"""

import operator
import torch
import torch.nn.functional as F
from queue import PriorityQueue
from torch.nn.utils.rnn import pad_sequence


class BeamSearchNode(object):
    def __init__(self, previousNode, wordId, logProb, length):
        """
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        """
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def __lt__(self, other):
        return self.logp < other.logp

    def eval(self, alpha=1.0):
        reward = 0
        T = 0.7  # [0.6, 0.8]
        return self.logp / float((self.leng - 1)**T + 1e-6) + alpha * reward
        # ln = ((5. + self.leng) / 6.)**T
        # return self.logp / float(ln + 1e-6)


def beam_decode(decoder, enc_feats, enc_mask, sos_ind, eos_ind, beam_width=5, top_k=1):
    """
    Args:
        enc_feats: src-side feats (B, T, D)
        model:
        sos_ind: index of '<sos>'
        eos_ind: index of '<eos>'
        beam_width: beam size
        top_k: how many sentences wanted to generate
    Returns:
    """
    decoded_batch = []
    batch_size = enc_feats.shape[0]
    # decoding goes sentence by sentence
    for idx in range(batch_size):
        enc_feat = enc_feats[idx, :].unsqueeze(0)
        # (1, time_frames, n_hid)

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([[sos_ind]]).to(enc_feats.device)

        # Number of sentence to generate
        endnodes = []
        number_required = min((top_k + 1), top_k - len(endnodes))

        # starting node -  previous node, word_id (sos_ind), logp, length
        node = BeamSearchNode(None, decoder_input, 0, 1)
        nodes = PriorityQueue()
        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize // (beam_width-1) >= 100:
                break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            if n.wordid[0, -1].item() == eos_ind and n.prevNode is not None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output = decoder(decoder_input, enc_feat, enc_mask[idx].unsqueeze(0))[0]
            log_prob = F.log_softmax(decoder_output[:, -1], dim=-1)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(log_prob, beam_width)
            nextnodes = []
            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()
                node = BeamSearchNode(n, torch.cat((decoder_input, decoded_t), dim=1), n.logp + log_p, n.leng + 1)
                nextnodes.append((-node.eval(), node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
            # increase qsize
            qsize += len(nextnodes) - 1

        # choose n_best paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(top_k)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterances.append(n.wordid[0, :])
            # # back trace
            # while n.prevNode != None:
            #     n = n.prevNode
            #     utterance.append(n.wordid)
            # utterance = utterance[::-1]
            # utterances.append(utterance)
        for i in range(top_k):
            decoded_batch.append(utterances[i])

    return pad_sequence(decoded_batch, batch_first=True, padding_value=eos_ind)[:, 1:]   # ignoring bos idx