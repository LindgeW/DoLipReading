import torch
from heapq import heappop, heappush, nlargest, nsmallest

alpha = 0.65
beta = 0.5
eps = 1e-5
gamma = 0.5


class Beam(object):
    def __init__(self,
                 tokens,
                 log_probs,
                 decoder_states,
                 coverage_vector):
        self.tokens = tokens
        self.log_probs = log_probs
        self.decoder_states = decoder_states
        self.coverage_vector = coverage_vector

    def extend(self,
               token,
               log_prob,
               decoder_states,
               coverage_vector):

        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    decoder_states=decoder_states,
                    coverage_vector=coverage_vector)

    def seq_score(self):
        """
        This function calculate the score of the current sequence.
        """
        len_Y = len(self.tokens)
        ln = (5 + len_Y) ** alpha / (5 + 1) ** alpha  # Lenth normalization
        cn = beta * torch.sum(  # Coverage normalization
            torch.log(
                eps + torch.where(
                    self.coverage_vector < 1.0,
                    self.coverage_vector,
                    torch.ones((1, self.coverage_vector.shape[1])).to(torch.device('cpu'))
                )
            )
        )
        score = sum(self.log_probs) / ln + cn
        return score

    def __lt__(self, other):
        return self.seq_score() < other.seq_score()

    def __le__(self, other):
        return self.seq_score() <= other.seq_score()


def best_k(self, beam, k, encoder_output, x_padding_masks, x, len_oovs):
    """Get best k tokens to extend the current sequence at the current time step.
    """
    # use decoder to generate vocab distribution for the next token
    x_t = torch.tensor(beam.tokens[-1]).reshape(1, 1).to(self.DEVICE)

    # Get context vector from attention network.
    context_vector, attention_weights, coverage_vector = self.model.attention(beam.decoder_states,
                                                                              encoder_output,
                                                                              x_padding_masks,
                                                                              beam.coverage_vector)
    # Replace the indexes of OOV words with the index of OOV token
    # to prevent index-out-of-bound error in the decoder.
    p_vocab, decoder_states, p_gen = self.model.decoder(replace_oovs(x_t, self.vocab),
                                                        beam.decoder_states,
                                                        context_vector)
    final_dist = self.model.get_final_distribution(x,
                                                   p_gen,
                                                   p_vocab,
                                                   attention_weights,
                                                   torch.max(len_oovs))
    # Calculate log probabilities.
    log_probs = torch.log(final_dist.squeeze())
    # Filter forbidden tokens.
    # EOS token penalty. Follow the definition in https://opennmt.net/OpenNMT/translation/beam_search
    log_probs[self.vocab.EOS] *= gamma * x.size()[1] / len(beam.tokens)
    log_probs[self.vocab.UNK] = -float('inf')
    # Get top k tokens and the corresponding logprob.
    topk_probs, topk_idx = torch.topk(log_probs, k)
    # Extend the current hypo with top k tokens, resulting k new hypos.
    best_k = [beam.extend(x, log_probs[x], decoder_states, coverage_vector) for x in topk_idx.tolist()]
    return best_k


# sigle sample
def beam_search(self,
                x,
                max_sum_len,
                beam_width,
                len_oovs,
                x_padding_masks):
    """Using beam search to generate summary.
    """
    # run body_sequence input through encoder
    encoder_output, encoder_states = self.model.encoder(replace_oovs(x, self.vocab))
    coverage_vector = torch.zeros((1, x.shape[1])).to(self.DEVICE)
    # initialize decoder states with encoder forward states
    decoder_states = self.model.reduce_state(encoder_states)

    # initialize the hypothesis with a class Beam instance.
    init_beam = Beam([self.vocab.SOS], [0], decoder_states, coverage_vector)
    # get the beam size and create a list for storing current candidates
    # and a list for completed hypothesis
    k = beam_width
    curr, completed = [init_beam], []
    # use beam search for max_sum_len (maximum length) steps
    for _ in range(max_sum_len):
        # get k best hypothesis when adding a new token
        topk = []
        for beam in curr:
            # When an EOS token is generated, add the hypo to the completed
            # list and decrease beam size.
            if beam.tokens[-1] == self.vocab.EOS:
                completed.append(beam)
                k -= 1
                continue
            for cand in best_k(beam, k, encoder_output, x_padding_masks, x, torch.max(len_oovs)):
                # Using topk as a heap to keep track of topk candidates.
                # Using the sequence scores of the hypos to compare and object ids to break ties.
                # add2heap(topk, (cand.seq_score(), id(cand), cand), k)
                heappush(topk, (-cand.seq_score(), id(cand), cand))
        # curr = [items[2] for items in topk]
        # curr = [items[2] for items in nsmallest(k, topk)]  # curr = [heappop(topk)[2] for _ in range(k)]
        # stop when there are enough completed hypothesis
        if len(completed) == beam_width:
            break

    # When there are not enough completed hypotheses,
    # take whatever when have in current best k as the final candidates.
    completed += curr
    # sort the hypothesis by normalized probability and choose the best one
    result = sorted(completed, key=lambda x: x.seq_score(), reverse=True)[0].tokens
    return result
