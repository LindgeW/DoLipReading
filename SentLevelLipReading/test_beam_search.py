from collections import namedtuple
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from utils import vocab_pad_idx, vocab_bos_idx, vocab_eos_idx, flatten, try_cuda

InferenceState = namedtuple("InferenceState",
                            "prev_inference_state, flat_index, last_word, word_count, score, last_alpha")


def backchain_inference_states(last_inference_state):
    word_indices = []
    inf_state = last_inference_state
    scores = []
    last_score = None
    attentions = []
    while inf_state is not None:
        word_indices.append(inf_state.last_word)
        attentions.append(inf_state.last_alpha)
        if last_score is not None:
            scores.append(last_score - inf_state.score)
        last_score = inf_state.score
        inf_state = inf_state.prev_inference_state
    scores.append(last_score)
    return list(reversed(word_indices))[1:], list(reversed(scores))[1:], list(reversed(attentions))[1:]  # exclude BOS


def beam_search(self, beam_size, path_obs, path_actions):
    assert len(path_obs) == len(path_actions)

    start_obs, batched_image_features, batched_action_embeddings, path_mask, \
    path_lengths, _, perm_indices = self._batch_observations_and_actions(path_obs, path_actions, None)

    batch_size = len(start_obs)
    assert len(perm_indices) == batch_size
    ctx, h_t, c_t = self.encoder(batched_action_embeddings, batched_image_features)

    # ===============================================================
    completed = []
    for _ in range(batch_size):
        completed.append([])

    beams = [
        [InferenceState(prev_inference_state=None,
                        flat_index=i,
                        last_word=vocab_bos_idx,
                        word_count=0,
                        score=0.0,
                        last_alpha=None)] for i in range(batch_size)
    ]

    for t in range(self.instruction_len):
        flat_indices = []
        beam_indices = []
        w_t_list = []
        for beam_index, beam in enumerate(beams):
            for inf_state in beam:
                beam_indices.append(beam_index)
                flat_indices.append(inf_state.flat_index)
                w_t_list.append(inf_state.last_word)
        w_t = try_cuda(Variable(torch.LongTensor(w_t_list), requires_grad=False))
        if len(w_t.shape) == 1:
            w_t = w_t.unsqueeze(0)

        h_t, c_t, alpha, logit = self.decoder(w_t.view(-1, 1), h_t[flat_indices], c_t[flat_indices], ctx[beam_indices],
                                              path_mask[beam_indices])

        log_probs = F.log_softmax(logit, dim=1).data
        _, word_indices = logit.data.topk(min(beam_size, logit.size()[1]), dim=1)
        word_scores = log_probs.gather(1, word_indices)

        start_index = 0
        all_successors = []
        for beam_index, beam in enumerate(beams):
            successors = []
            end_index = start_index + len(beam)
            if beam:
                for inf_index, (inf_state, word_score_row, word_index_row) in enumerate(zip(beam, word_scores[start_index:end_index], word_indices[start_index:end_index])):
                    for word_score, word_index in zip(word_score_row, word_index_row):
                        flat_index = start_index + inf_index
                        successors.append(
                            InferenceState(
                                prev_inference_state=inf_state,
                                flat_index=flat_index,
                                last_word=word_index,
                                word_count=inf_state.word_count + 1,
                                score=inf_state.score + word_score,
                                last_alpha=alpha[flat_index].data))
            start_index = end_index
            successors = sorted(successors, key=lambda t: t.score, reverse=True)[:beam_size]
            all_successors.append(successors)

        new_beams = []
        for beam_index, successors in enumerate(all_successors):
            new_beam = []
            for successor in successors:
                if successor.last_word == vocab_eos_idx or t == self.instruction_len - 1:
                    completed[beam_index].append(successor)
                else:
                    new_beam.append(successor)
            if len(completed[beam_index]) >= beam_size:
                new_beam = []
            new_beams.append(new_beam)

        beams = new_beams
        if not any(beam for beam in beams):
            break
    # ====================================================================

    outputs = []
    for _ in range(batch_size):
        outputs.append([])

    for perm_index, src_index in enumerate(perm_indices):
        this_outputs = outputs[src_index]
        assert len(this_outputs) == 0
        this_completed = completed[perm_index]
        instr_id = start_obs[perm_index]['instr_id']
        for inf_state in sorted(this_completed, key=lambda t: t.score, reverse=True)[:beam_size]:
            word_indices, scores, attentions = backchain_inference_states(inf_state)
            this_outputs.append({
                'instr_id': instr_id,
                'word_indices': word_indices,
                'score': inf_state.score,
                'scores': scores,
                'words': self.env.tokenizer.decode_sentence(word_indices, break_on_eos=True, join=False),
                'attentions': attentions,
            })

    return outputs
