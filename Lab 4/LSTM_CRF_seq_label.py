import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore.common.initializer import initializer, Uniform

#ms.set_context(device_target="Ascend")

def compute_score(emissions, tags, seq_ends, mask, trans, start_trans, end_trans):
    # emissions: (seq_length, batch_size, num_tags)
    # tags: (seq_length, batch_size)
    # mask: (seq_length, batch_size)

    seq_length, batch_size = tags.shape
    mask = mask.astype(emissions.dtype)

    # Set score to the initial transition probability.
    # shape: (batch_size,)
    score = start_trans[tags[0]]
    # score += Probability of the first emission
    # shape: (batch_size,)
    score += emissions[0, mnp.arange(batch_size), tags[0]]

    for i in range(1, seq_length):
        # Probability that the label is transited from i-1 to i (valid when mask == 1).
        # shape: (batch_size,)
        score += trans[tags[i - 1], tags[i]] * mask[i]

        # Emission probability of tags[i] prediction(valid when mask == 1).
        # shape: (batch_size,)
        score += emissions[i, mnp.arange(batch_size), tags[i]] * mask[i]

    # End the transition.
    # shape: (batch_size,)
    last_tags = tags[seq_ends, mnp.arange(batch_size)]
    # score += End transition probability
    # shape: (batch_size,)
    score += end_trans[last_tags]

    return score

def compute_normalizer(emissions, mask, trans, start_trans, end_trans):
    # emissions: (seq_length, batch_size, num_tags)
    # mask: (seq_length, batch_size)

    seq_length = emissions.shape[0]

    # Set score to the initial transition probability and add the first emission probability.
    # shape: (batch_size, num_tags)
    score = start_trans + emissions[0]

    for i in range(1, seq_length):
        # The score dimension is extended to calculate the total score.
        # shape: (batch_size, num_tags, 1)
        broadcast_score = score.expand_dims(2)

        # The emission dimension is extended to calculate the total score.
        # shape: (batch_size, 1, num_tags)
        broadcast_emissions = emissions[i].expand_dims(1)

        # Calculate score_i according to formula (7).
        # In this case, broadcast_score indicates all possible paths from token 0 to the current token.
        # log_sum_exp corresponding to score
        # shape: (batch_size, num_tags, num_tags)
        next_score = broadcast_score + trans + broadcast_emissions

        # Perform the log_sum_exp operation on score_i to calculate the score of the next token.
        # shape: (batch_size, num_tags)
        next_score = ops.logsumexp(next_score, axis=1)

        # The score changes only when mask == 1.
        # shape: (batch_size, num_tags)
        score = mnp.where(mask[i].expand_dims(1), next_score, score)

    # Add the end transition probability.
    # shape: (batch_size, num_tags)
    score += end_trans
    # Calculate log_sum_exp based on the scores of all possible paths.
    # shape: (batch_size,)
    return ops.logsumexp(score, axis=1)

def viterbi_decode(emissions, mask, trans, start_trans, end_trans):
    # emissions: (seq_length, batch_size, num_tags)
    # mask: (seq_length, batch_size)

    seq_length = mask.shape[0]

    score = start_trans + emissions[0]
    history = ()

    for i in range(1, seq_length):
        broadcast_score = score.expand_dims(2)
        broadcast_emission = emissions[i].expand_dims(1)
        next_score = broadcast_score + trans + broadcast_emission

        # Obtain the label with the maximum score corresponding to the current token and save the label.
        indices = next_score.argmax(axis=1)
        history += (indices,)

        next_score = next_score.max(axis=1)
        score = mnp.where(mask[i].expand_dims(1), next_score, score)

    score += end_trans

    return score, history

def post_decode(score, history, seq_length):
    # Use Score and History to calculate the optimal prediction sequence.
    batch_size = seq_length.shape[0]
    seq_ends = seq_length - 1
    # shape: (batch_size,)
    best_tags_list = []

    # Decode each sample in a batch in sequence.
    for idx in range(batch_size):
        # Search for the label that maximizes the prediction probability corresponding to the last token.
        # Add it to the list of best prediction sequence stores.
        best_last_tag = score[idx].argmax(axis=0)
        best_tags = [int(best_last_tag.asnumpy())]

        # Repeatedly search for the label with the maximum prediction probability corresponding to each token and add the label to the list.
        for hist in reversed(history[:seq_ends[idx]]):
            best_last_tag = hist[idx][best_tags[-1]]
            best_tags.append(int(best_last_tag.asnumpy()))

        # Reset the solved label sequence in reverse order to the positive sequence.
        best_tags.reverse()
        best_tags_list.append(best_tags)

    return best_tags_list

def sequence_mask(seq_length, max_length, batch_first=False):
    """Generate the mask matrix based on the actual length and maximum length of the sequence."""
    range_vector = mnp.arange(0, max_length, 1, seq_length.dtype)
    result = range_vector < seq_length.view(seq_length.shape + (1,))
    if batch_first:
        return result.astype(ms.int64)
    return result.astype(ms.int64).swapaxes(0, 1)

class CRF(nn.Cell):
    def __init__(self, num_tags: int, batch_first: bool = False, reduction: str = 'sum') -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.reduction = reduction
        self.start_transitions = ms.Parameter(initializer(Uniform(0.1), (num_tags,)), name='start_transitions')
        self.end_transitions = ms.Parameter(initializer(Uniform(0.1), (num_tags,)), name='end_transitions')
        self.transitions = ms.Parameter(initializer(Uniform(0.1), (num_tags, num_tags)), name='transitions')

    def construct(self, emissions, tags=None, seq_length=None):
        if tags is None:
            return self._decode(emissions, seq_length)
        return self._forward(emissions, tags, seq_length)

    def _forward(self, emissions, tags=None, seq_length=None):
        if self.batch_first:
            batch_size, max_length = tags.shape
            emissions = emissions.swapaxes(0, 1)
            tags = tags.swapaxes(0, 1)
        else:
            max_length, batch_size = tags.shape

        if seq_length is None:
            seq_length = mnp.full((batch_size,), max_length, ms.int64)

        mask = sequence_mask(seq_length, max_length)

        # shape: (batch_size,)
        numerator = compute_score(emissions, tags, seq_length-1, mask, self.transitions, self.start_transitions, self.end_transitions)
        # shape: (batch_size,)
        denominator = compute_normalizer(emissions, mask, self.transitions, self.start_transitions, self.end_transitions)
        # shape: (batch_size,)
        llh = denominator - numerator

        if self.reduction == 'none':
            return llh
        if self.reduction == 'sum':
            return llh.sum()
        if self.reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.astype(emissions.dtype).sum()

    def _decode(self, emissions, seq_length=None):
        if self.batch_first:
            batch_size, max_length = emissions.shape[:2]
            emissions = emissions.swapaxes(0, 1)
        else:
            batch_size, max_length = emissions.shape[:2]

        if seq_length is None:
            seq_length = mnp.full((batch_size,), max_length, ms.int64)

        mask = sequence_mask(seq_length, max_length)

        return viterbi_decode(emissions, mask, self.transitions, self.start_transitions, self.end_transitions)


class BiLSTM_CRF(nn.Cell):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Dense(hidden_dim, num_tags, 'he_uniform')
        self.crf = CRF(num_tags, batch_first=True)

    def construct(self, inputs, seq_length, tags=None):
        embeds = self.embedding(inputs)
        outputs, _ = self.lstm(embeds, seq_length=seq_length)
        feats = self.hidden2tag(outputs)

        crf_outs = self.crf(feats, tags, seq_length)
        return crf_outs


embedding_dim = 16
hidden_dim = 32

training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in Georgia".split(),
    "B I O O O O B".split()
)]

word_to_idx = {}
word_to_idx['<pad>'] = 0
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

tag_to_idx = {"B": 0, "I": 1, "O": 2}


model = BiLSTM_CRF(len(word_to_idx), embedding_dim, hidden_dim, len(tag_to_idx))
optimizer = nn.SGD(model.trainable_params(), learning_rate=0.01, weight_decay=1e-4)


grad_fn = ms.value_and_grad(model, None, optimizer.parameters)

def train_step(data, seq_length, label):
    loss, grads = grad_fn(data, seq_length, label)
    loss = ops.depend(loss, optimizer(grads))
    return loss


def prepare_sequence(seqs, word_to_idx, tag_to_idx):
    seq_outputs, label_outputs, seq_length = [], [], []
    max_len = max([len(i[0]) for i in seqs])

    for seq, tag in seqs:
        seq_length.append(len(seq))
        idxs = [word_to_idx[w] for w in seq]
        labels = [tag_to_idx[t] for t in tag]
        idxs.extend([word_to_idx['<pad>'] for i in range(max_len - len(seq))])
        labels.extend([tag_to_idx['O'] for i in range(max_len - len(seq))])
        seq_outputs.append(idxs)
        label_outputs.append(labels)

    return ms.Tensor(seq_outputs, ms.int64), \
            ms.Tensor(label_outputs, ms.int64), \
            ms.Tensor(seq_length, ms.int64)

def sequence_to_tag(sequences, idx_to_tag):
    outputs = []
    for seq in sequences:
        outputs.append([idx_to_tag[i] for i in seq])
    return outputs

from tqdm import tqdm

steps = 500
data, label, seq_length = prepare_sequence(training_data, word_to_idx, tag_to_idx)
with tqdm(total=steps) as t:
    for i in range(steps):
        loss = train_step(data, seq_length, label)
        t.set_postfix(loss=loss)
        t.update(1)


score, history = model(data, seq_length)

predict = post_decode(score, history, seq_length)


idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}

print(sequence_to_tag(predict, idx_to_tag))