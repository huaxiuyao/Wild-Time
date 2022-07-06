import pickle

class Vocabulary(object):

    def __init__(self):
        self.word2idx = {'<pad>': 0, '<cls>': 1, '<unk>': 2}
        self.idx2word = {0: '<pad>', 1: '<cls>', 2: '<unk>'}
        assert len(self.word2idx) == len(self.idx2word)
        self.idx = len(self.word2idx)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(all_words, output_filename):
    vocab = Vocabulary()
    for word in all_words:
        vocab.add_word(word)
    print(f"Vocab len:", len(vocab))

    # sanity check
    assert set(vocab.word2idx.keys()) == set(vocab.idx2word.values())
    assert set(vocab.word2idx.values()) == set(vocab.idx2word.keys())
    for word in vocab.word2idx.keys():
        assert word == vocab.idx2word[vocab(word)]

    pickle.dump(vocab, open(output_filename, 'wb'))
    return
