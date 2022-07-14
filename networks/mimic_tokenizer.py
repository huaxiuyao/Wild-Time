import os
import sys
from typing import List

import ipdb
import torch
from torch.nn.utils.rnn import pad_sequence

from networks.mimic_vocab import Vocabulary, build_vocab_mimic
import pickle

def to_index(sequence, vocab, prefix='', suffix=''):
    """ convert code to index """
    prefix = [vocab(prefix)] if prefix else []
    suffix = [vocab(suffix)] if suffix else []
    sequence = prefix + [vocab(token) for token in sequence] + suffix
    return sequence


class MIMICTokenizer:
    def __init__(self):
        build_vocab_mimic()
        self.vocab_dir = './Data/vocab.pkl'
        if not os.path.exists(self.vocab_dir):
            build_vocab_mimic()
        self.code_vocabs, self.code_vocabs_size = self._load_code_vocabs()
        self.type_vocabs, self.type_vocabs_size = self._load_type_vocabs()

    def _load_code_vocabs(self):

        vocabs = pickle.load(open(self.vocab_dir, 'rb'))
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def _load_type_vocabs(self):
        vocabs = Vocabulary()
        for word in ['dx', 'tr']:
            vocabs.add_word(word)
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def get_code_vocabs_size(self):
        return self.code_vocabs_size

    def get_type_vocabs_size(self):
        return self.type_vocabs_size

    def __call__(self,
                 batch_codes: List[str],
                 batch_types: List[str],
                 padding=True,
                 prefix='<cls>',
                 suffix=''):

        # to tensor
        batch_codes = [torch.tensor(to_index(c, self.code_vocabs, prefix=prefix, suffix=suffix)) for c in batch_codes]
        batch_types = [torch.tensor(to_index(t, self.type_vocabs, prefix=prefix, suffix=suffix)) for t in batch_types]

        # padding
        if padding:
            batch_codes = pad_sequence(batch_codes, batch_first=True)
            batch_types = pad_sequence(batch_types, batch_first=True)

        return batch_codes, batch_types
