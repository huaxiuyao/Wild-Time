import pickle
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence


def to_index(sequence, vocab, prefix='', suffix=''):
    """ convert code to index """
    prefix = [vocab(prefix)] if prefix else []
    suffix = [vocab(suffix)] if suffix else []
    sequence = prefix + [vocab(token) for token in sequence] + suffix
    return sequence


class MIMICTokenizer:
    def __init__(self):
        self.diagnosis_vocabs, self.diagnosis_vocabs_size = self._load_diagnosis_vocabs()
        self.treatment_vocabs, self.treatment_vocabs_size = self._load_treatment_vocabs()

    def _load_diagnosis_vocabs(self):
        vocab_dir = './Data/vocab_diagnosis.pkl'
        vocabs = pickle.load(open(vocab_dir))
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def _load_treatment_vocabs(self):
        vocab_dir = './Data/vocab_treatment.pkl'
        vocabs = pickle.load(open(vocab_dir))
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def get_code_diagnosis_size(self):
        return self.diagnosis_vocabs_size

    def get_type_vocabs_size(self):
        return self.treatment_vocabs_size

    def __call__(self,
                 batch_diagnosis: List[str],
                 batch_treatment: List[str],
                 padding=True,
                 prefix='<cls>',
                 suffix=''):

        # to tensor
        batch_diagnosis = [torch.tensor(to_index(d, self.diagnosis_vocabs, prefix=prefix, suffix=suffix)) for d in batch_diagnosis]
        batch_treatment = [torch.tensor(to_index(t, self.treatment_vocabs, prefix=prefix, suffix=suffix)) for t in batch_treatment]

        # padding
        if padding:
            batch_diagnosis = pad_sequence(batch_diagnosis, batch_first=True)
            batch_treatment = pad_sequence(batch_treatment, batch_first=True)

        return batch_diagnosis, batch_treatment
