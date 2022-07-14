# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Adapted by TDC.

import os

import torch
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
from data.utils import Mode

ID_HELD_OUT = 0.1

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
              'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']

smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
               '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
               'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
               'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
               'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']

MAX_SEQ_PROTEIN = 1000
MAX_SEQ_DRUG = 100

from sklearn.preprocessing import OneHotEncoder

enc_protein = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))
enc_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))


def protein_2_embed(x):
    return enc_protein.transform(np.array(x).reshape(-1, 1)).toarray().T


def drug_2_embed(x):
    return enc_drug.transform(np.array(x).reshape(-1, 1)).toarray().T


def trans_protein(x):
    temp = list(x.upper())
    temp = [i if i in amino_char else '?' for i in temp]
    if len(temp) < MAX_SEQ_PROTEIN:
        temp = temp + ['?'] * (MAX_SEQ_PROTEIN - len(temp))
    else:
        temp = temp[:MAX_SEQ_PROTEIN]
    return temp


def trans_drug(x):
    temp = list(x)
    temp = [i if i in smiles_char else '?' for i in temp]
    if len(temp) < MAX_SEQ_DRUG:
        temp = temp + ['?'] * (MAX_SEQ_DRUG - len(temp))
    else:
        temp = temp[:MAX_SEQ_DRUG]
    return temp


from torch.utils import data

class TdcDtiDgBase(data.Dataset):
    def __init__(self, args):
        super().__init__()

        self.ENV = [i for i in list(range(2013, 2021))]
        self.num_tasks = 8
        self.input_shape = [(26, 100), (63, 1000)]
        self.num_classes = 1
        self.current_time = 0
        self.mode = Mode.TRAIN

        self.task_idxs = {}
        self.datasets = pickle.load(open(os.path.join(args.data_dir,'drug_preprocessed.pkl'), 'rb'))
        start_idx = 0
        end_idx = 0

        if args.data_dir is None:
            raise ValueError('Data directory not specified!')

        for i in self.ENV:
            if i != 2019:
                start_idx = end_idx
                end_idx = start_idx + len(self.datasets[i][self.mode])
            elif i == 2019:
                start_idx = 0
                end_idx = len(self.datasets[i][self.mode])
            self.task_idxs[i]={}

            self.task_idxs[i][self.mode] = [start_idx, end_idx]

        self.datasets_copy = deepcopy(self.datasets)

    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.datasets[time][self.mode] = pd.concat([self.datasets[time][self.mode], self.datasets[prev_time][self.mode]])
        self.datasets[time][self.mode].reset_index()
        # self.index_mapping[time][self.mode] = {idx: value for idx, value in enumerate(self.datasets[time][self.mode].index.values)}
        if data_del:
            del self.datasets[time - 1]

    def update_historical_K(self, idx, K):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        if time >= K:
            last_K_timesteps = [self.datasets_copy[time - i][self.mode] for i in range(1, K + 1)]
            self.datasets[time][self.mode] = pd.concat(last_K_timesteps)
            del self.datasets[time - 1][self.mode]
        else:
            self.update_historical(time)

    def update_current_timestamp(self, time):
        self.current_time = time

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'drug'

class TdcDtiDg(TdcDtiDgBase):
    def __init__(self, args):
        super().__init__(args=args)

    def __getitem__(self, index):
        # index = self.index_mapping[self.current_time][self.mode][index]
        d = self.datasets[self.current_time][self.mode].iloc[index].Drug_Enc
        t = self.datasets[self.current_time][self.mode].iloc[index].Target_Enc

        d = drug_2_embed(d)
        t = protein_2_embed(t)

        y = self.datasets[self.current_time][self.mode].iloc[index].Y
        return (d, t), y

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode])

class TdcDtiDgGroup(TdcDtiDgBase):
    def __init__(self, args):
        super().__init__(args=args)
        self.num_groups = args.num_groups
        self.group_size = args.group_size
        self.window_end = self.ENV[0]
        self.train = True
        self.groupnum = 0

    def __getitem__(self, index):
        # index = self.index_mapping[self.current_time][self.mode][index]
        if self.mode == Mode.TRAIN:
            np.random.seed(index)
            # Select group ID
            idx = self.ENV.index(self.current_time)
            groupid = np.random.choice([i for i in range(max(1, idx - self.group_size + 1))])

            # Pick a time step in the sliding window
            window = np.arange(max(0, idx - groupid - self.group_size), idx + 1)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx = self.task_idxs[sel_time][self.mode][0]
            end_idx = self.task_idxs[sel_time][self.mode][1]

            # Pick an example in the time step
            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            d = drug_2_embed(self.datasets[self.current_time][self.mode].iloc[sel_idx].Drug_Enc)
            t = protein_2_embed(self.datasets[self.current_time][self.mode].iloc[sel_idx].Target_Enc)
            y = self.datasets[self.current_time][self.mode].iloc[sel_idx].Y
            g = torch.LongTensor([groupid])

            del window
            del sel_time
            del start_idx
            del end_idx
            del sel_idx
            del groupid

            return (d, t), y, g

        else:
            d = drug_2_embed(self.datasets[self.current_time][self.mode].iloc[index].Drug_Enc)
            t = protein_2_embed(self.datasets[self.current_time][self.mode].iloc[index].Target_Enc)
            y = self.datasets[self.current_time][self.mode].iloc[index].Y

            return (d, t), y

    def group_counts(self):
        idx = self.ENV.index(self.current_time)
        return torch.LongTensor([1 for _ in range(min(self.num_groups, idx + 1))])

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode])
