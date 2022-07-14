# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Adapted by TDC.

import os

import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
from data.utils import Mode
from torch.utils import data

ID_HELD_OUT = 0.2


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


class TdcDtiDgBase(data.Dataset):
    def __init__(self, data_dir):
        super().__init__()

        self.ENV = [i for i in list(range(2013, 2021))]
        self.num_tasks = 8
        self.input_shape = [(26, 100), (63, 1000)]
        self.num_classes = 1
        self.current_time = 0
        self.mode = Mode.TRAIN

        self.task_idxs = {}
        start_idx = 0
        end_idx=0

        if data_dir is None:
            raise ValueError('Data directory not specified!')

        ## create a datasets object
        self.datasets = {}
        from tdc import BenchmarkGroup
        self.group = BenchmarkGroup(name='DTI_DG_Group', path=data_dir)

        benchmark = self.group.get('BindingDB_Patent')
        train_val, test, name = benchmark['train_val'], benchmark['test'], benchmark['name']

        unique_drug = pd.Series(train_val['Drug'].unique()).apply(trans_drug)
        unique_dict_drug = dict(zip(train_val['Drug'].unique(), unique_drug))
        train_val['Drug_Enc'] = [unique_dict_drug[str(i)] for i in train_val['Drug']]

        unique_target = pd.Series(train_val['Target'].unique()).apply(trans_protein)
        unique_dict_target = dict(zip(train_val['Target'].unique(), unique_target))
        train_val['Target_Enc'] = [unique_dict_target[str(i)] for i in train_val['Target']]

        unique_drug = pd.Series(test['Drug'].unique()).apply(trans_drug)
        unique_dict_drug = dict(zip(test['Drug'].unique(), unique_drug))
        test['Drug_Enc'] = [unique_dict_drug[str(i)] for i in test['Drug']]

        unique_target = pd.Series(test['Target'].unique()).apply(trans_protein)
        unique_dict_target = dict(zip(test['Target'].unique(), unique_target))
        test['Target_Enc'] = [unique_dict_target[str(i)] for i in test['Target']]

        for i in self.ENV:
            self.datasets[i]={}
            if i < 2019:
                df_data = train_val[train_val.Year == i]
            else:
                df_data = test[test.Year == i]

            if i != 2019:
                start_idx = end_idx
                end_idx = start_idx + len(df_data)
            elif i==2019:
                start_idx = 0
                end_idx = len(df_data)

            self.task_idxs[i] = [start_idx, end_idx]

            num_samples = len(df_data)
            seed_ = np.random.get_state()
            np.random.seed(0)
            idxs = np.random.permutation(np.arange(start_idx, end_idx))
            np.random.set_state(seed_)
            num_train_samples = int((1 - ID_HELD_OUT) * num_samples)
            self.datasets[i][Mode.TRAIN] = df_data.loc[idxs[:num_train_samples],:].reset_index()
            self.datasets[i][Mode.TEST_ID] = df_data.loc[idxs[num_train_samples:], :].reset_index()
            self.datasets[i][Mode.TEST_OOD] = df_data.reset_index()

        with open('drug_preprocessed.pkl','wb') as f:
            pickle.dump(self.datasets, f)


if __name__ == '__main__':
    TdcDtiDgBase('TDC_OOD')