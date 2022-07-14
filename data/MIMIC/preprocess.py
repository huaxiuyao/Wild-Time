import pandas as pd
import pickle
import os

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Adapted by TDC.

import numpy as np
import pickle
from data.MIMIC.get_stay_dict import get_stay_dict
from data.utils import Mode
import os

ID_HELD_OUT = 0.2


def MIMICPreprocess(data, type):
    ENV = [i for i in list(range(2008, 2020))]
    num_tasks = len(ENV)

    datasets={}
    temp_datasets = {}


    for i in ENV:
        datasets[i] = {}
        temp_datasets[i] = {'code':[], 'labels':[]}

    for eachadmit in data:
        year = int(data[eachadmit].icu_timestamp)
        if year in temp_datasets:
            if type not in temp_datasets[year]:
                temp_datasets[year][type]=[]
            if type == 'mortality':
                temp_datasets[year]['labels'].append(data[eachadmit].mortality)
            elif type == 'readmission':
                temp_datasets[year]['labels'].append(data[eachadmit].readmission)
            dx_list = ['dx' for _ in data[eachadmit].diagnosis]
            tr_list = ['tr' for _ in data[eachadmit].treatment]
            temp_datasets[year]['code'].append([data[eachadmit].diagnosis + data[eachadmit].treatment, dx_list + tr_list])

    for eachyear in temp_datasets.keys():
        temp_datasets[eachyear]['labels'] = np.array(temp_datasets[eachyear]['labels'])
        temp_datasets[eachyear]['code'] = np.array(temp_datasets[eachyear]['code'])
        num_samples = temp_datasets[eachyear]['labels'].shape[0]
        seed_ = np.random.get_state()
        np.random.seed(0)
        idxs = np.random.permutation(np.arange(num_samples))
        np.random.set_state(seed_)
        num_train_samples = int((1 - ID_HELD_OUT) * num_samples)
        datasets[eachyear][Mode.TRAIN] = {}
        datasets[eachyear][Mode.TRAIN]['code'] = temp_datasets[eachyear]['code'][idxs[:num_train_samples]]
        datasets[eachyear][Mode.TRAIN]['labels'] = temp_datasets[eachyear]['labels'][idxs[:num_train_samples]]

        datasets[eachyear][Mode.TEST_ID] = {}
        datasets[eachyear][Mode.TEST_ID]['code'] = temp_datasets[eachyear]['code'][idxs[num_train_samples:]]
        datasets[eachyear][Mode.TEST_ID]['labels'] = temp_datasets[eachyear]['labels'][idxs[num_train_samples:]]

        datasets[eachyear][Mode.TEST_OOD] = {}
        datasets[eachyear][Mode.TEST_OOD]['code'] = temp_datasets[eachyear]['code']
        datasets[eachyear][Mode.TEST_OOD]['labels'] = temp_datasets[eachyear]['labels']

        print(eachyear, datasets[eachyear][Mode.TRAIN]['labels'].shape, datasets[eachyear][Mode.TEST_ID]['labels'].shape)

    with open('./Data/mimic_preprocessed_{}.pkl'.format(type),'wb') as f:
        pickle.dump(datasets, f)


def preprocess(args):
    if not os.path.exists('./Data/mimic_stay_dict.pkl'):
        get_stay_dict()
    data = pickle.load(open('./Data/mimic_stay_dict.pkl', 'rb'))
    MIMICPreprocess(data, 'readmission')
    MIMICPreprocess(data, 'mortality')