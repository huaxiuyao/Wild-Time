import pandas as pd
import pickle
import os

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Adapted by TDC.

import os

import ipdb
import torch
import numpy as np
import pickle
import sys
sys.path.append('/iris/u/huaxiu/Temporal_Robustness/archived/tl4h_eicu')

from data.utils import Mode

ID_HELD_OUT = 0.2

def MIMICPreprocess(type):
    ENV = [i for i in list(range(2008, 2020))]
    num_tasks = len(ENV)
    data = pickle.load(open('/iris/u/huaxiu/Data/MIMIC/mimic_stay_dict.pkl', 'rb'))
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

    with open('/iris/u/huaxiu/Data/MIMIC/mimic_preprocessed_{}.pkl'.format(type),'wb') as f:
        pickle.dump(datasets, f)

#
# def preprocess_mimic():
#     data_path = '/iris/u/cchoi1/Data'
#     df = pd.read_csv(os.path.join(data_path, 'mimic_data_preprocessed.csv'))
#     df = df.sort_values(by=['real_admit_year'])
#     df_years = df.groupby(pd.Grouper(key='real_admit_year'))
#     dfs = [group for _, group in df_years]
#
#     mimic_dataset = {}
#     for taskid, df in enumerate(dfs):
#         year = 2005 + taskid
#         mimic_dataset[year] = {}
#         all_icu_diagnoses = list(df['diagnoses'].str.split(" <sep> ", expand=False))
#         all_icu_procedures = list(df['procedure'].str.split(" <sep> ", expand=False))
#         mimic_dataset[year]['codes'] = {}
#         mimic_dataset[year]['types'] = {}
#         for icu_stay, (diagnoses_list, procedures_list) in enumerate(zip(all_icu_diagnoses, all_icu_procedures)):
#             icu_stay_codes = sorted(diagnoses_list + procedures_list)
#             mimic_dataset[year]['codes'][icu_stay] = icu_stay_codes
#             mimic_dataset[year]['types'][icu_stay] = ['dx' if code in diagnoses_list else 'tr' for code in icu_stay_codes]
#
#         mimic_dataset[year]['readmission'] = df['readmission'].to_numpy()
#         mimic_dataset[year]['mortality'] = df['mortality'].to_numpy()
#         assert len(mimic_dataset[year]['readmission']) == len(mimic_dataset[year]['mortality'])
#         assert len(mimic_dataset[year]['codes']) == len(mimic_dataset[year]['types'])
#         assert len(mimic_dataset[year]['readmission']) == len(mimic_dataset[year]['codes'])
#     with open(os.path.join(data_path, 'mimic_data_preprocessed_year.pkl'), 'wb') as f:
#         pickle.dump(mimic_dataset, f)
#
#     all_codes = []
#     for year in mimic_dataset.keys():
#         for codes_list in mimic_dataset[year]['codes'].values():
#             for code in codes_list:
#                 all_codes.append(code)
#     all_codes = list(set(all_codes))
#     with open(os.path.join(data_path, 'mimic_all_codes.pkl'), 'wb') as f:
#         pickle.dump(all_codes, f)
#
if __name__ == '__main__':
    MIMICPreprocess('readmission')
    MIMICPreprocess('mortality')