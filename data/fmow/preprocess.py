from collections import defaultdict

import os
import numpy as np
import pickle
import torch
from wilds import get_dataset

from data.utils import Mode

ID_HELD_OUT = 0.1

def preprocess(args):
    dataset = get_dataset(dataset="fmow", root_dir=args.data_dir, download=True)
    split_array = dataset.split_array
    split_dict = dataset.split_dict

    y_array = []
    datasets = {}
    unique_years = []
    image_idxs = defaultdict(list)
    labels = defaultdict(list)
    for split in ['train', 'val', 'test']:
        split_mask = split_array == split_dict[split]
        split_idx = np.where(split_mask)[0]
        y_array.append(dataset.y_array[split_idx])
        years = dataset.metadata_array[split_idx, 1]
        split_unique_years = torch.unique(years).detach().numpy().tolist()
        for year in split_unique_years:
            image_idxs[year].append(dataset.full_idxs[split_idx][torch.where(years == year)])
            labels[year].append(dataset.y_array[split_idx][torch.where(years == year)])
        unique_years = unique_years + split_unique_years

    for year in sorted(unique_years):
        if year not in datasets.keys():
            datasets[year] = {}
            datasets[year][Mode.TRAIN] = {}
            datasets[year][Mode.TRAIN]['image_idxs'] = []
            datasets[year][Mode.TRAIN]['labels'] = []
            datasets[year][Mode.TEST_ID] = {}
            datasets[year][Mode.TEST_ID]['image_idxs'] = []
            datasets[year][Mode.TEST_ID]['labels'] = []
            datasets[year][Mode.TEST_OOD] = {}
            datasets[year][Mode.TEST_OOD]['image_idxs'] = []
            datasets[year][Mode.TEST_OOD]['labels'] = []

        image_idxs_year = np.concatenate(image_idxs[year], axis=0)
        labels_year = np.concatenate(labels[year], axis=0)
        num_samples = len(labels_year)
        num_train_samples = int((1 - ID_HELD_OUT) * num_samples)
        seed_ = np.random.get_state()
        np.random.seed(0)
        idxs = np.random.permutation(np.arange(num_samples))
        np.random.set_state(seed_)
        train_image_idxs = image_idxs_year[idxs[:num_train_samples]]
        train_labels = labels_year[idxs[:num_train_samples]]
        test_image_idxs = image_idxs_year[idxs[num_train_samples:]]
        test_labels = labels_year[idxs[num_train_samples:]]
        datasets[year][Mode.TRAIN]['image_idxs'] = train_image_idxs
        datasets[year][Mode.TRAIN]['labels'] = train_labels
        datasets[year][Mode.TEST_ID]['image_idxs'] = test_image_idxs
        datasets[year][Mode.TEST_ID]['labels'] = test_labels
        datasets[year][Mode.TEST_OOD]['image_idxs'] = image_idxs_year
        datasets[year][Mode.TEST_OOD]['labels'] = labels_year
        del image_idxs_year, labels_year, train_image_idxs, train_labels, test_image_idxs, test_labels

    preprocessed_data_path = os.path.join(args.data_dir, 'fmow.pkl')
    pickle.dump(datasets, open(preprocessed_data_path, 'wb'))