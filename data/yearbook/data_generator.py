import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from data.utils import Mode
from data.yearbook.preprocess import preprocess

PREPROCESSED_FILE = 'yearbook.pkl'

class YearbookBase(Dataset):
    def __init__(self, args):
        super().__init__()

        if args.data_dir is None:
            raise ValueError('Data directory not specified!')
        self.data_file = os.path.join(args.data_dir, PREPROCESSED_FILE)
        if not os.path.isfile(self.data_file):
            print(f'Preprocessing data and saving to {self.data_file}')
            preprocess(args)
        self.datasets = pickle.load(open(self.data_file, 'rb'))

        self.args = args
        self.num_classes = 2
        self.current_time = 0
        self.resolution = 32
        self.mini_batch_size = args.mini_batch_size
        self.mode = Mode.TRAIN

        self.ENV = list(sorted(self.datasets.keys()))
        self.num_tasks = len(self.ENV)
        self.num_examples = {i: len(self.datasets[i][self.mode]['labels']) for i in self.ENV}

        ## create a datasets object
        self.class_id_list = {i: {} for i in range(self.num_classes)}
        self.input_dim = []
        cumulative_batch_size = 0
        self.task_idxs = {}
        start_idx = 0

        for i in self.ENV:
            end_idx = start_idx + len(self.datasets[i][self.mode]['labels'])
            self.task_idxs[i] = [start_idx, end_idx]
            start_idx = end_idx

            for classid in range(self.num_classes):
                sel_idx = np.nonzero(self.datasets[i][self.mode]['labels'] == classid)[0]
                self.class_id_list[classid][i] = sel_idx

            cumulative_batch_size += min(self.mini_batch_size, self.num_examples[i])
            if args.method in ['erm']:
                self.input_dim.append((cumulative_batch_size, 3, 32, 32))
            else:
                self.input_dim.append((min(self.mini_batch_size, self.num_examples[i]), 3, 32, 32))

    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.datasets[time][self.mode]['images'] = np.concatenate(
            (self.datasets[time][self.mode]['images'], self.datasets[prev_time][self.mode]['images']), axis=0)
        self.datasets[time][self.mode]['labels'] = np.concatenate(
            (self.datasets[time][self.mode]['labels'], self.datasets[prev_time][self.mode]['labels']), axis=0)
        if data_del:
            del self.datasets[prev_time]
        for classid in range(self.num_classes):
            sel_idx = np.nonzero(self.datasets[time][self.mode]['labels'] == classid)[0]
            self.class_id_list[classid][time] = sel_idx

    def update_historical_K(self, idx, K):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.window_start = self.ENV[max(0, idx - K)]
        if idx >= K:
            last_K_num_samples = self.input_dim[idx - K][0]
            self.datasets[time][self.mode]['images'] = np.concatenate(
                (self.datasets[time][self.mode]['images'], self.datasets[prev_time][self.mode]['images'][:-last_K_num_samples]), axis=0)
            self.datasets[time][self.mode]['labels'] = np.concatenate(
                (self.datasets[time][self.mode]['labels'], self.datasets[prev_time][self.mode]['labels'][:-last_K_num_samples]), axis=0)
            del self.datasets[prev_time]
            for classid in range(self.num_classes):
                sel_idx = np.nonzero(self.datasets[time][self.mode]['labels'] == classid)[0]
                self.class_id_list[classid][time] = sel_idx
        else:
            self.update_historical(idx)

    def update_current_timestamp(self, time):
        self.current_time = time

    def get_lisa_new_sample(self, time_idx, classid, num_sample):
        # time = self.ENV[time_idx]
        idx_all = self.class_id_list[classid][time_idx]
        sel_idx = np.random.choice(idx_all, num_sample, replace=True)
        image = self.datasets[time_idx][self.mode]['images'][sel_idx]
        label = self.datasets[time_idx][self.mode]['labels'][sel_idx]

        return torch.FloatTensor(image).permute(0, 3, 1, 2).cuda(), torch.LongTensor(label).unsqueeze(1).cuda()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'yearbook'


class Yearbook(YearbookBase):
    def __init__(self, args):
        super().__init__(args=args)

    def __getitem__(self, index):
        image = self.datasets[self.current_time][self.mode]['images'][index]
        label = self.datasets[self.current_time][self.mode]['labels'][index]
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
        label_tensor = torch.LongTensor([label])

        return image_tensor, label_tensor

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])


class YearbookGroup(YearbookBase):
    def __init__(self, args):
        super().__init__(args=args)
        self.num_groups = args.num_groups
        self.group_size = args.group_size
        self.window_end = self.ENV[0]
        self.groupnum = 0

    def __getitem__(self, index):
        if self.mode == Mode.TRAIN:
            np.random.seed(index)
            # Select group ID
            idx = self.ENV.index(self.current_time)
            groupid = np.random.choice([i for i in range(max(1, idx - self.group_size + 1))])

            # Pick a time step in the sliding window
            window = np.arange(max(0, idx - groupid - self.group_size), idx + 1)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx, end_idx = self.task_idxs[sel_time]

            # Pick an example in the time step
            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            image = self.datasets[self.current_time][self.mode]['images'][sel_idx]
            label = self.datasets[self.current_time][self.mode]['labels'][sel_idx]

            image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
            label_tensor = torch.LongTensor([label])
            group_tensor = torch.LongTensor([groupid])

            del groupid
            del window
            del sel_time
            del start_idx
            del end_idx
            del sel_idx

            return image_tensor, label_tensor, group_tensor

        else:
            image = self.datasets[self.current_time][self.mode]['images'][index]
            label = self.datasets[self.current_time][self.mode]['labels'][index]
            image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
            label_tensor = torch.LongTensor([label])

            return image_tensor, label_tensor

    def group_counts(self):
        idx = self.ENV.index(self.current_time)
        return torch.LongTensor([1 for _ in range(min(self.num_groups, idx + 1))])

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])
