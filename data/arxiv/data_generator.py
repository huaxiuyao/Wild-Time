import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from data.arxiv.preprocess import preprocess
from data.utils import initialize_distilbert_transform, Mode

PREPROCESSED_FILE = 'arxiv.pkl'
MAX_TOKEN_LENGTH = 300

class ArXivBase(Dataset):
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
        self.ENV = [year for year in range(2007, 2023)]
        self.num_tasks = len(self.ENV)
        self.num_classes = 172
        self.mini_batch_size = args.mini_batch_size
        self.task_indices = {}
        self.transform = initialize_distilbert_transform(max_token_length=MAX_TOKEN_LENGTH)
        self.mode = Mode.TRAIN

        self.class_id_list = {i: {} for i in range(self.num_classes)}
        start_idx = 0
        self.task_idxs = {}
        self.input_dim = []
        cumulative_batch_size = 0
        for i, year in enumerate(self.ENV):
            # Store task indices
            end_idx = start_idx + len(self.datasets[year][self.mode]['category'])
            self.task_idxs[year] = [start_idx, end_idx]
            start_idx = end_idx

            # Store class id list
            for classid in range(self.num_classes):
                sel_idx = np.nonzero(np.array(self.datasets[year][self.mode]['category']) == classid)[0]
                self.class_id_list[classid][year] = sel_idx
            print(f'Year {str(year)} loaded')

            # Store input dim
            num_examples = len(self.datasets[year][self.mode]['category'])
            cumulative_batch_size += min(self.mini_batch_size, num_examples)
            if args.method in ['erm']:
                self.input_dim.append(cumulative_batch_size)
            else:
                self.input_dim.append(min(self.mini_batch_size, num_examples))

    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.datasets[time][self.mode]['title'] = np.concatenate(
            (self.datasets[time][self.mode]['title'], self.datasets[prev_time][self.mode]['title']), axis=0)
        self.datasets[time][self.mode]['category'] = np.concatenate(
            (self.datasets[time][self.mode]['category'], self.datasets[prev_time][self.mode]['category']), axis=0)
        if data_del:
            del self.datasets[prev_time]
        for classid in range(self.num_classes):
            sel_idx = np.nonzero(self.datasets[time][self.mode]['category'] == classid)[0]
            self.class_id_list[classid][time] = sel_idx

    def update_historical_K(self, idx, K):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.window_start = self.ENV[max(0, idx - K)]
        if idx >= K:
            last_K_num_samples = self.input_dim[idx - K]
            self.datasets[time][self.mode]['title'] = np.concatenate(
                (self.datasets[time][self.mode]['title'], self.datasets[prev_time][self.mode]['title'][:-last_K_num_samples]), axis=0)
            self.datasets[time][self.mode]['category'] = np.concatenate(
                (self.datasets[time][self.mode]['category'], self.datasets[prev_time][self.mode]['category'][:-last_K_num_samples]), axis=0)
            del self.datasets[prev_time]
            for classid in range(self.num_classes):
                sel_idx = np.nonzero(self.datasets[time][self.mode]['category'] == classid)[0]
                self.class_id_list[classid][time] = sel_idx
        else:
            self.update_historical(idx)

    def update_current_timestamp(self, time):
        self.current_time = time

    def get_lisa_new_sample(self, time_idx, classid, num_sample):
        idx_all = self.class_id_list[classid][time_idx]
        if len(idx_all) == 0:
            return None, None
        sel_idx = np.random.choice(idx_all, num_sample, replace=True)[0]
        title = self.datasets[time_idx][self.mode]['title'][sel_idx]
        category = self.datasets[time_idx][self.mode]['category'][sel_idx]

        x = self.transform(text=title).unsqueeze(0).cuda()
        y = torch.LongTensor([category]).cuda()

        return x, y

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'arxiv'


class ArXiv(ArXivBase):
    def __init__(self, args):
        super().__init__(args=args)

    def __getitem__(self, index):
        title = self.datasets[self.current_time][self.mode]['title'][index]
        category = self.datasets[self.current_time][self.mode]['category'][index]

        x = self.transform(text=title)
        y = torch.LongTensor([category])

        return x, y

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['category'])


class ArXivGroup(ArXivBase):
    def __init__(self, args):
        super().__init__(args=args)
        self.num_groups = args.num_groups
        self.group_size = args.group_size
        self.window_end = self.ENV[0]
        self.train = True
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
            start_idx = self.task_idxs[sel_time][0]
            end_idx = self.task_idxs[sel_time][1]

            # Pick an example in the time step
            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            title = self.datasets[self.current_time][self.mode]['title'][sel_idx]
            category = self.datasets[self.current_time][self.mode]['category'][sel_idx]
            x = self.transform(text=title)
            y = torch.LongTensor([category])
            group_tensor = torch.LongTensor([groupid])

            del groupid
            del window
            del sel_time
            del start_idx
            del end_idx
            del sel_idx
            del title
            del category

            return x, y, group_tensor

        else:
            title = self.datasets[self.current_time][self.mode]['title'][index]
            category = self.datasets[self.current_time][self.mode]['category'][index]

            x = self.transform(text=title)
            y = torch.LongTensor([category])

            del title
            del category

            return x, y

    def group_counts(self):
        idx = self.ENV.index(self.current_time)
        return torch.LongTensor([1 for _ in range(min(self.num_groups, idx + 1))])

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['category'])