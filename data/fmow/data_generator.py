import os
import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from wilds import get_dataset

from data.fmow.preprocess import preprocess
from data.utils import Mode

PREPROCESSED_FILE = 'fmow.pkl'

class FMoWBase(Dataset):
    def __init__(self, args):
        super().__init__()

        if args.data_dir is None:
            raise ValueError('Data directory not specified!')
        self.data_file = os.path.join(args.data_dir, PREPROCESSED_FILE)
        if not os.path.isfile(self.data_file):
            print(f'Preprocessing data and saving to {self.data_file}')
            preprocess(args)

        self.datasets = pickle.load(open(self.data_file, 'rb'))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        dataset = get_dataset(dataset="fmow", root_dir=args.data_dir, download=True)
        self.root = dataset.root

        self.args = args
        self.num_classes = 62
        self.current_time = 0
        self.num_tasks = 17
        self.ENV = [year for year in range(0, 16)]
        self.resolution = 224
        self.mode = Mode.TRAIN

        self.class_id_list = {i: {} for i in range(self.num_classes)}
        self.task_idxs = {}
        self.input_dim = []
        cumulative_batch_size = 0
        start_idx = 0
        for year in sorted(self.datasets.keys()):
            count = len(self.datasets[year][self.mode]['labels'])
            print('year:', year, 'count', count)
            cumulative_batch_size += min(args.mini_batch_size, count)
            if args.method in ['erm']:
                self.input_dim.append((cumulative_batch_size, 3, 32, 32))
            else:
                self.input_dim.append((min(args.mini_batch_size, count), 3, 32, 32))

            end_idx = start_idx + len(self.datasets[year][self.mode]['labels'])
            self.task_idxs[year] = {}
            self.task_idxs[year][self.mode] = [start_idx, end_idx]
            start_idx = end_idx

            for classid in range(self.num_classes):
                sel_idx = np.nonzero(self.datasets[year][self.mode]['labels'] == classid)[0]
                self.class_id_list[classid][year] = sel_idx

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'fmow'

    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.datasets[time][self.mode]['image_idxs'] = np.concatenate(
            (self.datasets[time][self.mode]['image_idxs'], self.datasets[prev_time][self.mode]['image_idxs']), axis=0)
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
        if idx >= K:
            last_K_num_samples = self.input_dim[idx - K][0]
            self.datasets[time][self.mode]['image_idxs'] = np.concatenate(
                (self.datasets[time][self.mode]['image_idxs'], self.datasets[prev_time][self.mode]['image_idxs'][:-last_K_num_samples]), axis=0)
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

    def get_input(self, idx):
        """Returns x for a given idx."""
        idx = self.datasets[self.current_time][self.mode]['image_idxs'][idx]
        img = Image.open(self.root / 'images' / f'rgb_img_{idx}.png').convert('RGB')
        return img

    def get_lisa_new_sample(self, time_idx, classid, num_sample):
        idx_all = self.class_id_list[classid][time_idx]
        sel_idx = np.random.choice(idx_all, num_sample, replace=True)
        image = torch.stack([self.transform(self.get_input(idx)) for idx in sel_idx], dim=0)
        label = self.datasets[time_idx][self.mode]['labels'][sel_idx]

        return torch.FloatTensor(image).cuda(), torch.LongTensor(label).unsqueeze(1).cuda()


class FMoW(FMoWBase):
    def __init__(self, args):
        super().__init__(args=args)

    def __getitem__(self, idx):
        image_tensor = self.transform(self.get_input(idx))
        label = self.datasets[self.current_time][self.mode]['labels'][idx]

        return image_tensor, torch.LongTensor([label])

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])


class FMoWGroup(FMoWBase):
    def __init__(self, args):
        super().__init__(args=args)
        self.num_groups = args.num_groups
        self.group_size = args.group_size
        self.train = True
        self.groupnum = 0

    def __getitem__(self, idx):
        if self.mode == Mode.TRAIN:
            np.random.seed(idx)
            # Select group ID
            idx = self.ENV.index(self.current_time)
            groupid = np.random.choice([i for i in range(max(1, idx - self.group_size + 1))])
            # print(groupid)

            # Pick a time step in the sliding window
            window = np.arange(max(0, idx - groupid - self.group_size), idx + 1)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx, end_idx = self.task_idxs[sel_time][self.mode]

            # Pick an example in the time step
            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            label = self.datasets[self.current_time][self.mode]['labels'][sel_idx]
            image_tensor = self.transform(self.get_input(sel_idx))
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
            image_tensor = self.transform(self.get_input(idx))
            label = self.datasets[self.current_time][self.mode]['labels'][idx]
            label_tensor = torch.LongTensor([label])

            return image_tensor, label_tensor

    def group_counts(self):
        idx = self.ENV.index(self.current_time)
        return torch.LongTensor([1 for _ in range(min(self.num_groups, idx + 1))])

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])
