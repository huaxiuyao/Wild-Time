import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from data.precipitation.preprocess import preprocess
from data.utils import Mode

PREPROCESSED_FILE = 'weather.pkl'

continuous_cols = [
    "climate_pressure", "climate_temperature", "cmc_0_0_0_1000", "cmc_0_0_0_2", "cmc_0_0_0_2_grad",
    "cmc_0_0_0_2_interpolated", "cmc_0_0_0_2_next", "cmc_0_0_0_500", "cmc_0_0_0_700", "cmc_0_0_0_850",
    "cmc_0_0_0_925", "cmc_0_0_6_2", "cmc_0_0_7_1000", "cmc_0_0_7_2", "cmc_0_0_7_500", "cmc_0_0_7_700",
    "cmc_0_0_7_850", "cmc_0_0_7_925", "cmc_0_1_0_0", "cmc_0_1_11_0", "cmc_0_1_65_0", "cmc_0_1_65_0_grad",
    "cmc_0_1_65_0_next", "cmc_0_1_66_0", "cmc_0_1_66_0_grad", "cmc_0_1_66_0_next", "cmc_0_1_67_0",
    "cmc_0_1_67_0_grad", "cmc_0_1_67_0_next", "cmc_0_1_68_0", "cmc_0_1_68_0_grad", "cmc_0_1_68_0_next",
    "cmc_0_1_7_0", "cmc_0_2_2_10", "cmc_0_2_2_1000", "cmc_0_2_2_500", "cmc_0_2_2_700", "cmc_0_2_2_850",
    "cmc_0_2_2_925", "cmc_0_2_3_10", "cmc_0_2_3_1000", "cmc_0_2_3_500", "cmc_0_2_3_700", "cmc_0_2_3_850",
    "cmc_0_2_3_925", "cmc_0_3_0_0", "cmc_0_3_0_0_next", "cmc_0_3_1_0", "cmc_0_3_5_1000", "cmc_0_3_5_500",
    "cmc_0_3_5_700", "cmc_0_3_5_850", "cmc_0_3_5_925", "cmc_0_6_1_0", "cmc_available", "cmc_horizon_h",
    "cmc_precipitations", "cmc_timedelta_s", "gfs_2m_dewpoint", "gfs_2m_dewpoint_grad", "gfs_2m_dewpoint_next",
    "gfs_a_vorticity", "gfs_available", "gfs_cloudness", "gfs_clouds_sea", "gfs_horizon_h", "gfs_humidity",
    "gfs_precipitable_water", "gfs_precipitations", "gfs_pressure", "gfs_r_velocity", "gfs_soil_temperature",
    "gfs_soil_temperature_available", "gfs_temperature_10000", "gfs_temperature_15000", "gfs_temperature_20000",
    "gfs_temperature_25000", "gfs_temperature_30000", "gfs_temperature_35000", "gfs_temperature_40000",
    "gfs_temperature_45000", "gfs_temperature_5000", "gfs_temperature_50000", "gfs_temperature_55000", "gfs_temperature_60000",
    "gfs_temperature_65000", "gfs_temperature_7000", "gfs_temperature_70000", "gfs_temperature_75000", "gfs_temperature_80000",
    "gfs_temperature_85000", "gfs_temperature_90000", "gfs_temperature_92500", "gfs_temperature_95000", "gfs_temperature_97500",
    "gfs_temperature_sea", "gfs_temperature_sea_grad", "gfs_temperature_sea_interpolated", "gfs_temperature_sea_next",
    "gfs_timedelta_s", "gfs_total_clouds_cover_high", "gfs_total_clouds_cover_low", "gfs_total_clouds_cover_low_grad",
    "gfs_total_clouds_cover_low_next", "gfs_total_clouds_cover_middle", "gfs_u_wind", "gfs_v_wind", "gfs_wind_speed",
    "sun_elevation", "topography_bathymetry", "wrf_graupel", "wrf_hail", "wrf_psfc", "wrf_rain", "wrf_rh2", "wrf_snow",
    "wrf_t2", "wrf_t2_grad", "wrf_t2_interpolated", "wrf_t2_next", "wrf_wind_u", "wrf_wind_v"
]

categorical_cols = [
    "wrf_available"
]

def min_max_func(x, arr):
    max_val = np.max(arr)
    min_val = np.min(arr)
    new_val = (x - min_val) / (max_val - min_val)
    return new_val

def rev_min_max_func(x, arr):
    max_val = np.max(arr)
    min_val = np.min(arr)
    og_val = (x *(max_val - min_val)) + min_val
    return og_val

class PrecipitationBase(Dataset):
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
        self.num_classes = 9
        self.current_time = 0
        self.num_tasks = 12
        self.ENV = [month for month in range(1, self.num_tasks + 1)]
        self.mini_batch_size = args.mini_batch_size
        self.regression = True if self.args.regression else False
        self.mode = Mode.TRAIN

        self.class_id_list = {i: {} for i in range(self.num_classes)}
        self.input_dim = []
        cumulative_batch_size = 0
        self.num_examples = {}
        self.task_idxs = {}
        start_idx = 0
        self.scaler = MinMaxScaler()

        self.all_temps = np.concatenate([self.datasets[i][Mode.TRAIN]['data']['continuous'] for i in self.ENV], axis=0)
        self.scaler.fit(self.all_temps)
        if self.args.regression:
            for i in self.ENV:
                for type in [Mode.TRAIN, Mode.TEST_ID, Mode.TEST_OOD]:
                    self.datasets[i][type]['data']['continuous'] = self.scaler.transform(self.datasets[i][type]['data']['continuous'])

        for i in self.ENV:
            end_idx = start_idx + len(self.datasets[i][self.mode]['labels'])
            self.task_idxs[i] = [start_idx, end_idx]
            start_idx = end_idx

            for classid in range(self.num_classes):
                sel_idx = np.nonzero(self.datasets[i][self.mode]['labels'] == classid)[0]
                self.class_id_list[classid][i] = sel_idx
            print(f'Month {str(i)} loaded')

            self.num_examples[i] = len(self.datasets[i])

            cumulative_batch_size += min(self.mini_batch_size, self.num_examples[i])
            if args.method in ['erm']:
                self.input_dim.append((cumulative_batch_size, 123))
            else:
                self.input_dim.append((min(self.mini_batch_size, self.num_examples[i]), 123))


    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'precipitation'

    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.datasets[time][self.mode]['data']['categorical'] = np.concatenate(
            (self.datasets[time][self.mode]['data']['categorical'], self.datasets[prev_time][self.mode]['data']['categorical']), axis=0)
        self.datasets[time][self.mode]['data']['continuous'] = np.concatenate(
            (self.datasets[time][self.mode]['data']['continuous'], self.datasets[prev_time][self.mode]['data']['continuous']), axis=0)
        self.datasets[time][self.mode]['labels'] = np.concatenate(
            (self.datasets[time][self.mode]['labels'], self.datasets[prev_time][self.mode]['labels']), axis=0)
        self.datasets[time][self.mode]['temperatures'] = np.concatenate(
            (self.datasets[time][self.mode]['temperatures'], self.datasets[prev_time][self.mode]['temperatures']), axis=0)
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
            self.datasets[time][self.mode]['data']['categorical'] = np.concatenate(
                (self.datasets[time][self.mode]['data']['categorical'],
                 self.datasets[prev_time][self.mode]['data']['categorical'][:-last_K_num_samples]), axis=0)
            self.datasets[time][self.mode]['data']['continuous'] = np.concatenate(
                (self.datasets[time][self.mode]['data']['continuous'],
                 self.datasets[prev_time][self.mode]['data']['continuous'][:-last_K_num_samples]), axis=0)
            self.datasets[time][self.mode]['labels'] = np.concatenate(
                (self.datasets[time][self.mode]['labels'], self.datasets[prev_time][self.mode]['labels'][:-last_K_num_samples]), axis=0)
            self.datasets[time][self.mode]['temperatures'] = np.concatenate(
                (self.datasets[time][self.mode]['temperatures'], self.datasets[prev_time][self.mode]['temperatures'][:-last_K_num_samples]), axis=0)
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
        categorical_data = self.datasets[time_idx][self.mode]['data']['categorical'][sel_idx]
        continuous_data = self.datasets[time_idx][self.mode]['data']['continuous'][sel_idx]
        x = {}
        x['categorical'] = torch.FloatTensor(categorical_data).cuda()
        x['continuous'] = torch.FloatTensor(continuous_data).cuda()
        label = self.datasets[time_idx][self.mode]['labels'][sel_idx]

        return x, torch.LongTensor([label]).squeeze(0).unsqueeze(1).cuda()


class Precipitation(PrecipitationBase):
    def __init__(self, args):
        super().__init__(args=args)

    def __getitem__(self, index):
        categorical_data = self.datasets[self.current_time][self.mode]['data']['categorical'][index]
        continuous_data = self.datasets[self.current_time][self.mode]['data']['continuous'][index]
        x = {}
        x['categorical'] = torch.from_numpy(np.array(categorical_data))
        x['continuous'] = torch.from_numpy(continuous_data)
        if self.regression:
            label = self.datasets[self.current_time][self.mode]['temperatures'][index]
            return x, torch.FloatTensor([label])
        else:
            label = self.datasets[self.current_time][self.mode]['labels'][index]
            return x, torch.LongTensor([label])

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])


class PrecipitationGroup(PrecipitationBase):
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
            categorical_data = self.datasets[self.current_time][self.mode]['data']['categorical'][sel_idx]
            continuous_data = self.datasets[self.current_time][self.mode]['data']['continuous'][sel_idx]
            x = {}
            x['categorical'] = torch.from_numpy(np.array(categorical_data))
            x['continuous'] = torch.from_numpy(continuous_data)

            del window
            del sel_time
            del start_idx
            del end_idx

            if self.regression:
                label = self.datasets[self.current_time][self.mode]['temperatures'][sel_idx]
            else:
                label = self.datasets[self.current_time][self.mode]['labels'][sel_idx]

            return x, torch.LongTensor([label]), torch.LongTensor([groupid])

        else:
            categorical_data = self.datasets[self.current_time][self.mode]['data']['categorical'][index]
            continuous_data = self.datasets[self.current_time][self.mode]['data']['continuous'][index]
            x = {}
            x['categorical'] = torch.from_numpy(np.array(categorical_data))
            x['continuous'] = torch.from_numpy(continuous_data)
            if self.regression:
                label = self.datasets[self.current_time][self.mode]['temperatures'][index]
                return x, torch.FloatTensor([label])
            else:
                label = self.datasets[self.current_time][self.mode]['labels'][index]
                return x, torch.LongTensor([label])

    def group_counts(self):
        idx = self.ENV.index(self.current_time)
        return torch.LongTensor([1 for _ in range(min(self.num_groups, idx + 1))])

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])