import argparse
import bisect
import numpy as np
import os
import pandas as pd
import pickle
from datetime import datetime
from sklearn.utils import shuffle

from data.utils import Mode

RAW_DATA_FILE = 'full-dataset/shifts_precipitation.csv'
ID_HELD_OUT = 0.1


class Config():
    '''
    Define Configuration for partioning data
    '''

    def __init__(self, seed=1):
        '''
        time_splits: fractions associated with TRAIN, GAP, DEV_OUT, EVAL_OUT (split on time)
        climate_splits: number of climates kept for TRAIN, DEV_OUT, EVAL_OUT (from above time splits)
        in_domain_splits: Separation TRAIN time and specified climate segment block into TRAIN, DEV_IN, EVAL_IN
        eval_dev_overlap: Flag if TRUE, EVAL_OUT climates kept include the DEV_OUT climates kept.
        '''
        self.seed = seed


class Partitioner():
    '''
    Requires a block of data and partitions it into
    the following disjoint subsets:
    1) train.csv:
                    Data for training.

    2) dev_in.csv:
                    Development data from the same domain
                    in time and climate as of the train.csv
                    data.
    3) eval_in.csv:
                    Evaluation data from the same domain
                    in time and climate as of the train.csv
                    data.
    4) dev_out.csv:
                    Data distributionally shifted in time and climate
                    from train.csv.
    5) eval_out.csv:
                    Data further distributionally shifted in climate
                    and different time frame from train.csv and dev_out.csv.
                    Can be configured to have overlap in climates
                    with dev_out.csv.
    If no_meta == True, a further set of files will be generated:
    6) dev_in_no_meta.csv:
                    Same as dev_in.csv with meta data (first 6 features)
                    removed.
    7) eval_in_no_meta.csv:
                    Same as eval_in.csv with meta data (first 6 features)
                    removed.
    8) dev_out_no_meta.csv:
                    Same as dev_out.csv with meta data (first 6 features)
                    removed.

    9) eval_out_no_meta.csv:
                    Same as eval_out.csv with meta data (first 6 features)
                    removed.
    '''

    def __init__(self, data_path, config=Config()):
        '''
        unique labels in raw data for precipitation class [ 0. 10. 11. 12. 13. 20. 21. 22. 23.]
        rewrite as [0 1 2 3 4 5 6 7 8 9]
        '''
        self.config = config
        # Read in the raw data
        chunksize = 10 ** 6
        chunks = []
        num_chunks = 0
        reader = pd.read_csv(data_path, chunksize=chunksize)
        for chunk in reader:
            print(f'chunk {num_chunks} read')
            chunks.append(chunk)
            num_chunks += 1

        self.df = pd.concat(chunks)
        # Partition the data by time segments
        self.dfs_to_save = {}
        self._split_by_time()
        # Add dummy samples for unrepresented classifcation classes
        for taskid in self.dfs_to_save.keys():
            self.dfs_to_save[taskid] = self._add_dummy(self.dfs_to_save[taskid])

    def _split_by_time(self):
        """
        Partition the data into the main time segments.
        """
        # Sort all data in time order
        self.df = self.df.sort_values(by=['fact_time'])
        print('total rows', self.df.shape[0])

        # Convert timestamp to date
        self.df.fact_time = self.df.fact_time.apply(lambda x: datetime.fromtimestamp(x))
        df_months = self.df.groupby(pd.Grouper(key='fact_time', freq='M'))
        self.dfs = [group for _, group in df_months]
        num_nans = 0
        for i, df in enumerate(self.dfs):
            num_nans += self.dfs[i].shape[0] - self.dfs[i].dropna().shape[0]
            self.dfs_to_save[i] = self.dfs[i].dropna()
        print('num rows with NaN', num_nans)
        del df_months

    def _add_dummy(self, df_to_modify):
        '''
        Add dummy data for missing precipitation classes in df.
        '''
        # Identify list of all classification classes
        classes_to_check = set(list(self.df['fact_cwsm_class']))
        # Find the average of all data rows
        avg_row = df_to_modify.mean(axis=0)
        # Append averaged row for each classification class not present in the data
        for precip_class in classes_to_check:
            if precip_class not in df_to_modify['fact_cwsm_class']:
                print("Dummy added to training", precip_class)
                ind = len(df_to_modify)
                df_to_modify.loc[ind] = avg_row
                df_to_modify.at[ind, 'fact_cwsm_class'] = precip_class
        return df_to_modify

    def save(self, save_path):
        """
        Save all relevant data split files.
        """

        # Save all files
        precipitation_classes = {0.: 0, 10.: 1, 11.: 2, 12.: 3, 13.: 4, 20.: 5, 21.: 6, 22.: 7, 23.: 8}
        precipitation_dataset = {}
        for taskid in self.dfs_to_save.keys():
            print(taskid, self.dfs_to_save[taskid].shape[0])
            # Skip month 0 which has few data points
            if taskid == 0:
                continue
            data = self.dfs_to_save[taskid].iloc[:, 3:]  # first 3 columns are meta dat
            categorical_data = data['wrf_available']
            continuous_data = data.iloc[:, data.columns != 'wrf_available']
            precipitation_labels = self.dfs_to_save[taskid]['fact_cwsm_class'].apply(lambda x: precipitation_classes[x])
            temperatures = self.dfs_to_save[taskid]['fact_temperature']

            num_samples = len(precipitation_labels)
            num_train_images = int((1 - ID_HELD_OUT) * num_samples)
            seed_ = np.random.get_state()
            np.random.seed(0)
            idxs = np.random.permutation(np.arange(num_samples))
            np.random.set_state(seed_)
            train_idxs = idxs[:num_train_images].astype(int)
            test_idxs = idxs[num_train_images + 1:].astype(int)

            precipitation_dataset[taskid] = {}
            precipitation_dataset[taskid][Mode.TRAIN] = {}
            precipitation_dataset[taskid][Mode.TRAIN]['data'] = {}
            precipitation_dataset[taskid][Mode.TRAIN]['data']['categorical'] = np.array(categorical_data.to_numpy())[
                train_idxs]
            precipitation_dataset[taskid][Mode.TRAIN]['data']['continuous'] = np.array(continuous_data.to_numpy())[train_idxs]
            precipitation_dataset[taskid][Mode.TRAIN]['labels'] = np.array(precipitation_labels.to_numpy())[train_idxs]
            precipitation_dataset[taskid][Mode.TRAIN]['temperatures'] = np.array(temperatures.to_numpy())[train_idxs]
            precipitation_dataset[taskid][Mode.TEST_ID] = {}
            precipitation_dataset[taskid][Mode.TEST_ID]['data'] = {}
            precipitation_dataset[taskid][Mode.TEST_ID]['data']['categorical'] = np.array(categorical_data.to_numpy())[
                test_idxs]
            precipitation_dataset[taskid][Mode.TEST_ID]['data']['continuous'] = np.array(continuous_data.to_numpy())[
                test_idxs]
            precipitation_dataset[taskid][Mode.TEST_ID]['labels'] = np.array(precipitation_labels.to_numpy())[test_idxs]
            precipitation_dataset[taskid][Mode.TEST_ID]['temperatures'] = np.array(temperatures.to_numpy())[test_idxs]
            precipitation_dataset[taskid][Mode.TEST_OOD] = {}
            precipitation_dataset[taskid][Mode.TEST_OOD]['data'] = {}
            precipitation_dataset[taskid][Mode.TEST_OOD]['data']['categorical'] = np.array(categorical_data.to_numpy())
            precipitation_dataset[taskid][Mode.TEST_OOD]['data']['continuous'] = np.array(continuous_data.to_numpy())
            precipitation_dataset[taskid][Mode.TEST_OOD]['labels'] = precipitation_labels.to_numpy()
            precipitation_dataset[taskid][Mode.TEST_OOD]['temperatures'] = temperatures.to_numpy()

        with open(save_path, 'wb') as f:
            pickle.dump(precipitation_dataset, f)


def preprocess(args):
    '''Partitions tabular precipitation data for distributional shift'''

    # Load the configurable parameters
    config = Config()
    print(config)

    # Partition the raw precipitation data
    data_path = os.path.join(args.data_dir, RAW_DATA_FILE)
    partitioner = Partitioner(data_path, config)

    # Print number of data points in each data split
    for name, df in partitioner.dfs_to_save.items():
        print(name, df.shape[0])

    print()
    # Save all files
    preprocessed_data_path = os.path.join(args.data_dir, 'precipitation.pkl')
    partitioner.save(preprocessed_data_path)
