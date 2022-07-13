import os
import pickle
from collections import defaultdict

import numpy as np
from PIL import Image

from data.utils import Mode

RAW_DATA_FOLDER = 'faces_aligned_small_mirrored_co_aligned_cropped_cleaned'
RESOLUTION = 32
ID_HELD_OUT = 0.1

def preprocess(args):
    np.random.seed(0)
    raw_data_path = os.path.join(args.data_dir, RAW_DATA_FOLDER)
    if not os.path.exists(raw_data_path):
        raise ValueError(f'{RAW_DATA_FOLDER} is not in the data directory {args.data_dir}!')

    path = os.path.join(args.data_dir, RAW_DATA_FOLDER)
    dir_M = os.listdir(f'{path}/M')
    print('num male photos', len(dir_M))
    dir_F = os.listdir(f'{path}/F')
    print('num female photos', len(dir_F))

    images = defaultdict(list)
    labels = defaultdict(list)
    year_counts = {}
    for item in dir_M:
        year = int(item.split('_')[0])
        img = f'{path}/M/{item}'
        if os.path.isfile(img):
            img = Image.open(img)
            img_resize = img.resize((RESOLUTION, RESOLUTION), Image.ANTIALIAS)
            img_resize.save(f'{args.data_dir}/yearbook/{item}', 'PNG')
            images[year].append(np.array(img_resize))
            labels[year].append(0)
            if year not in year_counts.keys():
                year_counts[year] = {}
                year_counts[year]['m'] = 0
                year_counts[year]['f'] = 0
            year_counts[year]['m'] += 1

    for item in dir_F:
        year = int(item.split('_')[0])
        img = f'{path}/F/{item}'
        if os.path.isfile(img):
            img = Image.open(img)
            img_resize = img.resize((RESOLUTION, RESOLUTION), Image.ANTIALIAS)
            img_resize.save(f'{args.data_dir}/yearbook/{item}', 'PNG')
            images[year].append(np.array(img_resize))
            labels[year].append(1)
            if year not in year_counts.keys():
                year_counts[year] = {}
                year_counts[year]['m'] = 0
                year_counts[year]['f'] = 0
            year_counts[year]['f'] += 1

    dataset = {}
    for year in sorted(list(year_counts.keys())):
        # Ignore years 1905 - 1929, start at 1930
        if year < 1930:
            del year_counts[year]
            continue
        dataset[year] = {}
        num_samples = len(labels[year])
        num_train_images = int((1 - ID_HELD_OUT) * num_samples)
        idxs = np.random.permutation(np.arange(num_samples))
        train_idxs = idxs[:num_train_images].astype(int)
        print(train_idxs)
        test_idxs = idxs[num_train_images:].astype(int)
        print(test_idxs)
        train_images = np.array(images[year])[train_idxs]
        train_labels = np.array(labels[year])[train_idxs]
        test_images = np.array(images[year])[test_idxs]
        test_labels = np.array(labels[year])[test_idxs]
        dataset[year][Mode.TRAIN] = {}
        dataset[year][Mode.TRAIN]['images'] = np.stack(train_images, axis=0) / 255.0
        dataset[year][Mode.TRAIN]['labels'] = np.array(train_labels)
        dataset[year][Mode.TEST_ID] = {}
        dataset[year][Mode.TEST_ID]['images'] = np.stack(test_images, axis=0) / 255.0
        dataset[year][Mode.TEST_ID]['labels'] = np.array(test_labels)
        dataset[year][Mode.TEST_OOD] = {}
        dataset[year][Mode.TEST_OOD]['images'] = np.stack(images[year], axis=0) / 255.0
        dataset[year][Mode.TEST_OOD]['labels'] = np.array(labels[year])

    preprocessed_data_path = os.path.join(args.data_dir, 'yearbook.pkl')
    pickle.dump(dataset, open(preprocessed_data_path, 'wb'))
    np.random.seed(args.random_seed)