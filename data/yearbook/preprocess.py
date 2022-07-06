import os
import pickle
from collections import defaultdict

import numpy as np
from PIL import Image

from data.utils import Mode

RAW_DATA_FOLDER = 'faces_aligned_small_mirrored_co_aligned_cropped_cleaned'
RESOLUTION = 32
ID_HELD_OUT = 0.1

'''
Class Counts:
    M: 17673
    F: 20248

Year Counts:
    [(1905, 5), (1906, 3), (1908, 23), (1909, 32), (1910, 23), (1911, 17), (1912, 12), (1913, 32), (1914, 14),
    (1915, 32), (1916, 18), (1919, 18), (1922, 11), (1923, 47), (1924, 1), (1925, 109), (1926, 12), (1927, 48),
    (1928, 123), (1929, 152), (1930, 154), (1931, 176), (1932, 243), (1933, 272), (1934, 326), (1935, 492),
    (1936, 154), (1937, 307), (1938, 402), (1939, 160), (1940, 402), (1941, 93), (1942, 220), (1943, 704),
    (1944, 858), (1945, 834), (1946, 404), (1947, 379), (1948, 406), (1949, 178), (1950, 387), (1951, 222),
    (1952, 328), (1953, 267), (1954, 581), (1955, 748), (1956, 269), (1957, 233), (1958, 272), (1959, 503),
    (1960, 299), (1961, 367), (1962, 411), (1963, 259), (1964, 313), (1965, 651), (1966, 487), (1967, 1036),
    (1968, 336), (1969, 617), (1970, 828), (1971, 198), (1972, 478), (1973, 313), (1974, 769), (1975, 819),
    (1976, 358), (1977, 542), (1978, 234), (1979, 637), (1980, 602), (1981, 255), (1982, 494), (1983, 404),
    (1984, 1197), (1985, 497), (1986, 273), (1987, 583), (1988, 360), (1989, 777), (1990, 801), (1991, 428),
    (1992, 472), (1993, 223), (1994, 532), (1995, 637), (1996, 417), (1997, 507), (1998, 634), (1999, 656),
    (2000, 458), (2001, 627), (2002, 717), (2003, 411), (2004, 508), (2005, 442), (2006, 70), (2007, 243),
    (2008, 306), (2009, 505), (2010, 521), (2011, 87), (2012, 126), (2013, 493)]
'''

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