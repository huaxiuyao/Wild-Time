import numpy as np
import os
import pandas as pd
import pickle

from data.utils import Mode

RAW_DATA_FILE = 'arxiv-metadata-oai-snapshot.json'
ID_HELD_OUT = 0.1

def preprocess(args):
    data_file = os.path.join(args.data_dir, RAW_DATA_FILE)
    if not os.path.isfile(data_file):
        raise ValueError(f'{RAW_DATA_FILE} is not in the data directory {args.data_dir}!')

    # Load data frame from json file, group by year
    base_df = pd.read_json(data_file, lines=True)
    # Create a new column containing a paper's primary category
    base_df['category'] = base_df.categories.str.split().str.get(0)
    # Sort by year
    base_df['update_date'] = pd.to_datetime(base_df.update_date)
    base_df = base_df.sort_values(by=['update_date'])
    df_years = base_df.groupby(pd.Grouper(key='update_date', freq='Y'))
    dfs = [group for _, group in df_years]

    years = sorted(pd.unique(pd.DatetimeIndex(base_df['update_date']).year).tolist())
    categories = sorted(pd.unique(base_df['category']).tolist())
    categories_to_classids = {category: classid for classid, category in enumerate(categories)}

    dataset = {}
    for i, year in enumerate(years):
        # Store paper titles and category labels
        dataset[year] = {}
        df_year = dfs[i]
        titles = df_year['title'].str.lower().tolist()
        categories = [categories_to_classids[category] for category in df_year['category']]
        num_samples = len(categories)
        num_train_images = int((1 - ID_HELD_OUT) * num_samples)
        seed_ = np.random.get_state()
        np.random.seed(0)
        idxs = np.random.permutation(np.arange(num_samples))
        np.random.set_state(seed_)
        train_idxs = idxs[:num_train_images].astype(int)
        test_idxs = idxs[num_train_images + 1:].astype(int)
        titles_train = np.array(titles)[train_idxs]
        categories_train = np.array(categories)[train_idxs]
        titles_test_id = np.array(titles)[test_idxs]
        categories_test_id = np.array(categories)[test_idxs]

        dataset[year][Mode.TRAIN] = {}
        dataset[year][Mode.TRAIN]['title'] = titles_train
        dataset[year][Mode.TRAIN]['category'] = categories_train
        dataset[year][Mode.TEST_ID] = {}
        dataset[year][Mode.TEST_ID]['title'] = titles_test_id
        dataset[year][Mode.TEST_ID]['category'] = categories_test_id
        dataset[year][Mode.TEST_OOD] = {}
        dataset[year][Mode.TEST_OOD]['title'] = titles
        dataset[year][Mode.TEST_OOD]['category'] = categories


    preprocessed_data_path = os.path.join(args.data_dir, 'arxiv.pkl')
    pickle.dump(dataset, open(preprocessed_data_path, 'wb'))