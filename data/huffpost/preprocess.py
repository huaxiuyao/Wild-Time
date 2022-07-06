import numpy as np
import os
import pandas as pd
import pickle

from data.utils import Mode

RAW_DATA_FILE = 'News_Category_Dataset_v2.json'
ID_HELD_OUT = 0.1

'''
{'BLACK VOICES': 0, 'BUSINESS': 1, 'COMEDY': 2, 'CRIME': 3, 'ENTERTAINMENT': 4, 'IMPACT': 5, 'QUEER VOICES': 6, 'SCIENCE': 7, 'SPORTS': 8, 'TECH': 9, 'TRAVEL': 10}
'''

def preprocess(args):
    raw_data_path = os.path.join(args.data_dir, RAW_DATA_FILE)
    if not os.path.isfile(raw_data_path):
        raise ValueError(f'{RAW_DATA_FILE} is not in the data directory {args.data_dir}!')

    # Load data frame from json file, group by year
    base_df = pd.read_json(raw_data_path, lines=True)
    base_df = base_df.sort_values(by=['date'])
    df_years = base_df.groupby(pd.Grouper(key='date', freq='Y'))
    dfs = [group for _, group in df_years]
    years = [2012, 2013, 2014, 2015, 2016, 2017, 2018]

    # Identify class ids that appear in all years 2012 - 2018
    categories_to_classids = {category: classid for classid, category in
                              enumerate(sorted(base_df['category'].unique()))}
    classids_to_categories = {v: k for k, v in categories_to_classids.items()}
    classids = []
    num_classes = len(categories_to_classids.values())
    for classid in range(num_classes):
        class_count = 0
        for i, year in enumerate(years):
            year_classids = [categories_to_classids[category] for category in dfs[i]['category']]
            if classid in year_classids:
                class_count += 1
        if class_count == len(years):
            classids.append(classid)

    # Re-index the class ids that appear in all years 2012 - 2018 and store them
    classids_to_categories = {i: classids_to_categories[classid] for i, classid in enumerate(classids)}
    categories_to_classids = {v: k for k, v in classids_to_categories.items()}
    print(categories_to_classids)

    dataset = {}
    for i, year in enumerate(years):
        # Store news headlines and category labels
        dataset[year] = {}
        df_year = dfs[i]
        df_year = df_year[df_year['category'].isin(categories_to_classids.keys())]
        headlines = df_year['headline'].str.lower().tolist()
        categories = [categories_to_classids[category] for category in df_year['category']]

        num_samples = len(categories)
        num_train_images = int((1 - ID_HELD_OUT) * num_samples)
        seed_ = np.random.get_state()
        np.random.seed(0)
        idxs = np.random.permutation(np.arange(num_samples))
        np.random.set_state(seed_)
        train_idxs = idxs[:num_train_images].astype(int)
        test_idxs = idxs[num_train_images + 1:].astype(int)
        headlines_train = np.array(headlines)[train_idxs]
        categories_train = np.array(categories)[train_idxs]
        headlines_test_id = np.array(headlines)[test_idxs]
        categories_test_id = np.array(categories)[test_idxs]

        dataset[year][Mode.TRAIN] = {}
        dataset[year][Mode.TRAIN]['headline'] = headlines_train
        dataset[year][Mode.TRAIN]['category'] = categories_train
        dataset[year][Mode.TEST_ID] = {}
        dataset[year][Mode.TEST_ID]['headline'] = headlines_test_id
        dataset[year][Mode.TEST_ID]['category'] = categories_test_id
        dataset[year][Mode.TEST_OOD] = {}
        dataset[year][Mode.TEST_OOD]['headline'] = headlines
        dataset[year][Mode.TEST_OOD]['category'] = categories

    preprocessed_data_path = os.path.join(args.data_dir, 'huffpost.pkl')
    pickle.dump(dataset, open(preprocessed_data_path, 'wb'))