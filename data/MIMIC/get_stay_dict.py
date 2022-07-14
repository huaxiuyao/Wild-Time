"""
Reference: https://github.com/Google-Health/records-research/tree/master/graph-convolutional-transformer
"""

import pickle

class MIMICStay:

    def __init__(self,
                 icu_id,
                 icu_timestamp,
                 mortality,
                 readmission,
                 age,
                 gender,
                 ethnicity):
        self.icu_id = icu_id    # str
        self.icu_timestamp = icu_timestamp  # int
        self.mortality = mortality  # bool, end of icu stay mortality
        self.readmission = readmission  # bool, 15-day icu readmission
        self.age = age  # int
        self.gender = gender  # str
        self.ethnicity = ethnicity  # str

        self.diagnosis = []     # list of tuples (timestamp in min (int), diagnosis (str))
        self.treatment = []     # list of tuples (timestamp in min (int), treatment (str))

    def __repr__(self):
        return f'MIMIC ID-{self.icu_id}, mortality-{self.mortality}, readmission-{self.readmission}'


def get_stay_dict():
    mimic_dict = {}
    input_path = './data/MIMIC/processed/mimic4/data.csv'
    fboj = open(input_path)
    name_list = fboj.readline().strip().split(',')
    for eachline in fboj:
        t=eachline.strip().split(',')
        tempdata={eachname: t[idx] for idx, eachname in enumerate(name_list)}
        mimic_value = MIMICStay(icu_id=tempdata['hadm_id'],
                                 icu_timestamp=tempdata['real_admit_year'],
                                 mortality=tempdata['mortality'],
                                 readmission=tempdata['readmission'],
                                 age=tempdata['age'],
                                 gender=tempdata['gender'],
                                 ethnicity=tempdata['ethnicity'])
        mimic_value.diagnosis = tempdata['diagnoses'].split(' <sep> ')
        mimic_value.treatment = tempdata['procedure'].split(' <sep> ')
        mimic_dict[tempdata['hadm_id']]=mimic_value

    pickle.dump(mimic_dict, open('./Data/mimic_stay_dict.pkl', 'wb'))
