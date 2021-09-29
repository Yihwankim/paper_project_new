# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

#######################################################################################################################
# import data
df_seoul = pd.read_pickle('data_process/apt_data/seoul_interaction_term.pkl')
# df_seoul = pd.read_csv('data_process/apt_data/seoul_interaction_term.csv')

#######################################################################################################################
# Regression
df_seoul['log_per_Pr'] = np.log(df_seoul['per_Pr'])
df_seoul['log_num'] = np.log(df_seoul['num'])

for
X = sm.add_constant(df_seoul[['old', 'old_sq', 'log_num', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq',
                              'first', 'H2', 'H3', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency', 'dist_high',
                              'dist_sub', 'dist_park']])


