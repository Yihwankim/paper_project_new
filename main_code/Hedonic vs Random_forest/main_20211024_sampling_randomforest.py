# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

########################################################################################################################
# for random_forest
df_train = pd.read_pickle('data_process/apt_data/machine_learning/seoul_80_sample.pkl')
df_test = pd.read_pickle('data_process/apt_data/machine_learning/seoul_20_sample.pkl')

without_dummy = ['per_Pr', 'old', 'old_sq', 'num', 'car_per', 'area', 'room', 'toilet', 'floor',
                 'floor_sq', 'first', 'Heat', 'Type', 'C1', 'FAR', 'BC', 'Efficiency',
                 'lat', 'long', 'time_param']

distance = ['dist_high', 'dist_sub', 'dist_park']

len_gu = 25
gu_dum = []  # 25개 구 더미
for i in range(len_gu):
    a = 'GU' + str(i+1)
    gu_dum.append(a)

with_dummy = without_dummy + gu_dum + distance
without_distance = without_dummy + gu_dum

df_train_rfr_all = df_train[with_dummy]
df_test_rfr_all = df_test[with_dummy]

df_train_rfr_no_distance = df_train[without_distance]
df_test_rfr_no_distance = df_test[without_distance]

df_train_rfr_without = df_train[without_dummy]
df_test_rfr_without = df_test[without_dummy]

########################################################################################################################
df_train_rfr_all.to_pickle('data_process/conclusion/sample/rfr_all_train_data.pkl')
df_test_rfr_all.to_pickle('data_process/conclusion/sample/rfr_all_test_data.pkl')

df_train_rfr_no_distance.to_pickle('data_process/conclusion/sample/rfr_no_distance_train_data.pkl')
df_test_rfr_no_distance.to_pickle('data_process/conclusion/sample/rfr_no_distance_test_data.pkl')

df_train_rfr_without.to_pickle('data_process/conclusion/sample/rfr_without_train_data.pkl')
df_test_rfr_without.to_pickle('data_process/conclusion/sample/rfr_without_test_data.pkl')
