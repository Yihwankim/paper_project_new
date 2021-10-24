# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

########################################################################################################################
# for hedonic
# data
df_full = pd.read_pickle('data_process/apt_data/seoul_including_all_variables.pkl')
df_train = pd.read_pickle('data_process/apt_data/machine_learning/seoul_80_sample.pkl')
df_test = pd.read_pickle('data_process/apt_data/machine_learning/seoul_20_sample.pkl')

df_full['log_num'] = np.log(df_full['num'])
df_train['log_num'] = np.log(df_train['num'])
df_test['log_num'] = np.log(df_test['num'])

# variable check
physical_var = ['old', 'old_sq', 'log_num', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq', 'first', 'H1',
                'H2', 'H3', 'T1', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency', 'dist_high', 'dist_sub', 'dist_park']

time_dum = []  # 24개 반기 더미
gu_dum = []  # 25개 구 더미
inter_dum = []  # interaction term 600개

len_time = 24
len_gu = 25
# gu_dum.clear()
for i in range(len_gu):
    a = 'GU' + str(i+1)
    gu_dum.append(a)

for i in range(len_time):
    a = 'Half' + str(i+1)
    time_dum.append(a)

for i in range(len_gu):
    for j in range(len_time):
        a = 'i' + str(i+1) + ',' + str(j+1)
        inter_dum.append(a)

independent = physical_var + gu_dum + time_dum + inter_dum
independent.append('per_Pr')

df_full_hedonic = df_full[independent]
df_test_hedonic = df_test[independent]
df_train_hedonic = df_train[independent]

########################################################################################################################
# save the sample

df_full_hedonic.to_pickle('data_process/conclusion/sample/hedonic_full_data.pkl')
df_train_hedonic.to_pickle('data_process/conclusion/sample/hedonic_train_data.pkl')
df_test_hedonic.to_pickle('data_process/conclusion/sample/hedonic_test_data.pkl')


