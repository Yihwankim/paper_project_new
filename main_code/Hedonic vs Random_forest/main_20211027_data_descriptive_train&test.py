# 목표
# 전체 샘플에 대해서 interaction term 을 포함한 결과값들에 대한 기술통계량 값 서술

# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

########################################################################################################################
# summary
df_train = pd.read_pickle('data_process/apt_data/machine_learning/seoul_80_sample.pkl')
df_test = pd.read_pickle('data_process/apt_data/machine_learning/seoul_20_sample.pkl')

df_train = df_train.dropna()
df_test = df_test.dropna()

indep_var = ['old', 'old_sq', 'num', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq', 'first', 'H1', 'H2',
             'H3', 'T1', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency', 'dist_high', 'dist_sub', 'dist_park']

gu_dum = []
time_dum = []
inter_dum = []

len_gu = 25  # i
len_time = 24  # j

'''for i in range(len_gu):
    a = 'GU' + str(i+1)
    gu_dum.append(a)

for i in range(len_time):
    b = 'Half' + str(i+1)
    time_dum.append(b)'''

for i in range(len_gu):
    for j in range(len_time):
        a = 'i' + str(i + 1) + ',' + str(j + 1)
        inter_dum.append(a)

independent_part = indep_var + gu_dum + time_dum + inter_dum

df_train_des = df_train[independent_part]
df_test_des = df_test[independent_part]

summary_train = df_train_des.describe()
summary_test = df_test_des.describe()

summary_train.to_excel('data_process/conclusion/regression_result/descriptive_summary_train.xlsx')
summary_test.to_excel('data_process/conclusion/regression_result/descriptive_summary_test.xlsx')
