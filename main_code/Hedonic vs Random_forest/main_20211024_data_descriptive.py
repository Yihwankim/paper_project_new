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

df_full = pd.read_pickle('data_process/apt_data/seoul_including_all_variables.pkl')
df_full = df_full.dropna()

indep_var = ['old', 'old_sq', 'num', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq', 'first', 'H1', 'H2',
             'H3', 'T1', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency', 'dist_high', 'dist_sub', 'dist_park']

gu_dum = []
time_dum = []
inter_dum = []

len_gu = 25  # i
len_time = 24  # j
for i in range(len_gu):
    a = 'GU' + str(i+1)
    gu_dum.append(a)

for i in range(len_time):
    b = 'Half' + str(i+1)
    time_dum.append(b)

for i in range(len_gu):
    for j in range(len_time):
        a = 'i' + str(i + 1) + ',' + str(j + 1)
        inter_dum.append(a)

independent_part = indep_var + gu_dum + time_dum + inter_dum

df_seoul = df_full[independent_part]

summary = df_seoul.describe()
sum_variable = np.sum(df_seoul)

summary.to_excel('data_process/conclusion/regression_result/descriptive_summary.xlsx')
sum_variable.to_excel('data_process/conclusion/regression_result/descriptive_sum.xlsx')