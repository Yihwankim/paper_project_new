# 필요없어진 파일 제거요망

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
df_seoul = df_seoul.dropna()

# Regression
df_seoul['log_per_Pr'] = np.log(df_seoul['per_Pr'])
df_seoul['log_num'] = np.log(df_seoul['num'])

indep_var = ['old', 'old_sq', 'log_num', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq', 'first', 'H2',
             'H3', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency', 'dist_high', 'dist_sub', 'dist_park']

# dummy variable 만들기
gu_dum = []
time_dum = []
inter_dum = []

len_gu = 25  # i
len_time = 49  # j
for i in range(len_gu):
    a = 'GU' + str(i+1)
    gu_dum.append(a)

for i in range(len_time):
    b = 'D' + str(i+1)
    time_dum.append(b)

for i in tqdm(range(len_gu)):
    for j in range(len_time):
        c = 'i' + str(i+1) + ',' + str(j+1)
        inter_dum.append(c)

independent = indep_var + gu_dum + time_dum + inter_dum

X = sm.add_constant(df_seoul[independent])
Y = df_seoul['log_per_Pr']

ols_model = sm.OLS(Y, X.values)
df_full_res = ols_model.fit()
df_full_result = df_full_res.summary(xname=X.columns.tolist())


