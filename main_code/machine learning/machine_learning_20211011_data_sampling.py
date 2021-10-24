# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#######################################################################################################################
# data
df_full = pd.read_pickle('data_process/apt_data/seoul_year_interaction_term.pkl')

# time_dummy 수열로 표현
df_full['time_param'] = df_full['time(H)'].str.slice(start=4)
df_full['time_param'] = pd.to_numeric(df_full['time_param'])

# 난방 방식 label 로 표현
df_full['Heat'] = np.nan
length1 = len(df_full['gu'])
for i in tqdm(range(length1)):
    if df_full['H1'].iloc[i] == 1:
        df_full['Heat'].iloc[i] = 1

for i in tqdm(range(length1)):
    if df_full['H2'].iloc[i] == 1:
        df_full['Heat'].iloc[i] = 2

for i in tqdm(range(length1)):
    if df_full['H3'].iloc[i] == 1:
        df_full['Heat'].iloc[i] = 3

# 구조 더미 변수 label 로 표현
df_full['Type'] = np.nan
length1 = len(df_full['gu'])
for i in tqdm(range(length1)):
    if df_full['T1'].iloc[i] == 1:
        df_full['Type'].iloc[i] = 1

for i in tqdm(range(length1)):
    if df_full['T2'].iloc[i] == 1:
        df_full['Type'].iloc[i] = 2

for i in tqdm(range(length1)):
    if df_full['T3'].iloc[i] == 1:
        df_full['Type'].iloc[i] = 3

df_full.to_pickle('data_process/apt_data/seoul_including_all_variables.pkl')

# df_full = pd.read_pickle('data_process/apt_data/seoul_including_all_variables.pkl')

########################################################################################################################
# edit the variable for random forest regression
# time dummy --> time_param
# h1, h2, h3 --> heat
# t1, t2, t3 --> type
# distance to sth --> lat & long

physical_var = ['per_Pr', 'old', 'old_sq', 'num', 'car_per', 'area', 'room', 'toilet', 'floor',
                'floor_sq', 'first', 'Heat', 'Type', 'C1', 'FAR', 'BC', 'Efficiency',
                'lat', 'long', 'time_param']

'''
district_var = []
length1 = 25
for i in range(length1):
    a = 'GU' + str(i+1)
    district_var.append(a)

time_var = []
length2 = 24
for i in range(length2):
    a = 'Half' + str(i+1)
    time_var.append(a)


independent_var = physical_var + district_var + time_var
'''
df_seoul_full = df_full[physical_var]

df_seoul_full.to_pickle('data_process/apt_data/machine_learning/seoul_full_sample(district+half).pkl')

########################################################################################################################
# 전체 샘플을 train: 8, predict: 2 비율로 나누어두기
# df_seoul_full = df_full.copy()
df_seoul_full = df_seoul_full.sort_values(by=['gu'])
df_seoul_full = df_seoul_full.reset_index(drop='Ture')

df_seoul_80 = df_seoul_full.sample(frac=0.8, random_state=2)  # 399106 개 데이터
df_seoul_20 = df_seoul_full.drop(df_seoul_80.index)

df_seoul_80.to_pickle('data_process/apt_data/machine_learning/seoul_80_sample.pkl')
df_seoul_20.to_pickle('data_process/apt_data/machine_learning/seoul_20_sample.pkl')

########################################################################################################################
df_sample_1000 = df_seoul_80.sample(frac=0.0025056, replace=False, random_state=2)

df_sample_1000 = df_sample_1000.sort_values(by=['time_param'])
df_sample_1000 = df_sample_1000.reset_index(drop='Ture')

# sampling 저장
df_sample_1000.to_pickle('data_process/apt_data/machine_learning/seoul_sampling_1000unit.pkl')
