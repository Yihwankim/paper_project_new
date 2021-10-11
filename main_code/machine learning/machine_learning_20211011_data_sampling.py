# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#######################################################################################################################
# data
df_full = pd.read_pickle('data_process/apt_data/seoul_year_interaction_term.pkl')

physical_var = ['gu', 'dong', 'per_Pr', 'old', 'old_sq', 'num', 'car_per', 'area', 'room', 'toilet', 'floor',
                'floor_sq', 'first', 'H1', 'H2', 'H3', 'T1', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency',
                'dist_high', 'dist_sub', 'dist_park']

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

df_seoul_full = df_full[independent_var]

# df_seoul_full.to_pickle('data_process/apt_data/machine_learning/seoul_full_sample(district+half).pkl')

########################################################################################################################
# 전체 샘플을 train: 8, predict: 2 비율로 나누어두기
df_seoul_full = df_seoul_full.sort_values(by=['gu'])
df_seoul_full = df_seoul_full.reset_index(drop='Ture')

df_seoul_80 = df_seoul_full.sample(frac=0.8, random_state=2)  # 399106 개 데이터
df_seoul_20 = df_seoul_full.drop(df_seoul_80.index)

# 무작위 샘플링이 잘 되었는지 체크
a = [np.sum(df_seoul_80[district_var])]
b = [np.sum(df_seoul_20[district_var])]

df_seoul_80.to_pickle('data_process/apt_data/machine_learning/seoul_80_sample(district+half).pkl')
df_seoul_20.to_pickle('data_process/apt_data/machine_learning/seoul_20_sample(district+half).pkl')

########################################################################################################################
df_sample_1000 = df_seoul_80.sample(frac=0.0025056, replace=False, random_state=2)

# data 갯수 확인, 각 더미변수에 해당하는 값이 2494개가 나와야함
np.sum([np.sum(df_sample_1000[time_var])])
np.sum([np.sum(df_sample_1000[district_var])])
np.sum([np.sum(df_sample_1000[['H1', 'H2', 'H3']])])
np.sum([np.sum(df_sample_1000[['T1', 'T2', 'T3']])])

df_sample_1000 = df_sample_1000.sort_values(by=['gu'])
df_sample_1000 = df_sample_1000.reset_index(drop='Ture')
df_sample_80 = df_sample_1000.sample(frac=0.8, random_state=2)
df_sample_20 = df_sample_1000.drop(df_sample_80.index)

# sampling 저장
df_sample_1000.to_pickle('data_process/apt_data/machine_learning/seoul_sampling_1000unit.pkl')
df_sample_80.to_pickle('data_process/apt_data/machine_learning/seoul_sampling_800unit.pkl')
df_sample_20.to_pickle('data_process/apt_data/machine_learning/seoul_sampling_200unit.pkl')