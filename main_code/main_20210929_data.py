# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm

### 수정요망: 쥬피터에서 완전 실습 후 수정할 것

#######################################################################################################################
# import data
df_all = pd.read_excel('data_process/apt_data/seoul_full_variable.xlsx', header=0, skipfooter=0)

df_all.to_pickle('data_process/apt_data/seoul_variable_district_dummy.pkl')
#######################################################################################################################
# interaction term 만들기
# 지역구 개수 25개
# 시간 더미변수 개수 49개
df_inter = df_all

len_gu = 25  # i
len_time = 49  # j
# 시간과 지역의 interaction term 은 1,225개
len_inter = 1225
for i in tqdm(range(len_gu)):
    for j in range(len_time):
        df_inter['i' + str(i+1) + ',' + str(j+1)] = df_inter['GU' + str(i+1)] * df_inter['D' + str(j+1)]

number_data = []
for i in tqdm(range(len_gu)):
    for j in range(len_time):
        a = np.sum(df_inter['i' + str(i + 1) + ',' + str(j + 1)])
        number_data.append(a)

np.sum(number_data)  # 506,509와 일치하는지 확인

# 피클 파일로 저장
df_inter.to_pickle('data_process/apt_data/seoul_interaction_term.pkl')

# CSV 파일로 저장
df_inter.to_csv('data_process/apt_data/seoul_interaction_term.csv')
#######################################################################################################################
# Regression
df_inter['log_per_Pr'] = np.log(df_inter['per_Pr'])
df_inter['log_num'] = np.log(df_inter['num'])

X = sm.add_constant(df_inter[['old', 'old_sq', 'log_num', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq',
                              'first', 'H2', 'H3', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency', 'dist_high',
                              'dist_sub', 'dist_park']])






