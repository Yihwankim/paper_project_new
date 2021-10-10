# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#######################################################################################################################
# import data
df = pd.read_pickle('data_process/apt_data/seoul_year_interaction_term.pkl')

# speculative dummy
# 최초 투기지정 지역 11개구, 2차 투기지정 지역 4개구, 미지정 10개구
# 각각 Fist_spec, Second_spec, Non_spec

length1 = 11
first_spec = ['강남구', '서초구', '송파구', '강동구', '용산구', '성동구', '노원구', '마포구', '양천구', '영등포구', '강서구']

length2 = 4
second_spec = ['종로구', '중구', '동대문구', '동작구']

length3 = 10
non_spec = ['강북구', '광진구', '도봉구', '성북구', '중랑구', '서대문구', '은평구', '관악구', '구로구', '금천구']

# first_speculative
conditionlist = []
for i in range(length1):
    a = df['gu'] == first_spec[i]
    conditionlist.append(a)
for j in range(length2):
    b = df['gu'] == second_spec[j]
    conditionlist.append(b)
for k in range(length3):
    c = df['gu'] == non_spec[k]
    conditionlist.append(c)

choicelist = []
for i in range(length1):
    a = 'first_spec'
    choicelist.append(a)
for j in range(length2):
    b = 'second_spec'
    choicelist.append(b)
for k in range(length3):
    c = 'non_spec'
    choicelist.append(c)

df['spec_dummy'] = np.select(conditionlist, choicelist, default='wrong')

# dummy variable
df['First_spec'] = 0
df['Second_spec'] = 0
df['Non_spec'] = 0

length = len(df['gu'])
for k in range(length):
    if df['spec_dummy'].iloc[k] == 'first_spec':
        df['First_spec'].iloc[k] = 1

for k in range(length):
    if df['spec_dummy'].iloc[k] == 'second_spec':
        df['Second_spec'].iloc[k] = 1

for k in range(length):
    if df['spec_dummy'].iloc[k] == 'non_spec':
        df['Non_spec'].iloc[k] = 1

a = np.sum(df['First_spec']) + np.sum(df['Second_spec']) + np.sum(df['Non_spec'])

'''
num = df['spec_dummy'] == 'wrong'
df_wrong = df[num]
df_wrong['gu'].iloc[1]
'''

# DID
# 2017. 8. 3 --> Half18, D35
# spec_time
conditionlist = [df['time'] < 35]
choicelist = ['before']
df['spec_time'] = np.select(conditionlist, choicelist, default='after')

df['after'] = 0
for k in range(length):
    if df['spec_time'].iloc[k] == 'after':
        df['after'].iloc[k] = 1

# df.to_pickle('data_process/apt_data/for_did_speculative.pkl')
########################################################################################################################












