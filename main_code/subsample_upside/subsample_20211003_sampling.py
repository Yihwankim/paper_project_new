# sub-sample sampling

# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

#######################################################################################################################
# 반기 단위로 데이터 쪼개기

df_seoul = pd.read_pickle('data_process/apt_data/seoul_full_variable.pkl')

conditionlist = [(df_seoul['time'] < 3),
                 (df_seoul['time'] >= 3) & (df_seoul['time'] < 5),
                 (df_seoul['time'] >= 5) & (df_seoul['time'] < 7),
                 (df_seoul['time'] >= 7) & (df_seoul['time'] < 9),
                 (df_seoul['time'] >= 9) & (df_seoul['time'] < 11),
                 (df_seoul['time'] >= 11) & (df_seoul['time'] < 13),
                 (df_seoul['time'] >= 13) & (df_seoul['time'] < 15),
                 (df_seoul['time'] >= 15) & (df_seoul['time'] < 17),
                 (df_seoul['time'] >= 17) & (df_seoul['time'] < 19),
                 (df_seoul['time'] >= 19) & (df_seoul['time'] < 21),
                 (df_seoul['time'] >= 21) & (df_seoul['time'] < 23),
                 (df_seoul['time'] >= 23) & (df_seoul['time'] < 25),
                 (df_seoul['time'] >= 25) & (df_seoul['time'] < 27),
                 (df_seoul['time'] >= 27) & (df_seoul['time'] < 29),
                 (df_seoul['time'] >= 29) & (df_seoul['time'] < 31),
                 (df_seoul['time'] >= 31) & (df_seoul['time'] < 33),
                 (df_seoul['time'] >= 33) & (df_seoul['time'] < 35),
                 (df_seoul['time'] >= 35) & (df_seoul['time'] < 37),
                 (df_seoul['time'] >= 37) & (df_seoul['time'] < 39),
                 (df_seoul['time'] >= 39) & (df_seoul['time'] < 41),
                 (df_seoul['time'] >= 41) & (df_seoul['time'] < 43),
                 (df_seoul['time'] >= 43) & (df_seoul['time'] < 45),
                 (df_seoul['time'] >= 45) & (df_seoul['time'] < 47),
                 (df_seoul['time'] >= 47) & (df_seoul['time'] < 49)]

Half_dummy = []
for i in range(1, 25):
    half = 'half' + str(i)
    Half_dummy.append(half)

choicelist = Half_dummy

df_seoul['time(H)'] = np.select(conditionlist, choicelist, default='wrong')

# 기존의 21년 data 제거하기
num = df_seoul['time(H)'] != 'wrong'
df = df_seoul[num]
# df['time(H)'] = pd.to_numeric(df['time(H)'])

# year dummy 만들기
len_half = 24
for i in tqdm(range(len_half)):
    df['Half' + str(i + 1)] = 0

length = len(df['gu'])
for k in tqdm(range(len_half)):
    for i in range(length):
        if df['time(H)'].iloc[i] == 'half' + str(k + 1):
            df['Half' + str(k + 1)].iloc[i] = 1

# half dummy 갯수 확인
number_data = []
for i in range(len_half):
    a = np.sum(df['Half' + str(i + 1)])
    number_data.append(a)
num_half_dummy = np.sum(number_data)


# interaction term 만들기: 분기와 구 버전
len_gu = 25
len_time = 24
# 시간과 지역의 interaction term 은 600 개
# interaction_term : i25,20 ==> 2018년 2분기에 거래된 송파구(25)의 data
for i in tqdm(range(len_gu)):
    for j in range(len_time):
        df['i' + str(i+1) + ',' + str(j+1)] = df['Half' + str(j+1)] * df['GU' + str(i+1)]

# data 갯수 확인
number_data = []
for i in tqdm(range(len_gu)):
    for j in range(len_time):
        a = np.sum(df['i' + str(i+1) + ',' + str(j+1)])
        number_data.append(a)
num_interaction = np.sum(number_data)

index = []
for i in tqdm(range(len_gu)):
    for j in range(len_time):
        a = 'i' + str(i+1) + ',' + str(j+1)
        index.append(a)

df_check = pd.DataFrame(data=number_data, index=index)

df_check.to_excel('data_process/apt_data/descriptive_interactionterm.xlsx')
#index
#number_data
######################################################################################################################
# pickle로 저장
df.to_pickle('data_process/apt_data/seoul_year_interaction_term.pkl')
print(1)