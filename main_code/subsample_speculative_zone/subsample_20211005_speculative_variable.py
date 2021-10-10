
# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

#######################################################################################################################
df_spec = pd.read_pickle('data_process/apt_data/spec_variable.pkl')
df_non_spec = pd.read_pickle('data_process/apt_data/non_spec_variable.pkl')

'''def the_counting_number(length, df_name, column):
    number_data = []
    for i in range(length):
        a = np.sum(df_name[str(column) + str(i + 1)])
        number_data.append(a)
    print(np.sum(number_data))

the_counting_number(length1, df_spec, spec)  '''

#######################################################################################################################
# 최조 지정 투기지역: 11개
spec = ['강남구', '서초구', '송파구', '강동구', '용산구', '성동구', '노원구', '마포구', '양천구', '영등포구', '강서구']

# 지역별 더미변수 생성
length1 = 11
for i in tqdm(range(length1)):
    df_spec['spec' + str(i+1)] = 0

length2 = len(df_spec['gu'])

for k in tqdm(range(length1)):
    for i in range(length2):
        if df_spec['gu'].iloc[i] == spec[k]:
            df_spec['spec' + str(k+1)].iloc[i] = 1

# 갯수 확인
number_data = []
for i in range(length1):
    a = np.sum(df_spec['spec' + str(i + 1)])
    number_data.append(a)
num_count = np.sum(number_data)
print(num_count)

# half_dummy 생성하기
conditionlist = [(df_spec['time'] < 3),
                 (df_spec['time'] >= 3) & (df_spec['time'] < 5),
                 (df_spec['time'] >= 5) & (df_spec['time'] < 7),
                 (df_spec['time'] >= 7) & (df_spec['time'] < 9),
                 (df_spec['time'] >= 9) & (df_spec['time'] < 11),
                 (df_spec['time'] >= 11) & (df_spec['time'] < 13),
                 (df_spec['time'] >= 13) & (df_spec['time'] < 15),
                 (df_spec['time'] >= 15) & (df_spec['time'] < 17),
                 (df_spec['time'] >= 17) & (df_spec['time'] < 19),
                 (df_spec['time'] >= 19) & (df_spec['time'] < 21),
                 (df_spec['time'] >= 21) & (df_spec['time'] < 23),
                 (df_spec['time'] >= 23) & (df_spec['time'] < 25),
                 (df_spec['time'] >= 25) & (df_spec['time'] < 27),
                 (df_spec['time'] >= 27) & (df_spec['time'] < 29),
                 (df_spec['time'] >= 29) & (df_spec['time'] < 31),
                 (df_spec['time'] >= 31) & (df_spec['time'] < 33),
                 (df_spec['time'] >= 33) & (df_spec['time'] < 35),
                 (df_spec['time'] >= 35) & (df_spec['time'] < 37),
                 (df_spec['time'] >= 37) & (df_spec['time'] < 39),
                 (df_spec['time'] >= 39) & (df_spec['time'] < 41),
                 (df_spec['time'] >= 41) & (df_spec['time'] < 43),
                 (df_spec['time'] >= 43) & (df_spec['time'] < 45),
                 (df_spec['time'] >= 45) & (df_spec['time'] < 47),
                 (df_spec['time'] >= 47) & (df_spec['time'] < 49)]

half_dummy = []
for i in range(1, 25):
    half = 'half' + str(i)
    half_dummy.append(half)

choicelist = half_dummy

df_spec['time(H)'] = np.select(conditionlist, choicelist, default='wrong')

# 반기별로 나누게 되므로 짝이없는 49번째 분기 즉 21년 1분기 데이터는 제거
num = df_spec['time(H)'] != 'wrong'
df1 = df_spec[num]

# 이제 반기 더미변수 생성
len_half = 24
for i in tqdm(range(len_half)):
    df1['Half' + str(i + 1)] = 0

length = len(df1['gu'])
for k in tqdm(range(len_half)):
    for i in range(length):
        if df1['time(H)'].iloc[i] == 'half' + str(k + 1):
            df1['Half' + str(k + 1)].iloc[i] = 1

# half dummy 갯수 확인
number_data = []
for i in range(len_half):
    a = np.sum(df1['Half' + str(i + 1)])
    number_data.append(a)
num_half_dummy = np.sum(number_data)

# interaction term 만들기: 분기와 구 버전
len_gu = 11
len_time = 24
# 시간과 지역의 interaction term 은 600 개
# interaction_term : i25,20 ==> 2018년 2분기에 거래된 송파구(25)의 data
for i in tqdm(range(len_gu)):
    for j in range(len_time):
        df1['i' + str(i+1) + ',' + str(j+1)] = df1['Half' + str(j+1)] * df1['spec' + str(i+1)]

# data 갯수 확인
number_data = []
for i in tqdm(range(len_gu)):
    for j in range(len_time):
        a = np.sum(df1['i' + str(i+1) + ',' + str(j+1)])
        number_data.append(a)
num_interaction = np.sum(number_data)

index = []
for i in tqdm(range(len_gu)):
    for j in range(len_time):
        a = 'i' + str(i+1) + ',' + str(j+1)
        index.append(a)

df_check1 = pd.DataFrame(data=number_data, index=index)

df1.to_pickle('data_process/apt_data/spec_include_interaction.pkl')


#######################################################################################################################

# 투기지역으로 지정되지 않은 지역: 10개
nspec = ['강북구', '광진구', '도봉구', '성북구', '중랑구', '서대문구', '은평구', '관악구', '구로구', '금천구']

# 지역별 더미변수 생성
length1 = 10
for i in tqdm(range(length1)):
    df_non_spec['nspec' + str(i + 1)] = 0

length2 = len(df_non_spec['gu'])

for k in tqdm(range(length1)):
    for i in range(length2):
        if df_non_spec['gu'].iloc[i] == nspec[k]:
            df_non_spec['nspec' + str(k + 1)].iloc[i] = 1

# 갯수확인
number_data = []
for i in range(length1):
    a = np.sum(df_non_spec['nspec' + str(i + 1)])
    number_data.append(a)
num_count = np.sum(number_data)
print(num_count)

# half_dummy 생성하기
conditionlist = [(df_non_spec['time'] < 3),
                 (df_non_spec['time'] >= 3) & (df_non_spec['time'] < 5),
                 (df_non_spec['time'] >= 5) & (df_non_spec['time'] < 7),
                 (df_non_spec['time'] >= 7) & (df_non_spec['time'] < 9),
                 (df_non_spec['time'] >= 9) & (df_non_spec['time'] < 11),
                 (df_non_spec['time'] >= 11) & (df_non_spec['time'] < 13),
                 (df_non_spec['time'] >= 13) & (df_non_spec['time'] < 15),
                 (df_non_spec['time'] >= 15) & (df_non_spec['time'] < 17),
                 (df_non_spec['time'] >= 17) & (df_non_spec['time'] < 19),
                 (df_non_spec['time'] >= 19) & (df_non_spec['time'] < 21),
                 (df_non_spec['time'] >= 21) & (df_non_spec['time'] < 23),
                 (df_non_spec['time'] >= 23) & (df_non_spec['time'] < 25),
                 (df_non_spec['time'] >= 25) & (df_non_spec['time'] < 27),
                 (df_non_spec['time'] >= 27) & (df_non_spec['time'] < 29),
                 (df_non_spec['time'] >= 29) & (df_non_spec['time'] < 31),
                 (df_non_spec['time'] >= 31) & (df_non_spec['time'] < 33),
                 (df_non_spec['time'] >= 33) & (df_non_spec['time'] < 35),
                 (df_non_spec['time'] >= 35) & (df_non_spec['time'] < 37),
                 (df_non_spec['time'] >= 37) & (df_non_spec['time'] < 39),
                 (df_non_spec['time'] >= 39) & (df_non_spec['time'] < 41),
                 (df_non_spec['time'] >= 41) & (df_non_spec['time'] < 43),
                 (df_non_spec['time'] >= 43) & (df_non_spec['time'] < 45),
                 (df_non_spec['time'] >= 45) & (df_non_spec['time'] < 47),
                 (df_non_spec['time'] >= 47) & (df_non_spec['time'] < 49)]

Half_dummy = []
for i in range(1, 25):
    half = 'half' + str(i)
    Half_dummy.append(half)

choicelist = Half_dummy

df_non_spec['time(H)'] = np.select(conditionlist, choicelist, default='wrong')

# 기존의 21년 data 제거하기
num = df_non_spec['time(H)'] != 'wrong'
df2 = df_non_spec[num]
# df['time(H)'] = pd.to_numeric(df['time(H)'])

# year dummy 만들기
len_half = 24
for i in tqdm(range(len_half)):
    df2['Half' + str(i + 1)] = 0

length = len(df2['gu'])
for k in tqdm(range(len_half)):
    for i in range(length):
        if df2['time(H)'].iloc[i] == 'half' + str(k + 1):
            df2['Half' + str(k + 1)].iloc[i] = 1

# half dummy 갯수 확인
number_data = []
for i in range(len_half):
    a = np.sum(df2['Half' + str(i + 1)])
    number_data.append(a)
num_half_dummy = np.sum(number_data)

# interaction term 만들기: 분기와 구 버전
len_gu = 10
len_time = 24
# 시간과 지역의 interaction term 은 600 개
# interaction_term : i25,20 ==> 2018년 2분기에 거래된 송파구(25)의 data
for i in tqdm(range(len_gu)):
    for j in range(len_time):
        df2['i' + str(i+1) + ',' + str(j+1)] = df2['Half' + str(j+1)] * df2['nspec' + str(i+1)]

# data 갯수 확인
number_data = []
for i in tqdm(range(len_gu)):
    for j in range(len_time):
        a = np.sum(df2['i' + str(i+1) + ',' + str(j+1)])
        number_data.append(a)
num_interaction = np.sum(number_data)

index = []
for i in tqdm(range(len_gu)):
    for j in range(len_time):
        a = 'i' + str(i+1) + ',' + str(j+1)
        index.append(a)

df_check2 = pd.DataFrame(data=number_data, index=index)

df2.to_pickle('data_process/apt_data/non_spec_include_interaction.pkl')


#######################################################################################################################