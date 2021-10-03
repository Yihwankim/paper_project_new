'''
난방, 구조 더미변수 생성 후
* 건설사 더미의 경우 엑셀로 작업하는게 더욱 정확하여 엑셀로 작업함
seoul_replicating_duumy 파일 생성
'''

# import packages
import pandas as pd
import numpy as np
from haversine import haversine
from tqdm import tqdm

df_seoul = pd.read_excel('data_process/apt_data/Seoul_including_distance.xlsx', header=0, skipfooter=0)

# 난방 더미 만들기
# 개별난방 = H1, 지역난방 = H2, 중앙난방 = H3
df_seoul['H1'] = 0
df_seoul['H2'] = 0
df_seoul['H3'] = 0

df_seoul.loc[df_seoul['난방'].str.contains('개별'), 'H1'] = 1
df_seoul.loc[df_seoul['난방'].str.contains('지역'), 'H2'] = 1
df_seoul.loc[df_seoul['난방'].str.contains('중앙'), 'H3'] = 1

# 구조 더미 만들기
# 계단식 = T1, 복도식 = T2, 복합식 = T3
df_seoul['T1'] = 0
df_seoul['T2'] = 0
df_seoul['T3'] = 0

df_seoul.loc[df_seoul['구조'].str.contains('계단'), 'T1'] = 1
df_seoul.loc[df_seoul['구조'].str.contains('복도'), 'T2'] = 1
df_seoul.loc[df_seoul['구조'].str.contains('복합'), 'T3'] = 1


# 갯수 확인
H1 = np.sum(df_seoul['H1'])
H2 = np.sum(df_seoul['H2'])
H3 = np.sum(df_seoul['H3'])

T1 = np.sum(df_seoul['T1'])
T2 = np.sum(df_seoul['T2'])
T3 = np.sum(df_seoul['T3'])

# 난방구조 어디에도 속하지 않는 행 제거
dfs = []
for i in range(1, 4):
    heat = df_seoul['H' + str(i)] == 1
    df = df_seoul[heat]
    dfs.append(df)

df_seoul_edit = pd.concat(dfs, axis=0)

df_seoul_edit.to_excel('data_process/apt_data/seoul_replicating_dummy.xlsx', index=False)