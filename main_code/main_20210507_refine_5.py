# 2021-05-07
# Chapter 5

# 최종적으로 '연수' 를 정의하고
# 앞에서 미처 정의하지 못했던 각 컬럼들 별 성질을 재정의할것


# import packages
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import relativedelta
from dateutil import relativedelta

########################################################################################################################
# import excel file

df_seoul = pd.read_excel('Seoul.xlsx', header=0, skipfooter=0)
df_seoul = df_seoul.replace('-', np.nan)
df_seoul = df_seoul.dropna(axis=0)
df_seoul = df_seoul.reset_index(drop='Ture')

df = df_seoul

df.dtypes

date_now = datetime.strptime('2021-05', '%Y-%m')

df['날짜'] = df['사용승인일'].dt.to_period('M')
df['날짜'] = df['날짜'].apply(str)

day_len = len(df['날짜'])

time_month = []
for i in range(day_len):
    date1 = datetime.strptime('2021-05', '%Y-%m')
    date2 = datetime.strptime(df['날짜'].iloc[i], '%Y-%m')
    r = relativedelta.relativedelta(date1, date2)
    m = r.months + (12 * r.years)
    time_month.append(m)

df_seoul['연수'] = time_month

df_seoul = df_seoul[['지역구', '법정동', '아파트', '아파트코드', '사용승인일', '연수', '세대수', '저층', '고층', '주차대수_총',
                     '주차대수_세대', '용적률', '건폐율', '위도', '경도', '건설사', '난방', '구조', '면적유형', '해당면적 세대수',
                     '공급면적(㎡)', '전용면적(㎡)', '전용률(%)', '방 개수',
                     '화장실 개수']]

########################################################################################################################
# 중복제거
# nan 값 제거
df = df_seoul

df = df.reset_index(drop='Ture')

df['name'] = df['아파트'] + " " + df['면적유형']
df = df.sort_values(by=['name'])
df = df.drop_duplicates(['name'], keep='first')

df_seoul = df[['지역구', '법정동', '아파트', '아파트코드', '사용승인일', '연수', '세대수', '저층', '고층', '주차대수_총',
               '주차대수_세대', '용적률', '건폐율', '위도', '경도', '건설사', '난방', '구조', '면적유형', '해당면적 세대수',
               '공급면적(㎡)', '전용면적(㎡)', '전용률(%)', '방 개수', '화장실 개수']]

df_seoul.dtypes
# 해당면적 세대수, 공급면적, 방개수, 화장실 개수 수정
df_seoul = df_seoul.reset_index(drop='Ture')

df_seoul['해당면적 세대수'] = pd.to_numeric(df['해당면적 세대수'])
df_seoul['공급면적(㎡)'] = pd.to_numeric(df['공급면적(㎡)'])

df_seoul['방 개수'] = pd.to_numeric(df_seoul['방 개수'])
df_seoul['화장실 개수'] = pd.to_numeric(df_seoul['화장실 개수'])

df_seoul.dtypes

df_seoul = df_seoul[['지역구', '법정동', '아파트', '아파트코드', '사용승인일', '연수', '세대수', '저층', '고층', '주차대수_총',
                     '주차대수_세대', '용적률', '건폐율', '위도', '경도', '건설사', '난방', '구조', '면적유형',  '전용면적(㎡)',
                     '전용률(%)', '방 개수', '화장실 개수']]

df_seoul.to_excel('Seoul_last.xlsx', sheet_name='last', index=False)
