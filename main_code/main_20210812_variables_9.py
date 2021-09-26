# Chapter 9
# 권역별, 학군별, 시점별 time dummy 를 생성하고
# 시점별 time dummy 에 기인하여 분기별 엑셀 데이터를 형성
# 각종 변수들을 영문으로 수정 후 저장

'''
목표)
1. Time_dummy 만들기
2. 현재 추가된 더미변수: 건설사
3. 해야할 작업:
    a) 건설사 더미 이름 변경: C1
    b) 시점에 따라 Time _dummy 추가하기: D1 ~ D42 , clear
    c) 층, 연수 등의 변수를 로그화 시키기
    d) 최종 회귀분석
'''

# import packages
from tqdm import tqdm
import pandas as pd
import datetime
import numpy as np

########################################################################################################################
df_hedonic = pd.read_excel('real_transaction2/Seoul_index.xlsx', header=0, skipfooter=0)

df = df_hedonic

# 거래금액 숫자로 변환하기
price = []
for i in range(len(df['거래금액'])):
    p = df['거래금액'].iloc[i].lstrip()
    price.append(p)

df['price'] = price

cost = df['price'].str.split(',')
df['Pr'] = cost.str.get(0) + cost.str.get(1)
df['Pr'] = pd.to_numeric(df['Pr'])

# Time dummy 생성하기
length1 = 49
for i in tqdm(range(length1)):
    df['D' + str(i + 1)] = np.where(df['Time'] == i + 1, 1, 0)

# 건설사 더미(C1) 생성하기
df['C1'] = df_hedonic['건설사']

'''
# 각 지역구에 해당하는 key 값 만들기

도심권: 용산구 = 1, 종로구 = 2, 중구 = 3
동북권: 강북구 = 4, 광진구 = 5, 노원구 = 6, 도봉구 = 7, 동대문구 = 8, 성동구 = 9, 성북구 = 10, 중랑구 = 11
서북권: 마포구 = 12, 서대문구 = 13, 은평구 = 14
서남권: 강서구 = 15, 관악구 = 16, 구로구 = 17, 금천구 = 18, 동작구 = 19, 양천구 = 20, 영등포구 = 21
동남권: 강남구 = 22, 강동구 = 23, 서초구 = 24, 송파구 = 25

df['key'] = 0
length2 = len(df['지역구'])

for i in tqdm(range(length2)):
    if df['지역구'].iloc[i] == '용산구':
        df['key'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '종로구':
        df['key'].iloc[i] = 2

    elif df['지역구'].iloc[i] == '중구':
        df['key'].iloc[i] = 3

    elif df['지역구'].iloc[i] == '강북구':
        df['key'].iloc[i] = 4

    elif df['지역구'].iloc[i] == '광진구':
        df['key'].iloc[i] = 5

    elif df['지역구'].iloc[i] == '노원구':
        df['key'].iloc[i] = 6

    elif df['지역구'].iloc[i] == '도봉구':
        df['key'].iloc[i] = 7

    elif df['지역구'].iloc[i] == '동대문구':
        df['key'].iloc[i] = 8

    elif df['지역구'].iloc[i] == '성동구':
        df['key'].iloc[i] = 9

    elif df['지역구'].iloc[i] == '성북구':
        df['key'].iloc[i] = 10

    elif df['지역구'].iloc[i] == '중랑구':
        df['key'].iloc[i] = 11

    elif df['지역구'].iloc[i] == '마포구':
        df['key'].iloc[i] = 12

    elif df['지역구'].iloc[i] == '서대문구':
        df['key'].iloc[i] = 13

    elif df['지역구'].iloc[i] == '은평구':
        df['key'].iloc[i] = 14

    elif df['지역구'].iloc[i] == '강서구':
        df['key'].iloc[i] = 15

    elif df['지역구'].iloc[i] == '관악구':
        df['key'].iloc[i] = 16

    elif df['지역구'].iloc[i] == '구로구':
        df['key'].iloc[i] = 17

    elif df['지역구'].iloc[i] == '금천구':
        df['key'].iloc[i] = 18

    elif df['지역구'].iloc[i] == '동작구':
        df['key'].iloc[i] = 19

    elif df['지역구'].iloc[i] == '양천구':
        df['key'].iloc[i] = 20

    elif df['지역구'].iloc[i] == '영등포구':
        df['key'].iloc[i] = 21

    elif df['지역구'].iloc[i] == '강남구':
        df['key'].iloc[i] = 22

    elif df['지역구'].iloc[i] == '강동구':
        df['key'].iloc[i] = 23

    elif df['지역구'].iloc[i] == '서초구':
        df['key'].iloc[i] = 24

    elif df['지역구'].iloc[i] == '송파구':
        df['key'].iloc[i] = 25
'''
# 권역별 더미(G1 ~ G5) 형성하기
df['G1'] = 0  # 도심권
df['G2'] = 0  # 동북권
df['G3'] = 0  # 서북권
df['G4'] = 0  # 서남권
df['G5'] = 0  # 동남권
length2 = len(df['지역구'])

for i in tqdm(range(length2)):
    if df['지역구'].iloc[i] == '용산구':
        df['G1'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '종로구':
        df['G1'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '중구':
        df['G1'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '강북구':
        df['G2'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '광진구':
        df['G2'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '노원구':
        df['G2'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '도봉구':
        df['G2'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '동대문구':
        df['G2'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '성동구':
        df['G2'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '성북구':
        df['G2'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '중랑구':
        df['G2'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '마포구':
        df['G3'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '서대문구':
        df['G3'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '은평구':
        df['G3'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '강서구':
        df['G4'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '관악구':
        df['G4'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '구로구':
        df['G4'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '금천구':
        df['G4'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '동작구':
        df['G4'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '양천구':
        df['G4'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '영등포구':
        df['G4'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '강남구':
        df['G5'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '강동구':
        df['G5'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '서초구':
        df['G5'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '송파구':
        df['G5'].iloc[i] = 1


# 학군별 더미 생성하기
df['S1'] = 0  # 동부 학군
df['S2'] = 0  # 서부 학군
df['S3'] = 0  # 남부 학군
df['S4'] = 0  # 북부 학군
df['S5'] = 0  # 중부 학군
df['S6'] = 0  # 강동송파 학군
df['S7'] = 0  # 강서양천 학군
df['S8'] = 0  # 강남서초 학군
df['S9'] = 0  # 동작관악 학군
df['S10'] = 0  # 성동광진 학군
df['S11'] = 0  # 성북강북 학군

length2 = len(df['지역구'])

for i in tqdm(range(length2)):
    if df['지역구'].iloc[i] == '용산구':
        df['S5'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '종로구':
        df['S5'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '중구':
        df['S5'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '강북구':
        df['S11'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '광진구':
        df['S10'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '노원구':
        df['S4'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '도봉구':
        df['S4'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '동대문구':
        df['S1'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '성동구':
        df['S10'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '성북구':
        df['S11'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '중랑구':
        df['S1'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '마포구':
        df['S2'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '서대문구':
        df['S2'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '은평구':
        df['S2'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '강서구':
        df['S7'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '관악구':
        df['S9'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '구로구':
        df['S3'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '금천구':
        df['S3'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '동작구':
        df['S9'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '양천구':
        df['S7'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '영등포구':
        df['S3'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '강남구':
        df['S8'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '강동구':
        df['S6'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '서초구':
        df['S8'].iloc[i] = 1

    elif df['지역구'].iloc[i] == '송파구':
        df['S6'].iloc[i] = 1

df['yr'] = np.log(df['연수'])
df['num'] = np.log(df['세대수'])
df['car'] = np.log(df['주차대수_총'])
df['car_per'] = np.log(df['주차대수_세대'])
df['capacity'] = np.log(df['전용면적'])
df['room'] = np.log(df['방개수'])
df['toilet'] = np.log(df['화장실개수'])
df['floor'] = np.log(df['층'])

df_hedonic_index = df[['Pr', 'yr', 'num', 'car', 'car_per', 'capacity', 'room', 'toilet', 'floor',
                       'dist_elem', 'dist_middle', 'dist_high', 'dist_sub', 'dist_park',
                       '용적률', '건폐율', '전용률', 'H1', 'H2', 'H3', 'T1', 'T2', 'T3',
                       'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12',
                       'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22',
                       'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32', 'D33',
                       'D34', 'D35', 'D36', 'D37', 'D38', 'D39', 'D40', 'D41', 'D42',
                       'C1',
                       'G1', 'G2', 'G3', 'G4', 'G5',
                       'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11']]


#####################################################################################################################
df_hedonic_time = pd.DataFrame()
df_hedonic_time['gu'] = df['지역구']
df_hedonic_time['dong'] = df['법정동']
df_hedonic_time['per_Pr'] = df['Pr'] / df['전용면적']  # 면적당 가격
df_hedonic_time['year'] = df['연수']/12
df_hedonic_time['year_sq'] = df_hedonic_time['year']**2
df_hedonic_time['num'] = df['세대수']
df_hedonic_time['car'] = df['주차대수_총']
df_hedonic_time['car_per'] = df['주차대수_세대']
df_hedonic_time['area'] = df['전용면적']
df_hedonic_time['room'] = df['방개수']
df_hedonic_time['toilet'] = df['화장실개수']
df_hedonic_time['floor'] = df['층']
df_hedonic_time['floor_sq'] = df['층']**2
df_hedonic_time[['H1', 'H2', 'H3', 'T1', 'T2', 'T3', 'C1']] = df[['H1', 'H2', 'H3', 'T1', 'T2', 'T3', '건설사']]
df_hedonic_time[['FAR', 'BC', 'Efficiency']] = df[['용적률', '건폐율', '전용률']]
df_hedonic_time[['dist_elem', 'dist_middle', 'dist_high', 'dist_sub', 'dist_park']] = df[['dist_elem', 'dist_middle',
                                                                                          'dist_high', 'dist_sub',
                                                                                          'dist_park']]
df_hedonic_time[['G1', 'G2', 'G3', 'G4', 'G5']] = df[['G1', 'G2', 'G3', 'G4', 'G5']]

df_hedonic_time[['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11']] = df[['S1', 'S2', 'S3', 'S4',
                                                                                            'S5', 'S6', 'S7', 'S8',
                                                                                            'S9', 'S10', 'S11']]

df_hedonic_time[['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16',
                 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30',
                 'D31', 'D32', 'D33', 'D34', 'D35', 'D36', 'D37', 'D38', 'D39', 'D40', 'D41', 'D42', 'D43', 'D44',
                 'D45', 'D46', 'D47', 'D48', 'D49']]\
    = df[['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16',
          'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31',
          'D32', 'D33', 'D34', 'D35', 'D36', 'D37', 'D38', 'D39', 'D40', 'D41', 'D42', 'D43', 'D44', 'D45', 'D46',
          'D47', 'D48', 'D49']]

df_hedonic_time[['lat', 'long']] = df[['위도', '경도']]

df_hedonic_time.to_excel('real_transaction2/seoul_full_variable.xlsx', index=False)

for i in tqdm(range(length1)):
    num = df_hedonic_time['D' + str(i+1)] == 1
    df_quarter = df_hedonic_time[num]
    df_quarterly = df_quarter[['gu', 'dong', 'per_Pr', 'year', 'year_sq', 'num', 'car', 'car_per', 'area', 'room',
                               'toilet', 'floor', 'floor_sq', 'H1', 'H2', 'H3', 'T1', 'T2', 'T3', 'C1', 'FAR', 'BC',
                               'Efficiency', 'dist_elem', 'dist_middle', 'dist_high', 'dist_sub', 'dist_park', 'G1',
                               'G2', 'G3', 'G4', 'G5', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10',
                               'S11', 'lat', 'long']]
    df_quarterly.to_excel('real_transaction2/quarterly/seoul_apt' + str(i+1) + '.xlsx')
#####################################################################################################################
for i in tqdm(range(length1)):
    df_q = pd.read_excel('real_transaction2/quarterly/seoul_apt' + str(i+1) + '.xlsx', header=0, skipfooter=0)
    df = df_q[['gu', 'dong', 'per_Pr', 'year', 'year_sq', 'num', 'car', 'car_per', 'area', 'room', 'toilet', 'floor',
               'floor_sq', 'H1', 'H2', 'H3', 'T1', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency',
               'dist_elem', 'dist_middle', 'dist_high', 'dist_sub', 'dist_park', 'G1', 'G2', 'G3', 'G4', 'G5', 'S1',
               'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'lat', 'long']]
    df.to_excel('real_transaction2/quarterly_edit/seoul_apt' + str(i+1) + '.xlsx')


