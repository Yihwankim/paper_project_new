# 2021-06-29
# Chapter 6

# 개별 아파트까지의 거리를 추가
# 거리항목은 지하철, 초,중,고 까지의 거리, 공원까지의 거리로 정의

# import packages
import pandas as pd
import numpy as np
from haversine import haversine
from tqdm import tqdm


########################################################################################################################
# 함수 선언

'''
거리 구하기 예시
a = (37.552863, 126.87386)
b = (37.573943, 126.804646)
haversine(a, b, unit='km')
'''


def get_distance_apt(variable_length, variable_lat, variable_long, apt_to_variable, dist_variable):
    for i in tqdm(range(apt_length)):
        each_apt = (seoul_lat[i], seoul_long[i])

        for j in range(variable_length):
            each_variable = (variable_lat[j], variable_long[j])
            a = haversine(each_apt, each_variable, unit='km')
            apt_to_variable.append(a)

        b = min(apt_to_variable)
        apt_to_variable.clear()
        dist_variable.append(b)


########################################################################################################################
# 리스트 선언

# 개별 아파트와 개별 변수들까지의 거리
apt_to_elem = []
apt_to_middle = []
apt_to_high = []

apt_to_sub = []

apt_to_park = []

# 최단거리의 학교까지의 거리
dist_elem = []
dist_middle = []
dist_high = []

# 최단거리의 지하철까지의 거리
dist_sub = []

# 최단거리의 공원까지 거리
dist_park = []

########################################################################################################################
# 엑셀 호출

# 아파트 정보
df_seoul = pd.read_excel('Seoul_last.xlsx', header=0, skipfooter=0)

#df_seoul = df_seoul_before.drop_duplicates(['아파트코드'], keep='first')  # 계산을 더 빠르게 하기위해 면적유형 제거
#df_seoul = df_seoul.reset_index(drop='True')
#df_seoul.info()

# 학교 정보
df_elem = pd.read_excel('District data/학교현황.xlsx', sheet_name='초등학교', header=0, skipfooter=0)
df_middle = pd.read_excel('District data/학교현황.xlsx', sheet_name='중학교', header=0, skipfooter=0)
df_high = pd.read_excel('District data/학교현황.xlsx', sheet_name='고등학교', header=0, skipfooter=0)

# 지하철과의 거리
df_sub = pd.read_excel('District data/경기도 역사 정보.xlsx', header=0, skipfooter=0)
df_sub.info()

# 공원과의 거리
df_park = pd.read_excel('District data/공원현황.xlsx', header=0, skipfooter=0)
df_park.info()
df_park = df_park.dropna(axis=0)
df_park = df_park.reset_index(drop='True')
########################################################################################################################
# 각 변수별 위도와 경도 정리

# 아파트
seoul_lat = df_seoul.loc[:, '위도']
seoul_lat[0]

seoul_long = df_seoul.loc[:, '경도']
seoul_long[0]

# 학교
elem_lat = df_elem.loc[:, '위도']
elem_long = df_elem.loc[:, '경도']

middle_lat = df_middle.loc[:, '위도']
middle_long = df_middle.loc[:, '경도']

high_lat = df_high.loc[:, '위도']
high_long = df_high.loc[:, '경도']

# 지하철
sub_lat = df_sub.loc[:, '위도']
sub_long = df_sub.loc[:, '경도']

# 공원
park_lat = df_park.loc[:, '위도']
park_long = df_park.loc[:, '경도']
########################################################################################################################
# 아파트로 부터 최단거리에 위치한 각 학교, 지하철역, 공원까지의 거리 구하기

# 각 변수의 갯수
'''
length 를 정할 때 위도로 정하나 경도로 정하나 상관없음
'''
apt_length = len(seoul_lat)

elem_length = len(elem_lat)
middle_length = len(middle_lat)
high_length = len(high_lat)

sub_length = len(sub_lat)

park_length = len(park_lat)

# 코드 실행

get_distance_apt(elem_length, elem_lat, elem_long, apt_to_elem, dist_elem)

get_distance_apt(middle_length, middle_lat, middle_long, apt_to_middle, dist_middle)

get_distance_apt(high_length, high_lat, high_long, apt_to_high, dist_high)

get_distance_apt(sub_length, sub_lat, sub_long, apt_to_sub, dist_sub)

get_distance_apt(park_length, park_lat, park_long, apt_to_park, dist_park)

# 기존 파일에 append
df_seoul['dist_elem'] = dist_elem
df_seoul['dist_middle'] = dist_middle
df_seoul['dist_high'] = dist_high

df_seoul['dist_sub'] = dist_sub

df_seoul['dist_park'] = dist_park

df_seoul.to_excel('Seoul_including_distance.xlsx', sheet_name='including_distance', index=False)

'''
## 예시
apt_length = 5
elem_length = 5

for i in tqdm(range(apt_length)):
    each_apt = (seoul_lat[i], seoul_long[i])

    for j in range(elem_length):
        each_variable = (elem_lat[j], elem_long[j])
        a = haversine(each_apt, each_variable, unit='km')
        apt_to_elem.append(a)
        b = min(apt_to_elem)
        apt_to_elem.clear()

    dist_elem.append(b)

dist_elem

# dist_elem.clear()
'''