
# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

#######################################################################################################################
df = pd.read_pickle('data_process/apt_data/seoul_including_all_variables.pkl')  # edit is required
df_seoul = df.dropna()

'''
도심권: 용산구 = 1, 종로구 = 2, 중구 = 3
동북권: 강북구 = 4, 광진구 = 5, 노원구 = 6, 도봉구 = 7, 동대문구 = 8, 성동구 = 9, 성북구 = 10, 중랑구 = 11
서북권: 마포구 = 12, 서대문구 = 13, 은평구 = 14
서남권: 강서구 = 15, 관악구 = 16, 구로구 = 17, 금천구 = 18, 동작구 = 19, 양천구 = 20, 영등포구 = 21
동남권: 강남구 = 22, 강동구 = 23, 서초구 = 24, 송파구 = 25
'''

spec = []
# 용산구 (1)
yongsan = df_seoul['gu'] == '용산구'
spec.append(df_seoul[yongsan])
# 노원구 (6)
nowon = df_seoul['gu'] == '노원구'
spec.append(df_seoul[nowon])
# 성동구 (9)
seongdong = df_seoul['gu'] == '성동구'
spec.append(df_seoul[seongdong])
# 마포구 (12)
mapo = df_seoul['gu'] == '마포구'
spec.append(df_seoul[mapo])
# 강서구 (15)
gangseo = df_seoul['gu'] == '강서구'
spec.append(df_seoul[gangseo])
# 양천구 (20)
yangchun = df_seoul['gu'] == '양천구'
spec.append(df_seoul[yangchun])
# 영등포구 (21)
yeongdeungpo = df_seoul['gu'] == '영등포구'
spec.append(df_seoul[yeongdeungpo])
# 강남구 (22)
gangnam = df_seoul['gu'] == '강남구'
spec.append(df_seoul[gangnam])
# 강동구 (23)
gangdong = df_seoul['gu'] == '강동구'
spec.append(df_seoul[gangdong])
# 서초구 (24)
seocho = df_seoul['gu'] == '서초구'
spec.append(df_seoul[seocho])
# 송파구 (25)
songpa = df_seoul['gu'] == '송파구'
spec.append(df_seoul[songpa])

df_spec = pd.concat(spec, axis=0)

df_spec.to_pickle('data_process/apt_data/spec_variable.pkl')


#######################################################################################################################
'''
도심권: 용산구 = 1, 종로구 = 2, 중구 = 3
동북권: 강북구 = 4, 광진구 = 5, 노원구 = 6, 도봉구 = 7, 동대문구 = 8, 성동구 = 9, 성북구 = 10, 중랑구 = 11
서북권: 마포구 = 12, 서대문구 = 13, 은평구 = 14
서남권: 강서구 = 15, 관악구 = 16, 구로구 = 17, 금천구 = 18, 동작구 = 19, 양천구 = 20, 영등포구 = 21
동남권: 강남구 = 22, 강동구 = 23, 서초구 = 24, 송파구 = 25
'''

non_spec = []
# 강북구 (4)
gangbuk = df_seoul['gu'] == '강북구'
non_spec.append(df_seoul[gangbuk])
# 광진구 (5)
gwangjin = df_seoul['gu'] == '광진구'
non_spec.append(df_seoul[gwangjin])
# 도봉구 (7)
dobong = df_seoul['gu'] == '도봉구'
non_spec.append(df_seoul[dobong])
# 성북구 (10)
seongbuk = df_seoul['gu'] == '성북구'
non_spec.append(df_seoul[seongbuk])
# 중랑구 (11)
jungrang = df_seoul['gu'] == '중랑구'
non_spec.append(df_seoul[jungrang])
# 서대문구 (13)
seodaemun = df_seoul['gu'] == '서대문구'
non_spec.append(df_seoul[seodaemun])
# 은평구 (14)
eunpyung = df_seoul['gu'] == '은평구'
non_spec.append(df_seoul[eunpyung])
# 관악구 (16)
gwanak = df_seoul['gu'] == '관악구'
non_spec.append(df_seoul[gwanak])
# 구로구 (17)
guro = df_seoul['gu'] == '구로구'
non_spec.append(df_seoul[guro])
# 금천구 (18)
geumchun = df_seoul['gu'] == '금천구'
non_spec.append(df_seoul[geumchun])

df_non_spec = pd.concat(non_spec, axis=0)

df_non_spec.to_pickle('data_process/apt_data/non_spec_variable.pkl')


#######################################################################################################################
'''
도심권: 용산구 = 1, 종로구 = 2, 중구 = 3
동북권: 강북구 = 4, 광진구 = 5, 노원구 = 6, 도봉구 = 7, 동대문구 = 8, 성동구 = 9, 성북구 = 10, 중랑구 = 11
서북권: 마포구 = 12, 서대문구 = 13, 은평구 = 14
서남권: 강서구 = 15, 관악구 = 16, 구로구 = 17, 금천구 = 18, 동작구 = 19, 양천구 = 20, 영등포구 = 21
동남권: 강남구 = 22, 강동구 = 23, 서초구 = 24, 송파구 = 25
'''
spec2 = []
# 종로구
jongro = df_seoul['gu'] == '종로구'
spec2.append(df_seoul[jongro])
# 중구
jung = df_seoul['gu'] == '중구'
spec2.append(df_seoul[jung])
# 동대문구
dongdaemoon = df_seoul['gu'] == '동대문구'
spec2.append(df_seoul[dongdaemoon])
# 동작구
dongjak = df_seoul['gu'] == '동작구'
spec2.append(df_seoul[dongjak])

df_spec2 = pd.concat(spec2, axis=0)

df_spec2.to_pickle('data_process/apt_data/spec2_variable.pkl')

#######################################################################################################################

len(df_seoul['gu']) - len(df_spec['gu']) - len(df_non_spec['gu']) - len(df_spec2['gu'])