
# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

#######################################################################################################################
df = pd.read_pickle('data_process/apt_data/seoul_full_variable.pkl')
df_seoul = df.dropna()


spec = []
# 강남구
gangnam = df_seoul['gu'] == '강남구'
spec.append(df_seoul[gangnam])
# 서초구
seocho = df_seoul['gu'] == '서초구'
spec.append(df_seoul[seocho])
# 송파구
songpa = df_seoul['gu'] == '송파구'
spec.append(df_seoul[songpa])
# 강동구
gangdong = df_seoul['gu'] == '강동구'
spec.append(df_seoul[gangdong])
# 용산구
yongsan = df_seoul['gu'] == '용산구'
spec.append(df_seoul[yongsan])
# 성동구
seongdong = df_seoul['gu'] == '성동구'
spec.append(df_seoul[seongdong])
# 노원구
nowon = df_seoul['gu'] == '노원구'
spec.append(df_seoul[nowon])
# 마포구
mapo = df_seoul['gu'] == '마포구'
spec.append(df_seoul[mapo])
# 양천구
yangchun = df_seoul['gu'] == '양천구'
spec.append(df_seoul[yangchun])
# 영등포구
yeongdeungpo = df_seoul['gu'] == '영등포구'
spec.append(df_seoul[yeongdeungpo])
# 강서구
gangseo = df_seoul['gu'] == '강서구'
spec.append(df_seoul[gangseo])

df_spec = pd.concat(spec, axis=0)

df_spec.to_pickle('data_process/apt_data/spec_variable.pkl')

#######################################################################################################################
non_spec = []
# 강북구
gangbuk = df_seoul['gu'] == '강북구'
non_spec.append(df_seoul[gangbuk])
# 광진구
gwangjin = df_seoul['gu'] == '광진구'
non_spec.append(df_seoul[gwangjin])
# 도봉구
dobong = df_seoul['gu'] == '도봉구'
non_spec.append(df_seoul[dobong])
# 성북구
seongbuk = df_seoul['gu'] == '성북구'
non_spec.append(df_seoul[seongbuk])
# 중랑구
jungrang = df_seoul['gu'] == '중랑구'
non_spec.append(df_seoul[jungrang])
# 서대문구
seodaemun = df_seoul['gu'] == '서대문구'
non_spec.append(df_seoul[seodaemun])
# 은평구
eunpyung = df_seoul['gu'] == '은평구'
non_spec.append(df_seoul[eunpyung])
# 관악구
gwanak = df_seoul['gu'] == '관악구'
non_spec.append(df_seoul[gwanak])
# 구로구
guro = df_seoul['gu'] == '구로구'
non_spec.append(df_seoul[guro])
# 금천구
geumchun = df_seoul['gu'] == '금천구'
non_spec.append(df_seoul[geumchun])

df_non_spec = pd.concat(non_spec, axis=0)

df_non_spec.to_pickle('data_process/apt_data/non_spec_variable.pkl')





#######################################################################################################################