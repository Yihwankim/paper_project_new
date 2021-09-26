# Chapter 10
# 분기별로 수정된 자료들을 연도별로 묶고
# 연도별 자료들을 가지고 회귀분석 실시
# 우선 OLS 분석만 실시하고 추후 WLS 등으로 업그레이드

# import packages
from tqdm import tqdm
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from stargazer.stargazer import Stargazer

from urllib.request import urlopen
import json

import plotly.express as px  # 빠르게 사용
import plotly.graph_objects as go  # 디테일하게 설정해야할때
import plotly.figure_factory as ff
from plotly.subplots import make_subplots  # 여러 subplot을 그릴때
from plotly.validators.scatter.marker import SymbolValidator  # 마커사용

########################################################################################################################
# Before Covid
## Time dummy 를 만들어 before covid data 만들기
dfs = []
for i in range(9):
    df = pd.read_excel('yearly_edit/'+'seoul_20'+str(i+11)+'.xlsx')
    df['time'] = i+1
    dfs.append(df)

df_bf_covid = pd.concat(dfs, axis=0)

for i in range(9):
    df_bf_covid['D'+str(i+11)] = np.where(df_bf_covid['time'] == i+1, 1, 0)

########################################################################################################################
# 데이터 분석
est = sm.OLS(endog=df['target'], exog=sm.add_constant(df[df.columns[0:4]])).fit()
est2 = sm.OLS(endog=df['target'], exog=sm.add_constant(df[df.columns[0:6]])).fit()

df_bf_covid['log_per_Pr'] = np.log(df_bf_covid['per_Pr'])
df_bf_covid['log_num'] = np.log(df_bf_covid['num'])

X = sm.add_constant(df_bf_covid[['year', 'year_sq', 'log_num', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq',
                                 'H2', 'H3', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency', 'dist_elem', 'dist_middle',
                                 'dist_high', 'dist_sub', 'dist_park', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9',
                                 'S10', 'S11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19']])

Y = df_bf_covid['log_per_Pr']

bf_res = sm.OLS(endog=Y, exog=X).fit()
# ols_model = sm.OLS(Y, X.values)
# bf_res = ols_model.fit()
bf_result = bf_res.summary(xname=X.columns.tolist())
bf_result

########################################################################################################################
# After Covid
df_af_covid = pd.read_excel('yearly_edit/seoul_2020.xlsx')

df_af_covid['log_per_Pr'] = np.log(df_af_covid['per_Pr'])


# df_af_covid = df_af_covid.dropna()
# df_af_covid.to_excel('before_after/after_covid.xlsx')


df_af_covid['log_num'] = np.log(df_af_covid['num'])

X = sm.add_constant(df_af_covid[['year', 'year_sq', 'log_num', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq',
                                 'H2', 'H3', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency',
                                 'dist_elem', 'dist_middle', 'dist_high', 'dist_sub', 'dist_park',
                                 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11']])
Y = df_af_covid['log_per_Pr']

af_res = sm.OLS(endog=Y, exog=X).fit()
#ols_model = sm.OLS(Y, X.values)
#af_res = ols_model.fit()
af_result = af_res.summary(xname=X.columns.tolist())
af_result

########################################################################################################################
# 기술통계량 분석
pd.set_option('display.max_columns',None)

df_bf_covid.describe()
df_bf_covid.sum()

df_af_covid.describe()
df_af_covid.sum()

'''for i in tqdm(range(42)):
    df_q = pd.read_excel('Hedonic_index/Quarterly/seoul_apt' + str(i + 1) + '.xlsx', header=0, skipfooter=0,
                         usecols='B:AU')
    df_q['log_per_Pr'] = np.log(df_q['per_Pr'])

    X = sm.add_constant(df_q[['year', 'year_sq', 'num', 'car', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq',
                              'H1', 'H2', 'H3', 'T1', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency', 'dist_elem',
                              'dist_middle', 'dist_high', 'dist_sub', 'dist_park', 'G1', 'G2', 'G3', 'G4', 'G5', 'S1',
                              'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11']])
    Y = df_q['log_per_Pr']

    rlm_model = sm.RLM(Y, X.values, M=sm.robust.norms.HuberT())
    res = rlm_model.fit()
    print('Number' + str(i + 1) + 'regression result: ')
    print(res.summary(xname=X.columns.tolist()))
    print('#########################################################################################')'''



#####################################################################################################################
# Stargazer
from IPython.core.display import HTML

stargazer = Stargazer([bf_res, af_res])

HTML(stargazer.render_html())

html_file = open('html_file.html', 'w')
html_file.write(results)
html_file.close()

stargazer.render_latex()

