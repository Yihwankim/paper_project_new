# import packages
from tqdm import tqdm
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from IPython.core.display import HTML

from urllib.request import urlopen
import json

import plotly.express as px  # 빠르게 사용
import plotly.graph_objects as go  # 디테일하게 설정해야할때
import plotly.figure_factory as ff
from plotly.subplots import make_subplots  # 여러 subplot을 그릴때
from plotly.validators.scatter.marker import SymbolValidator  # 마커사용

#######################################################################################################################
# import data
df_mb = pd.read_excel('real_transaction2/yearly_edit/seoul_apt_09to12_edit.xlsx', header=0, skipfooter=0)
df_gh = pd.read_excel('real_transaction2/yearly_edit/seoul_apt_13to16_edit.xlsx', header=0, skipfooter=0)
df_ji = pd.read_excel('real_transaction2/yearly_edit/seoul_apt_17to20_edit.xlsx', header=0, skipfooter=0)

#######################################################################################################################
df_mb = df_mb.dropna()
df_gh = df_gh.dropna()
df_ji = df_ji.dropna()

# 2009 to 2012
df_mb['log_per_Pr'] = np.log(df_mb['per_Pr'])
df_mb['log_num'] = np.log(df_mb['num'])

X = sm.add_constant(df_mb[['year', 'year_sq', 'log_num', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq',
                           'H2', 'H3', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency', 'dist_elem','dist_high', 'dist_sub',
                           'dist_park', 'G2', 'G3', 'G4', 'G5',
                           'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',
                           'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16']])
Y = df_mb['log_per_Pr']

ols_model = sm.OLS(Y, X.values)
df_mb_res = ols_model.fit()
df_mb_result = df_mb_res.summary(xname=X.columns.tolist())

df_mb_output = pd.concat((df_mb_res.params, df_mb_res.tvalues, df_mb_res.pvalues), axis=1)
df_mb_output = df_mb_output.rename(columns={0: 'coef', 1: 't-value', 2: 'p-value'})

# 2013 to 2016
df_gh['log_per_Pr'] = np.log(df_gh['per_Pr'])
df_gh['log_num'] = np.log(df_gh['num'])

X = sm.add_constant(df_gh[['year', 'year_sq', 'log_num', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq',
                           'H2', 'H3', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency', 'dist_elem','dist_high', 'dist_sub',
                           'dist_park', 'G2', 'G3', 'G4', 'G5',
                           'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25',
                           'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']])
Y = df_gh['log_per_Pr']

ols_model = sm.OLS(Y, X.values)
df_gh_res = ols_model.fit()
df_gh_result = df_gh_res.summary(xname=X.columns.tolist())

df_gh_output = pd.concat((df_gh_res.params, df_gh_res.tvalues, df_gh_res.pvalues), axis=1)
df_gh_output = df_gh_output.rename(columns={0: 'coef', 1: 't-value', 2: 'p-value'})

# 2017 to 2020
df_ji['log_per_Pr'] = np.log(df_ji['per_Pr'])
df_ji['log_num'] = np.log(df_ji['num'])

X = sm.add_constant(df_ji[['year', 'year_sq', 'log_num', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq',
                           'H2', 'H3', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency', 'dist_elem','dist_high', 'dist_sub',
                           'dist_park', 'G2', 'G3', 'G4', 'G5',
                           'D34', 'D35', 'D36', 'D37', 'D38', 'D39', 'D40', 'D41',
                           'D42', 'D43', 'D44', 'D45', 'D46', 'D47', 'D48']])
Y = df_ji['log_per_Pr']

ols_model = sm.OLS(Y, X.values)
df_ji_res = ols_model.fit()
df_ji_result = df_ji_res.summary(xname=X.columns.tolist())

df_ji_output = pd.concat((df_ji_res.params, df_ji_res.tvalues, df_ji_res.pvalues), axis=1)
df_ji_output = df_ji_output.rename(columns={0: 'coef', 1: 't-value', 2: 'p-value'})

# 엑셀 파일로 변환
with pd.ExcelWriter('real_transaction2/yearly_edit/section_output.xlsx') as writer:
    df_mb_output.to_excel(writer, sheet_name='09to12')
    df_gh_output.to_excel(writer, sheet_name='13to16')
    df_ji_output.to_excel(writer, sheet_name='17to20')

#######################################################################################################################
# Descriptive statistics
mb_summary1 = df_mb.describe()
mb_summary2 = df_mb.sum()

gh_summary1 = df_gh.describe()
gh_summary2 = df_gh.sum()

ji_summary1 = df_ji.describe()
ji_summary2 = df_ji.sum()



