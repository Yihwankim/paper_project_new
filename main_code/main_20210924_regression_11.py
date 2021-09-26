# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm

### 수정요망: 쥬피터에서 완전 실습 후 수정할 것

#######################################################################################################################
# import data
df_mb = pd.read_excel('data_process/section_edit/seoul_apt_09to12_edit.xlsx', header=0, skipfooter=0)
df_gh = pd.read_excel('data_process/section_edit/seoul_apt_13to16_edit.xlsx', header=0, skipfooter=0)
df_ji = pd.read_excel('data_process/section_edit/seoul_apt_17to20_edit.xlsx', header=0, skipfooter=0)

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
df_mb_output = df_mb_output.rename(index={'x1': 'year', 'x2': 'year_sq', 'x3': 'log_num', 'x4': 'car_per',
                                          'x5': 'area', 'x6': 'room', 'x7': 'toilet', 'x8': 'floor',
                                          'x9': 'floor_sq', 'x10': 'H2', 'x11': 'H3', 'x12': 'T2', 'x13': 'T3',
                                          'x14': 'C1', 'x15': 'FAR', 'x16': 'BC', 'x17': 'Efficiency',
                                          'x18': 'dist_elem', 'x19': 'dist_high', 'x20': 'dist_sub', 'x21': 'dist_park',
                                          'x22': 'G2', 'x23': 'G3', 'x24': 'G4', 'X25': 'G5',
                                          'x26': 'D2', 'x27': 'D3', 'x28': 'D4', 'x29': 'D5', 'x30': 'D6',
                                          'x31': 'D7', 'x32': 'D8', 'x33': 'D9', 'x34': 'D10', 'x35': 'D11',
                                          'x36': 'D12', 'x37': 'D13', 'x38': 'D14', 'x39': 'D15', 'x40': 'D16'})

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
df_gh_output = df_gh_output.rename(index={'x1': 'year', 'x2': 'year_sq', 'x3': 'log_num', 'x4': 'car_per',
                                          'x5': 'area', 'x6': 'room', 'x7': 'toilet', 'x8': 'floor',
                                          'x9': 'floor_sq', 'x10': 'H2', 'x11': 'H3', 'x12': 'T2', 'x13': 'T3',
                                          'x14': 'C1', 'x15': 'FAR', 'x16': 'BC', 'x17': 'Efficiency',
                                          'x18': 'dist_elem', 'x19': 'dist_high', 'x20': 'dist_sub', 'x21': 'dist_park',
                                          'x22': 'G2', 'x23': 'G3', 'x24': 'G4', 'X25': 'G5',
                                          'x26': 'D18', 'x27': 'D19', 'x28': 'D20', 'x29': 'D21', 'x30': 'D22',
                                          'x31': 'D23', 'x32': 'D24', 'x33': 'D25', 'x34': 'D26', 'x35': 'D27',
                                          'x36': 'D28', 'x37': 'D29', 'x38': 'D30', 'x39': 'D31', 'x40': 'D32'})

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
df_ji_output = df_ji_output.rename(index={'x1': 'year', 'x2': 'year_sq', 'x3': 'log_num', 'x4': 'car_per',
                                          'x5': 'area', 'x6': 'room', 'x7': 'toilet', 'x8': 'floor',
                                          'x9': 'floor_sq', 'x10': 'H2', 'x11': 'H3', 'x12': 'T2', 'x13': 'T3',
                                          'x14': 'C1', 'x15': 'FAR', 'x16': 'BC', 'x17': 'Efficiency',
                                          'x18': 'dist_elem', 'x19': 'dist_high', 'x20': 'dist_sub', 'x21': 'dist_park',
                                          'x22': 'G2', 'x23': 'G3', 'x24': 'G4', 'X25': 'G5',
                                          'x26': 'D34', 'x27': 'D35', 'x28': 'D36', 'x29': 'D37', 'x30': 'D38',
                                          'x31': 'D39', 'x32': 'D40', 'x33': 'D41', 'x34': 'D42', 'x35': 'D43',
                                          'x36': 'D44', 'x37': 'D45', 'x38': 'D46', 'x39': 'D47', 'x40': 'D48'})

# 엑셀 파일로 변환
with pd.ExcelWriter('data_process/section_edit/section_output.xlsx') as writer:
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



