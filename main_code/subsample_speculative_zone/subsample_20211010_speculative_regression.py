# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

########################################################################################################################
# 'data_process/apt_data/spec_variable.pkl'
# 'data_process/apt_data/non_spec_variable.pkl'
# 'data_process/apt_data/spec2_variable.pkl'
df_spec = pd.read_pickle('data_process/apt_data/spec_variable.pkl')

# interaction term 을 포함한 regression, 포함하지 않은 regression 을 돌리고 포함하지 않은 regression 에 대해서는 vif test를 통해 변수의 다중공선성을 확인하기

df_spec = df_spec.dropna()

# Regression
df_spec['log_per_Pr'] = np.log(df_spec['per_Pr'])
df_spec['log_num'] = np.log(df_spec['num'])

indep_var = ['old', 'old_sq', 'log_num', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq', 'first', 'H2',
             'H3', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency', 'dist_elem', 'dist_high', 'dist_sub', 'dist_park']

# dummy variable
gu_dum = []
time_dum = []
inter_dum = []

len_spec = 25  # i
len_time = 24  # j

for i in range(len_spec):
    a = 'GU' + str(i + 1)
    gu_dum.append(a)

for i in range(len_time):
    b = 'Half' + str(i + 1)
    time_dum.append(b)

for i in tqdm(range(len_spec)):
    for j in range(len_time):
        c = 'i' + str(i + 1) + ',' + str(j + 1)
        inter_dum.append(c)

independent1 = indep_var + gu_dum[1:] + time_dum[1:] + inter_dum[1:]
independent2 = indep_var + gu_dum[1:] + time_dum[1:]

X1 = sm.add_constant(df_spec[independent1])
X2 = sm.add_constant(df_spec[independent2])
Y = df_spec['log_per_Pr']

# ols model including interaction term
ols_model1 = sm.OLS(Y, X1.values)
df_spec_res1 = ols_model1.fit()
df_spec_result1 = df_spec_res1.summary(xname=X1.columns.tolist())

# ols model without interaction term
ols_model2 = sm.OLS(Y, X2.values)
df_spec_res2 = ols_model2.fit()
df_spec_result2 = df_spec_res2.summary(xname=X2.columns.tolist())

# 회귀분석 결과값 출력 (with interaction term)
df_spec_output1 = pd.concat((df_spec_res1.params, df_spec_res1.bse, df_spec_res1.pvalues), axis=1)
df_spec_output1 = df_spec_output1.rename(columns={0: 'coef', 1: 'std', 2: 'p-value'})

x_variables = []
x_variables = ['const'] + independent1

df_spec_output1['variables'] = x_variables

df_spec_output1['coef'] = round(df_spec_output1['coef'], 4)
df_spec_output1['std'] = round(df_spec_output1['std'], 4)
df_spec_output1['p-value'] = round(df_spec_output1['p-value'], 4)

# 회귀분석 결과값 출력 (without interaction term)
df_spec_output2 = pd.concat((df_spec_res2.params, df_spec_res2.bse, df_spec_res2.pvalues), axis=1)
df_spec_output2 = df_spec_output2.rename(columns={0: 'coef', 1: 'std', 2: 'p-value'})

x_variables = []
x_variables = ['const'] + independent2

df_spec_output2['variables'] = x_variables

df_spec_output2['coef'] = round(df_spec_output2['coef'], 4)
df_spec_output2['std'] = round(df_spec_output2['std'], 4)
df_spec_output2['p-value'] = round(df_spec_output2['p-value'], 4)


## 합치기, 지금은 여기까지 했음


df = pd.DataFrame()
df = df_spec_output1[['variables', 'coef', 'std', 'p-value']]
df['coef'] = df_spec_output1['coef'].astype(str)
df['std'] = df_spec_output1['std'].astype(str)

df['beta'] = df['coef'] + ' (' + df['std'] + ')'

df['sig'] = np.nan
length = len(df['sig'])

for i in tqdm(range(length)):
    if df['p-value'].iloc[i] <= 0.1:
        df['sig'].iloc[i] = '*'

for i in tqdm(range(length)):
    if df['p-value'].iloc[i] <= 0.05:
        df['sig'].iloc[i] = '**'

for i in tqdm(range(length)):
    if df['p-value'].iloc[i] <= 0.01:
        df['sig'].iloc[i] = '***'


df_output = df[['variables', 'beta', 'sig']]


df_output.to_excel('data_process/regression_result_data/spec_half_time_interaction_regression_ols.xlsx')
