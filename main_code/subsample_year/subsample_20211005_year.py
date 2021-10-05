# old 로 subsample 분류하기
# old 75%: 19.916667
# old 25%: 8.750000

# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

#######################################################################################################################
df = pd.read_pickle('data_process/apt_data/seoul_year_interaction_term.pkl')
df_seoul = df.dropna()

old = df_seoul['old'] >= 19.916667
df_older_price = df_seoul[old]
df_older_price.to_csv('data_process/apt_data/sub_sample_data/old_price.csv', header=False, index=False)

younger = df_seoul['old'] <= 8.750000
df_younger_price = df_seoul[younger]
df_younger_price.to_csv('data_process/apt_data/sub_sample_data/younger_price.csv', header=False, index=False)

#######################################################################################################################
# Regression_older
df_older_price['log_per_Pr'] = np.log(df_older_price['per_Pr'])
df_older_price['log_num'] = np.log(df_older_price['num'])


indep_var = ['old', 'old_sq', 'log_num', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq', 'first', 'H2',
             'H3', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency', 'dist_high', 'dist_sub', 'dist_park']

# dummy variable 만들기
gu_dum = []
time_dum = []
inter_dum = []

len_gu = 25  # i
len_time = 24  # j
for i in range(len_gu):
    a = 'GU' + str(i+1)
    gu_dum.append(a)

for i in range(len_time):
    b = 'Half' + str(i+1)
    time_dum.append(b)

for i in tqdm(range(len_gu)):
    for j in range(len_time):
        c = 'i' + str(i+1) + ',' + str(j+1)
        inter_dum.append(c)

independent = indep_var + gu_dum[1:] + time_dum[1:] + inter_dum[1:]

X = sm.add_constant(df_older_price[independent])
Y = df_older_price['log_per_Pr']

ols_model = sm.OLS(Y, X.values)
df_older_res = ols_model.fit()
df_older_result = df_older_res.summary(xname=X.columns.tolist())

# 회귀분석 결과값 출력
df_older_output = pd.concat((df_older_res.params, df_older_res.bse, df_older_res.pvalues), axis=1)
df_older_output = df_older_output.rename(columns={0: 'coef', 1: 'std', 2: 'p-value'})
x_variables = []
x_variables = ['const'] + independent

df_older_output['variables'] = x_variables

df_older_output['coef'] = round(df_older_output['coef'], 4)
df_older_output['std'] = round(df_older_output['std'], 4)
df_older_output['p-value'] = round(df_older_output['p-value'], 4)

df = pd.DataFrame()
df = df_older_output[['variables', 'coef', 'std', 'p-value']]
df['coef'] = df_older_output['coef'].astype(str)
df['std'] = df_older_output['std'].astype(str)

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


df_output.to_excel('data_process/regression_result_data/older_price_regression_ols.xlsx')
########################################################################################################################
# Regression_younger
df_younger_price['log_per_Pr'] = np.log(df_younger_price['per_Pr'])
df_younger_price['log_num'] = np.log(df_younger_price['num'])


indep_var = ['old', 'old_sq', 'log_num', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq', 'first', 'H2',
             'H3', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency', 'dist_high', 'dist_sub', 'dist_park']

# dummy variable 만들기
gu_dum = []
time_dum = []
inter_dum = []

len_gu = 25  # i
len_time = 24  # j
for i in range(len_gu):
    a = 'GU' + str(i+1)
    gu_dum.append(a)

for i in range(len_time):
    b = 'Half' + str(i+1)
    time_dum.append(b)

for i in tqdm(range(len_gu)):
    for j in range(len_time):
        c = 'i' + str(i+1) + ',' + str(j+1)
        inter_dum.append(c)

independent = indep_var + gu_dum[1:] + time_dum[1:] + inter_dum[1:]

X = sm.add_constant(df_younger_price[independent])
Y = df_younger_price['log_per_Pr']

ols_model = sm.OLS(Y, X.values)
df_younger_res = ols_model.fit()
df_younger_result = df_younger_res.summary(xname=X.columns.tolist())

# 회귀분석 결과값 출력
df_younger_output = pd.concat((df_younger_res.params, df_younger_res.bse, df_younger_res.pvalues), axis=1)
df_younger_output = df_younger_output.rename(columns={0: 'coef', 1: 'std', 2: 'p-value'})
x_variables = []
x_variables = ['const'] + independent

df_younger_output['variables'] = x_variables

df_younger_output['coef'] = round(df_younger_output['coef'], 4)
df_younger_output['std'] = round(df_younger_output['std'], 4)
df_younger_output['p-value'] = round(df_younger_output['p-value'], 4)

df = pd.DataFrame()
df = df_younger_output[['variables', 'coef', 'std', 'p-value']]
df['coef'] = df_younger_output['coef'].astype(str)
df['std'] = df_younger_output['std'].astype(str)

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


df_output2 = df[['variables', 'beta', 'sig']]


df_output2.to_excel('data_process/regression_result_data/younger_price_regression_ols.xlsx')