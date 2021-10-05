# price 로 subsample 분류하기
# price 75%: 68400
# price 25%: 33600

# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

#######################################################################################################################
df = pd.read_pickle('data_process/apt_data/seoul_year_interaction_term.pkl')
df_seoul = df.dropna()

high = df_seoul['price'] >= 68400
df_high_price = df_seoul[high]
df_high_price.to_csv('data_process/apt_data/sub_sample_data/high_price.csv', header=False, index=False)

low = df_seoul['price'] <= 33600
df_low_price = df_seoul[low]
df_low_price.to_csv('data_process/apt_data/sub_sample_data/low_price.csv', header=False, index=False)

#######################################################################################################################
# Regression_high
df_high_price['log_per_Pr'] = np.log(df_high_price['per_Pr'])
df_high_price['log_num'] = np.log(df_high_price['num'])


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

X = sm.add_constant(df_high_price[independent])
Y = df_high_price['log_per_Pr']

ols_model = sm.OLS(Y, X.values)
df_high_res = ols_model.fit()
df_high_result = df_high_res.summary(xname=X.columns.tolist())

# 회귀분석 결과값 출력
df_high_output = pd.concat((df_high_res.params, df_high_res.bse, df_high_res.pvalues), axis=1)
df_high_output = df_high_output.rename(columns={0: 'coef', 1: 'std', 2: 'p-value'})
x_variables = []
x_variables = ['const'] + independent

df_high_output['variables'] = x_variables

df_high_output['coef'] = round(df_high_output['coef'], 4)
df_high_output['std'] = round(df_high_output['std'], 4)
df_high_output['p-value'] = round(df_high_output['p-value'], 4)

df = pd.DataFrame()
df = df_high_output[['variables', 'coef', 'std', 'p-value']]
df['coef'] = df_high_output['coef'].astype(str)
df['std'] = df_high_output['std'].astype(str)

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


df_output.to_excel('data_process/regression_result_data/high_price_regression_ols.xlsx')
########################################################################################################################
# Regression_low
df_low_price['log_per_Pr'] = np.log(df_low_price['per_Pr'])
df_low_price['log_num'] = np.log(df_low_price['num'])


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

X = sm.add_constant(df_low_price[independent])
Y = df_low_price['log_per_Pr']

ols_model = sm.OLS(Y, X.values)
df_low_res = ols_model.fit()
df_low_result = df_low_res.summary(xname=X.columns.tolist())

# 회귀분석 결과값 출력
df_low_output = pd.concat((df_low_res.params, df_low_res.bse, df_low_res.pvalues), axis=1)
df_low_output = df_low_output.rename(columns={0: 'coef', 1: 'std', 2: 'p-value'})
x_variables = []
x_variables = ['const'] + independent

df_low_output['variables'] = x_variables

df_low_output['coef'] = round(df_low_output['coef'], 4)
df_low_output['std'] = round(df_low_output['std'], 4)
df_low_output['p-value'] = round(df_low_output['p-value'], 4)

df = pd.DataFrame()
df = df_low_output[['variables', 'coef', 'std', 'p-value']]
df['coef'] = df_low_output['coef'].astype(str)
df['std'] = df_low_output['std'].astype(str)

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


df_output2.to_excel('data_process/regression_result_data/low_price_regression_ols.xlsx')












