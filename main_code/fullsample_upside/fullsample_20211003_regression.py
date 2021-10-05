# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#######################################################################################################################
# regression
df = pd.read_pickle('data_process/apt_data/seoul_year_interaction_term.pkl')

df_seoul = df.dropna()

df_seoul['price'] = df_seoul['per_Pr'] * df_seoul['area']

# Regression
df_seoul['log_per_Pr'] = np.log(df_seoul['per_Pr'])
df_seoul['log_num'] = np.log(df_seoul['num'])


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

X = sm.add_constant(df_seoul[independent])
Y = df_seoul['log_per_Pr']

ols_model = sm.OLS(Y, X.values)
df_full_res = ols_model.fit()
df_full_result = df_full_res.summary(xname=X.columns.tolist())

# 회귀분석 결과값 출력
df_full_output = pd.concat((df_full_res.params, df_full_res.bse, df_full_res.pvalues), axis=1)
df_full_output = df_full_output.rename(columns={0: 'coef', 1: 'std', 2: 'p-value'})
x_variables = []
x_variables = ['const'] + independent

df_full_output['variables'] = x_variables

df_full_output['coef'] = round(df_full_output['coef'], 4)
df_full_output['std'] = round(df_full_output['std'], 4)
df_full_output['p-value'] = round(df_full_output['p-value'], 4)

df = pd.DataFrame()
df = df_full_output[['variables', 'coef', 'std', 'p-value']]
df['coef'] = df_full_output['coef'].astype(str)
df['std'] = df_full_output['std'].astype(str)

df['beta'] = df['coef'] + ' (' + df['std'] + ')'

length = len(df['sig'])
df['sig'] = np.nan
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


df_output.to_excel('data_process/regression_result_data/half_time_interaction_regression_ols.xlsx')
########################################################################################################################
summary = df_seoul.describe()
sum = np.sum(X)

summary.to_excel('data_process/regression_result_data/descriptive_half_time_interaction_regression_full.xlsx')
sum.to_excel('data_process/regression_result_data/sum_half_time_interaction_regression_ols.xlsx')
########################################################################################################################

# Regression2
indep_var = ['old', 'old_sq', 'log_num', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq', 'first', 'H2',
             'H3', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency', 'dist_high', 'dist_sub', 'dist_park']

independent_part = indep_var + gu_dum[1:] + time_dum[1:]

X = sm.add_constant(df_seoul[independent_part])
Y = df_seoul['log_per_Pr']

ols_model2 = sm.OLS(Y, X.values)
df_part_res = ols_model2.fit()
df_part_result = df_part_res.summary(xname=X.columns.tolist())

# 회귀분석 결과값 출력
df_part_output = pd.concat((df_part_res.params, df_part_res.bse, df_part_res.pvalues), axis=1)
df_part_output = df_part_output.rename(columns={0: 'coef', 1: 'std', 2: 'p-value'})
x_variables = []
x_variables = ['const'] + independent_part

df_part_output['variables'] = x_variables

df_part_output['coef'] = round(df_part_output['coef'], 4)
df_part_output['std'] = round(df_part_output['std'], 4)
df_part_output['p-value'] = round(df_part_output['p-value'], 4)

df = pd.DataFrame()
df = df_part_output[['variables', 'coef', 'std', 'p-value']]
df['coef'] = df_part_output['coef'].astype(str)
df['std'] = df_part_output['std'].astype(str)

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


df_output.to_excel('data_process/regression_result_data/half_time_no_interaction_regression_ols.xlsx')
print(1)
# 분산팽창계수 확인
vif = pd.DataFrame()
vif['vif'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['variable'] = X.columns
vif.to_excel('data_process/regression_result_data/half_time_no_interaction_regression_ols_vif.xlsx')
df_output['vif'] = vif['vif']

