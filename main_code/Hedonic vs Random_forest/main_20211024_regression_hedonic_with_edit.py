# interaction term 에 대하여 Hedonic regression 실시

# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

########################################################################################################################
# regression_ data load
df_with = pd.read_pickle('data_process/conclusion/sample/hedonic_full_data.pkl')
df_with = df_with.dropna()

# variable setting
df_with['log_per_Pr'] = np.log(df_with['per_Pr'])

indep_var = ['old', 'old_sq', 'log_num', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq', 'first', 'H2',
             'H3', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency', 'dist_high', 'dist_sub', 'dist_park']

gu_dum = []
time_dum = []
inter_dum = []

len_gu = 25  # i
len_time = 24  # j
'''for i in range(len_gu):
    a = 'GU' + str(i+1)
    gu_dum.append(a)

for i in range(len_time):
    b = 'Half' + str(i+1)
    time_dum.append(b)'''

for i in range(len_gu):
    for j in range(len_time):
        a = 'i' + str(i + 1) + ',' + str(j + 1)
        inter_dum.append(a)

# independent_part = indep_var + gu_dum[1:] + time_dum[1:] + inter_dum[1:]
independent_part = indep_var + inter_dum[1:]

# Hedonic regression with interaction term
X = sm.add_constant(df_with[independent_part])
Y = df_with['log_per_Pr']

ols_model = sm.OLS(endog=Y, exog=X.values)
df_full_res = ols_model.fit()
df_full_result = df_full_res.summary(xname=X.columns.tolist())

# 회귀분석 결과값 출력
df_full_output = pd.concat((df_full_res.params, df_full_res.bse, df_full_res.pvalues), axis=1)
df_full_output = df_full_output.rename(columns={0: 'coef', 1: 'std', 2: 'p-value'})
x_variables = []
x_variables = ['const'] + independent_part

df_full_output['variables'] = x_variables

df_full_output['coef'] = round(df_full_output['coef'], 4)
df_full_output['std'] = round(df_full_output['std'], 4)
df_full_output['p-value'] = round(df_full_output['p-value'], 4)

df = pd.DataFrame()
df = df_full_output[['variables', 'coef', 'std', 'p-value']]
df['coef'] = df_full_output['coef'].astype(str)
df['std'] = df_full_output['std'].astype(str)

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

# VIF test
vif = pd.DataFrame()
vif['vif'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['variable'] = X.columns

#######################################################################################################################
df_output.to_excel('data_process/conclusion/regression_result/full_with_interactionterm_regression_ols_results_edit.xlsx')














