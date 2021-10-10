# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle

########################################################################################################################
df_nspec = pd.read_pickle('data_process/apt_data/non_spec_include_interaction.pkl')

df_nspec = df_nspec.dropna()

# Regression
df_nspec['log_per_Pr'] = np.log(df_nspec['per_Pr'])
df_nspec['log_num'] = np.log(df_nspec['num'])

indep_var = ['old', 'old_sq', 'log_num', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq', 'first', 'H2',
             'H3', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency', 'dist_high', 'dist_sub', 'dist_park']

# dummy variable
nspec_dum = []
time_dum = []
inter_dum = []

len_nspec = 10  # i
len_time = 24  # j

for i in range(len_nspec):
    a = 'nspec' + str(i + 1)
    nspec_dum.append(a)

for i in range(len_time):
    b = 'Half' + str(i + 1)
    time_dum.append(b)

for i in tqdm(range(len_nspec)):
    for j in range(len_time):
        c = 'i' + str(i + 1) + ',' + str(j + 1)
        inter_dum.append(c)

independent = indep_var + nspec_dum[1:] + time_dum[1:] + inter_dum[1:]

X = sm.add_constant(df_nspec[independent])
Y = df_nspec['log_per_Pr']

ols_model = sm.OLS(Y, X.values)
df_nspec_res = ols_model.fit()
df_nspec_result = df_nspec_res.summary(xname=X.columns.tolist())

# 회귀분석 결과값 출력
df_nspec_output = pd.concat((df_nspec_res.params, df_nspec_res.bse, df_nspec_res.pvalues), axis=1)
df_nspec_output = df_nspec_output.rename(columns={0: 'coef', 1: 'std', 2: 'p-value'})
x_variables = []
x_variables = ['const'] + independent

df_nspec_output['variables'] = x_variables

df_nspec_output['coef'] = round(df_nspec_output['coef'], 4)
df_nspec_output['std'] = round(df_nspec_output['std'], 4)
df_nspec_output['p-value'] = round(df_nspec_output['p-value'], 4)

df = pd.DataFrame()
df = df_nspec_output[['variables', 'coef', 'std', 'p-value']]
df['coef'] = df_nspec_output['coef'].astype(str)
df['std'] = df_nspec_output['std'].astype(str)

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


df_output.to_excel('data_process/regression_result_data/non_spec_half_time_interaction_regression_ols.xlsx')
