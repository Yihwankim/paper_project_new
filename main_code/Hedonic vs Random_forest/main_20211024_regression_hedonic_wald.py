# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

########################################################################################################################
# regression_ data load
df_full = pd.read_pickle('data_process/conclusion/sample/hedonic_full_data.pkl')
df_full = df_full.dropna()

# variable setting
df_full['log_per_Pr'] = np.log(df_full['per_Pr'])

indep_var = ['old', 'old_sq', 'log_num', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq', 'first', 'H2',
             'H3', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency', 'dist_high', 'dist_sub', 'dist_park']

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

for i in range(len_gu):
    for j in range(len_time):
        a = 'i' + str(i + 1) + ',' + str(j + 1)
        inter_dum.append(a)
########################################################################################################################
independent_part1 = indep_var + gu_dum[1:] + time_dum[1:]
independent_part2 = indep_var + gu_dum[1:] + time_dum[1:] + inter_dum[1:]

# Hedonic regression without interaction term
X = sm.add_constant(df_full[independent_part1])
Y = df_full['log_per_Pr']

ols_model = sm.OLS(endog=Y, exog=X.values)
df_full_res = ols_model.fit()
df_full_result = df_full_res.summary(xname=X.columns.tolist())

# 회귀분석 결과값 출력
df_full_output1 = pd.concat((df_full_res.params, df_full_res.bse, df_full_res.pvalues), axis=1)
df_full_output1 = df_full_output1.rename(columns={0: 'coef', 1: 'std', 2: 'p-value'})
x_variables = []
x_variables = ['const'] + independent_part1

df_full_output1['variables'] = x_variables

df_full_output1['coef'] = round(df_full_output1['coef'], 4)
df_full_output1['std'] = round(df_full_output1['std'], 4)
df_full_output1['p-value'] = round(df_full_output1['p-value'], 4)

df1 = pd.DataFrame()
df1 = df_full_output1[['variables', 'coef', 'std', 'p-value']]

########################################################################################################################
# Hedonic regression with interaction term
X = sm.add_constant(df_full[independent_part2])
Y = df_full['log_per_Pr']

ols_model = sm.OLS(endog=Y, exog=X.values)
df_full_res = ols_model.fit()
df_full_result = df_full_res.summary(xname=X.columns.tolist())

# 회귀분석 결과값 출력
df_full_output2 = pd.concat((df_full_res.params, df_full_res.bse, df_full_res.pvalues), axis=1)
df_full_output2 = df_full_output2.rename(columns={0: 'coef', 1: 'std', 2: 'p-value'})
x_variables = []
x_variables = ['const'] + independent_part2

df_full_output2['variables'] = x_variables

df_full_output2['coef'] = round(df_full_output2['coef'], 4)
df_full_output2['std'] = round(df_full_output2['std'], 4)
df_full_output2['p-value'] = round(df_full_output2['p-value'], 4)

df2 = pd.DataFrame()
df2 = df_full_output2[['variables', 'coef', 'std', 'p-value']]

df2 = df2.iloc[:69, :]

########################################################################################################################
# wald test
df_output = pd.DataFrame()

df_output['variable'] = df1['variables']
df_output['diff.'] = (df2['coef'] - df1['coef'])**2
df_output['var'] = (df2['std'])**2 + (df1['std'])**2
df_output['wald stat.'] = df_output['diff.'] / df_output['var']

length = len(df_output['variable'])

chi_p_value = []
for i in range(length):
    a = 1 - stats.chi2.cdf(df_output['wald stat.'].iloc[i], 1)
    chi_p_value.append(a)

df_output['chi_p_value'] = chi_p_value

df_output['sig'] = np.nan
for i in range(length):
    if df_output['chi_p_value'].iloc[i] <= 0.1:
        df_output['sig'].iloc[i] = '*'

for i in range(length):
    if df_output['chi_p_value'].iloc[i] <= 0.05:
        df_output['sig'].iloc[i] = '**'

for i in range(length):
    if df_output['chi_p_value'].iloc[i] <= 0.01:
        df_output['sig'].iloc[i] = '***'

df_output.to_excel('data_process/conclusion/regression_result/wald_test_regression_ols_results.xlsx')









