# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle
from scipy.stats import chisquare
from scipy import stats
########################################################################################################################
df_spec = pd.read_pickle('data_process/apt_data/spec_include_interaction.pkl')

df_spec = df_spec.dropna()

# Regression
df_spec['log_per_Pr'] = np.log(df_spec['per_Pr'])
df_spec['log_num'] = np.log(df_spec['num'])

indep_var = ['old', 'old_sq', 'log_num', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq', 'first', 'H2',
             'H3', 'T2', 'T3', 'C1', 'FAR', 'BC', 'Efficiency', 'dist_high', 'dist_sub', 'dist_park']

# dummy variable
spec_dum = []
time_dum = []
inter_dum = []

len_spec = 11  # i
len_time = 24  # j

for i in range(len_spec):
    a = 'spec' + str(i + 1)
    spec_dum.append(a)

for i in range(len_time):
    b = 'Half' + str(i + 1)
    time_dum.append(b)

for i in tqdm(range(len_spec)):
    for j in range(len_time):
        c = 'i' + str(i + 1) + ',' + str(j + 1)
        inter_dum.append(c)

independent = indep_var + spec_dum[1:] + time_dum[1:] + inter_dum[1:]

X = sm.add_constant(df_spec[independent])
Y = df_spec['log_per_Pr']

ols_model = sm.OLS(Y, X.values)
df_spec_res = ols_model.fit()
df_spec_result = df_spec_res.summary(xname=X.columns.tolist())

# 회귀분석 결과값 출력
df_spec_output = pd.concat((df_spec_res.params, df_spec_res.bse, df_spec_res.pvalues), axis=1)
df_spec_output = df_spec_output.rename(columns={0: 'coef', 1: 'std', 2: 'p-value'})
x_variables = []
x_variables = ['const'] + independent

df_spec_output['variables'] = x_variables

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

########################################################################################################################
df_output3 = pd.DataFrame()

length = 21
variable = []
for i in range(length):
    a = 'x' + str(i+1)
    variable.append(a)

df_spec_wald = df_spec_output.loc[variable]
df_nspec_wald = df_nspec_output.loc[variable]

df_output3['variable'] = df_spec_wald['variables']
df_output3['diff.'] = (df_spec_wald['coef'] - df_nspec_wald['coef'])**2
df_output3['var'] = (df_spec_wald['std'])**2 + (df_nspec_wald['std'])**2
df_output3['wald stat.'] = df_output3['diff.'] / df_output3['var']

chisquare(df_output3['wald stat.'], ddof=1)

chi_p_value = []
for i in range(length):
    a = 1 - stats.chi2.cdf(df_output3['wald stat.'].iloc[i], 1)
    chi_p_value.append(a)

df_output3['chi_p_value'] = chi_p_value

df_output3['sig'] = np.nan
for i in range(length):
    if df_output3['chi_p_value'].iloc[i] <= 0.1:
        df_output3['sig'].iloc[i] = '*'

for i in range(length):
    if df_output3['chi_p_value'].iloc[i] <= 0.05:
        df_output3['sig'].iloc[i] = '**'

for i in range(length):
    if df_output3['chi_p_value'].iloc[i] <= 0.01:
        df_output3['sig'].iloc[i] = '***'

df_output3.to_excel('data_process/regression_result_data/speculative_chisquare.xlsx')