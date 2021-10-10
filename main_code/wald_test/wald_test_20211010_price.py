# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle
from scipy.stats import chisquare
from scipy import stats

########################################################################################################################
# price 로 subsample 분류하기
# price 75%: 68400
# price 25%: 33600

#######################################################################################################################
df = pd.read_pickle('data_process/apt_data/seoul_year_interaction_term.pkl')
df_seoul = df.dropna()

df_seoul['price'] = df_seoul['per_Pr'] * df_seoul['area']

high = df_seoul['price'] >= 68400
df_high_price = df_seoul[high]
# df_high_price.to_csv('data_process/apt_data/sub_sample_data/high_price.csv', header=False, index=False)

low = df_seoul['price'] <= 33600
df_low_price = df_seoul[low]
# df_low_price.to_csv('data_process/apt_data/sub_sample_data/low_price.csv', header=False, index=False)

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

########################################################################################################################
df_output3 = pd.DataFrame()

length = 667
variable = []
for i in range(length):
    a = 'x' + str(i+1)
    variable.append(a)

df_high_wald = df_high_output.loc[variable]
df_low_wald = df_low_output.loc[variable]

df_output3['variable'] = df_high_wald['variables']
df_output3['diff.'] = (df_high_wald['coef'] - df_low_wald['coef'])**2
df_output3['var'] = (df_high_wald['std'])**2 + (df_low_wald['std'])**2
df_output3['wald stat.'] = df_output3['diff.'] / df_output3['var']

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

df_output3.to_excel('data_process/regression_result_data/price_chisquare.xlsx')














