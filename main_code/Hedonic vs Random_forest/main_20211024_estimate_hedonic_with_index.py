# with / without interaction term 에 대하여 Hedonic regression 실시

# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

########################################################################################################################
# regression_ data load
df_train = pd.read_pickle('data_process/apt_data/machine_learning/seoul_80_sample.pkl')
df_test = pd.read_pickle('data_process/apt_data/machine_learning/seoul_20_sample.pkl')

df_train = df_train.dropna()
df_test = df_test.dropna()

df_train['log_num'] = np.log(df_train['num'])
df_test['log_num'] = np.log(df_test['num'])

# variable setting
df_train['log_per_Pr'] = np.log(df_train['per_Pr'])
df_test['log_per_Pr'] = np.log(df_test['per_Pr'])

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

independent_part1 = indep_var + gu_dum[1:] + time_dum[1:]
independent_part2 = indep_var + inter_dum[1:]

# Hedonic regression with interaction term
# train & test 1: without interaction term
# train & test 2: with interaction term
X_train1 = sm.add_constant(df_train[independent_part1])
X_train2 = sm.add_constant(df_train[independent_part2])
Y_train = df_train['log_per_Pr']

X_test1 = sm.add_constant(df_test[independent_part1])
X_test2 = sm.add_constant(df_test[independent_part2])
Y_test = df_test['log_per_Pr']

line_fitting1 = LinearRegression()
line_fitting1.fit(X_train1, Y_train)

line_fitting2 = LinearRegression()
line_fitting2.fit(X_train2, Y_train)

x_index = pd.read_excel('data_process/conclusion/summary_full_rfr.xlsx', header=0, skipfooter=0)
x_index = pd.read_excel('data_process/conclusion/summary_rfr.xlsx', header=0, skipfooter=0)
line_fitting1.predict()
line_fitting2.predict()

# hedonic_outcome.to_excel('data_process/conclusion/regression_result/hedonic_estimation.xlsx')














