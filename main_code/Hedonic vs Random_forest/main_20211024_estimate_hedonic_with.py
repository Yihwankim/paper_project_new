# with / without interaction term 에 대하여 Hedonic regression 실시

# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

# R-squared
score = []
a = line_fitting1.score(X_train1, Y_train)
score.append(a)

b = line_fitting2.score(X_train2, Y_train)
score.append(b)

########################################################################################################################
# Estimation
Y_predict1 = line_fitting1.predict(X_test1)
Y_predict2 = line_fitting2.predict(X_test2)

# model 평가
# 1. mean squared error
mse = []
a = mean_squared_error(Y_predict1, Y_test)
mse.append(a)

b = mean_squared_error(Y_predict2, Y_test)
mse.append(b)

# 2. root mean squared error
rmse = []
rmse.append(a**(1/2))
rmse.append(b**(1/2))

# 3. mean absolute percentage error
mape = []
a = np.mean(np.abs((Y_test - Y_predict1) / Y_test)) * 100
mape.append(a)

b = np.mean(np.abs((Y_test - Y_predict2) / Y_test)) * 100
mape.append(b)

# 4. Correlation
correlation = []

df1 = pd.DataFrame({'y_true': Y_test, 'y_pred': Y_predict1})
a = df1['y_true'].corr(df1['y_pred'])
correlation.append(a)  # 예측값과 실제값 사이의 correlation

df2 = pd.DataFrame({'y_true': Y_test, 'y_pred': Y_predict2})
b = df2['y_true'].corr(df2['y_pred'])
correlation.append(b)  # 예측값과 실제값 사이의 correlation

########################################################################################################################
# outcome
hedonic_outcome = pd.DataFrame()
hedonic_outcome['r-squared'] = score
hedonic_outcome['mse'] = mse
hedonic_outcome['rmse'] = rmse
hedonic_outcome['mape'] = mape
hedonic_outcome['correlation'] = correlation

hedonic_outcome = hedonic_outcome.rename(index={0: 'without_interaction_term', 1: 'with_interaction_term'})


#######################################################################################################################
hedonic_outcome.to_excel('data_process/conclusion/regression_result/hedonic_estimation.xlsx')














