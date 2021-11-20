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

indep_var = ['old', 'old_sq', 'log_num', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq', 'first', 'H1',
             'H2', 'T1', 'T2', 'C1', 'FAR', 'BC', 'Efficiency', 'dist_high', 'dist_sub', 'dist_park']

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

########################################################################################################################
indep_var = ['const', 'old', 'old_sq', 'log_num', 'car_per', 'area', 'room', 'toilet',
             'floor', 'floor_sq', 'first', 'H1', 'H2', 'T1', 'T2', 'C1',
             'FAR', 'BC', 'Efficiency', 'dist_high', 'dist_sub', 'dist_park']
independent_part1 = indep_var + gu_dum[1:] + time_dum[1:]
independent_part2 = indep_var + inter_dum[1:]

# Without interaction term estimate
df_data1 = pd.read_excel\
    ('data_process/conclusion/predict_data/index_predict_input/basic/nn_index_data_no_interaction.xlsx',
     header=0, skipfooter=0)
df_data1['const'] = 1.0

X_index_without = df_data1[independent_part1]

y_predict_without = line_fitting1.predict(X_index_without)
y_predict_without = pd.DataFrame(y_predict_without)
y_predict_without.columns = ['without']

df_without = y_predict_without.copy()
df_without['real_values'] = np.exp(df_without['without'])
df_without['index'] = (df_without['real_values']/df_without['real_values'].loc[0]) *100

df_without.to_excel('data_process/conclusion/predict_data/index_predict_output/hedonic_without_basic_index.xlsx')
########################################################################################################################
# Without interaction term estimate with trend
df_data1 = pd.read_excel\
    ('data_process/conclusion/predict_data/index_predict_input/trend/without_interaction_NN_edit.xlsx',
     header=0, skipfooter=0)

df_data1['const'] = 1.0
X_index_without = df_data1[independent_part1]

y_predict_without = line_fitting1.predict(X_index_without)
y_predict_without = pd.DataFrame(y_predict_without)
y_predict_without.columns = ['without']

df_without = y_predict_without.copy()
df_without['real_values'] = np.exp(df_without['without'])
df_without['index'] = (df_without['real_values']/df_without['real_values'].loc[0]) *100

df_without.to_excel('data_process/conclusion/predict_data/index_predict_output/hedonic_without_trend_index.xlsx')

########################################################################################################################
# With interaction term estimate
df_data2 = pd.read_excel\
    ('data_process/conclusion/predict_data/index_predict_input/basic/nn_index_data_interaction.xlsx',
     header=0, skipfooter=0)

df_data2['const'] = 1.0
X_index_with = df_data2[independent_part2]

y_predict_with = line_fitting2.predict(X_index_with)
y_predict_with = pd.DataFrame(y_predict_with)
y_predict_with.columns = ['with']

df_with = y_predict_with.copy()
df_with['real_values'] = np.exp(df_with['with'])

length = 600
df_with['index'] = 0
for i in range(length):
    df_with['index'].loc[i] = (df_with['real_values'].loc[i]/df_with['real_values'].loc[int(i/24)*24]) * 100


df_with.to_excel('data_process/conclusion/predict_data/index_predict_output/hedonic_with_basic_index.xlsx')

########################################################################################################################
# With interaction term estimate
df_data2 = pd.read_excel\
    ('data_process/conclusion/predict_data/index_predict_input/trend/with_interaction_NN_edit.xlsx',
     header=0, skipfooter=0)

df_data2['const'] = 1.0
X_index_with = df_data2[independent_part2]

y_predict_with = line_fitting2.predict(X_index_with)
y_predict_with = pd.DataFrame(y_predict_with)
y_predict_with.columns = ['with']

df_with = y_predict_with.copy()
df_with['real_values'] = np.exp(df_with['with'])

length = 600
df_with['index'] = 0
for i in range(length):
    df_with['index'].loc[i] = (df_with['real_values'].loc[i]/df_with['real_values'].loc[int(i/24)*24]) * 100


df_with.to_excel('data_process/conclusion/predict_data/index_predict_output/hedonic_with_trend_index.xlsx')












