# Import packages
import numpy as np
import tensorflow as tf
import random as rn
from sklearn import metrics
from sklearn.model_selection import train_test_split  # train / test
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import r2_score

######################################################################################################################
# with & without 결과값
# normalization 과 no_normalization 은 있음

# epoch 별로 성과를 측정
## with & normalization 에서 성과가 가장 좋았으므로 해당 set 에 대해 시행
### epoch를 50, 100, 150, 200 으로 측정

r2_score_train = []
r2_score_predict = []
mse = []
rmse = []
mape = []
corr = []


df_data = pd.read_pickle('data_process/conclusion/NN/normalization_without_interaction.pkl')
df_data = df_data.dropna()

df_data_sample = df_data.sample(frac=0.00200449, replace=False, random_state=2)
df_data_sample = df_data_sample.sort_values(by=['old'])
df_data_sample = df_data_sample.reset_index(drop='Ture')

model_name = "NN"

# StopWatch: 코드 시작
time_PredictionANN_start = datetime.now()
print("PredictionANN started at: " + str(time_PredictionANN_start))



X = df_data.iloc[:, 1:]
y = np.log(df_data.iloc[:, 0:1])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=2)

# sample set to array
X_train = np.array(x_train)
X_test = np.array(x_test)
X_valid = np.array(x_valid)

Y_train = np.array(y_train)
Y_test = np.array(y_test)
Y_valid = np.array(y_valid)

# 랜덤 시드 고정
ann_seed_num = 50
np.random.seed(ann_seed_num)
tf.random.set_seed(ann_seed_num)
rn.seed(ann_seed_num)

# ANN 모형 설정
# epoch = 100, 150, 200, 250, 300
for epoch in [100, 150, 200, 250, 300]:
    model = Sequential()

    # Normalization
    model.add(Dense(X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(150, kernel_initializer='normal', activation='relu'))
    # model.add(Dropout(0.2))

    model.add(Dense(75, kernel_initializer='normal', activation='relu'))
    # model.add(Dropout(0.2))

    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    # model.add(Dropout(0.2))

    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))

    model.compile(
    optimizer=Adam(0.00001),
    loss=tf.keras.losses.mean_squared_error)

    r = model.fit(X_train, y_train, validation_data=(X_valid, Y_valid), batch_size=10, epochs=epoch)

    test_pred = model.predict(X_test)

    # 훈련 결과를 이용하여 예측 실시
    df_predict = pd.DataFrame(test_pred)
    df_predict.columns = ['per_Pr']

    # test value
    df_test = pd.DataFrame(Y_test)
    df_test.columns = ['per_Pr']

    # 성과평가
    a = mean_squared_error(test_pred, Y_test)
    mse.append(a)

    b = a ** (1 / 2)
    rmse.append(b)

    a = np.mean(np.abs((Y_test - test_pred) / Y_test)) * 100
    mape.append(a)

    df = pd.DataFrame({'y_true': df_test['per_Pr'], 'y_pred': df_predict['per_Pr']})
    a = df['y_pred'].corr(df['y_true'])
    corr.append(a)

    print(model.summary())

    # r-square 도출
    a = r2_score(Y_test, test_pred)
    r2_score_predict.append(a)

    train_pred = model.predict(X_train)
    a = r2_score(Y_train, train_pred)
    r2_score_train.append(a)

# StopWatch: 코드 종료
time_PredictionANN_end = datetime.now()
print("PredictionANN finished at: " + str(time_PredictionANN_end))
print("Elapsed (in PredictionANN): " + str(time_PredictionANN_end - time_PredictionANN_start))



nn_outcome = pd.DataFrame()
nn_outcome['R_squared'] = r2_score_train
nn_outcome['MSE'] = mse
nn_outcome['RMSE'] = rmse
nn_outcome['Correlation'] = corr
nn_outcome['MAPE'] = mape
nn_outcome['est_R_squared'] = r2_score_predict


nn_outcome.to_excel('data_process/conclusion/regression_result/nn_results_sample_normalization.xlsx')