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

#######################################################################################################################
#df_sample = pd.read_pickle('data_process/apt_data/machine_learning/seoul_sampling_1000unit.pkl')  # 1000 개 서브샘플
df_train = pd.read_pickle('data_process/conclusion/sample/rfr_all_train_data.pkl')
df_test = pd.read_pickle('data_process/conclusion/sample/rfr_all_test_data.pkl')

#df_sample = df_sample.dropna()
df_train = df_train.dropna()
df_test = df_test.dropna()

X = df_train.iloc[:, 1:]
y = np.log(df_train.iloc[:, 0:1])

x_test = df_test.iloc[:, 1:]
y_test = np.log(df_test.iloc[:, 0:1])

x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=2)

# StopWatch: 코드 시작
time_PredictionANN_start = datetime.now()
print("PredictionANN started at: " + str(time_PredictionANN_start))

model_name = "NN"


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

model = Sequential()

model.add(Dense(X_train.shape[1], activation='relu'))
model.add(Dense(150, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(75, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(10, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(optimizer=Adam(0.00001), loss='mse')

r = model.fit(X_train, y_train, validation_data=(X_valid, Y_valid), batch_size=10, epochs=100)

#pd.DataFrame({'True_Values': y_test, 'Predicte_Values': pred}).hvplot.scatter(x='True_Values', y='Predicted_Values')

test_pred = model.predict(X_test)

# 훈련 결과를 이용하여 예측 실시
df_predict = pd.DataFrame(test_pred)
df_predict.columns = ['per_Pr']

# test value
df_test = pd.DataFrame(Y_test)
df_test.columns = ['per_Pr']

# 성과평가
mse = mean_squared_error(test_pred, Y_test)
rmse = mse**(1/2)
mape = np.mean(np.abs((Y_test - test_pred) / Y_test)) * 100

df = pd.DataFrame({'y_true': df_test['per_Pr'], 'y_pred': df_predict['per_Pr']})
corr = df['y_pred'].corr(df['y_true'])

print('mean_squared_error : ', mse)
print('root_mean_squared_error : ', rmse)
print('mean_absolute_percentage_error : ', mape)
print('correlation : ', corr)


a = pd.DataFrame(r.history).plot()
plt.ylabel('accuracy')
plt.xlabel('epoch')

# r-square 도출
r2_score(Y_test, test_pred)
# batch 1: 실제값, 예측값: 0.8948353606750048
# batch 10: 실제값, 예측값 0.7810577059355858

train_pred = model.predict(X_train)
r2_score(Y_train, train_pred)
# batch 1 훈련샘플에 대한 실제값, 예측값: 0.8961584672923316
# batch 10: 훈련샘플에 대한 실제값 예측값 0.7808119947625769

model.summary()