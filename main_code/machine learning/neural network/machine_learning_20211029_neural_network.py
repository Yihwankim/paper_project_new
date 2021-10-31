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

#######################################################################################################################
df_sample = pd.read_pickle('data_process/apt_data/machine_learning/seoul_sampling_1000unit.pkl')

df_sample = df_sample.dropna()
X = df_sample.iloc[:, 1:]
y = np.log(df_sample.iloc[:, 0:1])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=2)

col_train_x = list(x_train.columns)
col_train_y = list(y_train.columns)

# StopWatch: 코드 시작
time_PredictionANN_start = datetime.now()
print("PredictionANN started at: " + str(time_PredictionANN_start))

model_name = "NN"

# 랜덤 시드 고정
ann_seed_num = 50
np.random.seed(ann_seed_num)
tf.random.set_seed(ann_seed_num)
rn.seed(ann_seed_num)

# ANN 모형 설정
# input output 차원 맞추기에 주의
x_train.shape[-1]
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(100, kernel_initializer='normal', activation='relu'),
        tf.keras.layers.Dense(50, kernel_initializer='normal', activation='relu'),
        # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, kernel_initializer='normal', activation='relu'),
        tf.keras.layers.Dense(1, kernel_initializer='normal')
    ])

# 메트릭 정의
my_metrics = [
    tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
    tf.keras.metrics.MeanSquaredError(name='mse'),
    tf.keras.metrics.RootMeanSquaredError(name='rmse')]

model.compile(
    loss=tf.keras.losses.mean_squared_error,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=my_metrics
)

# ANN 네트워크 훈련 실시 shuffle=False,
history = model.fit(x_train, y_train, epochs=100, verbose=2, validation_data=(x_valid, y_valid))
model.summary()

# StopWatch: 코드 종료
time_PredictionANN_end = datetime.now()
print("PredictionANN finished at: " + str(time_PredictionANN_end))
print("Elapsed (in PredictionANN): " + str(time_PredictionANN_end - time_PredictionANN_start))

# 훈련 결과를 이용하여 예측 실시
y_predict = model.predict(x_test)
df_predict = pd.DataFrame(y_predict)
df_predict.columns = ['per_Pr']


# 성과평가
mse = mean_squared_error(y_predict, y_test)
rmse = mse**(1/2)
mape = np.mean(np.abs((y_test - y_predict) / y_test)) * 100

df = pd.DataFrame({'y_true': y_test['per_Pr'], 'y_pred': df_predict['per_Pr']})
corr = df['y_pred'].corr(df['y_true'])

########################################################################################################################
print('mean_squared_error : ', mse)
print('root_mean_squared_error : ', rmse)
print('mean_absolute_percentage_error : ', mape)
print('correlation : ', corr)


a = pd.DataFrame(history.history).plot()
plt.ylabel('accuracy')
plt.xlabel('epoch')

########################################################################################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

df_sample = pd.read_pickle('data_process/apt_data/machine_learning/seoul_sampling_1000unit.pkl')

df_sample = df_sample.dropna()
X = df_sample.iloc[:, 1:]
y = np.log(df_sample.iloc[:, 0:1])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=2)

X_train = np.array(x_train)
X_test = np.array(x_test)
X_valid = np.array(x_valid)

Y_train = np.array(y_train)
Y_test = np.array(y_test)
Y_valid = np.array(y_valid)

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

r = model.fit(X_train, y_train, validation_data=(X_valid, Y_valid), batch_size=1, epochs=100)

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