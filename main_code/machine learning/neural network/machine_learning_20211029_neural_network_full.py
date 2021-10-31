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

# 랜덤 시드 고정
ann_seed_num = 50
np.random.seed(ann_seed_num)
tf.random.set_seed(ann_seed_num)
rn.seed(ann_seed_num)

# ANN 모형 설정
# input output 차원 맞추기에 주의
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
    optimizer=tf.keras.optimizers.Adam(lr=0.002),
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