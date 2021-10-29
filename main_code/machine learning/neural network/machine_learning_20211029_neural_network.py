# Import packages
import numpy as np
import tensorflow as tf
import random as rn
from sklearn import metrics
from sklearn.model_selection import train_test_split  # train / test
from datetime import datetime
import pandas as pd

#######################################################################################################################
df_sample = pd.read_pickle('data_process/apt_data/machine_learning/seoul_sampling_1000unit.pkl')

df_sample = df_sample.dropna()
X = df_sample.iloc[:, 1:]
y = df_sample.iloc[:, 0:1]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=2)


class PredictionANN:
    def __init__(self, x_train, y_train, x_valid, y_valid, x_test, y_test):
        # Created by Kim Hyeongjun on 15/09/2020.
        # Copyright ©️ 2021 dr-hkim.github.io. All rights reserved.
        # StopWatch: 코드 시작
        time_PredictionANN_start = datetime.now()
        print("PredictionANN started at: " + str(time_PredictionANN_start))

        self.model_name = "NN"

        # 랜덤 시드 고정
        ann_seed_num = 50
        np.random.seed(ann_seed_num)
        tf.random.set_seed(ann_seed_num)
        rn.seed(ann_seed_num)

        # ANN 모형 설정 input output 차원 맞추기에 주의
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=16, activation='relu', input_shape=(x_train.shape[-1],)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(units=1, activation='sigmoid'),
        ])

# 메트릭 정의
        my_metrics = [
            tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
            tf.keras.metrics.MeanSquaredError(name='mse'),
            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
            tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
        ]

        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                           optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                           metrics=my_metrics)

        self.model.summary()

        # 조기종료(early stopping) 정의
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', verbose=1, patience=10, mode='max', restore_best_weights=True)

        # ANN 네트워크 훈련 실시 shuffle=False,
        self.history = self.model.fit(x_train, y_train, epochs=100, batch_size=2048, verbose=2,
                                      callbacks=[early_stopping], validation_data=(x_valid, y_valid))

        # 훈련 결과를 이용하여 예측 실시
        predicted_Y_prob0 = self.model.predict(x_test, batch_size=2048)
        self.predicted_Y_prob = predicted_Y_prob0.reshape(np.shape(predicted_Y_prob0)[0],)
        self.predicted_Y = np.where(self.predicted_Y_prob > 0.5, 1, 0)

        # 예측 성과(Performance) 평가
        self.accuracy = metrics.accuracy_score(y_test, self.predicted_Y)
        self.precision = metrics.precision_score(y_test, self.predicted_Y)
        self.recall = metrics.recall_score(y_test, self.predicted_Y)
        self.f1 = metrics.f1_score(y_test, self.predicted_Y)
        self.auc = metrics.roc_auc_score(y_test, self.predicted_Y)

        # 실제값
        self.y_test = y_test

        # StopWatch: 코드 종료
        time_PredictionANN_end = datetime.now()
        print("PredictionANN finished at: " + str(time_PredictionANN_end))
        print("Elapsed (in PredictionANN): " + str(time_PredictionANN_end - time_PredictionANN_start))

# 클래스 호출하여 사용
tmp = PredictionANN(x_train, y_train, x_valid, y_valid, x_test, y_test)
tmp.accuracy
tmp.precision