# Import packages
import numpy as np
import tensorflow as tf
import random as rn
from sklearn import metrics
from datetime import datetime

class PredictionANN:
    def __init__(self, train_X, train_Y, valid_X, valid_Y, test_X, test_Y):
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
            tf.keras.layers.Dense(units=16, activation='relu', input_shape=(train_X.shape[-1])),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(units=1, activation='sigmoid'),
        ])

# 메트릭 정의
        my_metrics = [
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
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
        self.history = self.model.fit(train_X, train_Y, epochs=100, batch_size=2048, verbose=2,
                                      callbacks=[early_stopping], validation_data=(valid_X, valid_Y))

        # 훈련 결과를 이용하여 예측 실시
        predicted_Y_prob0 = self.model.predict(test_X, batch_size=2048)
        self.predicted_Y_prob = predicted_Y_prob0.reshape(np.shape(predicted_Y_prob0)[0],)
        self.predicted_Y = np.where(self.predicted_Y_prob > 0.5, 1, 0)

        # 예측 성과(Performance) 평가
        self.accuracy = metrics.accuracy_score(test_Y, self.predicted_Y)
        self.precision = metrics.precision_score(test_Y, self.predicted_Y)
        self.recall = metrics.recall_score(test_Y, self.predicted_Y)
        self.f1 = metrics.f1_score(test_Y, self.predicted_Y)
        self.auc = metrics.roc_auc_score(test_Y, self.predicted_Y)

        # 실제값
        self.test_Y = test_Y

        # StopWatch: 코드 종료
        time_PredictionANN_end = datetime.now()
        print("PredictionANN finished at: " + str(time_PredictionANN_end))
        print("Elapsed (in PredictionANN): " + str(time_PredictionANN_end - time_PredictionANN_start))

# 클래스 호출하여 사용
tmp = PredictionANN(train_X, train_Y, valid_X, valid_Y, test_X, test_Y)
tmp.accuracy
tmp.precision

