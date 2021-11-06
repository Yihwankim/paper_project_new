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
# 결과값을 저장할 set
sample_name = ['without', 'with']

for i in sample_name:
    if i == 'without':

        # StopWatch: 코드 시작
        time_PredictionANN_start = datetime.now()
        print("PredictionANN started at: " + str(time_PredictionANN_start))

        df_data = pd.read_pickle('data_process/conclusion/NN/normalization_' + i + '_interaction.pkl')

        df_data = df_data.dropna()

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
        # With normalization
        model1 = Sequential()

        model1.add(Dense(X_train.shape[1], kernel_initializer='normal', activation='relu'))
        model1.add(Dense(150, kernel_initializer='normal', activation='relu'))
        # model.add(Dropout(0.2))

        model1.add(Dense(75, kernel_initializer = 'normal', activation='relu'))
        # model.add(Dropout(0.2))

        model1.add(Dense(50, kernel_initializer = 'normal', activation='relu'))
        # model.add(Dropout(0.2))

        model1.add(Dense(10, kernel_initializer = 'normal', activation='relu'))
        model1.add(Dropout(0.1))
        model1.add(Dense(1))

        model1.compile(
            optimizer=Adam(0.00001),
            loss=tf.keras.losses.mean_squared_error)

        without_model = model1.fit(X_train, y_train, validation_data=(X_valid, Y_valid), batch_size=10, epochs=100)

        df_data = pd.read_excel('data_process/conclusion/NN/nn_index_data_no_interaction.xlsx', header=0, skipfooter=0)
        X_without = df_data.copy()

        # sample set to array
        X_without_test = np.array(X_without)

        index_pred = model1.predict(X_without_test)
        df_without = pd.DataFrame(index_pred)

        # StopWatch: 코드 종료
        time_PredictionANN_end = datetime.now()
        print("PredictionANN finished at: " + str(time_PredictionANN_end))
        print("Elapsed (in PredictionANN): " + str(time_PredictionANN_end - time_PredictionANN_start))

    else:

        # StopWatch: 코드 시작
        time_PredictionANN_start = datetime.now()
        print("PredictionANN started at: " + str(time_PredictionANN_start))

        df_data = pd.read_pickle('data_process/conclusion/NN/normalization_' + i + '_interaction.pkl')

        df_data = df_data.dropna()

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
        # With normalization
        model2 = Sequential()

        model2.add(Dense(X_train.shape[1], kernel_initializer='normal', activation='relu'))
        model2.add(Dense(150, kernel_initializer='normal', activation='relu'))
        # model.add(Dropout(0.2))

        model2.add(Dense(75, kernel_initializer='normal', activation='relu'))
        # model.add(Dropout(0.2))

        model2.add(Dense(50, kernel_initializer='normal', activation='relu'))
        # model.add(Dropout(0.2))

        model2.add(Dense(10, kernel_initializer='normal', activation='relu'))
        model2.add(Dropout(0.1))
        model2.add(Dense(1))

        model2.compile(
            optimizer=Adam(0.00001),
            loss=tf.keras.losses.mean_squared_error)

        with_model = model2.fit(X_train, y_train, validation_data=(X_valid, Y_valid), batch_size=10, epochs=100)

        df_data = pd.read_excel('data_process/conclusion/NN/nn_index_data_interaction.xlsx', header=0, skipfooter=0)
        X_with = df_data.copy()

        # sample set to array
        X_with_test = np.array(X_with)

        index_pred = model2.predict(X_with_test)
        df_with = pd.DataFrame(index_pred)

        # StopWatch: 코드 종료
        time_PredictionANN_end = datetime.now()
        print("PredictionANN finished at: " + str(time_PredictionANN_end))
        print("Elapsed (in PredictionANN): " + str(time_PredictionANN_end - time_PredictionANN_start))

# 지수화를 위한 세팅
df_without.columns = ['without']
df_without['real_values'] = np.exp(df_without['without'])
df_without['index'] = (df_without['real_values']/df_without['real_values'].loc[0]) *100

df_with.columns = ['with']
df_with['real_values'] = np.exp(df_with['with'])

length = 600
df_with['index'] = 0
for i in range(length):
    df_with['index'].loc[i] = (df_with['real_values'].loc[i]/df_with['real_values'].loc[int(i/24)*24]) *100

df_without.to_excel('data_process/conclusion/NN/nn_index_without.xlsx')
df_with.to_excel('data_process/conclusion/NN/nn_index_with.xlsx')