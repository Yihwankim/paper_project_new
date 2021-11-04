# 목표)
## 1. 좌표로 시행했을 때와 지역 더미를 추가했을 때 거리 변수를 추가했을 때 예측값 비교
### 1-1. 여기서는 좌표값으로만 했을 때의 결과
## 2. 전체 샘플로 회귀를 실시했을 때 오류가 나는 문제를 해결
## 3. 각 tree 의 갯수별 차이를 분석
## 4. feature selection 의 경우 tree 개수 별로 어떻게 나오는지 보여주기

# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor  # 회귀트리 모델
from sklearn.model_selection import train_test_split  # train / test
from sklearn.datasets import fetch_california_housing, load_boston  # dataset
from sklearn.metrics import mean_squared_error  # 평균제곱오차
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn import set_config
from sklearn.metrics import r2_score

########################################################################################################################
# 전체적인 모델링 순서
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

# 1. data loading
# 전체 데이터 중 1000개를 무작위로 sampling
# 설명변수의 갯수는 72개: 물리적 특성변수 20개, 거리변수 3개, 지역 더미변수 25개, 시간 더미변수 24개
# 종속변수는 per_Pr --> log(per_Pr)

# df_sample = pd.read_pickle('data_process/apt_data/machine_learning/seoul_sampling_1000unit.pkl')  # 1000 개 서브샘플
df_train = pd.read_pickle('data_process/conclusion/sample/rfr_without_train_data.pkl')
df_test = pd.read_pickle('data_process/conclusion/sample/rfr_without_test_data.pkl')

# df_sample = df_sample.dropna()
df_train = df_train.dropna()
df_test = df_test.dropna()

########################################################################################################################
# Sample test
'''X = df_sample.iloc[:, 1:]  # 'gu' 와 'dong' 그리고 종속변수를 제외한 나머지 값들을 설명변수로 입력
y = df_sample.iloc[:, 0:1]  # per_Price (면적당 가격)을 종속변수로 입력

# 2. normalization
y = np.log(y)  # per_Pr 에 log 를 취한 값을 최종 종속변수로 선정
y.columns = ['log_per_Pr']


# 3. model 생성
# 참고: https://statkclee.github.io/model/model-python-predictive-model.html
rfr_outcome = pd.DataFrame()
sum_variable_important = pd.DataFrame()

r_square = []
mean_sq_er = []
root_mse = []
correlation = []
mean_ape = []

features = int(np.sqrt(19))  # tree 에 들어가는 변수의 갯수 선정

for number_estimator in [100, 150, 200]:  # 최적의 tree 갯수를 찾아보기 위해 100, 150, 200개에 대해 test
    print('Case: number of estimators = ' + str(number_estimator))
    model = RandomForestRegressor(n_estimators=number_estimator, max_features=features, criterion='mse', random_state=2)
    # n_estimators: 랜덤 포레스트 안의 결정 트리 갯수
    # max_features: 무작위로 선택할 Feature 의 개수
    # criterion: model 선정 기준_ mse: mean squared error
    # random_state: 일관성있는 샘플링을 위함

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    # test_size => test sample 의 비율
    # train 과 test 의 비율 => train : test = 8:2

    # 4. model 학습
    model.fit(x_train, y_train)  # 상기 조건대로 random_forest_regression_model 학습

    # 5. model 검증
    score = model.score(x_train, y_train)  # 학습된 모델의 설명계수 값
    r_square.append(score)

    # 5-1. model feature selection
    X_columns = X.columns
    importance_var = list(zip(X_columns, model.feature_importances_))
    variable_important = pd.DataFrame(importance_var)
    variable_important = variable_important.sort_values(by=[1], axis=0, ascending=False)
    variable_important = variable_important.reset_index(drop='Ture')

    sum_variable_important = pd.concat([sum_variable_important, variable_important], axis=1)


    # 6. model 예측
    y_pred = model.predict(x_test)  # test sample 의 값을 model 에 넣어 산출한 값

    # 7. model 평가
    mse = mean_squared_error(y_pred, y_test)
    rmse = mse**(1/2)
    mape = np.mean(np.abs((y_test['log_per_Pr'] - y_pred) / y_test['log_per_Pr'])) * 100
    mean_sq_er.append(mse)
    root_mse.append(rmse)
    mean_ape.append(mape)

    df = pd.DataFrame({'y_true': y_test['log_per_Pr'], 'y_pred': y_pred})
    cor = df['y_true'].corr(df['y_pred'])
    correlation.append(cor)  # 예측값과 실제값 사이의 correlation

    # 8. plot 그리기
    # 참고: https://www.datatechnotes.com/2020/09/regression-example-with-randomforestregressor.html
    x_axis = range(len(df['y_true']))
    plt.plot(x_axis, df['y_true'], linewidth=1, label="original")
    plt.plot(x_axis, df['y_pred'], linewidth=1.1, label="predicted")
    plt.title("y-test and y-predicted data " + 'case: ' + str(number_estimator))
    plt.xlabel('X-axis: # data')  # x 축은 각 데이터의 순번
    plt.ylabel('Y-axis: value')  # y 축은 예측 값과 실제 값의 value
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.figure()

# 9. 결과값
rfr_outcome['R_squared'] = r_square
rfr_outcome['MSE'] = mean_sq_er
rfr_outcome['RMSE'] = root_mse
rfr_outcome['Correlation'] = correlation
rfr_outcome['MAPE'] = mean_ape
rfr_outcome = rfr_outcome.rename(index={0: 'case 1: tree100', 1: 'case 2: tree150', 2: 'case 3: tree200'})

rfr_outcome.to_excel('data_process/conclusion/regression_result/rfr_sample_test.xlsx')

sum_variable_important.to_excel('data_process/conclusion/regression_result/rfr_sample_test_featureselection.xlsx')'''
########################################################################################################################
# full sample test
# train 과 test 의 비율 => train : test = 8:2

# 2. normalization
x_train = df_train.iloc[:, 1:]  # 'gu' 와 'dong' 그리고 종속변수를 제외한 나머지 값들을 설명변수로 입력
y_train = np.log(df_train.iloc[:, 0:1])  # per_Price (면적당 가격)을 종속변수로 입력
y_train.columns = ['log_per_Pr']

x_test = df_test.iloc[:, 1:]  # 'gu' 와 'dong' 그리고 종속변수를 제외한 나머지 값들을 설명변수로 입력
y_test = np.log(df_test.iloc[:, 0:1])  # per_Price (면적당 가격)을 종속변수로 입력
y_test.columns = ['log_per_Pr']

# 3. model 생성
# 참고: https://statkclee.github.io/model/model-python-predictive-model.html
rfr_outcome = pd.DataFrame()
sum_variable_important = pd.DataFrame()

r_square = []
mean_sq_er = []
root_mse = []
correlation = []
mean_ape = []
r_square2 = []

features = int(np.sqrt(19))  # tree 에 들어가는 변수의 갯수 선정

model = RandomForestRegressor(n_estimators=100, max_features=features, criterion='mse', random_state=2)
# n_estimators: 랜덤 포레스트 안의 결정 트리 갯수
# max_features: 무작위로 선택할 Feature 의 개수
# criterion: model 선정 기준_ mse: mean squared error
# random_state: 일관성있는 샘플링을 위함

# 4. model 학습
model.fit(x_train, y_train)  # 상기 조건대로 random_forest_regression_model 학습

# 5. model 검증
score = model.score(x_train, y_train)  # 학습된 모델의 설명계수 값
r_square.append(score)

# 5-1. model feature selection
X_columns = x_train.columns
importance_var = list(zip(X_columns, model.feature_importances_))
variable_important = pd.DataFrame(importance_var)
variable_important = variable_important.sort_values(by=[1], axis=0, ascending=False)
variable_important = variable_important.reset_index(drop='Ture')

sum_variable_important = pd.concat([sum_variable_important, variable_important], axis=1)


# 6. model 예측
y_pred = model.predict(x_test)  # test sample 의 값을 model 에 넣어 산출한 값

# 7. model 평가
mse = mean_squared_error(y_pred, y_test)
rmse = mse**(1/2)
mape = np.mean(np.abs((y_test['log_per_Pr'] - y_pred) / y_test['log_per_Pr'])) * 100
mean_sq_er.append(mse)
root_mse.append(rmse)
mean_ape.append(mape)

est_score = r2_score(y_test['log_per_Pr'], y_pred)
r_square2.append(est_score)

df = pd.DataFrame({'y_true': y_test['log_per_Pr'], 'y_pred': y_pred})
cor = df['y_true'].corr(df['y_pred'])
correlation.append(cor)  # 예측값과 실제값 사이의 correlation

# 8. plot 그리기
# 참고: https://www.datatechnotes.com/2020/09/regression-example-with-randomforestregressor.html
x_axis = range(len(df['y_true']))
plt.plot(x_axis, df['y_true'], linewidth=1, label="original")
plt.plot(x_axis, df['y_pred'], linewidth=1.1, label="predicted")
plt.title("y-test and y-predicted data ")
plt.xlabel('X-axis: # data')  # x 축은 각 데이터의 순번
plt.ylabel('Y-axis: value')  # y 축은 예측 값과 실제 값의 value
plt.legend(loc='best', fancybox=True, shadow=True)
plt.grid(True)
plt.figure()

# 9. 결과값
rfr_outcome['R_squared'] = r_square
rfr_outcome['MSE'] = mean_sq_er
rfr_outcome['RMSE'] = root_mse
rfr_outcome['Correlation'] = correlation
rfr_outcome['MAPE'] = mean_ape
rfr_outcome['est_R_squared'] = r_square2

rfr_outcome.to_excel('data_process/conclusion/regression_result/rfr_with_lat_long_test.xlsx')
sum_variable_important.to_excel('data_process/conclusion/regression_result/rfr_mini_test_featureselection.xlsx')

