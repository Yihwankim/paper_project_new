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

features = 8  # tree 에 들어가는 변수의 갯수 선정

model = RandomForestRegressor(n_estimators=150, max_features=features, criterion='mse', random_state=2)
# n_estimators: 랜덤 포레스트 안의 결정 트리 갯수
# max_features: 무작위로 선택할 Feature 의 개수
# criterion: model 선정 기준_ mse: mean squared error
# random_state: 일관성있는 샘플링을 위함

# 4. model 학습
model.fit(x_train, y_train)  # 상기 조건대로 random_forest_regression_model 학습

# 5. model 예측
df_data_basic_index = pd.read_excel('data_process/conclusion/predict_data/index_predict_input/basic/summary_rfr.xlsx',
                                    header=0, skipfooter=0)
x_index = df_data_basic_index.iloc[:, 1:]
y_basic_index = model.predict(x_index)  # test sample 의 값을 model 에 넣어 산출한 값

########################################################################################################################
df_data_trend_index = pd.read_excel\
    ('data_process/conclusion/predict_data/index_predict_input/trend/with_lat_long_rfr_edit.xlsx',
     header=0, skipfooter=0)
x_index = df_data_trend_index.iloc[:, 1:]
y_trend_index = model.predict(x_index)


# 6. index 도출 및 출력
df_index_basic = pd.DataFrame(y_basic_index)
df_index_basic.columns = ['raw_value']

df_index_basic['real_value'] = np.exp(df_index_basic['raw_value'])
df_index_basic['index'] = (df_index_basic['real_value']/df_index_basic['real_value'].loc[0]) * 100

df_index_basic.to_excel('data_process/conclusion/predict_data/index_predict_output/basic/rfr_basic_latlong_index.xlsx')
########################################################################################################################
df_index_trend = pd.DataFrame(y_trend_index)
df_index_trend.columns = ['raw_value']

df_index_trend['real_value'] = np.exp(df_index_trend['raw_value'])
df_index_trend['index'] = (df_index_trend['real_value']/df_index_trend['real_value'].loc[0]) * 100

df_index_trend.to_excel('data_process/conclusion/predict_data/index_predict_output/trend/rfr_trend_latlong_index.xlsx')