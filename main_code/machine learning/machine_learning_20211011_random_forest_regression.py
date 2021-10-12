# 랜덤 포레스트 회귀
## 참고: https://joyfuls.tistory.com/62
## 전반적으로 위의 링크에서 사용한 방법을 차용했습니다.

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

########################################################################################################################
# 전체적인 모델링 순서
# 참고: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

# 1. data loading
# 전체 데이터 중 1000개를 무작위로 sampling
# 설명변수의 갯수는 72개: 물리적 특성변수 20개, 거리변수 3개, 지역 더미변수 25개, 시간 더미변수 24개
# 종속변수는 per_Pr --> log(per_Pr)
df_sample = pd.read_pickle('data_process/apt_data/machine_learning/seoul_sampling_1000unit.pkl')
# df_sample = pd.read_pickle('data_process/apt_data/machine_learning/seoul_full_sample(district+half).pkl') #  풀 샘플
df_sample = df_sample.dropna()
X = df_sample.iloc[:, 3:]  # 'gu' 와 'dong' 그리고 종속변수를 제외한 나머지 값들을 설명변수로 입력
y = df_sample.iloc[:, 2:3]  # per_Price (면적당 가격)을 종속변수로 입력

# 2. normalization
y = np.log(y)  # per_Pr 에 log 를 취한 값을 최종 종속변수로 선정
y.columns = ['log_per_Pr']


# 3. model 생성
# 참고: https://statkclee.github.io/model/model-python-predictive-model.html
features = int(np.sqrt(72))  # tree 에 들어가는 변수의 갯수 선정

for number_estimator in [100, 150, 200]:
    print('Case: number of estimators = ' + str(number_estimator))
    model = RandomForestRegressor(n_estimators=number_estimator, max_features=features, criterion='mse', random_state=2)
    # n_estimators: 랜덤 포레스트 안의 결정 트리 갯수
    # max_features: 무작위로 선택할 Feature 의 개수

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    # test_size => test sample 의 비율

    # 4. model 학습
    model.fit(x_train, y_train)

    # 5. model 검증
    score = model.score(x_train, y_train)
    print('R-squared: ', score)

    # 6. model 예측
    y_pred = model.predict(x_test)

    # 7. model 평가
    mse = mean_squared_error(y_pred, y_test)
    rmse = mse**(1/2)
    print('MSE = ', mse)
    print('RMSE = ', rmse)

    df = pd.DataFrame({'y_true': y_test['log_per_Pr'], 'y_pred': y_pred})
    cor = df['y_true'].corr(df['y_pred'])
    print('Correlation = ', cor)
    print('###########################################################')

    # 8. plot 그리기
    # 참고: https://www.datatechnotes.com/2020/09/regression-example-with-randomforestregressor.html
    x_axis = range(len(df['y_true']))
    plt.plot(x_axis, df['y_true'], linewidth=1, label="original")
    plt.plot(x_axis, df['y_pred'], linewidth=1.1, label="predicted")
    plt.title("y-test and y-predicted data " + 'case: ' + str(number_estimator))
    plt.xlabel('X-axis: # data')
    plt.ylabel('Y-axis: value')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.figure()
