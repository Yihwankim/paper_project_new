# RandomForest Regressor Practice

# import packages
from sklearn.ensemble import RandomForestRegressor  # 회귀트리 모델
from sklearn.model_selection import train_test_split  # train / test
from sklearn.datasets import fetch_california_housing, load_boston  # dataset
from sklearn.metrics import mean_squared_error  # 평균제곱오차

import numpy as np
import pandas as pd
########################################################################################################################
# 1. dataset loading
X, y = fetch_california_housing(return_X_y=True)
X.shape
y.shape  # Ok

# 2. normalization
y = np.log(y)  # 추후 부동산 데이터로 test 할 때도 종속변수 정규화가 필요

# 3. model 생성
model = RandomForestRegressor()

model.fit(X=X, y=y)

y_pred = model.predict(X)  # 훈련 후 predict 으로 예측하는 y
y_true = y  # 실제 y

# 4-1. model 평가: 평균제곱오차가 작을수록 정확
mse = mean_squared_error(y_true, y_pred)
print('mse = ', mse)

# 4-2. model 평가: 상관관계가 높을수록 정확
df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
cor = df['y_true'].corr(df['y_pred'])

########################################################################################################################
# 1. dataset load
X, y = load_boston(return_X_y=True)
X.shape # (506, 13)

boston = load_boston()
X = boston.data
y = boston.target
colnames = boston.feature_names  # 13개 칼럼 이름 가져올때
colnames
# ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']


# 2. train/test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
x_train.shape  # (354, 13)


# 3. model
model = RandomForestRegressor(n_estimators=400, min_samples_split=3)
model.fit(X = x_train, y = y_train)

RandomForestRegressor()
# 4. model의 중요변수
imp = model.feature_importances_
imp
len(imp) # 13 => 13개 칼럼들
colnames

import matplotlib.pyplot as plt
plt.barh(range(13), imp)  # (x, y) # 중요도 (y에 얼마나 영향을 미치는지)
plt.yticks(range(13), colnames)  # 축 이름




