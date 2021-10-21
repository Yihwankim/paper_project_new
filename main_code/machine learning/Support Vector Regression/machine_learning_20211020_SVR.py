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
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, cross_validate

########################################################################################################################
# data loading
df_sample = pd.read_pickle('data_process/apt_data/machine_learning/seoul_sampling_1000unit.pkl')  # 1000 개 서브샘플
df_sample = df_sample.dropna()
X = df_sample.iloc[:, 1:]  # 'gu' 와 'dong' 그리고 종속변수를 제외한 나머지 값들을 설명변수로 입력
y = df_sample.iloc[:, 0:1]  # per_Price (면적당 가격)을 종속변수로 입력

# 2. normalization
y = np.log(y)  # per_Pr 에 log 를 취한 값을 최종 종속변수로 선정
y.columns = ['log_per_Pr']

svr_outcome = pd.DataFrame()

cross_validate_score = []
mean_sq_er = []
root_mse = []
correlation = []
mean_ape = []

# 3. model 생성
model_type = ['linear', 'poly', 'rbf']
for type_kernel in model_type:
    model = SVR(kernel=type_kernel)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    # 4. model 학습
    model.fit(x_train, y_train)

    # 5. model 검증
    scores = cross_val_score(model, x_train, y_train, cv=5)
    cross_validate_score.append(scores)

    # 6. model 예측
    y_pred = model.predict(x_test)

    # 7. model 평가
    mse = mean_squared_error(y_pred, y_test)
    mean_sq_er.append(mse)
    rmse = mse ** (1 / 2)
    root_mse.append(rmse)
    mape = np.mean(np.abs((y_test['log_per_Pr'] - y_pred) / y_test['log_per_Pr'])) * 100
    mean_ape.append(mape)

    df = pd.DataFrame({'y_true': y_test['log_per_Pr'], 'y_pred': y_pred})
    cor = df['y_true'].corr(df['y_pred'])
    correlation.append(cor)

    # 8. plot 그리기
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
svr_outcome['R_squared'] = cross_validate_score
svr_outcome['MSE'] = mean_sq_er
svr_outcome['RMSE'] = root_mse
svr_outcome['Correlation'] = correlation
svr_outcome['MAPE'] = mean_ape
svr_outcome = svr_outcome.rename(index={0: 'linear', 1: 'polynomial', 2: 'rbf'})

#####################################################################################################################


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
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, cross_validate

########################################################################################################################
# data loading
df_sample = pd.read_pickle('data_process/apt_data/machine_learning/seoul_sampling_1000unit.pkl')  # 1000 개 서브샘플
df_sample = df_sample.dropna()
X = df_sample.iloc[:, 1:]  # 'gu' 와 'dong' 그리고 종속변수를 제외한 나머지 값들을 설명변수로 입력
y = df_sample.iloc[:, 0:1]  # per_Price (면적당 가격)을 종속변수로 입력

# 2. normalization
y = np.log(y)  # per_Pr 에 log 를 취한 값을 최종 종속변수로 선정
y.columns = ['log_per_Pr']

svr_outcome = pd.DataFrame()

cross_validate_score = []
mean_sq_er = []
root_mse = []
correlation = []
mean_ape = []

# 3. model 생성
model = SVR(kernel='linear')
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# 4. model 학습
model.fit(x_train, y_train)

# 5. model 검증
score = model.score(x_train, y_train)  # 학습된 모델의 설명계수 값
pd.DataFrame(cross_validate(model, X, y, cv=5))

# 6. model 예측
y_pred = model.predict(x_test)

# 7. model 평가
mse = mean_squared_error(y_pred, y_test)
mean_sq_er.append(mse)
rmse = mse ** (1 / 2)
root_mse.append(rmse)
mape = np.mean(np.abs((y_test['log_per_Pr'] - y_pred) / y_test['log_per_Pr'])) * 100
mean_ape.append(mape)

df = pd.DataFrame({'y_true': y_test['log_per_Pr'], 'y_pred': y_pred})
cor = df['y_true'].corr(df['y_pred'])
correlation.append(cor)

# 8. plot 그리기
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
svr_outcome['R_squared'] = cross_validate_score
svr_outcome['MSE'] = mean_sq_er
svr_outcome['RMSE'] = root_mse
svr_outcome['Correlation'] = correlation
svr_outcome['MAPE'] = mean_ape
svr_outcome = svr_outcome.rename(index={0: 'linear', 1: 'polynomial', 2: 'rbf'})
