# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

########################################################################################################################
# regression_ data load
df_full = pd.read_pickle('data_process/apt_data/seoul_including_all_variables.pkl')
df_full = df_full.dropna()

# Without 에 대해 각 평균값 구하기
without_inter = pd.DataFrame()
length_time = 24
for i in range(length_time):
    half = df_full['Half' + str(i+1)] == 1
    half_edit = df_full[half]
    summary = half_edit.describe()
    summary_edit = summary.transpose()
    summary_mean = summary_edit['mean']

    without_inter['Half' + str(i+1)] = summary_mean

without_inter = without_inter.transpose()
without_inter.to_excel('data_process/conclusion/predict_data/without_interaction.xlsx')

# With 에 대해 각 평균값 구하기
with_inter = pd.DataFrame()
length_gu = 25

for i in tqdm(range(length_gu)):
    for j in range(length_time):
        inter = df_full['i' + str(i+1) + ',' + str(j+1)] == 1
        inter_edit = df_full[inter]
        summary = inter_edit.describe()
        summary_edit = summary.transpose()
        summary_mean = summary_edit['mean']

        with_inter['i' + str(i+1) + ',' + str(j+1)] = summary_mean

with_inter = with_inter.transpose()
with_inter.to_excel('data_process/conclusion/predict_data/with_interaction.xlsx')












