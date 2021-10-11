# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#######################################################################################################################
# data
# 1000개 sample 과 sample 을 8대 2로 나눈 subsample 2개
df_sample_1000 = pd.read_pickle('data_process/apt_data/machine_learning/seoul_sampling_1000unit.pkl')

df_sample_80 = pd.read_pickle('data_process/apt_data/machine_learning/seoul_sampling_800unit.pkl')
df_sample_20 = pd.read_pickle('data_process/apt_data/machine_learning/seoul_sampling_200unit.pkl')






