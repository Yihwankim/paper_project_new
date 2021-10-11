# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#######################################################################################################################
# data
df_full = pd.read_pickle('data_process/apt_data/seoul_year_interaction_term.pkl')
df_sample_1000.to_pickle('data_process/apt_data/machine_learning/seoul_sampling_1000unit.pkl')
df_sample_80.to_pickle('data_process/apt_data/machine_learning/seoul_sampling_800unit.pkl')
df_sample_20.to_pickle('data_process/apt_data/machine_learning/seoul_sampling_200unit.pkl')