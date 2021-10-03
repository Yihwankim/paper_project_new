# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm

#######################################################################################################################
# regression
df = pd.read_pickle('data_process/apt_data/seoul_year_interaction_term.pkl')

len_gu = 25
len_time = 12

number_data = []
for i in tqdm(range(len_time)):
    for j in range(len_gu):
        a = np.sum(df['i' + str(j + 1) + ',' + str(i + 9)])
        number_data.append(a)

np.sum(number_data)




