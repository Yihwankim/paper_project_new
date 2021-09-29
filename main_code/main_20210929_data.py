# import packages
from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm

### 수정요망: 쥬피터에서 완전 실습 후 수정할 것

#######################################################################################################################
# import data
df_all = pd.read_excel('data_process/apt_data/seoul_full_variable.xlsx', header=0, skipfooter=0)

#######################################################################################################################
# interaction term 만들기
# 지역구 개수 25개
# 시간 더미변수 개수 49개

len_gu = 25  # i
len_time = 49  # j
# 시간과 지역의 interaction term은 1,225개
len_inter = 1225
for i in tqdm(range(len_gu)):
    for j in range(len_time):
        df_all['i' + str(i+1) + ',' + str(j+1)] = df_all['GU' + str(i+1)] * df_all['D' + str(j+1)]

df_all.to_excel('data_process/apt_data/seoul_interaction_term.xlsx')







