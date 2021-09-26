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
# 데이터 권역별로 분류하기
length = 5
for i in tqdm(range(length)):
    num = df_all['G' + str(i+1)] == 1
    df_place = df_all[num]
    df_district = df_place[['gu', 'dong', 'per_Pr', 'year', 'year_sq', 'num', 'car', 'car_per', 'area', 'room',
                            'toilet', 'floor', 'floor_sq', 'H1', 'H2', 'H3', 'T1', 'T2', 'T3', 'C1', 'FAR', 'BC',
                            'Efficiency', 'dist_elem', 'dist_middle', 'dist_high', 'dist_sub', 'dist_park',
                            'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13',
                            'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25',
                            'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32', 'D33', 'D34', 'D35', 'D36', 'D37',
                            'D38', 'D39', 'D40', 'D41', 'D42', 'D43', 'D44', 'D45', 'D46', 'D47', 'D48',
                            'lat', 'long']]

    df_district.to_excel('data_process/district_edit/seoul_apt_G' + str(i+1) + '.xlsx')




