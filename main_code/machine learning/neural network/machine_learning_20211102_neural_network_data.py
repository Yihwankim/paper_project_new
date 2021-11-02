# Import packages
import numpy as np
import pandas as pd


#######################################################################################################################
df_data = pd.read_pickle('data_process/conclusion/sample/hedonic_full_data.pkl')

# normalization 이 필요한 변수 select
indep_var = ['per_Pr', 'old', 'old_sq', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq',
             'FAR', 'BC', 'Efficiency']

distance_var = ['dist_high', 'dist_sub', 'dist_park']

df_data_edit = df_data[indep_var + distance_var]

# data normalization
df_data_physical = np.log(df_data_edit)
df_data_norm = df_data_physical.copy()

# add normalization variables
df_data_norm['log_num'] = df_data['log_num']
df_data_norm['first'] = df_data['first']

df_data_norm['H1'] = df_data['H1']
df_data_norm['H2'] = df_data['H2']
df_data_norm['T1'] = df_data['T1']
df_data_norm['T2'] = df_data['T2']
df_data_norm['C1'] = df_data['C1']

len_gu = 25  # i
len_time = 24  # j
for i in range(len_gu):
    a = 'GU' + str(i+1)
    df_data_norm[a] = df_data[a]

for i in range(len_time):
    b = 'Half' + str(i+1)
    df_data_norm[b] = df_data[b]

df_data_norm.to_pickle('data_process/conclusion/NN/normalization_without_interaction.pkl')

#######################################################################################################################
# with interaction term
df_data_norm2 = df_data_physical.copy()

# add normalization variables
df_data_norm2['log_num'] = df_data['log_num']
df_data_norm2['first'] = df_data['first']

df_data_norm2['H1'] = df_data['H1']
df_data_norm2['H2'] = df_data['H2']
df_data_norm2['T1'] = df_data['T1']
df_data_norm2['T2'] = df_data['T2']
df_data_norm2['C1'] = df_data['C1']

len_gu = 25  # i
len_time = 24  # j

for i in range(len_gu):
    for j in range(len_time):
        c = 'i' + str(i+1) + ',' + str(j+1)
        df_data_norm2[c] = df_data[c]

df_data_norm2.to_pickle('data_process/conclusion/NN/normalization_with_interaction.pkl')