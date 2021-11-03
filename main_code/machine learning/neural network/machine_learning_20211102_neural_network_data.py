# Import packages
import numpy as np
import pandas as pd


#######################################################################################################################
df_data = pd.read_pickle('data_process/conclusion/sample/hedonic_full_data.pkl')

# normalization 이 필요한 변수 select
indep_var = ['per_Pr', 'old', 'old_sq', 'car_per', 'area', 'room', 'toilet', 'floor', 'floor_sq',
             'FAR', 'BC', 'Efficiency', 'H1', 'H2', 'T1', 'T2', 'C1', 'first', 'log_num']

distance_var = ['dist_high', 'dist_sub', 'dist_park']

gu = []
len_gu = 25  # i
len_time = 24  # j
for i in range(len_gu):
    a = 'GU' + str(i+1)
    gu.append(a)

time = []
for i in range(len_time):
    b = 'Half' + str(i+1)
    time.append(b)

variables = indep_var + distance_var +gu[1:] + time[1:]

df_without = df_data[variables]

df_without.to_pickle('data_process/conclusion/NN/normalization_without_interaction.pkl')

#######################################################################################################################
# with interaction term
len_gu = 25  # i
len_time = 24  # j

inter = []
for i in range(len_gu):
    for j in range(len_time):
        c = 'i' + str(i+1) + ',' + str(j+1)
        inter.append(c)

variables2 = indep_var + distance_var + inter[1:]

df_with = df_data[variables2]

df_with.to_pickle('data_process/conclusion/NN/normalization_with_interaction.pkl')