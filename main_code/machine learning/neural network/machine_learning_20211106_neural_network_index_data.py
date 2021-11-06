# Import packages
import pandas as pd

#######################################################################################################################
# 결과값을 저장할 set
sample_name = ['without', 'with']

i = 'without'
df_data = pd.read_pickle('data_process/conclusion/NN/normalization_' + i + '_interaction.pkl')
df_without = df_data.iloc[0:1, :]
df_without.to_excel('data_process/conclusion/NN/nn_index_data_no_interaction.xlsx')


i = 'with'
df_data = pd.read_pickle('data_process/conclusion/NN/normalization_' + i + '_interaction.pkl')
df_with = df_data.iloc[0:1, :]
df_with.to_excel('data_process/conclusion/NN/nn_index_data_interaction.xlsx')
