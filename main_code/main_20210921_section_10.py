# Chapter 9-2
# regression
## 09-12, 13-16, 17-20


# import packages
from tqdm import tqdm
import pandas as pd
import datetime
import numpy as np

#####################################################################################################################
dfs1 = []
for i in range(1, 17):
    df = pd.read_excel('real_transaction2/quarterly_edit/seoul_apt' + str(i+1) + '.xlsx', header=0, skipfooter=0)
    df['time'] = i
    dfs1.append(df)
df_09to12 = pd.concat(dfs1, axis=0)
df_09to12.to_excel('real_transaction2/yearly/seoul_apt_09to12.xlsx')

dfs2 = []
for i in range(17, 33):
    df = pd.read_excel('real_transaction2/quarterly_edit/seoul_apt' + str(i + 1) + '.xlsx', header=0, skipfooter=0)
    df['time'] = i
    dfs2.append(df)
df_13to16 = pd.concat(dfs2, axis=0)
df_13to16.to_excel('real_transaction2/yearly/seoul_apt_13to16.xlsx')

dfs3 = []
for i in range(33, 49):
    df = pd.read_excel('real_transaction2/quarterly_edit/seoul_apt' + str(i + 1) + '.xlsx', header=0, skipfooter=0)
    df['time'] = i
    dfs3.append(df)
df_17to20 = pd.concat(dfs3, axis=0)
df_17to20.to_excel('real_transaction2/yearly/seoul_apt_17to20.xlsx')

#####################################################################################################################
for i in range(1, 17):
    df_09to12['D' + str(i)] = np.where(df_09to12['time'] == i, 1, 0)

for i in range(17, 33):
    df_13to16['D' + str(i)] = np.where(df_13to16['time'] == i, 1, 0)

for i in range(33, 49):
    df_17to20['D' + str(i)] = np.where(df_17to20['time'] == i, 1, 0)

df_09to12.to_excel('real_transaction2/yearly_edit/seoul_apt_09to12_edit.xlsx')
df_13to16.to_excel('real_transaction2/yearly_edit/seoul_apt_13to16_edit.xlsx')
df_17to20.to_excel('real_transaction2/yearly_edit/seoul_apt_17to20_edit.xlsx')
