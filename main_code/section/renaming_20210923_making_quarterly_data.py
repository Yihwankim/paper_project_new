# for matching
# edit the file name to number

# import packages
from tqdm import tqdm
import pickle
import pandas as pd
import datetime
import numpy as np


########################################################################################################################

'''
49 * 3 으로 실거래가 데이터를 1부터 147으로 명명 
따라서 2009년 1월의 실거래가 자료가 1의 값을 갖게된다.
'''

length = 126
# df_dataset.to_pickle("./data_raw/df_dataset_" + str(yyyymm) + ".pkl")
for i in tqdm(range(length)):
    data01 = pd.read_pickle('real_transaction_data/df_dataset_' + str(i + 1) + '.pkl')
    data01.to_pickle("./real_transaction2/df_dataset_" + str(i + 22) + ".pkl")

