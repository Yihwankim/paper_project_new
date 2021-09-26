# Chapter 8
# 더미변수를 추가한 seoul apt data 와 실거래가 데이터를 matching
# 이 과정에서 층 정보가 포함되고, 같은 면적유형의 아파트여도 실거래 측면에서는 각기 다른 아파트로 분류될 수 있기 때문에 데이터 수가 확장됨
# 매칭을 하게 하는 key 값은 법정동 + 전용면적 유형 + 건축년도로 구성되어 있음
# 예를들어 2010년에 건축된 회기동 85m^2의 아파트는 실거래가에서 동일 정보를 갖는 아파트와 동일한 정보로 취급

# import packages
import pickle
from tqdm import tqdm
import pandas as pd
import datetime
import numpy as np

########################################################################################################################
# 크롤링 데이터 불러오기
df_seoul = pd.read_excel('Hedonic_index/seoul_replicating_dummy.xlsx', header=0, skipfooter=0)

# 매칭을 위해 KEY 값 만들기
df_seoul['건축년도'] = df_seoul['사용승인일'].dt.year
# 참고 :
# https://www.delftstack.com/ko/howto/python-pandas/how-to-extract-month-and-year-separately-from-datetime-column-in-pandas/

df_seoul['전용면적'] = round(df_seoul['전용면적'], 2)
df_seoul['전용면적1'] = df_seoul['전용면적']

# df_seoul['전용면적1'] = df_seoul['전용면적'].astype(int)
# 참고 : https://www.javaer101.com/ko/article/54022504.html

# 데이터 값을 합쳐주기 위해 문자열로 지정
df_seoul['건축년도'] = df_seoul['건축년도'].astype(str)
df_seoul['전용면적1'] = df_seoul['전용면적1'].astype(str)

# 'matching' KEY 값 만들기
df_seoul['matching'] = df_seoul['법정동'] + " " + df_seoul['전용면적1'] + " " + df_seoul['건축년도']
df_seoul = df_seoul.sort_values(by=['matching'])

########################################################################################################################
length = 49  # 2009년 1분기부터 2021년 1분기까지의 데이터를 활용할 예정, 총 42분기 월단위로는 42*3=126

for i in tqdm(range(length)):
    if i == 0:
        data01 = pd.read_pickle('real_transaction2/df_dataset_' + str(i + 1) + '.pkl')
        data02 = pd.read_pickle('real_transaction2/df_dataset_' + str(i + 2) + '.pkl')
        data03 = pd.read_pickle('real_transaction2/df_dataset_' + str(i + 3) + '.pkl')

        # 월별 데이터를 각 분기별 데이터로 만들기
        data_quarter = pd.concat([data01, data02, data03], axis=0)

        # 크롤링 데이터와 매칭 시키기
        df = data_quarter.loc[data_quarter['법정동시군구코드'].str.contains("11")]  # 서울지역은 11 포함
        df = df.sort_values(by=['법정동시군구코드'])
        df = df.reset_index(drop='Ture')

        df = df[df['법정동시군구코드'].str.startswith("11")]

        df_edit = df[['아파트', '법정동', '전용면적', '건축년도', '거래금액', '월', '층']]

        # 숫자로 바꾸어 추출
        df_edit['전용면적'] = pd.to_numeric(df_edit['전용면적'])
        df_edit['전용면적1'] = round(df_edit['전용면적'], 2)
        df_edit['전용면적1'] = df_edit['전용면적1'].astype(str)  # matching 을 위해 다시 합치기

        df_edit['법정동'] = df_edit['법정동'].str.slice(start=1)

        # 'matching' KEY 값 만들기
        df_edit['matching'] = df_edit['법정동'] + " " + df_edit['전용면적1'] + " " + df_edit['건축년도']
        df_edit['matching'].astype('str')
        df_edit = df_edit.sort_values(by=['matching'])

        df_price_quarter = df_edit[['matching', '아파트', '거래금액', '월', '층']]
        df_price_quarter = df_price_quarter.sort_values(by=['층'])
        # 간혹 층이 음수값으로 기입된 경우가 있을 수 있으니 별도로 확인이 필요하다.

        # 'matching' 을 기준으로 합쳐보기
        df_seoul_1q = pd.merge(df_seoul, df_price_quarter, on="matching")

        # 데이터 전처리
        # 중복제거 (1) : 동일 아파트에서 비슷한 유형의 경우 같은 매물로 묶이는 문제를 제거하기 위함
        df_seoul_1q['sorting1'] = df_seoul_1q['아파트_y'] + " " + df_seoul_1q['층'] + df_seoul_1q['거래금액']
        df_seoul_1q = df_seoul_1q.sort_values(by=['sorting1'])
        df_seoul_1q = df_seoul_1q.drop_duplicates(['sorting1'], keep='first')

        # 중북제거 (2) : 데이터 중에서 KEY 값과 층 정보가 같다면 동일한 매물로서 처리하여 중복값 제거
        df_seoul_1q['sorting2'] = df_seoul_1q['matching'] + " " + df_seoul_1q['층']
        df_seoul_1q = df_seoul_1q.sort_values(by=['sorting2'])
        df_seoul_1q = df_seoul_1q.drop_duplicates(['sorting2'], keep='first')

        # 세부적인 조정
        df_seoul_Q = df_seoul_1q[['지역구', '법정동', '아파트_x', '아파트_y', '아파트코드', '사용승인일', '연수', '세대수',
                                  '저층', '고층', '주차대수_총', '주차대수_세대', '용적률', '건폐율', '위도', '경도', '건설사',
                                  '난방', '구조', '면적유형', '전용면적', '전용률', '방개수', '화장실개수',
                                  'dist_elem', 'dist_middle', 'dist_high', 'dist_sub', 'dist_park', '층', '거래금액',
                                  'H1', 'H2', 'H3', 'T1', 'T2', 'T3', 'G1']]

        df_seoul_Q['층'] = pd.to_numeric(df_seoul_Q['층'])
        df_seoul_Q['거래금액'].iloc[0]
        '''
        df_seoul_1Q['거래금액'] = df_seoul_1Q['거래금액'].str.slice(start=4)
        df_seoul_1Q['거래금액'].iloc[0]

        cost = df_seoul_1Q['거래금액'].str.split(',')
        df_seoul_1Q['거래금액'] = cost.str.get(0) + cost.str.get(1)
        df_seoul_1Q['거래금액'] = pd.to_numeric(df_seoul_1Q['거래금액'])
        '''

        # df_seoul_2020_4q = df_seoul_2020_4q.sort_values(by=['아파트_x'])

        df_seoul_Q = df_seoul_Q.dropna(axis=0)

        number = len(df_seoul_Q['거래금액'])
        df_seoul_Q['Time'] = np.linspace(i + 1, i + 1, number)

        df_index = df_seoul_Q

    else:
        data01 = pd.read_pickle('real_transaction2/df_dataset_' + str(i * 3 + 1) + '.pkl')
        data02 = pd.read_pickle('real_transaction2/df_dataset_' + str(i * 3 + 2) + '.pkl')
        data03 = pd.read_pickle('real_transaction2/df_dataset_' + str(i * 3 + 3) + '.pkl')

        # 월별 데이터를 각 분기별 데이터로 만들기
        data_quarter = pd.concat([data01, data02, data03], axis=0)

        # 크롤링 데이터와 매칭 시키기
        df = data_quarter.loc[data_quarter['법정동시군구코드'].str.contains("11")]  # 서울지역은 11 포함
        df = df.sort_values(by=['법정동시군구코드'])
        df = df.reset_index(drop='Ture')

        df = df[df['법정동시군구코드'].str.startswith("11")]

        df_edit = df[['아파트', '법정동', '전용면적', '건축년도', '거래금액', '월', '층']]

        # 숫자로 바꾸어 추출
        df_edit['전용면적'] = pd.to_numeric(df_edit['전용면적'])
        df_edit['전용면적1'] = round(df_edit['전용면적'], 2)
        df_edit['전용면적1'] = df_edit['전용면적1'].astype(str)  # matching 을 위해 다시 합치기

        df_edit['법정동'] = df_edit['법정동'].str.slice(start=1)

        # 'matching' KEY 값 만들기
        df_edit['matching'] = df_edit['법정동'] + " " + df_edit['전용면적1'] + " " + df_edit['건축년도']
        df_edit['matching'].astype('str')
        df_edit = df_edit.sort_values(by=['matching'])

        df_price_quarter = df_edit[['matching', '아파트', '거래금액', '월', '층']]
        df_price_quarter = df_price_quarter.sort_values(by=['층'])
        # 간혹 층이 음수값으로 기입된 경우가 있을 수 있으니 별도로 확인이 필요하다.

        # 'matching' 을 기준으로 합쳐보기
        df_seoul_q = pd.merge(df_seoul, df_price_quarter, on="matching")

        # 데이터 전처리
        # 중복제거 (1) : 동일 아파트에서 비슷한 유형의 경우 같은 매물로 묶이는 문제를 제거하기 위함
        df_seoul_q['sorting1'] = df_seoul_q['아파트_y'] + " " + df_seoul_q['층'] + df_seoul_q['거래금액']
        df_seoul_q = df_seoul_q.sort_values(by=['sorting1'])
        df_seoul_q = df_seoul_q.drop_duplicates(['sorting1'], keep='first')

        # 중북제거 (2) : 데이터 중에서 KEY 값과 층 정보가 같다면 동일한 매물로서 처리하여 중복값 제거
        df_seoul_q['sorting2'] = df_seoul_q['matching'] + " " + df_seoul_q['층']
        df_seoul_q = df_seoul_q.sort_values(by=['sorting2'])
        df_seoul_q = df_seoul_q.drop_duplicates(['sorting2'], keep='first')

        # 세부적인 조정
        df_seoul_Q = df_seoul_q[['지역구', '법정동', '아파트_x', '아파트_y', '아파트코드', '사용승인일', '연수', '세대수',
                                 '저층', '고층', '주차대수_총', '주차대수_세대', '용적률', '건폐율', '위도', '경도', '건설사',
                                 '난방', '구조', '면적유형', '전용면적', '전용률', '방개수', '화장실개수',
                                 'dist_elem', 'dist_middle', 'dist_high', 'dist_sub', 'dist_park', '층', '거래금액',
                                 'H1', 'H2', 'H3', 'T1', 'T2', 'T3', 'G1']]

        df_seoul_Q['층'] = pd.to_numeric(df_seoul_Q['층'])
        df_seoul_Q['거래금액'].iloc[0]
        '''
        df_seoul_Q['거래금액'] = df_seoul_1Q['거래금액'].str.slice(start=4)
        df_seoul_Q['거래금액'].iloc[0]

        cost = df_seoul_1Q['거래금액'].str.split(',')
        df_seoul_Q['거래금액'] = cost.str.get(0) + cost.str.get(1)
        df_seoul_Q['거래금액'] = pd.to_numeric(df_seoul_1Q['거래금액'])
        '''

        # df_seoul_2020_4q = df_seoul_2020_4q.sort_values(by=['아파트_x'])

        df_seoul_Q = df_seoul_Q.dropna(axis=0)

        number = len(df_seoul_Q['거래금액'])
        df_seoul_Q['Time'] = np.linspace(i + 1, i + 1, number)

        df_index = pd.concat([df_index, df_seoul_Q])

df_index.to_excel('real_transaction2/Seoul_index.xlsx')
