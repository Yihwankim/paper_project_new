# 2021-05-07
# Chapter 4

# melt 가 이루어진 edit_4 엑셀 파일들을 하나의 엑셀파일로 통합하기
# 이때 각 컬럼들을 숫자, 문자열 등으로 구분하여 타입을 재지정해주기


# Import packages
import pandas as pd
import numpy as np

########################################################################################################################
# 엑셀 파일 한번에 불러오기
# 강북 엑셀파일 불러오기

filenames_GB = [
    'Dobonggu_edit4', 'Dongdaemoongu_edit4',
    'Eunpyeonggu_edit4', 'Gangbukgu_edit4',
    'Gwangjingu_edit4', 'Jongnogu_edit4',
    'Junggu_edit4', 'Jungnanggu_edit4',
    'Mapogu_edit4', 'Nowongu_edit4',
    'Seodaemungu_edit4', 'Seongbukgu_edit4',
    'Seongdonggu_edit4', 'Yongsangu_edit4'
]

dfs = []

for fname in filenames_GB:
    print('Loading {}'.format(fname))

    df = pd.read_excel('Gangbuk_edit4/{}.xlsx'.format(fname))
    # df.columns = [fname]

    dfs.append(df)

print('Data loading is completed!')

df_GB = pd.concat(dfs, axis=0)  # axis=0 : 밑으로 붙이기

# 강남 엑셀파일 불러오기

filenames_GN = [
    'Dongjakgu_edit4', 'Gangdonggu_edit4',
    'Gangnamgu_edit4', 'Gangseogu_edit4',
    'Geumcheongu_edit4', 'Gurogu_edit4',
    'Gwanakgu_edit4', 'Seochogu_edit4',
    'Songpagu_edit4', 'Yangcheongu_edit4',
    'Yeongdeungpogu_edit4']

dfs2 = []

for fname in filenames_GN:
    print('Loading {}'.format(fname))

    df = pd.read_excel('Gangnam_edit4/{}.xlsx'.format(fname))
    # df.columns = [fname]

    dfs2.append(df)

print('Data loading is completed!')

df_GN = pd.concat(dfs2, axis=0)  # axis=0 : 밑으로 붙이기

########################################################################################################################

# 강북데이터와 강남데이터를 각각 edit4 폴더에 새롭게 저장한 후 합치기

df_GB.to_excel('Gangbuk_edit4/Gangbuk_total.xlsx', sheet_name='Gangbuk', index=False)
df_GN.to_excel('Gangnam_edit4/Gangnam_total.xlsx', sheet_name='Gangnam', index=False)

df_GB = pd.read_excel('Gangbuk_edit4/Gangbuk_total.xlsx', header=0, skipfooter=0)
df_GN = pd.read_excel('Gangnam_edit4/Gangnam_total.xlsx', header=0, skipfooter=0)


df_seoul = pd.concat([df_GB, df_GN], axis=0)

df_seoul = df_seoul.sort_values(by=['Apt_name'])
df_seoul = df_seoul.reset_index(drop='Ture')

# seoul data 편집하기

df_seoul = df_seoul.replace('-', np.nan)

df = df_seoul[['Gu', '읍면동', 'Apt_name']]
df.columns = ['지역구', '법정동', '아파트']

# number 편집
se_number = df_seoul['number'].str.split('(', n=1, expand=True)
df['세대수'] = se_number[0]
df['세대수'] = df['세대수'].str.slice(start=0, stop=-2)
df['세대수'] = pd.to_numeric(df['세대수'])

# floor 편집
df[['저층', '고층']] = df_seoul['floor'].str.split('/', expand=True)
df['저층'] = df['저층'].str.slice(start=0, stop=-1)
df['저층'] = pd.to_numeric(df['저층'])
df['고층'] = df['고층'].str.slice(start=0, stop=-1)
df['고층'] = pd.to_numeric(df['고층'])

# car 편집
df[['주차대수_총', '주차대수_세대']] = df_seoul['car'].str.split('(', n=1, expand=True)
df['주차대수_총'] = df['주차대수_총'].str.slice(start=0, stop=-1)
df['주차대수_총'] = pd.to_numeric(df['주차대수_총'])

df['주차대수_세대'] = df['주차대수_세대'].str.slice(start=4, stop=-2)
df['주차대수_세대'] = pd.to_numeric(df['주차대수_세대'])

# FAR (용적률) 편집
df['용적률'] = pd.to_numeric(df_seoul['FAR'].str.slice(start=0, stop=-1))

# BC (건폐율) 편집
df['건폐율'] = pd.to_numeric(df_seoul['BC'].str.slice(start=0, stop=-1))

# 건설사, 난방
df[['건설사', '난방']] = df_seoul[['con', 'heat']]

# 아파트코드, 위도, 경도
df['아파트코드'] = pd.to_numeric(df_seoul['code'])
df['위도'] = pd.to_numeric(df_seoul['lat'])
df['경도'] = pd.to_numeric(df_seoul['long'])

# 면적정보
df['면적유형'] = df_seoul['type_capacity']

# 크기
area_inform = df_seoul['area'].str.split('/')
df['공급면적(㎡)'] = area_inform.str.get(0)
df['공급면적(㎡)'] = df['공급면적(㎡)'].str.slice(start=0, stop=-1)
# df['공급면적(㎡)'] = pd.to_numeric(df['공급면적(㎡)'])  # 오류발생

area_inform2 = area_inform.str.get(1)
area_inform2 = area_inform2.str.split('(')

df['전용면적(㎡)'] = area_inform2.str.get(0)
df['전용면적(㎡)'] = df['전용면적(㎡)'].str.slice(start=0, stop=-1)
df['전용면적(㎡)'] = pd.to_numeric(df['전용면적(㎡)'])

df['전용률(%)'] = area_inform2.str.get(1)
df['전용률(%)'] = df['전용률(%)'].str.slice(start=4, stop=-2)

# 방, 화장실 개수
df['방 개수'] = df_seoul['room'].str.slice(start=0, stop=-1)
# df['방 개수'] = pd.to_numeric(df['방 개수'])  # 오류발생

df['화장실 개수'] = df_seoul['toilet'].str.slice(start=0, stop=-1)
# df['화장실 개수'] = pd.to_numeric(df['화장실 개수'])  # 오류발생

# 구조, 해당면적 세대수
df['구조'] = df_seoul['structure']

df['해당면적 세대수'] = df_seoul['n_this_area'].str.slice(start=0, stop=-2)
# df['해당면적 세대수'] = pd.to_numeric(df['해당면적 세대수'])  # 오류발생

# 사용승인일 참고[https://m.blog.naver.com/wideeyed/221603778414]
df['사용승인일'] = df_seoul['confirm_date']
df.replace({'사용승인일': {'년 ': '-'}}, inplace=True)
df['사용승인일'] = df['사용승인일'].str.replace(pat='년 ', repl='-', regex=False)
df['사용승인일'] = df['사용승인일'].str.replace(pat='월', repl='', regex=False)
df['사용승인일'] = df['사용승인일'].str.replace(pat='월 ', repl='-', regex=False)
df['사용승인일'] = df['사용승인일'].str.replace(pat='일', repl='', regex=False)

df['사용승인일'] = pd.to_datetime(df['사용승인일'])

# 해당면적 세대수, 화장실 개수, 방 개수, 공급면적이 숫자로 처리되지 않고 문자열로 처리되어 있는 상태
# 확인 및 변경이 요구됨


# 열 순서 변경
df = df[['지역구', '법정동', '아파트', '아파트코드', '사용승인일', '세대수', '저층', '고층', '주차대수_총', '주차대수_세대', '용적률',
         '건폐율', '위도', '경도', '건설사', '난방', '구조', '면적유형', '해당면적 세대수',
         '공급면적(㎡)', '전용면적(㎡)', '전용률(%)', '방 개수',
         '화장실 개수']]

df.to_excel('Seoul.xlsx', sheet_name='total', index=False)


