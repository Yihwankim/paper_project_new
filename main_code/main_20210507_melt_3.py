# 2021-05-07
# Chapter 3
# edit_3로 만든 파일들을 melt 를 활용하여 면적별 정보로 나열
# 최종적으로 edit_4로 저장하기

# 최종 비교시 고려할 사항
# 1. 아파트 이름(인덱스)과 읍면동, 아파트 정보가 동일한지 여부
# 2. 아파트 세대수가 면적별 세대수의 합과 동일한지 여부
# 3. type of capacity 의 모든 element 가 nan 으로 채워져 있을 경우 오류가 발생하므로 주의


# Import packages
import pandas as pd
import numpy as np

#############################################################################################################
# 엑셀 파일 불러오기
df_data = pd.read_excel('Gangnam_edit3/Gwanakgu_edit3.xlsx')

df_Gu = pd.read_excel('Gangnam_edit3/Gwanakgu_edit3.xlsx', usecols='A:P')

#############################################################################################################
# 6개의 컬럼으로 나누어진 면적별 정보를 하나의 칼럼으로 합쳐서 시리즈로 저장하기
# 해당 시리즈는 앞으로 10개 이상의 시리즈로 확장되어야함
se_type1 = df_data['type_capacity1'].str.cat(df_data[['area1',
                                                      'room1',
                                                      'toilet1',
                                                      'structure1',
                                                      'n_this_area1']], sep=',')

se_type2 = df_data['type_capacity2'].str.cat(df_data[['area2',
                                                      'room2',
                                                      'toilet2',
                                                      'structure2',
                                                      'n_this_area2']], sep=',')

se_type3 = df_data['type_capacity3'].str.cat(df_data[['area3',
                                                      'room3',
                                                      'toilet3',
                                                      'structure3',
                                                      'n_this_area3']], sep=',')

se_type4 = df_data['type_capacity4'].str.cat(df_data[['area4',
                                                      'room4',
                                                      'toilet4',
                                                      'structure4',
                                                      'n_this_area4']], sep=',')

se_type5 = df_data['type_capacity5'].str.cat(df_data[['area5',
                                                      'room5',
                                                      'toilet5',
                                                      'structure5',
                                                      'n_this_area5']], sep=',')

se_type6 = df_data['type_capacity6'].str.cat(df_data[['area6',
                                                      'room6',
                                                      'toilet6',
                                                      'structure6',
                                                      'n_this_area6']], sep=',')

se_type7 = df_data['type_capacity7'].str.cat(df_data[['area7',
                                                      'room7',
                                                      'toilet7',
                                                      'structure7',
                                                      'n_this_area7']], sep=',')

se_type8 = df_data['type_capacity8'].str.cat(df_data[['area8',
                                                      'room8',
                                                      'toilet8',
                                                      'structure8',
                                                      'n_this_area8']], sep=',')

se_type9 = df_data['type_capacity9'].str.cat(df_data[['area9',
                                                      'room9',
                                                      'toilet9',
                                                      'structure9',
                                                      'n_this_area9']], sep=',')

se_type10 = df_data['type_capacity10'].str.cat(df_data[['area10',
                                                        'room10',
                                                        'toilet10',
                                                        'structure10',
                                                        'n_this_area10']], sep=',')

se_type11 = df_data['type_capacity11'].str.cat(df_data[['area11',
                                                        'room11',
                                                        'toilet11',
                                                        'structure11',
                                                        'n_this_area11']], sep=',')

se_type12 = df_data['type_capacity12'].str.cat(df_data[['area12',
                                                        'room12',
                                                        'toilet12',
                                                        'structure12',
                                                        'n_this_area12']], sep=',')

se_type13 = df_data['type_capacity13'].str.cat(df_data[['area13',
                                                        'room13',
                                                        'toilet13',
                                                        'structure13',
                                                        'n_this_area13']], sep=',')

se_type14 = df_data['type_capacity14'].str.cat(df_data[['area14',
                                                        'room14',
                                                        'toilet14',
                                                        'structure14',
                                                        'n_this_area14']], sep=',')

se_type15 = df_data['type_capacity15'].str.cat(df_data[['area15',
                                                        'room15',
                                                        'toilet15',
                                                        'structure15',
                                                        'n_this_area15']], sep=',')

se_type16 = df_data['type_capacity16'].str.cat(df_data[['area16',
                                                        'room16',
                                                        'toilet16',
                                                        'structure16',
                                                        'n_this_area16']], sep=',')

se_type17 = df_data['type_capacity17'].str.cat(df_data[['area17',
                                                        'room17',
                                                        'toilet17',
                                                        'structure17',
                                                        'n_this_area17']], sep=',')

# 오류발생 가능
# type_capacity_number 의 모든 값이 nan 으로 채워져있을 경우 에러가 발생함

# 해당 오류 발생시 밑의 코드를 탄력적으로 수정
# 원본
df_Gu = pd.concat([df_Gu, se_type1, se_type2, se_type3, se_type4, se_type5, se_type6, se_type7, se_type8,
                  se_type9, se_type10, se_type11, se_type12, se_type13, se_type14, se_type15, se_type16, se_type17],
                  axis=1)


#df_Gu = pd.concat([df_Gu, se_type1, se_type2, se_type3, se_type4, se_type5, se_type6, se_type7, se_type8,
#                   se_type9, se_type10, se_type11, se_type12, se_type13],
#                  axis=1)  # 만들어 둔 시리즈를 기존의 데이터 프레임에 합치기

#############################################################################################################

# 하나의 컬럼으로 만들어 둔 면적별 정보(type_capacity)에 따라 아파트 기본 정보들을 stack 시키기
# 에러 처리
# 원본
df_edit = df_Gu.melt(id_vars=['읍면동', '아파트', '세대수', '입주년월', 'Apt_name', 'number', 'floor',
                              'confirm_date', 'car', 'FAR', 'BC', 'con', 'heat', 'code', 'lat',
                              'long'], value_vars=['type_capacity1', 'type_capacity2', 'type_capacity3',
                                                   'type_capacity4', 'type_capacity5', 'type_capacity6',
                                                   'type_capacity7', 'type_capacity8', 'type_capacity9',
                                                   'type_capacity10', 'type_capacity11', 'type_capacity12',
                                                   'type_capacity13', 'type_capacity14', 'type_capacity15',
                                                   'type_capacity16', 'type_capacity17'])

# 수정
#df_edit = df_Gu.melt(id_vars=['읍면동', '아파트', '세대수', '입주년월', 'Apt_name', 'number', 'floor',
#                              'confirm_date', 'car', 'FAR', 'BC', 'con', 'heat', 'code', 'lat',
#                              'long'], value_vars=['type_capacity1', 'type_capacity2', 'type_capacity3',
#                                                   'type_capacity4', 'type_capacity5', 'type_capacity6',
#                                                   'type_capacity7', 'type_capacity8', 'type_capacity9',
#                                                   'type_capacity10', 'type_capacity11', 'type_capacity12',
#                                                   'type_capacity13'])

############################################################################################

df_edit = df_edit.sort_values(by='Apt_name')  # 아파트 이름에 따라 행을 정렬
df_edit2 = df_edit.dropna(axis=0)  # nan 값이 있는 행 제거: 면적이 3개인 아파트의 경우 4~10번째 면적정보는 drop 된다.

type_information = df_edit2['value'].str.split(',')  # 하나의 컬럼으로 만들어둔 면적별 정보를 다시 6개로 나누기

# 데이터 프레임에 각각 고유의 이름으로 합치기
df_edit2['type_capacity'] = type_information.str.get(0)
df_edit2['area'] = type_information.str.get(1)
df_edit2['room'] = type_information.str.get(2)
df_edit2['toilet'] = type_information.str.get(3)
df_edit2['structure'] = type_information.str.get(4)
df_edit2['n_this_area'] = type_information.str.get(5)

df_Gu_last = df_edit2.drop(['value', 'variable', '아파트',
                            '세대수', '입주년월'], axis=1)  # 필요없어진 값 제거

df_Gu_last.insert(0, 'Gu', '관악구')  # 이름 확인 주의

df_Gu_last.to_excel('Gangnam_edit4/Gwanakgu_edit4.xlsx', sheet_name='edit4', index=False)  # 이름 확인 주의
