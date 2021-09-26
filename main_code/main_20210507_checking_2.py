# 2021-05-07
# Chapter 2
# edit_1을 전처리 하여 edit_2를 형성
# edit_2를 수작업하여 새롭게 크롤링을 돌린 후 기존의 edit_1과 합치한 edit_3를 만드는 과정

# 1. number 변수에 값이 nan 일 경우 해당 행을 엑셀에서 제거하여 새로운 dataframe 으로 생성
# 2. type_capacity10에 값이 있을 경우 해당 행을 엑셀에서 제거하여 새로운 dataframe 에 추가
# 3. 전처리 대상들은 edit_2라는 이름으로 저장
# 4. 면적별 정보를 17개까지 확인할 수 있도록 코드 수정, 인식이 안되는 이름들을 하나하나 인식이 되는 이름으로 바꿔주기 (수작업)
# 5. edit_2상의 파일들을 다시 크롤링 후 기존의 edit_1 엑셀과 합쳐주기. 이때 인덱스는 edit_2가 기준이 되야함.
# 6. 최종 결과 값을 edit_3에 넣고 melt 돌리기.


# Import Packages
from selenium import webdriver
import time
import openpyxl
from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.common.keys import Keys
import numpy as np
from urllib.parse import urlparse  # 출처: https://datamasters.co.kr/67 [데이터마스터]
from datetime import datetime  # HKim: 코드 내에 타이머를 사용하면 여러모로 편리합니다.


########################################################################################################################
# 함수 선언
# HKim: 클래스 및 함수 선언은 패키지 임포트 바로 아래에 있어야 합니다.

# 함수 선언 1: 단지정보 추출
def get_apt_info():
    apt_name_selector = "#complexTitle"
    apt_name.append(chrome.find_element_by_css_selector(apt_name_selector).text)
    number_selector = "#detailContents1 > div.detail_box--complex > table > tbody > tr:nth-child(1) > td:nth-child(2)"
    number.append(chrome.find_element_by_css_selector(number_selector).text)
    floor_selector = "#detailContents1 > div.detail_box--complex > table > tbody > tr:nth-child(1) > td:nth-child(4)"
    floor.append(chrome.find_element_by_css_selector(floor_selector).text)
    confirm_date_selector = "#detailContents1 > div.detail_box--complex > table > tbody > tr:nth-child(2) > " \
                            "td:nth-child(2) "
    confirm_date.append(chrome.find_element_by_css_selector(confirm_date_selector).text)
    car_selector = "#detailContents1 > div.detail_box--complex > table > tbody > tr:nth-child(2) > td:nth-child(4)"
    car.append(chrome.find_element_by_css_selector(car_selector).text)
    FAR_selector = "#detailContents1 > div.detail_box--complex > table > tbody > tr:nth-child(3) > td:nth-child(2)"
    FAR.append(chrome.find_element_by_css_selector(FAR_selector).text)
    BC_selector = "#detailContents1 > div.detail_box--complex > table > tbody > tr:nth-child(3) > td:nth-child(4)"
    BC.append(chrome.find_element_by_css_selector(BC_selector).text)
    con_selector = "#detailContents1 > div.detail_box--complex > table > tbody > tr:nth-child(4) > td"
    con.append(chrome.find_element_by_css_selector(con_selector).text)
    heat_selector = "#detailContents1 > div.detail_box--complex > table > tbody > tr:nth-child(5) > td"
    heat.append(chrome.find_element_by_css_selector(heat_selector).text)


# 함수선언 2: url 정보 추출
# HKim: PEP8에서 함수 이름은 lowercase_underscore 를 사용합니다.
def get_url_info():
    current_url = chrome.current_url  # url 확인하기
    df_url = pd.DataFrame([urlparse(current_url)])

    path = df_url.loc[0]['path']
    code.append(path.split('/')[2])  # code 에 아파트 코드 담기

    query = df_url.loc[0]['query']
    check = query.split('=')[1]
    lat.append(check.split(',')[0])
    long.append(check.split(',')[1])  # lat, long에 각각 위도와 경도 담기


# 함수선언 3: nan 값 입력
def input_nan_if_null():
    apt_name.append(np.nan)
    number.append(np.nan)
    floor.append(np.nan)
    confirm_date.append(np.nan)
    car.append(np.nan)
    FAR.append(np.nan)
    BC.append(np.nan)
    con.append(np.nan)
    heat.append(np.nan)
    code.append(np.nan)
    lat.append(np.nan)
    long.append(np.nan)


# 함수선언 4: 면적 정보 추출
def input_value_in_vars(type_capacity, area, room, toilet, n_this_area, structure, i):
    try:
        if i == 0:
            type_capacity_selector = "#tab0 > span"
            type_capacity.append(chrome.find_element_by_css_selector(type_capacity_selector).text)

            area_selector = "#tabpanel > table > tbody > tr:nth-child(1) > td"
            area.append(chrome.find_element_by_css_selector(area_selector).text)

            rt_selector = "#tabpanel > table > tbody > tr:nth-child(2) > td"  # 방 개수와 화장실 개수
            rt = chrome.find_element_by_css_selector(rt_selector).text
            room.append(rt.split('/')[0])
            toilet.append(rt.split('/')[1])

            n_this_area_selector = "#tabpanel > table > tbody > tr:nth-child(3) > td"
            n_this_area.append(chrome.find_element_by_css_selector(n_this_area_selector).text)

            structure_selector = "#tabpanel > table > tbody > tr:nth-child(4) > td"
            structure.append(chrome.find_element_by_css_selector(structure_selector).text)

            time.sleep(2)

        elif i == 7:
            down_scroll = "#detailContents1 > div.detail_box--floor_plan > div.detail_sorting_tabs > div > " \
                          "div.btn_moretab_box > button "
            chrome.find_element_by_css_selector(down_scroll).click()
            time.sleep(1)

            n = str(i)
            num_capacity = "#tab" + n + "> span"
            chrome.find_element_by_css_selector(num_capacity).click()

            type_capacity.append(chrome.find_element_by_css_selector(num_capacity).text)

            area_selector = "#tabpanel > table > tbody > tr:nth-child(1) > td"
            area.append(chrome.find_element_by_css_selector(area_selector).text)

            rt_selector = "#tabpanel > table > tbody > tr:nth-child(2) > td"  # 방 개수와 화장실 개수
            rt = chrome.find_element_by_css_selector(rt_selector).text
            room.append(rt.split('/')[0])
            toilet.append(rt.split('/')[1])

            n_this_area_selector = "#tabpanel > table > tbody > tr:nth-child(3) > td"
            n_this_area.append(chrome.find_element_by_css_selector(n_this_area_selector).text)

            structure_selector = "#tabpanel > table > tbody > tr:nth-child(4) > td"
            structure.append(chrome.find_element_by_css_selector(structure_selector).text)

            time.sleep(2)


        else:
            n = str(i)
            num_capacity = "#tab" + n + "> span"
            chrome.find_element_by_css_selector(num_capacity).click()

            type_capacity.append(chrome.find_element_by_css_selector(num_capacity).text)

            area_selector = "#tabpanel > table > tbody > tr:nth-child(1) > td"
            area.append(chrome.find_element_by_css_selector(area_selector).text)

            rt_selector = "#tabpanel > table > tbody > tr:nth-child(2) > td"  # 방 개수와 화장실 개수
            rt = chrome.find_element_by_css_selector(rt_selector).text
            room.append(rt.split('/')[0])
            toilet.append(rt.split('/')[1])

            n_this_area_selector = "#tabpanel > table > tbody > tr:nth-child(3) > td"
            n_this_area.append(chrome.find_element_by_css_selector(n_this_area_selector).text)

            structure_selector = "#tabpanel > table > tbody > tr:nth-child(4) > td"
            structure.append(chrome.find_element_by_css_selector(structure_selector).text)

            time.sleep(2)

    except Exception as ex:
        type_capacity.append(np.nan)
        area.append(np.nan)
        room.append(np.nan)
        toilet.append(np.nan)
        n_this_area.append(np.nan)
        structure.append(np.nan)
        time.sleep(1)


# 함수선언 5: 면적정보 1~10 리스트에 append 하기
def get_capacity_info():
    input_value_in_vars(type_capacity1, area1, room1, toilet1, n_this_area1, structure1, i=0)
    input_value_in_vars(type_capacity2, area2, room2, toilet2, n_this_area2, structure2, i=1)
    input_value_in_vars(type_capacity3, area3, room3, toilet3, n_this_area3, structure3, i=2)
    input_value_in_vars(type_capacity4, area4, room4, toilet4, n_this_area4, structure4, i=3)
    input_value_in_vars(type_capacity5, area5, room5, toilet5, n_this_area5, structure5, i=4)
    input_value_in_vars(type_capacity6, area6, room6, toilet6, n_this_area6, structure6, i=5)
    input_value_in_vars(type_capacity7, area7, room7, toilet7, n_this_area7, structure7, i=6)
    input_value_in_vars(type_capacity8, area8, room8, toilet8, n_this_area8, structure8, i=7)
    input_value_in_vars(type_capacity9, area9, room9, toilet9, n_this_area9, structure9, i=8)
    input_value_in_vars(type_capacity10, area10, room10, toilet10, n_this_area10, structure10, i=9)
    input_value_in_vars(type_capacity11, area11, room11, toilet11, n_this_area11, structure11, i=10)
    input_value_in_vars(type_capacity12, area12, room12, toilet12, n_this_area12, structure12, i=11)
    input_value_in_vars(type_capacity13, area13, room13, toilet13, n_this_area13, structure13, i=12)
    input_value_in_vars(type_capacity14, area14, room14, toilet14, n_this_area14, structure14, i=13)
    input_value_in_vars(type_capacity15, area15, room15, toilet15, n_this_area15, structure15, i=14)
    input_value_in_vars(type_capacity16, area16, room16, toilet16, n_this_area16, structure16, i=15)
    input_value_in_vars(type_capacity17, area17, room17, toilet17, n_this_area17, structure17, i=16)


# 함수선언 6: 크롤링한 data DataFrame 에 append 하기

def append_to_df2():
    df_Gu2['Apt_name'] = apt_name
    df_Gu2['number'] = number
    df_Gu2['floor'] = floor
    df_Gu2['confirm_date'] = confirm_date
    df_Gu2['car'] = car
    df_Gu2['FAR'] = FAR
    df_Gu2['BC'] = BC
    df_Gu2['con'] = con
    df_Gu2['heat'] = heat
    df_Gu2['code'] = code
    df_Gu2['lat'] = lat
    df_Gu2['long'] = long

    df_Gu2['type_capacity1'] = type_capacity1
    df_Gu2['area1'] = area1
    df_Gu2['room1'] = room1
    df_Gu2['toilet1'] = toilet1
    df_Gu2['structure1'] = structure1
    df_Gu2['n_this_area1'] = n_this_area1

    df_Gu2['type_capacity2'] = type_capacity2
    df_Gu2['area2'] = area2
    df_Gu2['room2'] = room2
    df_Gu2['toilet2'] = toilet2
    df_Gu2['structure2'] = structure2
    df_Gu2['n_this_area2'] = n_this_area2

    df_Gu2['type_capacity3'] = type_capacity3
    df_Gu2['area3'] = area3
    df_Gu2['room3'] = room3
    df_Gu2['toilet3'] = toilet3
    df_Gu2['structure3'] = structure3
    df_Gu2['n_this_area3'] = n_this_area3

    df_Gu2['type_capacity4'] = type_capacity4
    df_Gu2['area4'] = area4
    df_Gu2['room4'] = room4
    df_Gu2['toilet4'] = toilet4
    df_Gu2['structure4'] = structure4
    df_Gu2['n_this_area4'] = n_this_area4

    df_Gu2['type_capacity5'] = type_capacity5
    df_Gu2['area5'] = area5
    df_Gu2['room5'] = room5
    df_Gu2['toilet5'] = toilet5
    df_Gu2['structure5'] = structure5
    df_Gu2['n_this_area5'] = n_this_area5

    df_Gu2['type_capacity6'] = type_capacity6
    df_Gu2['area6'] = area6
    df_Gu2['room6'] = room6
    df_Gu2['toilet6'] = toilet6
    df_Gu2['structure6'] = structure6
    df_Gu2['n_this_area6'] = n_this_area6

    df_Gu2['type_capacity7'] = type_capacity7
    df_Gu2['area7'] = area7
    df_Gu2['room7'] = room7
    df_Gu2['toilet7'] = toilet7
    df_Gu2['structure7'] = structure7
    df_Gu2['n_this_area7'] = n_this_area7

    df_Gu2['type_capacity8'] = type_capacity8
    df_Gu2['area8'] = area8
    df_Gu2['room8'] = room8
    df_Gu2['toilet8'] = toilet8
    df_Gu2['structure8'] = structure8
    df_Gu2['n_this_area8'] = n_this_area8

    df_Gu2['type_capacity9'] = type_capacity9
    df_Gu2['area9'] = area9
    df_Gu2['room9'] = room9
    df_Gu2['toilet9'] = toilet9
    df_Gu2['structure9'] = structure9
    df_Gu2['n_this_area9'] = n_this_area9

    df_Gu2['type_capacity10'] = type_capacity10
    df_Gu2['area10'] = area10
    df_Gu2['room10'] = room10
    df_Gu2['toilet10'] = toilet10
    df_Gu2['structure10'] = structure10
    df_Gu2['n_this_area10'] = n_this_area10

    df_Gu2['type_capacity11'] = type_capacity11
    df_Gu2['area11'] = area11
    df_Gu2['room11'] = room11
    df_Gu2['toilet11'] = toilet11
    df_Gu2['structure11'] = structure11
    df_Gu2['n_this_area11'] = n_this_area11

    df_Gu2['type_capacity12'] = type_capacity12
    df_Gu2['area12'] = area12
    df_Gu2['room12'] = room12
    df_Gu2['toilet12'] = toilet12
    df_Gu2['structure12'] = structure12
    df_Gu2['n_this_area12'] = n_this_area12

    df_Gu2['type_capacity13'] = type_capacity13
    df_Gu2['area13'] = area13
    df_Gu2['room13'] = room13
    df_Gu2['toilet13'] = toilet13
    df_Gu2['structure13'] = structure13
    df_Gu2['n_this_area13'] = n_this_area13

    df_Gu2['type_capacity14'] = type_capacity14
    df_Gu2['area14'] = area14
    df_Gu2['room14'] = room14
    df_Gu2['toilet14'] = toilet14
    df_Gu2['structure14'] = structure14
    df_Gu2['n_this_area14'] = n_this_area14

    df_Gu2['type_capacity15'] = type_capacity15
    df_Gu2['area15'] = area15
    df_Gu2['room15'] = room15
    df_Gu2['toilet15'] = toilet15
    df_Gu2['structure15'] = structure15
    df_Gu2['n_this_area15'] = n_this_area15

    df_Gu2['type_capacity16'] = type_capacity16
    df_Gu2['area16'] = area16
    df_Gu2['room16'] = room16
    df_Gu2['toilet16'] = toilet16
    df_Gu2['structure16'] = structure16
    df_Gu2['n_this_area16'] = n_this_area16

    df_Gu2['type_capacity17'] = type_capacity17
    df_Gu2['area17'] = area17
    df_Gu2['room17'] = room17
    df_Gu2['toilet17'] = toilet17
    df_Gu2['structure17'] = structure17
    df_Gu2['n_this_area17'] = n_this_area17
########################################################################################################################
# 엑셀값 출력

df_Gu = pd.read_excel('Gangbuk_edit1/Eunpyeonggu_edit1.xlsx', sheet_name='edit1', header=0, skipfooter=0)
########################################################################################################################
# 전처리 대상 값들을 새로운 데이터프레임으로 지정하기

# 전처리 1단계, 데이터 프레임에서 number 컬럼에 값이 nan 인 경우 해당 row 를 새로운 dataframe 에 저장
df_Gu2 = df_Gu[df_Gu['number'].isnull()]

# 전처리 2단계, 데이터 프레임에서 type_capacity10 열이 nan 값이 아닌 경우를 추출
df = df_Gu.loc[df_Gu['type_capacity10'].str.contains('㎡', na=False)]

# 다시 크롤링을 돌려야하는 파일을 df_Gu2 으로 저장
df_Gu2 = pd.concat([df_Gu2, df])

# 전처리 대상 엑셀파일을 edit2로 저장, 개별 수정
df_Gu2.to_excel('Gangbuk_edit2/Eunpyeonggu_edit2.xlsx', sheet_name='edit2', index=True)
########################################################################################################################
# 전처리 대상이 아닌 값들을 따로 보관해두기

# 기존 데이터 프레임에서 nan 값이 있는 행 제거 ('전처리 1단계' 에서 걸러진 데이터들 보전)
df_Gu = df_Gu.dropna(subset=['number'])

# 기존 데이터 프레임에서 type capacity10 가 nan 값인 행 추출 ('전처리 2단계' 에서 걸러진 데이터들 보전)
df_Gu = df_Gu[df_Gu['type_capacity10'].isnull()]

# 이제 수정된 df_Gu2에 대해서 크롤링을 다시 한 뒤, 해당 df_Gu 값을 합쳐줘야 함
########################################################################################################################
'''
개별 수정 단계 : 네이버 부동산에서 input value 로 인식하지 못하는 Apt 이름과 
면적별 유형이 10개가 넘어가는 아파트들에 대한 정리 
'''

# 수정된 엑셀파일 다시 불러오고, sorting 다시 하기
df_Gu2 = pd.read_excel('Gangbuk_edit2/Eunpyeonggu_edit2.xlsx', sheet_name='edit2', header=0, skipfooter=0,
                       usecols='B:E')

df_name = df_Gu2[['읍면동', '아파트']]  # 여러 열을 추출하고 싶을때는 [[ 두개를 사용 ]]

df_name = df_name.astype('str')
se_name = df_name['읍면동'] + " " + df_name['아파트']
########################################################################################################################
# 크롤링 정보를 담을 리스트 선언

# 1. 단지 정보 리스트
apt_name = []  # 아파트 이름; input 과 output 이 제대로 일치하는지 확인하기 위함
number = []  # 세대수
floor = []  # 저/최고층
confirm_date = []  # 사용승인일
car = []  # 주차대수
FAR = []  # 용적률 (Floor Area Ratio)
BC = []  # 건폐율 (Building Coverage)
con = []  # 건설사
heat = []  # 난방 / 난방방식
lat = []  # 위도
long = []  # 경도
code = []  # 아파트 코드

# 2. 면적별 정보 리스트
# 1)
type_capacity1 = []
area1 = []  # 면적 : 공급/전용(전용률)
room1 = []  # 방 갯수
toilet1 = []  # 화장실 개수
structure1 = []  # 현관구조
n_this_area1 = []  # 해당면적 세대수
# 2)
type_capacity2 = []
area2 = []
room2 = []
toilet2 = []
structure2 = []
n_this_area2 = []
# 3)
type_capacity3 = []
area3 = []
room3 = []
toilet3 = []
structure3 = []
n_this_area3 = []
# 4)
type_capacity4 = []
area4 = []
room4 = []
toilet4 = []
structure4 = []
n_this_area4 = []
# 5)
type_capacity5 = []
area5 = []
room5 = []
toilet5 = []
structure5 = []
n_this_area5 = []
# 6)
type_capacity6 = []
area6 = []
room6 = []
toilet6 = []
structure6 = []
n_this_area6 = []
# 7)
type_capacity7 = []
area7 = []
room7 = []
toilet7 = []
structure7 = []
n_this_area7 = []
# 8)
type_capacity8 = []
area8 = []
room8 = []
toilet8 = []
structure8 = []
n_this_area8 = []
# 9)
type_capacity9 = []
area9 = []
room9 = []
toilet9 = []
structure9 = []
n_this_area9 = []
# 10)
type_capacity10 = []
area10 = []
room10 = []
toilet10 = []
structure10 = []
n_this_area10 = []
# 11)
type_capacity11 = []
area11 = []
room11 = []
toilet11 = []
structure11 = []
n_this_area11 = []
# 12)
type_capacity12 = []
area12 = []
room12 = []
toilet12 = []
structure12 = []
n_this_area12 = []
# 13)
type_capacity13 = []
area13 = []
room13 = []
toilet13 = []
structure13 = []
n_this_area13 = []
# 14)
type_capacity14 = []
area14 = []
room14 = []
toilet14 = []
structure14 = []
n_this_area14 = []
# 15)
type_capacity15 = []
area15 = []
room15 = []
toilet15 = []
structure15 = []
n_this_area15 = []
# 16)
type_capacity16 = []
area16 = []
room16 = []
toilet16 = []
structure16 = []
n_this_area16 = []
# 17)
type_capacity17 = []
area17 = []
room17 = []
toilet17 = []
structure17 = []
n_this_area17 = []
########################################################################################################################
# 스크래핑 시작

# StopWatch: 코드 시작
time_start = datetime.now()
print("Procedure started at: " + str(time_start))

apt_len = len(se_name)  # 단지명 리스트의 길이.
chrome = webdriver.Chrome('chromedriver.exe')

for i in range(apt_len):
    apt = se_name[i]

    try:
        if i == 0:
            chrome.get('https://land.naver.com/')  # 네이버 부동산 실행
            time.sleep(1)
            # apt = df_name[0]
            # Copy selector 을 해서 원하는 '검색창' 의 정보를 불러온다.
            # queryInputHeader = 해당 검색창의 selector
            input_engine = chrome.find_element_by_css_selector('#queryInputHeader')
            input_engine.clear()
            input_engine.send_keys(apt)  # enter 키를 누르기 전 상태
            input_engine.send_keys(Keys.ENTER)  # 특정키를 입력하고 싶은 경우
            link = chrome.find_element_by_css_selector(
                '#summaryInfo > div.complex_summary_info > div.complex_detail_link > button:nth-child(1)')
            link.click()  # '단지정보' 클릭

            time.sleep(1)

            get_apt_info()  # 단지 정보 가져오기
            get_url_info()  # url 에서 위도와 경도 가져오기
            get_capacity_info()  # 면적별 정보 가져오기

            chrome.find_element_by_css_selector('#search_input').clear  # 검색창 초기화

            time.sleep(1)

        else:
            search = chrome.find_element_by_css_selector('#search_input')
            search.clear()
            search.send_keys(apt)  # enter 키를 누르기 전 상태
            # input.submit()
            search.send_keys(Keys.ENTER)  # 특정키를 입력하고 싶은 경우
            time.sleep(4)
            link = chrome.find_element_by_css_selector(
                '#summaryInfo > div.complex_summary_info > div.complex_detail_link > button:nth-child(1)')
            link.click()
            time.sleep(4)

            get_apt_info()  # 단지 정보 가져오기
            get_url_info()  # url 에서 위도와 경도 가져오기
            get_capacity_info()  # 면적별 정보 가져오기

            chrome.find_element_by_css_selector('#search_input').clear  # 검색창 초기화


    except:  # 검색 시 여러개의 창이 뜰 때 기능
        try:
            search.clear()
            choice = chrome.find_element_by_css_selector(
                '#ct > div.map_wrap > div.search_panel > div.list_contents > div > div > div:nth-child(2) > div > a')
            choice.click()
            # summaryInfo > div.complex_summary_info > div.complex_detail_link > button:nth-child(1)
            link = chrome.find_element_by_css_selector(
                '#summaryInfo > div.complex_summary_info > div.complex_detail_link > button:nth-child(1)')
            link.click()

            time.sleep(3)

            get_apt_info()  # 단지 정보 가져오기
            get_url_info()  # url 에서 위도와 경도 가져오기
            get_capacity_info()  # 면적별 정보 가져오기


        except Exception as ex:
            research = chrome.find_element_by_css_selector('#search_input')
            research.clear()

            input_nan_if_null()  # 원하는 정보를 얻지 못했으므로 전체 nan 값 출력
            get_capacity_info()  # 입력값에 해당하는 정보가 없는 상황에서 자동으로 nan 값 출력

            chrome.back()

            try:
                time.sleep(3)
                input_engine = chrome.find_element_by_css_selector('#queryInputHeader')
                input_engine.send_keys(Keys.ENTER)
            except:
                search = chrome.find_element_by_css_selector('#search_input')
                search.send_keys(Keys.ENTER)

# StopWatch: 코드 종료
time_end = datetime.now()
print("Procedure finished at: " + str(time_end))
print("Elapsed (in this Procedure): " + str(time_end - time_start))

# 엑셀에 append 시키기
append_to_df2()

# 스크래핑 종료
########################################################################################################################

# 각 항목의 length 확인하기
print('apt_name: ', len(apt_name))
print('number: ', len(number))
print('floor: ', len(floor))
print('confirm_date: ', len(confirm_date))
print('car: ', len(car))
print('FAR: ', len(FAR))
print('BC: ', len(BC))
print('con: ', len(con))
print('heat: ', len(heat))
print('lat: ', len(lat))
print('long: ', len(long))
print('code: ', len(code))

#############################################################
# apt_name 의 length 가 맞지 않아 데이터프레임이 합쳐지지 않을 경우.
# apt_name 일치 안할 경우, 해당 항목을 drop 시켜야 한다.
# 확인하기 위한 코드

# df_apt_name = pd.DataFrame(apt_name)
# df_check = pd.concat([df_Gu, df_apt_name], axis=1)
# df_check
# print(apt_name[?])
# apt_name.pop(?)
#############################################################

# 전처리 후 새롭게 크롤링을 돌린 값과 기존의 값을 하나의 dataframe 으로 합쳐주기
# df_Gu : 전처리 후 보전된 데이터
# df_Gu2 : 전처리 과정에서 새롭게 정제된 데이터
df_Gu3 = pd.concat([df_Gu2, df_Gu])

# 결과값 엑셀로 내보내기
df_Gu3.to_excel('Gangbuk_edit3/Eunpyeonggu_edit3.xlsx', sheet_name='edit3', index=False)
