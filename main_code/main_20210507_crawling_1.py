# 2021-05-07
# Chapter 1
# input data 를 활용하여 네이버 부동산 크롤링 하기
# 2021-04-05
# selenium 을 이용한 크롤링 일반화 작업
# 면적별 정보를 불러올 때 6개가 넘어가는 경우 화살표 버튼을 활성화하는 기능
# 면적별 정보가 10개가 넘어갈 때(10개를 다 채운 아파트의 경우 엑셀에서 직접 전처리 과정을 실행할 것)
# 입력값과 출력값이 상응하는지 확인하기 위해 아파트 명을 dataframe 에 추가
# 아파트 명의 경우 엑셀로 변환시 (= dataframe 에서 최종 위치가) 기존 엑셀 데이터의 아파트 뒤에 오도록 위치 (입력값과 출력값을 확인하기 위함)
# 면적별 정보를 append 할 때 type_capacity 이라는 이름을 붙여주어 각 정보가 어떤 면적에 대한 정보인지 가늠할 수 있도록 조치
# 엑셀 시트를 하나의 dataframe 으로 합쳐 최종적으로 구별 아파트 정보를 출력 및 저장

# HKim: PEP8 스타일 가이드를 가급적 준수하세요.
# 1. 패키지를 불러온 이후에는 두 줄 띄웁니다.
# 2. 클래스 및 함수 정의는 패키지 불러온 아래에 붙습니다. 이후 두 줄 띄웁니다.
# 3. 코멘트를 쓸 때에는 샵(#) 붙이고 한칸 띄웁니다.
# 4. 라인 안에서 코멘트를 쓸 때에는 코드 맨 뒤에 두칸 띄우고 # 를 시작합니다.


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


# 함수선언 6: 크롤링한 data DataFrame 에 append 하기

def append_to_df():
    df_Gu['Apt_name'] = apt_name
    df_Gu['number'] = number
    df_Gu['floor'] = floor
    df_Gu['confirm_date'] = confirm_date
    df_Gu['car'] = car
    df_Gu['FAR'] = FAR
    df_Gu['BC'] = BC
    df_Gu['con'] = con
    df_Gu['heat'] = heat
    df_Gu['code'] = code
    df_Gu['lat'] = lat
    df_Gu['long'] = long

    df_Gu['type_capacity1'] = type_capacity1
    df_Gu['area1'] = area1
    df_Gu['room1'] = room1
    df_Gu['toilet1'] = toilet1
    df_Gu['structure1'] = structure1
    df_Gu['n_this_area1'] = n_this_area1

    df_Gu['type_capacity2'] = type_capacity2
    df_Gu['area2'] = area2
    df_Gu['room2'] = room2
    df_Gu['toilet2'] = toilet2
    df_Gu['structure2'] = structure2
    df_Gu['n_this_area2'] = n_this_area2

    df_Gu['type_capacity3'] = type_capacity3
    df_Gu['area3'] = area3
    df_Gu['room3'] = room3
    df_Gu['toilet3'] = toilet3
    df_Gu['structure3'] = structure3
    df_Gu['n_this_area3'] = n_this_area3

    df_Gu['type_capacity4'] = type_capacity4
    df_Gu['area4'] = area4
    df_Gu['room4'] = room4
    df_Gu['toilet4'] = toilet4
    df_Gu['structure4'] = structure4
    df_Gu['n_this_area4'] = n_this_area4

    df_Gu['type_capacity5'] = type_capacity5
    df_Gu['area5'] = area5
    df_Gu['room5'] = room5
    df_Gu['toilet5'] = toilet5
    df_Gu['structure5'] = structure5
    df_Gu['n_this_area5'] = n_this_area5

    df_Gu['type_capacity6'] = type_capacity6
    df_Gu['area6'] = area6
    df_Gu['room6'] = room6
    df_Gu['toilet6'] = toilet6
    df_Gu['structure6'] = structure6
    df_Gu['n_this_area6'] = n_this_area6

    df_Gu['type_capacity7'] = type_capacity7
    df_Gu['area7'] = area7
    df_Gu['room7'] = room7
    df_Gu['toilet7'] = toilet7
    df_Gu['structure7'] = structure7
    df_Gu['n_this_area7'] = n_this_area7

    df_Gu['type_capacity8'] = type_capacity8
    df_Gu['area8'] = area8
    df_Gu['room8'] = room8
    df_Gu['toilet8'] = toilet8
    df_Gu['structure8'] = structure8
    df_Gu['n_this_area8'] = n_this_area8

    df_Gu['type_capacity9'] = type_capacity9
    df_Gu['area9'] = area9
    df_Gu['room9'] = room9
    df_Gu['toilet9'] = toilet9
    df_Gu['structure9'] = structure9
    df_Gu['n_this_area9'] = n_this_area9

    df_Gu['type_capacity10'] = type_capacity10
    df_Gu['area10'] = area10
    df_Gu['room10'] = room10
    df_Gu['toilet10'] = toilet10
    df_Gu['structure10'] = structure10
    df_Gu['n_this_area10'] = n_this_area10


########################################################################################################################
# 엑셀값 출력

df_Gu = pd.read_excel('Gangbuk/Eunpyeonggu.xlsx', sheet_name=None, header=0, skipfooter=0,
                      usecols='C:D, G:H')

# 출력한 엑셀값 하나로 합치기
df_Gu = pd.concat(df_Gu, ignore_index='Ture')

df_Gu = df_Gu.drop_duplicates(['아파트'], keep='first')
df_Gu = df_Gu.sort_values(by=['아파트'])
df_Gu = df_Gu.reset_index(drop='Ture')

df_name = df_Gu[['읍면동', '아파트']]  # 여러 열을 추출하고 싶을때는 [[ 두개를 사용 ]]

df_name = df_name.astype('str')
se_name = df_name['읍면동'] + " " + df_name['아파트']

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
append_to_df()


# 스크래핑 종료
########################################################################################################################

# 각 항목의 length 확인하기
def check_length():
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
    print('type_capacity1: ', len(type_capacity1))
    print('area1: ', len(area1))
    print('room1: ', len(room1))
    print('toilet1: ', len(toilet1))
    print('structure1: ', len(structure1))
    print('n_this_area1: ', len(n_this_area1))
    print('type_capacity2: ', len(type_capacity2))
    print('area2: ', len(area2))
    print('room2: ', len(room2))
    print('toilet2: ', len(toilet2))
    print('structure2: ', len(structure2))
    print('n_this_area2: ', len(n_this_area2))
    print('type_capacity3: ', len(type_capacity3))
    print('area3: ', len(area3))
    print('room3: ', len(room3))
    print('toilet3: ', len(toilet3))
    print('structure3: ', len(structure3))
    print('n_this_area3: ', len(n_this_area3))
    print('type_capacity4: ', len(type_capacity4))
    print('area4: ', len(area4))
    print('room4: ', len(room4))
    print('toilet4: ', len(toilet4))
    print('structure4: ', len(structure4))
    print('n_this_area4: ', len(n_this_area4))
    print('type_capacity5: ', len(type_capacity5))
    print('area5: ', len(area5))
    print('room5: ', len(room5))
    print('toilet5: ', len(toilet5))
    print('structure5: ', len(structure5))
    print('n_this_area5: ', len(n_this_area5))
    print('type_capacity6: ', len(type_capacity6))
    print('area6: ', len(area6))
    print('room6: ', len(room6))
    print('toilet6: ', len(toilet6))
    print('structure6: ', len(structure6))
    print('n_this_area6: ', len(n_this_area6))
    print('type_capacity7: ', len(type_capacity7))
    print('area7: ', len(area7))
    print('room7: ', len(room7))
    print('toilet7: ', len(toilet7))
    print('structure7: ', len(structure7))
    print('n_this_area7: ', len(n_this_area7))
    print('type_capacity8: ', len(type_capacity8))
    print('area8: ', len(area8))
    print('room8: ', len(room8))
    print('toilet8: ', len(toilet8))
    print('structure8: ', len(structure8))
    print('n_this_area8: ', len(n_this_area8))
    print('type_capacity9: ', len(type_capacity9))
    print('area9: ', len(area9))
    print('room9: ', len(room9))
    print('toilet9: ', len(toilet9))
    print('structure9: ', len(structure9))
    print('n_this_area9: ', len(n_this_area9))
    print('type_capacity10: ', len(type_capacity10))
    print('area10: ', len(area10))
    print('room10: ', len(room10))
    print('toilet10: ', len(toilet10))
    print('structure10: ', len(structure10))
    print('n_this_area10: ', len(n_this_area10))


#######################################################################################################################
# apt_name 의 length 가 맞지 않아 데이터프레임이 합쳐지지 않을 경우.
# apt_name 일치 안할 경우, 해당 항목을 drop 시켜야 한다.
# 확인하기 위한 코드

# df_apt_name = pd.DataFrame(apt_name)
# df_check = pd.concat([df_Gu, df_apt_name], axis=1)
# df_check
# print(apt_name[?])
# apt_name.pop(?)
#######################################################################################################################

# 결과값 엑셀로 내보내기
df_Gu.to_excel('Gangbuk_edit1/Eunpyeonggu_edit1.xlsx', sheet_name='edit1', index=False)
