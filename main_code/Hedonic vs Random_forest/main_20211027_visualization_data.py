# import packages
import numpy as np
import pandas as pd
from urllib.request import urlopen
import json

import plotly.io as pio
import plotly.express as px #빠르게 사용
import plotly.graph_objects as go  #디테일하게 설정해야할때
import plotly.figure_factory as ff
from plotly.subplots import make_subplots # 여러 subplot을 그릴때
from plotly.validators.scatter.marker import SymbolValidator # 마커사용

########################################################################################################################
df_full = pd.read_pickle('data_process/apt_data/seoul_including_all_variables.pkl')
df_train = pd.read_pickle('data_process/apt_data/machine_learning/seoul_80_sample.pkl')
df_test = pd.read_pickle('data_process/apt_data/machine_learning/seoul_20_sample.pkl')

df_full = df_full.dropna()
df_train = df_train.dropna()
df_test = df_test.dropna()

# full
full_fig = px.scatter_mapbox(df_full, lat='lat', lon="long", color='gu', size="area",
                             color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10,
                             mapbox_style="carto-positron")
full_fig.show()

# train
train_fig = px.scatter_mapbox(df_train, lat='lat', lon="long", color='gu', size="area",
                              color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10,
                              mapbox_style="carto-positron")
train_fig.show()

# test
test_fig = px.scatter_mapbox(df_test, lat='lat', lon="long", color='gu', size="area",
                             color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10,
                             mapbox_style="carto-positron")
test_fig.show()
