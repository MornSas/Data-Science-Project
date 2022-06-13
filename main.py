import json
import sqlite3
import pandas_ta as ta
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry
import requests
import yfinance as yf
import networkx as nx
import streamlit as st
import altair as alt
from bs4 import BeautifulSoup
from keras.layers import LSTM
from keras.layers.core import Dense
from keras.models import Sequential
from shapely.geometry import shape, Polygon
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

url = 'http://dataservices.imf.org/REST/SDMX_JSON.svc/'
key = 'CompactData/FDI/'
period = 'A'
time = '?startPeriod=2010&endPeriod=2021'
base = f"{url}{key}{period}.{time}"
r = requests.get(base)
data = r.json()['CompactData']['DataSet']['Series']
obs = {}
obs2 = []
for i in data:
    if i['@INDICATOR'] != 'FD_FD_IX':
        pass
    else:
        obs2 = []
        obs[i['@REF_AREA']] = 0
        for j in i['Obs']:
            obs2 = [j['@TIME_PERIOD'], j['@OBS_VALUE']]
            if obs[i['@REF_AREA']] == 0:
                obs[i['@REF_AREA']] = obs2
            else:
                obs[i['@REF_AREA']].append(obs2[0])
                obs[i['@REF_AREA']].append(obs2[1])
df = pd.DataFrame.from_dict(obs, orient='index')
renamer = {}
for i in df.columns:
    if int(i)%2 == 0:
        renamer[i+1] = df[i][0]
        df = df.drop(i, axis=1)
df = df.rename(columns=renamer)
countries = {}
for i in df.index:
    try:
        countries[i] = pycountry.countries.get(alpha_2=i).name
    except AttributeError:
        pass
df = df.rename(index=countries)
df.reset_index(inplace=True)
df.rename(columns={'index': 'Country'}, inplace=True)

df_complaints = pd.read_csv('/Users/nikitakhomenko/Downloads/consumer_complaints.csv')
conn = sqlite3.connect('database.sqlite')
try:
    df_complaints.to_sql('complaints', conn)
except ValueError:
    pass
c = conn.cursor()
df_product = pd.read_sql('''
select product, count(distinct complaint_id) as complaints
from complaints
group by product
order by complaints desc
limit 5;
''', conn,)
df_company = pd.read_sql(
    """
select company, count(distinct complaint_id) as complaints
from complaints
group by company
order by complaints desc
limit 5;
""",
    conn,
)
df_states = pd.read_sql(
    """
select state, count(distinct complaint_id) as complaints
from complaints
group by state;
""",
    conn,
).dropna()
conn.close()
with open(''https://raw.githubusercontent.com/MornSas/DSProject/master/gz_2010_us_040_00_500k.json', encoding = 'utf-8') as f:
    a = json.load(f)
placedata = []
for i in range(len(a['features'])):
    if a['features'][i]['geometry']['type'] == 'MultiPolygon':
        poly = shape(a['features'][i]['geometry'])
    elif a['features'][i]['geometry']['type'] == 'Polygon':
        poly = Polygon(a['features'][i]['geometry']['coordinates'][0])
    placedata.append([a['features'][i]['properties']['NAME'], poly])
df_places = pd.DataFrame(placedata, columns=['state', 'poly'])
###FROM: https://gist.github.com/JeffPaine/3083347
states = {
    'AK': 'Alaska',
    'AL': 'Alabama',
    'AR': 'Arkansas',
    'AZ': 'Arizona',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DC': 'District of Columbia',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'IA': 'Iowa',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'MA': 'Massachusetts',
    'MD': 'Maryland',
    'ME': 'Maine',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MO': 'Missouri',
    'MS': 'Mississippi',
    'MT': 'Montana',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'NE': 'Nebraska',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NV': 'Nevada',
    'NY': 'New York',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VA': 'Virginia',
    'VT': 'Vermont',
    'WA': 'Washington',
    'WI': 'Wisconsin',
    'WV': 'West Virginia',
    'WY': 'Wyoming'
}
###END FROM
df_states['state'].replace(states, inplace=True)
df_sc = df_states.merge(df_places, how='outer', on='state').fillna(0)
df_sc = df_sc[df_sc.poly!=0]
gdf = gpd.GeoDataFrame(df_sc, geometry='poly')

headers = {'User-agent': 'Mozilla/5.0'}
r = requests.get('https://www.slickcharts.com/nasdaq100', headers=headers)
page = BeautifulSoup(r.text, 'html.parser')
data = []
for row in page({'tr': 'td'}):
    data.append(row.text.split('\n'))
df_nasdaq = pd.DataFrame(data, columns=data[0])
df_nasdaq.drop(columns=[''], inplace=True)
df_nasdaq.drop([0, 103, 104, 105, 106], inplace=True)
df_nasdaq.drop(axis=1, columns=['Chg', '% Chg'], inplace=True)
df_nasdaq.set_index('#', inplace=True)
data = yf.download(
    tickers = 'AAPL MSFT AMZN TSLA GOOG GOOGL META NVDA PEP AVGO',
    period = '1y',
    auto_adjust=True)
names = list(data['Close'])
stat = []
for name in names:
    n = []
    prices = list(data['Close'][name])
    for i in range(0, len(prices)-1):
        n.append((prices[i+1]-prices[i])/prices[i])
    nums = np.array(n)
    stat.append([name, np.std(nums), np.mean(nums)])
for comp in stat:
    comp.append((comp[2]-0.02/365)/comp[1])
df_stat = pd.DataFrame(stat, columns=['Company', 'Volatility StDev', 'ExpRet day', 'Sharpe Ratio'])
corr = []
for name in names:
    for name2 in names:
        if name != name2 and data['Close'][name].corr(data['Close'][name2])>0.5:
            corr.append([name, name2])
df_corr = pd.DataFrame(corr, columns=['from', 'to'])
data_ml = yf.download(
    tickers='PEP',
    start='2021-06-12',
    end='2022-06-13',
    auto_adjust=True)
data_ml['EMA'] = data_ml['Close'].ewm(span=14, adjust=False).mean()
data_ml.ta.rsi(close='Close', append=True)
data_ml=data_ml.dropna()
df_ml = data_ml[['Close', 'EMA', 'RSI_14']]
train_df, test_df = train_test_split(df_ml, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(train_df.drop(columns=['Close']), train_df['Close'])
predclose = model.predict(test_df.drop(columns=['Close']))
error = ((test_df['Close'] - predclose)**2).mean()
data_ml['Predicted Close'] = data_ml['Close']
data_ml.loc['2022-04-04':, 'Predicted Close'] = predclose
errors = 0
df_predict = data_ml['2022-04-04':]
for index, row in df_predict.iterrows():
    if (row['Close'] - row['Open'])*(row['Predicted Close'] - row['Open']) < 0:
        errors += 1
###FROM: https://github.com/sonaam1234/DeepLearningInFinance/blob/master/ReturnPrediction/ReturnPrediction_LSTM.py
traini_df = train_df.copy()
traini_df.reset_index(inplace=True)
traini_df.drop(columns=['Date'], inplace=True)
sc = MinMaxScaler(feature_range = (0, 1))
train_df_scaled = sc.fit_transform(traini_df)
lookback = 12
timeseries = np.asarray(train_df_scaled)
timeseries = np.atleast_2d(timeseries)
if timeseries.shape[0] == 1:
        timeseries = timeseries.T
X = np.atleast_3d(np.array([timeseries[start:start + lookback] for start in range(0, timeseries.shape[0] - lookback)]))
y = timeseries[lookback:]
###END FROM
model = Sequential()
model.add(LSTM(1, batch_input_shape=(1, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X,
          y,
          epochs=200,
          batch_size=1)
traintest_df = train_df.tail(12)
testtrain_df = pd.concat([traintest_df, test_df], axis=0)
tst_df = testtrain_df.copy()
tst_df.reset_index(inplace=True)
tst_df.drop(columns=['Date'], inplace=True)
test_df_scaled = sc.fit_transform(tst_df)
timeseries = np.asarray(test_df_scaled)
timeseries = np.atleast_2d(timeseries)
if timeseries.shape[0] == 1:
        timeseries = timeseries.T
X_test = np.atleast_3d(np.array([timeseries[start:start + lookback] for start in range(0, timeseries.shape[0] - lookback)]))
y_test = timeseries[lookback:]
predicted_stock_price = model.predict(X_test, batch_size=1)
for i in range(len(y_test)):
    y_test[i][0] = predicted_stock_price[i]
prediction = sc.inverse_transform(y_test)
pred = [prediction[i][0] for i in range(len(prediction))]
df_ml['Neuropredicted Close'] = df_ml['Close']
df_ml.loc['2022-04-04':, 'Neuropredicted Close'] = pred

with st.echo(code_location='below'):
    '''
    ## Немного финансов вам в ленту
    '''
    country = st.selectbox(
        'Country', df.sort_values('Country')['Country']
    )
    df_selection = df[lambda x: x['Country'] == country]
    df_selection


    '''
    ## Теперь посмотрим, на какие финансовые услуги жалуются люди в США
    '''

    '''
    ## Топ-5 услуг по количеству жалоб
    '''
    df_product
    x = list(df_product['product'])
    y = list(df_product['complaints'])
    fig = plt.figure()
    fig.set_figwidth(7.5)
    fig.set_figheight(7.5)
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Product name')
    ax1.set_xlabel('Number of complaints')
    ax1.set_title('smh')
    for i, v in enumerate(y):
        ax1.text(v, i, str(v), color='red', fontweight='bold')
    plt.barh(x, y, color='red')
    st.pyplot(fig)

    '''
    ## Топ-5 корпораций по количеству жалоб
    '''
    df_company

    x = list(df_company['company'])
    y = list(df_company['complaints'])
    fig = plt.figure()
    fig.set_figwidth(7.5)
    fig.set_figheight(7.5)
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Company name')
    ax1.set_xlabel('Number of complaints')
    ax1.set_title('smh')
    for i, v in enumerate(y):
        ax1.text(v, i, str(v), color='red', fontweight='bold')
    plt.barh(x, y, color='red')
    st.pyplot(fig)

    '''
    ## А вот как выглядит карта жалоб по штатам
    '''
    fig, compmap = plt.subplots()
    gdf.plot(column='complaints', ax=compmap, legend=True, figsize=(20, 10))
    compmap.set_title('Количество жалоб')
    plt.xlim([-170, -60])
    plt.ylim([10, 75])
    st.pyplot(fig)

    '''
    ## Теперь немного посмотрим на акции. Вот какие акции входят в индекс NASDAQ 100
    '''
    company = st.selectbox(
        'Company', df_nasdaq.sort_values('Weight')['Company']
    )
    df_selection1 = df_nasdaq[lambda x: x['Company'] == company]
    df_selection1

    '''
    ## А вот как выглядят веса 10 наибольших в индексе
    '''

    labels = list(df_nasdaq['Company'][:10])
    labels.append('Other')
    w = df_nasdaq['Weight']
    values = list(w[:10].astype(float))
    values.append(w[10:].astype(float).sum())
    fig1, ax1 = plt.subplots(figsize=(20, 10))
    ax1.pie(values, labels=labels, autopct='%1.2f%%')
    ax1.axis('equal')
    st.pyplot(fig1)

    '''
    Посмотрим поближе на эти 10 компаний
    '''
    company1 = st.selectbox(
        'Company', list(data['Close'])
    )
    fig1, stocks = plt.subplots(figsize=(20, 10))
    stocks = plt.plot(data['Close'][company1])
    st.pyplot(fig1)

    '''
    ## А вот несколько их параметров
    '''
    df_stat

    '''
    ## А вот компании, корреляция между ценами которых больше 0.5
    '''
    G = nx.DiGraph([(frm, to) for (frm, to) in df_corr.values])
    fig, ax = plt.subplots()
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    st.pyplot(fig)
    
    '''
    ## Посмотрим поближе на PepsiCo, как на компанию с наибольшим Sharpe Ratio
    '''
    '''
    ##Вот как выглядит предугадывание цен с помощью линейной регрессии
    '''
    lin = plt.figure()
    lin.set_figwidth(20)
    lin.set_figheight(10)
    plt.plot(data_ml['Predicted Close'], color='red')
    plt.plot(data_ml['Close'], color='green')
    st.pyplot(lin)
    '''
    ## А вот как оно будет выглядеть, если написать небольшую нейронку
    '''
    neu = plt.figure()
    neu.set_figwidth(20)
    neu.set_figheight(10)
    plt.plot(df_ml['Neuropredicted Close'], color='red')
    plt.plot(df_ml['Close'], color='green')
    st.pyplot(neu)
