import yfinance as yf
import pandas as pd
import numpy as np
import ta
from scipy.interpolate import CubicSpline
from fredapi import Fred
import pymongo
from pymongo import MongoClient
from datetime import datetime
from datetime import date

# 🔹 MongoDB 연결
# MONGO_URI = "mongodb+srv://kanzki001:ahdrhelql@cluster0.b3gwf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGO_URI = "mongodb+srv://kanzki247:ahdrhelql@cluster0.b3gwf.mongodb.net/"
client = pymongo.MongoClient(MONGO_URI)
db = client["stock_data"]  # 데이터베이스 이름
collection_testD = db["testD"]
collection_testQ = db["testQ"]

# MongoDB 데이터 가져오기
def fetch_data_from_mongo(collection_name):
    client = pymongo.MongoClient(MONGO_URI)
    db = client["stock_data"]  # 데이터베이스 이름
    collection = db[collection_name]  # 컬렉션 이름

    # 데이터 불러오기 (_id 제외)
    data = list(collection.find({}, {"_id": 0}))

    return data


# 데이터를 변환하여 DataFrame으로 만드는 함수
def transform_to_dataframe(data):
    # 데이터를 데이터프레임으로 변환
    df = pd.DataFrame(data)

    # 'date'를 인덱스로 설정
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])  # datetime 형식으로 변환
        # df.set_index('date', inplace=True)  # 'date' 열을 인덱스로 설정

    # 'data' 필드를 열로 확장
    if 'data' in df.columns:
        # 'data' 필드 내부 JSON 확장
        expanded_data = pd.json_normalize(df['data'])
        # 원본 데이터프레임에서 'data'를 제거 후 확장 필드 병합
        df = pd.concat([df.drop(columns=['data']), expanded_data], axis=1)
        df.set_index('date', inplace=True)
    return df


# 실행
#if __name__ == "__main__":
# MongoDB 데이터 가져오기
mongo_data1 = fetch_data_from_mongo("testQ")
mongo_data2 = fetch_data_from_mongo("testD")
# 데이터프레임으로 변환
df_Q = transform_to_dataframe(mongo_data1)
df_D = transform_to_dataframe(mongo_data2)


# 결과 확인
df_Q.info()
df_D.info()

df_Q1 = df_Q.set_index('date').copy()
df_D1 = df_D.set_index('date').copy()

df_Q1 = df_Q1.drop(columns=['KOR_CPI', 'US_CPI'])

# 일별 데이터 null 값 처리
from prophet import Prophet
import pandas as pd

def fill_missing_with_prophet(df, column_name):
    """
    Prophet 모델을 사용하여 시계열 데이터프레임의 특정 컬럼의 결측치를 채웁니다.
    (인덱스가 이미 DatetimeIndex인 경우)

    Args:
        df (pd.DataFrame): 시계열 데이터프레임. 인덱스는 DatetimeIndex여야 합니다.
        column_name (str): 결측치를 채울 컬럼 이름.

    Returns:
        pd.DataFrame: 결측치가 Prophet 모델로 예측된 값으로 채워진 데이터프레임.
    """
    # Prophet 모델에 맞는 형식으로 데이터프레임 준비
    prophet_df = df[[column_name]].reset_index().rename(columns={'date': 'ds', column_name: 'y'})

    # 결측치가 아닌 데이터만 사용해 모델 학습
    prophet_df = prophet_df[prophet_df['y'].notnull()]

    # 데이터가 충분한지 확인
    if prophet_df.shape[0] < 2:
        print(f"'{column_name}' 컬럼에 유효한 데이터가 부족하여 결측치를 채울 수 없습니다.")
        return df.copy()

    # Prophet 모델 초기화 및 학습
    model = Prophet(
        yearly_seasonality=True,  # 연간 주기성 활성화
        weekly_seasonality=True,  # 주간 주기성 활성화 (일별 데이터에 적합)
        daily_seasonality=True,   # 일일 주기성 활성화
        growth='linear'          # 선형 성장 모델
    )
    model.fit(prophet_df)

    # 결측치가 있는 날짜만 추출하여 Prophet 모델에 예측 요청
    missing_dates = df[df[column_name].isnull()].index.to_frame(index=False, name='ds')

    if not missing_dates.empty:
        future = model.predict(missing_dates)
        predicted_values = future[['ds', 'yhat']].set_index('ds')

        # 원래 데이터프레임에 예측값 반영
        df_filled = df.copy()
        df_filled.loc[df[column_name].isnull(), column_name] = predicted_values['yhat']
        return df_filled
    else:
        print(f"'{column_name}' 컬럼에 결측치가 없습니다.")
        return df.copy()

def fill_all_columns(df):
    df_filled = df.copy()
    for column in df.columns:
        print(f"Processing column: {column}")
        df_filled = fill_missing_with_prophet(df_filled, column)
    return df_filled

# 실행
#if __name__ == '__main__':
# daily_df의 결측치를 채움
df_D_data = fill_all_columns(df_D1)
df_D_data.info()  # 결과 확인


# 일별 데이터 -> df_D_data
# 일별 데이터 기술적 지표 추가
def add_ta_features(df, base='USDKRW=X_Close'):
        # 기본 지표
        #5일 단기 ema
        df[f'{base}_MA5'] = df[f'{base}'].ewm(span=5, adjust=False).mean()
        # 20,60 중기 sma
        df[f'{base}_MA20'] = ta.trend.SMAIndicator(df[base], 20).sma_indicator()
        df[f'{base}_MA60'] = ta.trend.SMAIndicator(df[base], 60).sma_indicator()
        # 120일 장기 sma
        df[f'{base}_MA120'] = ta.trend.SMAIndicator(df[base], 120).sma_indicator()


        # RSI, MACD
        df[f'{base}_RSI'] = ta.momentum.RSIIndicator(df[base], 14).rsi()
        macd = ta.trend.MACD(df[base])
        df[f'{base}_MACD'] = macd.macd_diff()
        # RSI(Relative Strength Index): 상대강도지수
        # → 주식의 과매수(overbought) 또는 과매도(oversold) 상태를 나타내는 지표입니다.
        # 일반적으로 70 이상이면 과매수, 30 이하면 과매도로 판단합니다.
        # MACD(Moving Average Convergence Divergence): 이동평균 수렴·확산 지표
        # → 두 이동평균선 간의 관계를 통해 추세의 방향과 전환 시점을 파악하는 지표입니다.

        # 볼린저 밴드
        bb = ta.volatility.BollingerBands(df[base], 20)
        df[f'{base}_BB_H'] = bb.bollinger_hband()
        df[f'{base}_BB_L'] = bb.bollinger_lband()
        # Bollinger Bands: 볼린저 밴드 → 주가의 표준편차를 이용해 주가의 변동성과
        # 과매수·과매도 구간을 시각화한 지표입니다. 중심선은 주로 20일 이동평균입니다.

        return df

# 기술 지표 생성
D_df = add_ta_features(df_D_data, base='USDKRW=X_Close')    

# 기술 지표 null 값 처리
def fill_selected_columns(df):
    df_filled = df.copy()
    columns_to_fill = ['USDKRW=X_Close_BB_H', 'USDKRW=X_Close_BB_L', 'USDKRW=X_Close_MACD', 'USDKRW=X_Close_RSI']

    for column in columns_to_fill:
        print(f"Processing column: {column}")
        df_filled = fill_missing_with_prophet(df_filled, column)

    return df_filled

# 실행
#if __name__ == '__main__':
# indicator_df의 결측치를 채움
D_df = fill_selected_columns(D_df)
print("결측치 채우기 완료!")
D_df.info()  # 결과 확인

# 분기 데이터 null

def fill_missing_with_prophet(df, column_name):
    """
    Prophet 모델을 사용하여 분기별 데이터프레임의 특정 컬럼의 결측치를 채웁니다.

    Args:
        df (pd.DataFrame): 시계열 데이터프레임. 인덱스는 DatetimeIndex여야 합니다.
        column_name (str): 결측치를 채울 컬럼 이름.

    Returns:
        pd.DataFrame: 결측치가 채워진 데이터프레임.
    """
    prophet_df = df[[column_name]].reset_index().rename(columns={'date': 'ds', column_name: 'y'})
    prophet_df = prophet_df[prophet_df['y'].notnull()]

    if prophet_df.shape[0] < 2:
        print(f"'{column_name}' 컬럼에 유효한 데이터가 부족하여 결측치를 채울 수 없습니다.")
        return df.copy()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        growth='linear'
    )
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)

    model.fit(prophet_df)

    missing_dates = df[df[column_name].isnull()].index.to_frame(index=False, name='ds')

    if not missing_dates.empty:
        future = model.predict(missing_dates)
        predicted_values = future[['ds', 'yhat']].set_index('ds')
        df_filled = df.copy()
        df_filled.loc[df[column_name].isnull(), column_name] = predicted_values['yhat']
        return df_filled
    else:
        print(f"'{column_name}' 컬럼에 결측치가 없습니다.")
        return df.copy()

def fill_all_columns(df):
    """
    데이터프레임의 모든 컬럼의 결측치를 Prophet으로 채웁니다.

    Args:
        df (pd.DataFrame): 시계열 데이터프레임.

    Returns:
        pd.DataFrame: 결측치가 채워진 데이터프레임.
    """
    df_filled = df.copy()
    for column in df.columns:
        print(f"Processing column: {column}")
        df_filled = fill_missing_with_prophet(df_filled, column)
    return df_filled

# 실행
Q_df = fill_all_columns(df_Q1)
Q_df.info()

# NEW26 추가
import pandas as pd
import requests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.filters.hp_filter import hpfilter
import pymongo
import warnings
warnings.filterwarnings("ignore")

# API 키 설정
FRED_API_KEY = "8fefd7e8b91eb249e46de9c6577e0ed1"
ECOS_API_KEY = "98PNCEB9CC7IQHQ8O9BF"
START_DATE = "1990-01-01"  # 학습용 과거 데이터
PREDICT_START = "2000-01-01"  # 예측 시작
END_DATE = "2025-03-30"  # 실제 데이터 끝
FORECAST_END = "2025-03-30"  # 미래 예측 끝

# 공통 함수 정의
def get_fred_data(series_id, start_date, end_date):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&observation_start={start_date}&observation_end={end_date}"
    response = requests.get(url)
    data = response.json()
    try:
        df = pd.DataFrame(data['observations'])
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df[['date', 'value']].set_index('date')
        return df.resample('QE').mean()  # 분기별로 통일
    except KeyError:
        print(f"Error retrieving FRED data for {series_id}")
        return None

def get_ecos_data(stat_code, item_code, start_date, end_date, freq='M'):
    url = f"http://ecos.bok.or.kr/api/StatisticSearch/{ECOS_API_KEY}/json/kr/1/1000/{stat_code}/{freq}/{start_date}/{end_date}/{item_code}"
    response = requests.get(url)
    data = response.json()
    try:
        rows = data['StatisticSearch']['row']
        df = pd.DataFrame(rows)
        if freq == 'M':  # 월간 데이터
            df['date'] = pd.to_datetime(df['TIME'], format='%Y%m')
        elif freq == 'Q':  # 분기 데이터
            # "YYYYQn" 형식에서 연도와 분기 추출
            df['date'] = df['TIME'].apply(lambda x: pd.to_datetime(f"{x[:4]}-{(int(x[5:]) * 3)-1:02d}-01") + pd.offsets.MonthEnd(0))
        df['value'] = pd.to_numeric(df['DATA_VALUE'], errors='coerce')
        df = df[['date', 'value']].set_index('date')
        return df.resample('QE').mean() if freq == 'M' else df  # 월간은 분기별로 변환
    except KeyError:
        print(f"Error retrieving ECOS data for {stat_code}/{item_code}")
        return None

def arima_predict(data, order=(2,1,2), forecast_steps=6):
    quarterly_index = pd.date_range(start=PREDICT_START, end=FORECAST_END, freq='QE')
    pred_df = pd.DataFrame(index=quarterly_index, columns=['value'])
    for date in quarterly_index:
        train_end = date - pd.offsets.QuarterEnd(1)
        train_data = data.loc[:train_end]['value'].dropna()
        if len(train_data) > 10:
            model = ARIMA(train_data, order=order)
            result = model.fit()
            pred = result.forecast(steps=1).iloc[0]
            pred_df.loc[date, 'value'] = pred
    return pred_df

# 데이터 수집 및 처리
data_dict = {}

# 1. 경상수지 (ECOS, 월간 → 분기별)(계절조정)백만달러
# 국가가 외국과의 거래에서 발생한 수입과 지출의 차액
current_account = get_ecos_data("301Y017", "SA000", "200001", "202511")
if current_account is not None:
    data_dict['current_account'] = current_account

# 2. 가계신용 (ECOS, 분기별)십억원
household_debt = get_ecos_data("151Y004", "1000000", "2000Q1", "2025Q1", "Q")
if household_debt is not None:
    data_dict['household_debt'] = household_debt

# 3. 주택가격지수 (ECOS, 월간 → 분기별)총지수(Housing Price Index) 기준년 2020 예상
housing_price = get_ecos_data("901Y062", "P63A", "200001", "202511")
if housing_price is not None:
    data_dict['housing_price'] = housing_price

# 4. 외환보유액 (ECOS, 월간 → 분기별) 외환 천달러
foreign_reserves = get_ecos_data("732Y001", "04", "200001", "202501")
if foreign_reserves is not None:
    data_dict['foreign_reserves'] = foreign_reserves

# 5. 주요국 GDP 갭 (FRED, 분기별)(Major Country GDP Gap)
gdp_series = {
    'korea': "NGDPRSAXDCKRQ",
    'us_real': "GDPC1",
    'us_potential': "GDPPOT",
    'china': "NGDPRXDCCNA",
    'euro': "CLVMEURSCAB1GQEA19",
    'japan': "JPNRGDPEXP"
}
for country, series in gdp_series.items():
    df = get_fred_data(series, START_DATE, END_DATE)
    if df is not None:
        data_dict[f"{country}_gdp"] = df
        if country == 'us_real':
            us_potential = data_dict.get('us_potential_gdp')
            if us_potential is not None:
                data_dict['us_gdp_gap'] = ((df['value'] - us_potential['value']) / us_potential['value']) * 100
        elif country == 'china':
            df.loc[df.index < "2011-01-01", 'value'] = df['value'].dropna().iloc[0]
            quarterly_index = pd.date_range(start=START_DATE, end=END_DATE, freq='QE')
            china_q = pd.DataFrame(index=quarterly_index)
            for year in df.index.year.unique():
                year_value = df.loc[df.index.year == year, 'value'].iloc[0]
                china_q.loc[quarterly_index[quarterly_index.year == year], 'value'] = year_value
            data_dict['china_gdp'] = china_q
            cycle, potential = hpfilter(china_q['value'], lamb=1600)
            data_dict['china_gdp_gap'] = ((china_q['value'] - potential) / potential) * 100
        else:
            cycle, potential = hpfilter(df['value'], lamb=1600)
            data_dict[f"{country}_gdp_gap"] = ((df['value'] - potential) / potential) * 100

# 6. 정책 기대 (금리, FRED+ECOS, 분기별)
korea_rate_fred = get_fred_data("INTDSRKRM193N", START_DATE, END_DATE)
korea_rate_ecos = get_ecos_data("722Y001", "0101000", "200001", "202412")
us_rate = get_fred_data("FEDFUNDS", START_DATE, END_DATE)
if korea_rate_fred is not None and korea_rate_ecos is not None:
    korea_rate = pd.concat([korea_rate_fred.loc[:'1999-12-31'], korea_rate_ecos.loc['2000-01-01':]])
    data_dict['korea_interest_rate'] = korea_rate_ecos
    data_dict['korea_interest_rate_pred'] = arima_predict(korea_rate)
if us_rate is not None:
    data_dict['us_interest_rate'] = us_rate
    data_dict['us_interest_rate_pred'] = arima_predict(us_rate)

# 7. 석유 수입 가격 (ECOS, 월간 → 분기별)
oil_volume = get_ecos_data("403Y004", "201121AA", "200001", "202503")
oil_price = get_ecos_data("403Y003", "201121AA", "200001", "202503")
if oil_volume is not None and oil_price is not None:
    df = pd.merge(oil_volume.rename(columns={'value': 'volume'}), oil_price.rename(columns={'value': 'price'}), left_index=True, right_index=True)
    volume_base, price_base = 980259000, 44427432000
    df['total_volume'] = (df['volume'] / 100) * volume_base
    df['total_price'] = (df['price'] / 100) * price_base
    df['oil_import_price'] = df['total_price'] / df['total_volume']
    data_dict['oil_import_price'] = df[['oil_import_price']]

# 8. 한국 투자 (ECOS, 분기별)주체별 총 고정가본형성(계절조정,실질,분기)십억원
investment = get_ecos_data("200Y134", "10101", "2000Q1", "2025Q1", freq='Q')
if investment is not None:
    data_dict['investment_korea'] = investment

# 9. 한국 소비 (ECOS, 분기별) 국내 총생산에 대한 지출 (실질,계절 조정,전기비)
consumption = get_ecos_data("200Y102", "10122", "2000Q1", "2025Q1", freq='Q')
if consumption is not None:
    data_dict['consumption_korea'] = consumption

# 10. 실질 유효 환율 (FRED, 월간 → 분기별)
reer_us = get_fred_data("RBUSBIS", START_DATE, END_DATE)
reer_korea = get_fred_data("RBKRBIS", START_DATE, END_DATE)
if reer_us is not None:
    data_dict['reer_us'] = reer_us
if reer_korea is not None:
    data_dict['reer_korea'] = reer_korea

# 11. 대한민국 총 수입수출 (FRED, 분기별)
korea_exports = get_fred_data("XTEXVA01KRQ667N", START_DATE, END_DATE)
if korea_exports is not None:
    data_dict['korea_exports'] = korea_exports

korea_imports = get_fred_data("XTIMVA01KRQ667S", START_DATE, END_DATE)
if korea_imports is not None:
    data_dict['korea_imports'] = korea_imports

# 12. 한미 CPI (ECOS, 분기별)
KOR_CPI = get_ecos_data("902Y008", "KR", "2000Q1", "2025Q1", freq='Q')
if KOR_CPI is not None:
    data_dict['KOR_CPI'] = KOR_CPI
US_CPI = get_ecos_data("902Y008", "US", "2000Q1", "2025Q1", freq='Q')
if US_CPI is not None:
    data_dict['US_CPI'] = US_CPI


# 병합: 2000년 이후 데이터만 포함
quarterly_index = pd.date_range(start="2000-01-01", end=FORECAST_END, freq='QE')
final_df = pd.DataFrame(index=quarterly_index)

for key, df in data_dict.items():
    # 'value' 컬럼이 있으면 사용, 없으면 Series로 변환
    if isinstance(df, pd.DataFrame) and 'value' in df.columns:
        series = df['value']
    else:
        series = df
    # 2000년 이후로 필터링하고 인덱스 맞춤
    series = series.reindex(quarterly_index, method='ffill')
    final_df=final_df.bfill()
    final_df.isna().sum()
    final_df[key] = series

# 결과 확인 및 저장
print("Final DataFrame Columns:", final_df.columns.tolist())

# 변경하려는 날짜들 (월, 일)의 목록
target_dates = [(3, 31), (6, 30), (9, 30), (12, 31)]

# 새로운 인덱스 값을 담을 리스트 생성
new_index_list = []

# 기존 인덱스를 하나씩 순회하면서 새로운 인덱스 값을 계산
for current_date in final_df.index:
    # 현재 날짜가 변경 대상 날짜인지 확인
    if (current_date.month, current_date.day) in target_dates:
        # 변경 대상이면 하루를 더한 날짜를 새로운 값으로 사용
        # pd.DateOffset(days=1)을 사용하면 월, 년 변경이 자동으로 처리됩니다.
        new_date = current_date + pd.DateOffset(days=1)
    else:
        # 변경 대상이 아니면 원래 날짜를 그대로 사용
        new_date = current_date

    # 계산된 새로운 날짜를 리스트에 추가
    new_index_list.append(new_date)

# 계산된 새로운 날짜 리스트로 데이터프레임의 인덱스를 교체
final_df.index = pd.DatetimeIndex(new_index_list)

final_df = final_df.loc['2005-01-01':'2024-10-02']

Q_df_merged = Q_df.join(final_df)

Q_df = Q_df_merged.copy()
Q_df.info()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# 분기별 데이터를 일별 데이터에 통합하기 위한 함수
def merge_quarterly_daily_data(quarterly_df, daily_df):
    # 분기별 데이터의 각 인덱스를 해당 분기의 시작일로 간주
    q_data = quarterly_df.copy()
    d_data = daily_df.copy()

    # 일별 데이터에 사용할 날짜 범위
    min_date = d_data.index.min()
    max_date = d_data.index.max()

    # 분기별 데이터 최신화 (일별 데이터 범위에 맞게)
    q_data = q_data.loc[(q_data.index >= min_date) & (q_data.index <= max_date)]

    # 일별 데이터에 분기 열 추가
    d_data['quarter'] = d_data.index.to_period('Q')

    # 분기별 데이터에 분기 정보 추가
    q_data['quarter'] = q_data.index.to_period('Q')

    # 분기를 기준으로 일별 데이터에 분기별 데이터 병합
    merged_data = pd.merge(
        d_data.reset_index(),
        q_data.reset_index(),
        on='quarter',
        how='left',
        suffixes=('', '_quarterly')
    )

    # 중복된 date 열 삭제하고 인덱스 설정
    merged_data = merged_data.drop('date_quarterly', axis=1).set_index('date')

    # 분기를 나타내는 임시 열 삭제
    merged_data = merged_data.drop('quarter', axis=1)

    return merged_data

# 시차 변수(Lag Features) 생성 함수
def create_lag_features(df, lag_columns, lag_periods):
    df_copy = df.copy()

    for col in lag_columns:
        for lag in lag_periods:
            lag_col_name = f'{col}_lag_{lag}'
            df_copy[lag_col_name] = df_copy[col].shift(lag)

    return df_copy

# 롤링 윈도우 특성 생성 함수
def create_rolling_features(df, roll_columns, windows):
    df_copy = df.copy()

    for col in roll_columns:
        for window in windows:
            # 평균
            df_copy[f'{col}_roll_mean_{window}'] = df_copy[col].rolling(window=window).mean()
            # 표준 편차
            df_copy[f'{col}_roll_std_{window}'] = df_copy[col].rolling(window=window).std()
            # 최대값과 최소값의 차이
            df_copy[f'{col}_roll_range_{window}'] = df_copy[col].rolling(window=window).max() - df_copy[col].rolling(window=window).min()

    return df_copy

# 데이터 차분 특성 생성 함수
def create_diff_features(df, diff_columns, periods):
    df_copy = df.copy()

    for col in diff_columns:
        for period in periods:
            diff_col_name = f'{col}_diff_{period}'
            df_copy[diff_col_name] = df_copy[col].diff(period)

            # 퍼센트 변화율 (실수 연산 에러 방지)
            pct_col_name = f'{col}_pct_{period}'
            df_copy[pct_col_name] = df_copy[col].pct_change(period)

    return df_copy

# 날짜 관련 특성 추가 함수
def add_date_features(df):
    df_copy = df.copy()

    # 날짜 관련 특성 추가
    df_copy['day_of_week'] = df_copy.index.dayofweek
    df_copy['month'] = df_copy.index.month
    df_copy['quarter'] = df_copy.index.quarter
    df_copy['year'] = df_copy.index.year
    df_copy['day_of_year'] = df_copy.index.dayofyear
    df_copy['week_of_year'] = df_copy.index.isocalendar().week

    # 사이클링 표현 (주기적 특성을 더 잘 포착하기 위해)
    df_copy['day_of_week_sin'] = np.sin(df_copy['day_of_week'] * (2 * np.pi / 7))
    df_copy['day_of_week_cos'] = np.cos(df_copy['day_of_week'] * (2 * np.pi / 7))
    df_copy['month_sin'] = np.sin((df_copy['month'] - 1) * (2 * np.pi / 12))
    df_copy['month_cos'] = np.cos((df_copy['month'] - 1) * (2 * np.pi / 12))

    return df_copy

# 경제 지표간 비율 및 교차 특성 생성 함수
def create_ratio_features(df):
    df_copy = df.copy()

    # 한국과 미국 금리 차이
    df_copy['interest_rate_diff'] = df_copy['korea_interest_rate'] - df_copy['us_interest_rate']

    # CPI 비율 (인플레이션 비교)
    df_copy['cpi_ratio'] = df_copy['KOR_CPI'] / df_copy['US_CPI']

    # 수출입 비율
    df_copy['export_import_ratio'] = df_copy['korea_exports'] / df_copy['korea_imports']

    # 실질환율 비율
    df_copy['reer_ratio'] = df_copy['reer_korea'] / df_copy['reer_us']

    return df_copy

# 데이터 전처리 메인 함수
def preprocess_data(quarterly_df, daily_df):
    # 무빙 에버리지 컬럼 제거
    ma_columns = [col for col in daily_df.columns if '_MA' in col]
    daily_data = daily_df.drop(columns=ma_columns, errors='ignore')

    # 분기별 데이터와 일별 데이터 통합
    merged_data = merge_quarterly_daily_data(quarterly_df, daily_data)

    # 타겟 변수 정의
    target_column = 'USDKRW=X_Close'

    # 날짜 특성 추가
    merged_data = add_date_features(merged_data)

    # 시차 특성 생성 (타겟 및 핵심 특성들에 대해)
    lag_columns = [target_column, 'DX-Y.NYB_Close', 'EURUSD=X_Close', '^GSPC_Close', '^VIX_Close']
    merged_data = create_lag_features(merged_data, lag_columns, [1, 3, 5, 7, 14, 21])

    # 롤링 특성 생성
    roll_columns = [target_column, 'DX-Y.NYB_Close', 'EURUSD=X_Close', '^VIX_Close']
    merged_data = create_rolling_features(merged_data, roll_columns, [5, 10, 20])

    # 차분 특성 생성
    diff_columns = [target_column, 'DX-Y.NYB_Close', 'EURUSD=X_Close', '^GSPC_Close', '^VIX_Close', 'GC=F_Close', 'CL=F_Close']
    merged_data = create_diff_features(merged_data, diff_columns, [1, 5])

    # 경제 지표간 비율 특성 생성
    merged_data = create_ratio_features(merged_data)

    # NA 값 처리 (전진 채우기 후 후진 채우기)
    merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')

    return merged_data, target_column

# 데이터 준비
processed_data, target_column = preprocess_data(Q_df, D_df)

# 학습/테스트 분할 (시간 기반)
split_date = processed_data.index.max() - timedelta(days=60)  # 최근 60일을 테스트 세트로 사용
train_data = processed_data.loc[processed_data.index <= split_date]
test_data = processed_data.loc[processed_data.index > split_date]

# 특성 및 타겟 분리
X_train = train_data.drop(target_column, axis=1)
y_train = train_data[target_column]
X_test = test_data.drop(target_column, axis=1)
y_test = test_data[target_column]

# 예측할 미래 날짜 생성 (지난 거래일로부터 5일)
last_date = processed_data.index.max()
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='B')


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
from datetime import timedelta
import uuid
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 최적 하이퍼파라미터
best_params_lgb = {
    'n_estimators': 423,
    'learning_rate': 0.19301667354007498,
    'num_leaves': 25,
    'max_depth': 7,
    'min_child_samples': 26,
    'subsample': 0.9166693062792854,
    'colsample_bytree': 0.9304717428288543,
    'reg_alpha': 0.005446718336403275,
    'reg_lambda': 2.368998364165221,
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'random_state': 42,
    'verbose': -1
}

# LightGBM 모델 학습
print("LightGBM 모델 학습 시작...")
lgb_model = lgb.LGBMRegressor(**best_params_lgb)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='rmse'
)
print("LightGBM 모델 학습 완료.")

# 학습 및 테스트 데이터 예측
train_preds = lgb_model.predict(X_train)
test_preds = lgb_model.predict(X_test)

# 성능 평가
lgb_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
lgb_mse = mean_squared_error(y_test, test_preds)
lgb_mae = mean_absolute_error(y_test, test_preds)
lgb_r2 = r2_score(y_test, test_preds)

print(f"\nLightGBM 모델 성능:")
print(f"RMSE: {lgb_rmse:.4f}")
print(f"MSE: {lgb_mse:.4f}")
print(f"MAE: {lgb_mae:.4f}")
print(f"R²: {lgb_r2:.4f}")

# 날짜 형식 확인 및 변환
if not pd.api.types.is_datetime64_any_dtype(processed_data.index):
    print("인덱스를 datetime으로 변환합니다...")
    processed_data.index = pd.to_datetime(processed_data.index)

# 미래 5일 예측을 위한 날짜 생성
last_data_date = processed_data.index.max()
future_dates = []
current_date = last_data_date + pd.DateOffset(days=1)
for _ in range(5):
    while current_date.weekday() >= 5:  # 주말 건너뛰기
        current_date += pd.DateOffset(days=1)
    future_dates.append(current_date)
    current_date += pd.DateOffset(days=1)

# 미래 5일 예측을 위한 특성 업데이트 함수
def update_features_for_forecast_lgb(history_df, new_date, target_column):
    last_data = history_df.iloc[[-1]].copy()
    updated_features = last_data.copy()
    updated_features.index = [new_date]
    
    # 날짜 특성 업데이트
    updated_features['day_of_week'] = new_date.dayofweek
    updated_features['month'] = new_date.month
    updated_features['quarter'] = new_date.quarter
    updated_features['year'] = new_date.year
    updated_features['day_of_year'] = new_date.dayofyear
    updated_features['week_of_year'] = new_date.isocalendar()[1]
    updated_features['day_of_week_sin'] = np.sin(updated_features['day_of_week'] * (2 * np.pi / 7))
    updated_features['day_of_week_cos'] = np.cos(updated_features['day_of_week'] * (2 * np.pi / 7))
    updated_features['month_sin'] = np.sin((updated_features['month'] - 1) * (2 * np.pi / 12))
    updated_features['month_cos'] = np.cos((updated_features['month'] - 1) * (2 * np.pi / 12))
    
    # 시차 특성 업데이트
    lag_periods = [1, 3, 5, 7, 14, 21]
    for lag in lag_periods:
        lag_col = f'{target_column}_lag_{lag}'
        past_date_for_lag = new_date - pd.DateOffset(days=lag)
        closest_past_date_idx = history_df.index.get_indexer([past_date_for_lag], method='ffill')
        if closest_past_date_idx[0] != -1 and closest_past_date_idx[0] < len(history_df):
            updated_features[lag_col] = history_df.iloc[closest_past_date_idx[0]][target_column]
        else:
            updated_features[lag_col] = history_df[target_column].iloc[-1]
    
    # 타겟 열 제거 (예측에 불필요)
    if target_column in updated_features.columns:
        updated_features = updated_features.drop(columns=[target_column])
    
    # X_train의 열 순서와 일치시키기
    updated_features = updated_features.reindex(columns=X_train.columns)
    
    return updated_features

# 미래 5일 예측
max_lag = max([1, 3, 5, 7, 14, 21])
future_features_history = processed_data.iloc[-max_lag:].copy()
lgb_future_preds = []

print("\nLightGBM 미래 5일 예측 시작...")
for i in range(5):
    current_date = future_dates[i]
    current_step_features = update_features_for_forecast_lgb(
        future_features_history,
        current_date,
        target_column
    )
    
    # 예측
    pred = lgb_model.predict(current_step_features)[0]
    lgb_future_preds.append(pred)
    
    # 다음 예측을 위해 히스토리 업데이트
    predicted_row = current_step_features.copy()
    predicted_row[target_column] = pred
    future_features_history = pd.concat([future_features_history, predicted_row])

# 모든 예측값과 날짜 결합
train_dates = X_train.index
test_dates = X_test.index
all_dates = np.concatenate([train_dates, test_dates, future_dates])
all_predictions = np.concatenate([train_preds, test_preds, lgb_future_preds])

# 실측값 결합 (학습 + 테스트)
all_actuals = np.concatenate([y_train, y_test, [np.nan] * 5])  # 미래 5일은 실측값 없음

# 날짜를 datetime으로 보장
all_dates = [pd.Timestamp(date) for date in all_dates]

# 데이터프레임 생성 (CSV 저장용)
output_df = pd.DataFrame({
    'Date': all_dates,
    'Actual': all_actuals,
    'Prediction': all_predictions
})
# rpa로 실행
def main():
    # CSV 파일로 저장
    output_filename = 'lgbm_predictions.csv'
    output_df.to_csv(output_filename, index=False)
    print(f"\n예측값이 {output_filename}에 저장되었습니다.")
    result = "학습 완료."
    return result
'''
# 그래프 시각화 (가독성 개선)
plt.figure(figsize=(14, 8))  # 그래프 크기 증가

# 실측값과 예측값 (학습 + 테스트)
historical_dates = all_dates[:-5]  # 미래 5일 제외
plt.plot(historical_dates, all_actuals[:-5], label='Actual', color='navy', linewidth=2.5, alpha=0.9)
plt.plot(historical_dates, all_predictions[:-5], label='Predicted', color='orange', linewidth=1.5, alpha=0.7)

# 미래 5일 예측 (점선 + 마커)
future_dates = all_dates[-5:]
plt.plot(future_dates, all_predictions[-5:], label='Forecast', color='orange', linestyle='--', linewidth=2, marker='o', markersize=6, alpha=0.8)

# 그래프 설정
plt.xlabel('Date', fontsize=12)
plt.ylabel('USDKRW=X_Close', fontsize=12)
plt.title('LightGBM: Actual vs Predicted (with 5-Day Forecast)', fontsize=14, pad=15)
plt.grid(True, linestyle='--', alpha=0.5)

# X축 날짜 포맷팅
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))  # 날짜 간격 조정
plt.xticks(rotation=45, fontsize=10)

# Y축 범위 동적 조정
y_min = min(np.nanmin(all_actuals), np.nanmin(all_predictions)) * 0.99
y_max = max(np.nanmax(all_actuals), np.nanmax(all_predictions)) * 1.01
plt.ylim(y_min, y_max)

# 범례를 그래프 외부에 배치
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

# 레이아웃 조정
plt.tight_layout()

# 그래프 표시
plt.show()
'''
# 미래 예측 결과 출력
print("\nLightGBM 미래 5일 예측:")
for date, pred in zip(future_dates, lgb_future_preds):
    print(f"{date.date()}: {pred:.2f}")