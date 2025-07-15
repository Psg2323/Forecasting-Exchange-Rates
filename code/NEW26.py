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
print("Final DataFrame (tail):")
print(final_df.tail())

# MongoDB 업로드 (필요 시 사용)
client = pymongo.MongoClient("mongodb+srv://kanzki247:ahdrhelql@cluster0.b3gwf.mongodb.net/")
db = client["stock_data"]
collection = db["NEW24"]
records = final_df.reset_index().rename(columns={'index': 'date'}).to_dict('records')
collection.delete_many({})
collection.insert_many(records)
print(f"✅ 데이터 업로드 완료 ({len(records)}건)")