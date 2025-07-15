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

# 일별 데이터 목록
tickers = {
    'DXY': 'DX-Y.NYB',  # 달러 인덱스 (Dollar Index)
    'EURUSD': 'EURUSD=X',  # 유로/미국 달러 환율 (EUR/USD)
    'JPYUSD': 'JPY=X',  # 일본 엔/미국 달러 환율 (JPY/USD)
    'GBPUSD': 'GBPUSD=X',  # 영국 파운드/미국 달러 환율 (GBP/USD)
    'CHFUSD': 'CHFUSD=X',  # 스위스 프랑/미국 달러 환율 (CHF/USD)
    'USDKRW': 'USDKRW=X',  # 미국 달러/한국 원 환율 (USD/KRW)
    'CNYUSD': 'CNYUSD=X',  # 중국 위안/미국 달러 환율 (CNY/USD)
    'SP500': '^GSPC',  # S&P 500 지수
    'VIX': '^VIX',  # 변동성 지수 (VIX, 공포지수)
    'Gold': 'GC=F',  # 금 선물 (Gold Futures)
    'Oil': 'CL=F',  # 서부 텍사스산 원유 선물 (WTI Crude Oil Futures)
    'KOSPI': '^KS11',  # 코스피 종합지수   
}
# 데이터 기간 설정 -> 20년
start_date = '2004-01-01'  # 시작 날짜
end_date = date.today()

# 🔹 데이터 다운로드
def download_data(tickers, start_date, end_date):
    data = {}
    for name, ticker in tickers.items():
        df = yf.download(ticker, start=start_date, end=end_date)
        df['Ticker'] = name
        df.reset_index(inplace=True)
        data[name] = df
    return data

pd.set_option('display.max_rows', None)

# yfinance 데이터 수집
yf_data = yf.download(list(tickers.values()), start=start_date, end=end_date, group_by='ticker')
yf_data.info()

yf_data.head(10)

# 컬럼 이름 변경
def process_yfinance(data, ticker):
        df = data[ticker][['Close', 'High', 'Low']]
        df.columns = [f'{ticker}_Close', f'{ticker}_High',
                      f'{ticker}_Low']
        return df

df_yf = pd.concat([process_yfinance(yf_data, t) for t in tickers.values()], axis=1)

# 종가만 추출
df_yf.info()
close_columns = [col for col in df_yf.columns if col.endswith('_Close')]
daily_df = df_yf[close_columns].copy()

# 🔹 FRED API 키
api_key = "8fefd7e8b91eb249e46de9c6577e0ed1"

fred_series = {
    'GDP':'US GDP', # US 국내총 생산
    'FEDFUNDS': 'Federal Funds Rate', # 연방기금금리
    'DTWEXAFEGS': 'Trade Weighted Dollar', # 무역가중달러지수
    'DGS10': '10-Year Treasury Yield', # 10년 만기 국채 수익률
    'PCEPI': 'PCE Price Index', # 개인소비지출 물가지수(PCE)
    'PPIACO': 'PPI', # 생산자물가지수(PPI)
    'UNRATE': 'Unemployment Rate', # 실업률
    'INDPRO': 'Industrial Production', # 산업생산지수
    'T10Y2Y': '10Y-2Y Yield Spread', # 10년-2년 금리차 (수익률 스프레드)
    'XTEXVA01KRQ659S':'총 수출 (상품)', #분기별	계절 조정됨	명목 값 (USD)',
    'XTIMVA01KRQ667S':'총 수입 (상품)', #분기별	계절 조정됨	명목 값 (USD)'
}

start_date = '2005-01-01'  # 시작 날짜
end_date = date.today()

# fred 데이터 다운로드 및 전처리
df_fred = pd.DataFrame()
fred = Fred(api_key=api_key)

for code, name in fred_series.items():
    try:
        data = fred.get_series(code, observation_start=start_date, observation_end=end_date)
        df_fred[code] = data
    except Exception as e:
        print(f"❌ {name}({code}) 가져오기 실패: {e}")
        
df_fred.index.name = 'date'
df_fred.reset_index(inplace=True)
df_fred = df_fred.set_index('date')

# 데이터 확인
df_fred.info()

# 최신 날짜를 조회하는 함수
def get_latest_date(collection):
    """
    MongoDB 컬렉션에서 가장 최근의 date를 반환합니다.
    
    Args:
        collection: MongoDB 컬렉션 객체
    
    Returns:
        datetime: 가장 최근의 date, 데이터가 없으면 None 반환
    """
    latest_doc = collection.find_one(sort=[("date", -1)])  # date 기준 내림차순 정렬
    if latest_doc and "date" in latest_doc:
        return pd.to_datetime(latest_doc["date"])
    return None

# 새로운 데이터만 MongoDB에 저장하는 함수
def save_new_data_to_mongo(df, collection, collection_name):
    """
    DataFrame에서 기존 데이터의 최신 날짜 이후의 데이터만 MongoDB에 저장합니다.
    
    Args:
        df (pd.DataFrame): 저장할 데이터프레임 (인덱스는 datetime)
        collection: MongoDB 컬렉션 객체
        collection_name (str): 컬렉션 이름 (로그용)
    
    Returns:
        None
    """
    # 최신 날짜 조회
    latest_date = get_latest_date(collection)
    
    # 새로운 데이터 필터링
    if latest_date is not None:
        new_data = df[df.index > latest_date]
    else:
        new_data = df  # 기존 데이터가 없으면 전체 데이터 저장
    
    # 새로운 데이터가 있는 경우에만 저장
    if not new_data.empty:
        documents = []
        for date, row in new_data.to_dict('index').items():
            document = row.copy()
            document['date'] = date.isoformat()  # datetime을 ISO 형식 문자열로 변환
            documents.append(document)
        
        collection.insert_many(documents)
        print(f"{collection_name}: {len(documents)}개의 새로운 문서가 MongoDB에 저장되었습니다.")
        result = f"{collection_name}: {len(documents)}개의 새로운 문서가 MongoDB에 저장되었습니다."
    else:
        print(f"{collection_name}: 새로운 데이터가 없어 저장하지 않습니다.")
        result = f"{collection_name}: 새로운 데이터가 없어 저장하지 않습니다."
    return result

# 실행
def main():
    # daily_df (testD) 저장
    result_d = save_new_data_to_mongo(daily_df, collection_testD, "testD")
    
    # df_fred (testQ) 저장
    result_q = save_new_data_to_mongo(df_fred, collection_testQ, "testQ")

    # 두 결과를 문자열로 합쳐서 반환
    return result_d + "\n" + result_q