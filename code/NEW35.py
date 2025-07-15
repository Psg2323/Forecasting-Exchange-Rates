# 35개 필수컬럼 저장코드
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from scipy.interpolate import CubicSpline
from fredapi import Fred
import pymongo
from pymongo import MongoClient
from datetime import datetime

# 🔹 MongoDB 연결
# MONGO_URI = "mongodb+srv://kanzki001:ahdrhelql@cluster0.b3gwf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGO_URI = "mongodb+srv://kanzki247:ahdrhelql@cluster0.b3gwf.mongodb.net/"
client = pymongo.MongoClient(MONGO_URI)
db = client["stock_data"]  # 데이터베이스 이름
collection_full = db["daily_prices"]  # 모든 컬럼 저장 컬렉션
collection_selected = db["NEW35"]    # 35개 선택 컬럼 저장 컬렉션

# 🔹 FRED API 키
api_key = "8fefd7e8b91eb249e46de9c6577e0ed1"

# 🔹 관심 종목 및 시리즈
tickers = {
    'DXY': 'DX-Y.NYB',  # 달러 인덱스 (Dollar Index)
    'EURUSD': 'EURUSD=X',  # 유로/미국 달러 환율 (EUR/USD)
    'JPYUSD': 'JPY=X',  # 일본 엔/미국 달러 환율 (JPY/USD)
    'GBPUSD': 'GBPUSD=X',  # 영국 파운드/미국 달러 환율 (GBP/USD)
    'CADUSD': 'CADUSD=X',  # 캐나다 달러/미국 달러 환율 (CAD/USD)
    'SEKUSD': 'SEKUSD=X',  # 스웨덴 크로나/미국 달러 환율 (SEK/USD)
    'CHFUSD': 'CHFUSD=X',  # 스위스 프랑/미국 달러 환율 (CHF/USD)
    'KRWUSD': 'KRWUSD=X',  # 한국 원/미국 달러 환율 (KRW/USD)
    'USDKRW': 'USDKRW=X',  # 미국 달러/한국 원 환율 (USD/KRW)
    'CNYUSD': 'CNYUSD=X',  # 중국 위안/미국 달러 환율 (CNY/USD)
    'SP500': '^GSPC',  # S&P 500 지수
    'Nasdaq': '^IXIC',  # 나스닥 종합지수
    'Daw': '^DJI',  # 다우존스 산업평균지수 (Dow Jones Industrial Average)
    'VIX': '^VIX',  # 변동성 지수 (VIX, 공포지수)
    'Gold': 'GC=F',  # 금 선물 (Gold Futures)
    'Oil': 'CL=F',  # 서부 텍사스산 원유 선물 (WTI Crude Oil Futures)
    'KOSPI': '^KS11',  # 코스피 종합지수
    'KOSDAQ': '^KQ11',  # 코스닥 종합지수
    'KOSPI200': '^KS200',  # 코스피200 지수
}

fred_series = {
    'GDP':'US GDP', # US 국내총 생산
    'FEDFUNDS': 'Federal Funds Rate', # 연방기금금리
    'DTWEXAFEGS': 'Trade Weighted Dollar', # 무역가중달러지수
    'DGS10': '10-Year Treasury Yield', # 10년 만기 국채 수익률
    'CPIAUCSL': 'CPI All Items', # 소비자물가지수(CPI) - 전체 항목
    'PCEPI': 'PCE Price Index', # 개인소비지출 물가지수(PCE)
    'PPIACO': 'PPI', # 생산자물가지수(PPI)
    'UNRATE': 'Unemployment Rate', # 실업률
    'INDPRO': 'Industrial Production', # 산업생산지수
    'DPCERA3M086SBEA': 'Real Personal Consumption', # 실질 개인소비지출
    'M2SL': 'M2 Money Stock', # M2 통화공급량
    'T10Y2Y': '10Y-2Y Yield Spread', # 10년-2년 금리차 (수익률 스프레드)
    'NGDPRSAXDCKRQ': '대한민국 실질 GDP',  # 대한민국 실질 GDP
    'INTDSRKRM193N': '대한민국의 이자율',
    'XTEXVA01KRM664S':'국제 상품 무역 통계',
    'KORCPALTT01CTGYM':'소비자물가지수(CPI)',
    'IRLTLT01KRM156N':'장기 정부 채권 수익률: 10년',
    'LRUNTTTTKRM156S':'실업률 총계',
    'XTEXVA01KRQ659S':'총 수출 (상품)', #분기별	계절 조정됨	명목 값 (USD)',
    'XTIMVA01KRQ667S':'총 수입 (상품)', #분기별	계절 조정됨	명목 값 (USD)'
}

# 선택할 컬럼 리스트
columns_to_select = [
    "USDKRW=X_Close", "CNYUSD=X_Close", "Federal Funds Rate", "10-Year Treasury Yield",
    "US GDP", "Trade Weighted Dollar", "CPI All Items", "PCE Price Index", "PPI",
    "Unemployment Rate", "Industrial Production", "Real Personal Consumption", "M2 Money Stock",
    "10Y-2Y Yield Spread", "대한민국 실질 GDP", "대한민국의 이자율", "국제 상품 무역 통계",
    "소비자물가지수(CPI)", "장기 정부 채권 수익률: 10년", "실업률 총계", "CL=F_Close",
    "^VIX_Close", "USDKRW=X_Close_RSI", "USDKRW=X_Close_MACD", "ATR",
    "EURUSD=X_Close", "JPY=X_Close", "^GSPC_Close", "^IXIC_Close", "^DJI_Close",
    "^KS11_Close", "^KQ11_Close", "USDKRW=X_Close_SMA5", "USDKRW=X_Close_SMA20",
    "CL=F_High"
]

def fetch_and_store_data(full_history=False):
    """
    YFinance 및 FRED 데이터를 가져와 MongoDB에 저장하는 함수
    - full_history=True: 2000년 이후 모든 데이터 저장 (최초 실행 시)
    - full_history=False: 최신 데이터만 업데이트 (매일 실행 시)
    """
    print(f"🔄 데이터 업데이트 시작: {datetime.now()} (full_history={full_history})")

    # 1. 기본 설정
    start_date = '2000-01-01'
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    # 2. yfinance 데이터 수집
    yf_data = yf.download(list(tickers.values()), start=start_date, end=end_date, group_by='ticker')

    # 3. FRED 데이터 수집
    fred = Fred(api_key)
    fred_data = pd.DataFrame()
    for code in fred_series.keys():
        series = fred.get_series(code, start=start_date, end=end_date)
        series.name = fred_series[code]
        fred_data = pd.concat([fred_data, series], axis=1)

    # 4. 데이터 병합 및 전처리
    def process_yfinance(data, ticker):
        df = data[ticker][['Close', 'High', 'Low', 'Open']]
        df.columns = [f'{ticker}_Close', f'{ticker}_High',
                      f'{ticker}_Low', f'{ticker}_Open']
        return df

    merged = pd.concat([process_yfinance(yf_data, t) for t in tickers.values()], axis=1)

    # FRED 데이터 일별 보간
    daily_idx = pd.date_range(start=start_date, end=end_date, freq='D')
    merged = merged.reindex(daily_idx)
    fred_daily = fred_data.reindex(merged.index)

    for col in fred_daily.columns:
        if fred_daily[col].count() < len(fred_daily) * 0.3:
            # Cubic Spline 보간
            valid_idx = fred_daily[col].dropna().index
            cs = CubicSpline(valid_idx.to_julian_date(),
                             fred_daily[col].dropna().values)
            fred_daily[col] = cs(merged.index.to_julian_date())
        else:
            fred_daily[col] = fred_daily[col].ffill().interpolate(method='time')

    # 5. 최종 병합
    final_df = pd.concat([merged, fred_daily], axis=1).ffill().bfill()

    # 6. 기술적 지표 추가
    def add_ta_features(df, base='USDKRW=X_Close'):
        # 기본 지표
        df[f'{base}_SMA5'] = ta.trend.SMAIndicator(df[base], 5).sma_indicator()
        df[f'{base}_SMA20'] = ta.trend.SMAIndicator(df[base], 20).sma_indicator()
        # SMA5: 단순이동평균선(5일) → 최근 5일간의 종가 평균을 나타냅니다. 단기 추세를 확인할 때 사용합니다.
        # SMA20: 단순이동평균선(20일) → 최근 20일간의 종가 평균으로, 중기 추세 파악에 사용됩니다.

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

        # ATR
        high = df['USDKRW=X_High']
        low = df['USDKRW=X_Low']
        close = df['USDKRW=X_Close']
        df['ATR'] = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range()
        # ATR (Average True Range): 평균 진폭 범위
        # → 시장의 변동성을 측정하는 지표로, 값이 클수록 시장의 변동이
        # 크다는 것을 의미합니다

        return df

    final_df = add_ta_features(final_df)
    final_df = final_df.ffill().bfill()

    # 7. 모든 컬럼을 daily_prices에 저장
    full_records = []
    for index, row in final_df.iterrows():
        record = {
            "date": index,
            "data": row.to_dict()
        }
        full_records.append(record)

    collection_full.delete_many({})  # 기존 데이터 전체 삭제
    collection_full.insert_many(full_records)  # 모든 컬럼 데이터 삽입
    print(f"✅ daily_prices에 모든 컬럼 저장 완료 ({len(full_records)}건)")

    # 8. 선택한 컬럼만 추출 및 NEW35에 저장
    selected_df = final_df[columns_to_select]
    print("Selected Columns:", selected_df.columns.tolist())
    print("Selected DataFrame (tail):")
    print(selected_df.tail())

    selected_records = []
    for index, row in selected_df.iterrows():
        record = {
            "date": index,
            "data": row.to_dict()
        }
        selected_records.append(record)

    collection_selected.delete_many({})  # 기존 데이터 전체 삭제
    collection_selected.insert_many(selected_records)  # 선택된 컬럼 데이터 삽입
    print(f"✅ NEW35에 선택된 35개 컬럼 저장 완료 ({len(selected_records)}건)")

    print("✅ 모든 데이터 업데이트 완료")

# 🔹 최초 실행 시 2000년 이후 전체 데이터 저장
fetch_and_store_data(full_history=True)