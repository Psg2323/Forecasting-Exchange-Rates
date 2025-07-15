import pandas as pd
import pymongo
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter

# 📌 MongoDB 연결 (사용자 환경에 맞게 URI 변경 가능)
client = pymongo.MongoClient("mongodb+srv://kanzki247:ahdrhelql@cluster0.b3gwf.mongodb.net/")
db = client["stock_data"]
coll_macro = db["NEW24"]  # 분기별 경제지표
coll_market = db["NEW35"]  # 일일 시장 데이터 + 기술적 지표

# 🧱 1. 분기 데이터 불러오기 → DataFrame으로 변환
df_quarter = pd.DataFrame(list(coll_macro.find())).drop("_id", axis=1)
df_quarter['date'] = pd.to_datetime(df_quarter['date'])
df_quarter.set_index('date', inplace=True)

# 🧱 2. 일일 데이터 불러오기 → 최근 데이터 보간
# df_daily_raw = pd.DataFrame(list(coll_market.find())).drop("_id", axis=1)
# df_daily = pd.DataFrame([{'date': doc['date'], **doc['data']} for doc in df_daily_raw])
# df_daily['date'] = pd.to_datetime(df_daily['date'])
# df_daily.set_index('date', inplace=True)
# df_daily = df_daily.resample('QE').mean().ffill()
# 🧱 2. 일일 데이터 불러오기 → 최근 데이터 보간
# MongoDB에서 데이터를 JSON 리스트로 불러오기 및 _id 제거
df_daily_raw = pd.json_normalize(
    list(coll_market.find({}, {"_id": 0})))

# 컬럼 이름에서 'data.' 접두사 제거
df_daily_raw.columns = df_daily_raw.columns.str.replace('^data\.', '', regex=True)

# 날짜 변환 및 인덱스 설정
df_daily_raw['date'] = pd.to_datetime(df_daily_raw['date'], errors='coerce')
df_daily_raw.dropna(subset=['date'], inplace=True)  # 날짜가 없는 데이터 제거
df_daily_raw.set_index('date', inplace=True)

# 숫자 데이터만 선택 후 리샘플
df_daily = df_daily_raw.select_dtypes(include='number')  # 숫자 컬럼만 추출
df_daily = df_daily.resample('QE').mean().ffill()



# 🎯 최종 DataFrame 병합
final_df = df_quarter.join(df_daily, how='outer').sort_index().ffill()
final_df.columns
# ✅ 내생변수 생성 시작

# 1. 🇰🇷 GDP 갭 (실질 GDP - HP 필터 기반 잠재 GDP)
korea_gdp = final_df['korea_gdp']
korea_cycle, korea_trend = hpfilter(korea_gdp, lamb=1600)
final_df['korea_gdp_gap'] = ((korea_gdp - korea_trend) / korea_trend) * 100  # 백분율 표현
# → 경기 과열/침체 여부를 보여주는 핵심 거시 내생변수

# 2. 🇺🇸 정책금리 차이 (한-미 정책금리 차)
final_df['policy_rate_diff'] = final_df['korea_interest_rate'] - final_df['us_interest_rate']
# → 금리 차는 자본 이동을 유발하고 환율에 직접 영향

# 3. 📉 실질 실효환율 갭 (REER 갭)
reer_k, reer_u = final_df['reer_korea'], final_df['reer_us']
k_reer_cycle, k_reer_trend = hpfilter(reer_k, lamb=1600)
final_df['reer_gap'] = ((reer_k - k_reer_trend) / k_reer_trend) * 100
# → 원화가 이론적 가치보다 과대평가 or 저평가 되었는지를 나타냄

# 4. 🏠 가계부채 비율 (GDP 대비 가계부채)
final_df['debt_to_gdp'] = (final_df['household_debt'] * 1000) / (korea_gdp * 1000)
# → 금융위험의 주요 지표이며, 통화정책과 환율의 상호작용에 영향

# 5. 🛢️ 원유 수입 가격 → 수입 인플레 요인
final_df['oil_price_inflation'] = final_df['oil_import_price'].pct_change() * 100
# → 국제 유가 변동은 수입물가 및 환율에 영향

# 6. 🇰🇷 실질 소비 증가율 (소비 지출 전기비 변화율)
final_df['consumption_growth'] = final_df['consumption_korea'].pct_change() * 100
final_df = final_df.replace([np.inf, -np.inf], np.nan)
final_df['consumption_korea'].replace(0, np.nan).ffill()
# → 내수 경기 반영 지표이며, 환율과 수입 수요 간 연결

# 7. 💹 수출입 갭 (순수출 비율)
exports = final_df['korea_exports']
imports = final_df['korea_imports']
final_df['net_export_ratio'] = ((exports - imports) / (exports + imports)) * 100
# → 경상수지에 영향을 주는 요인으로 환율과 밀접한 관련

# 8. 🪙 기술적 지표 예: 환율의 RSI
final_df['fx_rsi'] = final_df['USDKRW=X_Close_RSI']
final_df['fx_macd'] = final_df['USDKRW=X_Close_MACD']
# → 시장 심리를 반영하는 단기 예측 변수

# 9. 📊 VIX (변동성 지수 → 글로벌 리스크 프록시)
final_df['vix_index'] = final_df['^VIX_Close']
# → 위험 회피 심리를 통해 환율에 간접 영향

# 결과 확인
final_vars = final_df[['korea_gdp_gap', 'policy_rate_diff', 'reer_gap', 'debt_to_gdp',
                      'oil_price_inflation', 'consumption_growth', 'net_export_ratio',
                      'fx_rsi', 'fx_macd', 'vix_index']]

print("✅ 내생변수 생성 완료. 예시 데이터:")
print(final_vars.tail())

# korea_gdp_gap	한국의 GDP 갭 (실질 vs 잠재)
# policy_rate_diff	한미 정책금리 차이
# reer_gap	실질 실효환율의 과대/과소 평가 지표
# debt_to_gdp	가계부채 / GDP 비율
# oil_price_inflation	유가 상승률 (인플레이션 유발 변수)
# consumption_growth	실질 소비 증가율
# net_export_ratio	순수출 비율 (경상수지 대리 변수)
# fx_rsi, fx_macd	기술적 환율 지표 (심리 반영)
# vix_index	글로벌 불확실성 (위험 회피 지표)