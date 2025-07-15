import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter
import pymongo
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import jax
import numpyro
from numpyro import sample
from numpyro.distributions import Normal, HalfNormal
from numpyro.infer import MCMC, NUTS
import pytensor.tensor as pt


# 📌 MongoDB 연결
client = pymongo.MongoClient("mongodb+srv://kanzki247:ahdrhelql@cluster0.b3gwf.mongodb.net/")
db = client["stock_data"]
coll_macro = db["NEW24"]
coll_market = db["NEW35"]

# 🧱 데이터 로딩 및 병합
# 1. 분기 데이터
df_quarter = pd.DataFrame(list(coll_macro.find({}, {"_id": 0})))
df_quarter['date'] = pd.to_datetime(df_quarter['date'])
df_quarter.set_index('date', inplace=True)

# 2. 일일 데이터 → 분기별 리샘플링
df_daily_raw = pd.json_normalize(list(coll_market.find({}, {"_id": 0})))
df_daily_raw.columns = df_daily_raw.columns.str.replace('^data\.', '', regex=True)
df_daily_raw['date'] = pd.to_datetime(df_daily_raw['date'], errors='coerce')
df_daily_raw.dropna(subset=['date'], inplace=True)
df_daily_raw.set_index('date', inplace=True)
df_daily = df_daily_raw.select_dtypes(include='number').resample('QE').mean().ffill()

# 병합
final_df = df_quarter.join(df_daily, how='outer').sort_index().ffill()

# ✅ 핵심 컬럼 설정
final_df['target_fx'] = final_df['USDKRW=X_Close']
final_df['us_cpi'] = final_df['US_CPI']
final_df['kr_cpi'] = final_df['KOR_CPI']
final_df['us_rate'] = final_df['us_interest_rate']
final_df['kr_rate'] = final_df['korea_interest_rate']

# ✅ 기대 인플레이션 및 차이 계산
final_df['exp_infl_us'] = final_df['us_cpi'].pct_change() * 100
final_df['exp_infl_kr'] = final_df['kr_cpi'].pct_change() * 100
final_df['inflation_diff'] = final_df['exp_infl_us'] - final_df['exp_infl_kr']
final_df['rate_diff'] = final_df['us_rate'] - final_df['kr_rate']

# ✅ 실질환율 갭 계산 (REER)
final_df['reer_gap'] = ((final_df['reer_korea'] - hpfilter(final_df['reer_korea'],
        1600)[1]) / hpfilter(final_df['reer_korea'], 1600)[1]) * 100
final_df['reer_gap_us'] = ((final_df['reer_us'] - hpfilter(final_df['reer_us'],
        1600)[1]) / hpfilter(final_df['reer_us'], 1600)[1]) * 100
final_df['relative_reer_gap'] = final_df['reer_gap'] - final_df['reer_gap_us']

# ✅ (1) 금리차 + 인플레차 기반 이론 환율
start_fx = final_df['target_fx'].iloc[-12]  # 최근 값 기준
final_df['fx_theory_uip'] = start_fx + (final_df['rate_diff']
                                        + final_df['inflation_diff']).cumsum()

# ✅ (2) 실질환율 갭 기반 이론 환율
final_df['fx_theory_reer'] = final_df['target_fx'].iloc[-12] * (1 + (-final_df['relative_reer_gap'] / 100).cumsum())


# 데이터 준비 (NaN 제거 및 선형 보간)
data = final_df[['target_fx', 'fx_theory_uip', 'fx_theory_reer', 'us_potential_gdp_gap',
                 'current_account', 'oil_import_price', 'china_gdp_gap']].dropna()
data = data.interpolate(method='linear')

# NumPyro 모델 정의
def model(data):
    # 계수의 사전분포 정의 (정규분포)
    alpha = sample('alpha', Normal(0.1, 0.05))
    beta = sample('beta', Normal(0.05, 0.02))
    gamma = sample('gamma', Normal(0.03, 0.01))
    delta = sample('delta', Normal(0.02, 0.01))
    epsilon = sample('epsilon', Normal(0.04, 0.02))

    # 지연 변수 준비
    fx_lag = jnp.array(data['target_fx'].values[:-1])
    fx_theory_u_lag = jnp.array(data['fx_theory_uip'].values[:-1])
    fx_theory_r_lag = jnp.array(data['fx_theory_reer'].values[:-1])
    us_gap_lag = jnp.array(data['us_potential_gdp_gap'].values[:-1])
    ca_lag = jnp.array(data['current_account'].values[:-1])
    oil_lag = jnp.array(data['oil_import_price'].values[:-1])
    china_gap_lag = jnp.array(data['china_gdp_gap'].values[:-1])

    # PAC 방정식 (UIP 기반)
    pac_pred_uip = fx_lag + alpha * (fx_theory_u_lag - fx_lag) + beta * us_gap_lag + gamma * ca_lag + delta * oil_lag + epsilon * china_gap_lag
    # PAC 방정식 (REER 기반)
    pac_pred_reer = fx_lag + alpha * (fx_theory_r_lag - fx_lag) + beta * us_gap_lag + gamma * ca_lag + delta * oil_lag + epsilon * china_gap_lag

    # 우도 정의 (Normal 분포)
    sigma = sample('sigma', HalfNormal(50))  # 오차 표준편차
    sample('likelihood_uip', Normal(pac_pred_uip, sigma), obs=jnp.array(data['target_fx'].values[1:]))
    sample('likelihood_reer', Normal(pac_pred_reer, sigma), obs=jnp.array(data['target_fx'].values[1:]))

# MCMC 샘플링 설정
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_warmup=1500, num_samples=3000, num_chains=4)
mcmc.run(jax.random.PRNGKey(0), data=data)

# 결과 요약 출력
mcmc.print_summary()

# 최적화된 계수 추출
optimal_params = mcmc.get_samples()

# 최적화된 계수를 사용해 예측값 계산
alpha_opt = optimal_params['alpha'].mean()
beta_opt = optimal_params['beta'].mean()
gamma_opt = optimal_params['gamma'].mean()
delta_opt = optimal_params['delta'].mean()
epsilon_opt = optimal_params['epsilon'].mean()

pac_data = data.iloc[1:].copy()
pac_data['pac_fx_uip_opt'] = (data['target_fx'].values[:-1] +
                              alpha_opt * (data['fx_theory_uip'].values[:-1] - data['target_fx'].values[:-1]) +
                              beta_opt * data['us_potential_gdp_gap'].values[:-1] +
                              gamma_opt * data['current_account'].values[:-1] +
                              delta_opt * data['oil_import_price'].values[:-1] +
                              epsilon_opt * data['china_gdp_gap'].values[:-1])

pac_data['pac_fx_reer_opt'] = (data['target_fx'].values[:-1] +
                               alpha_opt * (data['fx_theory_reer'].values[:-1] - data['target_fx'].values[:-1]) +
                               beta_opt * data['us_potential_gdp_gap'].values[:-1] +
                               gamma_opt * data['current_account'].values[:-1] +
                               delta_opt * data['oil_import_price'].values[:-1] +
                               epsilon_opt * data['china_gdp_gap'].values[:-1])

# RMSE 계산
from sklearn.metrics import mean_squared_error
rmse_uip_opt = np.sqrt(mean_squared_error(pac_data['target_fx'], pac_data['pac_fx_uip_opt']))
rmse_reer_opt = np.sqrt(mean_squared_error(pac_data['target_fx'], pac_data['pac_fx_reer_opt']))

print(f"Optimized PAC UIP RMSE: {rmse_uip_opt:.4f}")
print(f"Optimized PAC REER RMSE: {rmse_reer_opt:.4f}")

# Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(pac_data.index, pac_data['target_fx'], label='Actual FX', color='blue')
plt.plot(pac_data.index, pac_data['pac_fx_uip_opt'], label='Optimized PAC UIP', color='red', linestyle='--')
plt.plot(pac_data.index, pac_data['pac_fx_reer_opt'], label='Optimized PAC REER', color='green', linestyle='--')
plt.title('Optimized PAC Exchange Rate Predictions vs Actual')
plt.xlabel('Date')
plt.ylabel('USD/KRW Exchange Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()