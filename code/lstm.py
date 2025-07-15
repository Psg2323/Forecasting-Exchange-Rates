import pymongo
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from pymongo import MongoClient

# MongoDB 연결
MONGO_URI = "mongodb+srv://kanzki001:ahdrhelql@cluster0.b3gwf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)
db = client["stock_data"]  # 데이터베이스 이름
collection = db["NEW35"]  # 컬렉션 이름
try:
    client = MongoClient(MONGO_URI)
    # 서버 정보 가져오기 (연결 확인)
    print("MongoDB 서버 정보:", client.server_info())
    print("✅ MongoDB 연결 성공!")
except Exception as e:
    print("❌ MongoDB 연결 실패:", e)

# MongoDB 데이터 가져오기
def fetch_data_from_mongo():
    client = pymongo.MongoClient(MONGO_URI)
    db = client["stock_data"]  # 데이터베이스 이름
    collection = db["NEW35"]  # 컬렉션 이름

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
if __name__ == "__main__":
    # MongoDB 데이터 가져오기
    mongo_data = fetch_data_from_mongo()

    # 데이터프레임으로 변환
    df = transform_to_dataframe(mongo_data)

    # 결과 확인
    print(df.head())  # 데이터 확인

df = df[~df.index.weekday.isin([5, 6])]

df = df.sort_index(ascending=True)
start_date = "2004-01-01"
end_date = "2025-04-07" 

df = df[start_date:end_date]

# 1. 설정
target_col = 'USDKRW=X_Close'
sequence_length = 30  # 초기값, Optuna에서 튜닝 예정

# 2. 피처 선택
features = [col for col in df.columns if 'Close' in col]
features.remove(target_col)  # target은 별도 분리
input_cols = features + [target_col]  # 순서 중요: target 맨 뒤

# 3. 스케일링
scalers = {}
scaled_data = pd.DataFrame()

for col in input_cols:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[[col]])
    scaled_data[col] = scaled.flatten()
    scalers[col] = scaler  # 저장

# 4. 시계열 윈도우 생성
def create_sequences(data, target_col, seq_len=10):
    X, y = [], []
    for i in range(len(data) - seq_len - 1):
        window = data.iloc[i:i+seq_len]
        target = data.iloc[i+seq_len][target_col]
        X.append(window.values)
        y.append(target)
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data[input_cols], target_col, sequence_length)

# 5. Train / Val / Test Split
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1),
                                 initializer='normal')
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1),
                                 initializer='zeros')
        super(Attention, self).build(input_shape)
    
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)  # score
        a = K.softmax(e, axis=1)  # attention weights
        output = x * a  # weighted sequence
        return K.sum(output, axis=1)  # context vector
    
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_lstm_attention_model(input_shape, lstm_units=177, dropout_rate=0.07878081274907092, learning_rate=0.005357395669049675):
    inputs = Input(shape=input_shape)
    x = LSTM(lstm_units, return_sequences=True)(inputs)
    x = Attention()(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(1)(x)

    model = Model(inputs, output)
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate), metrics=['mae'])
    return model

import optuna
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from math import sqrt
'''
def objective(trial):
    # 튜닝 파라미터 정의
    lstm_units = trial.suggest_int('lstm_units', 32, 256)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    sequence_length = trial.suggest_int('sequence_length', 10, 60)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # 시퀀스 재생성
    X_, y_ = create_sequences(scaled_data[input_cols], target_col, sequence_length)
    train_size = int(len(X_) * 0.7)
    val_size = int(len(X_) * 0.15)

    X_train, y_train = X_[:train_size], y_[:train_size]
    X_val, y_val = X_[train_size:train_size+val_size], y_[train_size:train_size+val_size]

    # 모델 빌드
    model = build_lstm_attention_model(
        input_shape=(sequence_length, len(input_cols)),
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    # 학습
    es = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
    model.fit(X_train, y_train, epochs=50, batch_size=batch_size,
              validation_data=(X_val, y_val), callbacks=[es], verbose=0)

    # 예측 및 평가
    preds = model.predict(X_val)
    rmse = sqrt(mean_squared_error(y_val, preds))
    return rmse
'''

'''
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
'''

'''
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
if completed_trials:
    best = study.best_trial
    print("Best trial:")
    print(f"  Value: {best.value}")
    print(f"  Params: {best.params}")
else:
    print("❌ 완료된 trial이 없습니다. objective 함수를 확인하세요.")
'''

best_params = {
    'lstm_units': 177,
    'dropout_rate': 0.07878081274907092,
    'learning_rate': 0.005357395669049675,
    'sequence_length': 10,
    'batch_size': 64
}

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

# 1. 데이터 재생성 (최적 sequence_length)
X_final, y_final = create_sequences(scaled_data[input_cols], target_col, best_params['sequence_length'])

# 2. Train / Test Split
train_size = int(len(X_final) * 0.85)
X_train_final, y_train_final = X_final[:train_size], y_final[:train_size]
X_test_final, y_test_final = X_final[train_size:], y_final[train_size:]

# 3. 모델 빌드 및 학습
model_final = build_lstm_attention_model(
    input_shape=(best_params['sequence_length'], len(input_cols)),
    lstm_units=best_params['lstm_units'],
    dropout_rate=best_params['dropout_rate'],

    learning_rate=best_params['learning_rate']
)

early_stop = EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss')

model_final.fit(
    X_train_final, y_train_final,
    validation_split=0.1,
    epochs=100,
    batch_size=best_params['batch_size'],
    callbacks=[early_stop],
    verbose=1
)

# 4. 예측 및 RMSE 평가
y_pred = model_final.predict(X_test_final)

y_test_final = y_test_final.reshape(-1,1)
y_pred = y_pred.reshape(-1,1)

y_test_final_inverse=scaler.inverse_transform(y_test_final)
y_pred_inverse = scaler.inverse_transform(y_pred)

rmse_final = sqrt(mean_squared_error(y_test_final_inverse, y_pred_inverse))
mae_final = mean_absolute_error(y_test_final_inverse, y_pred_inverse)
r2_final = r2_score(y_test_final_inverse, y_pred_inverse)

print(f"LSTM 모델 평가")
print(f"RMSE: {rmse_final:.4f}, MAE: {mae_final:.4f}, R2: {r2_final:.4f}") 


import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt

# 1. 역정규화
y_test_inv = scalers[target_col].inverse_transform(y_test_final.reshape(-1, 1)).flatten()
y_pred_inv = scalers[target_col].inverse_transform(y_pred.reshape(-1, 1)).flatten()

# 2. 날짜 정보 생성
start_index = len(scaled_data) - len(y_test_inv)
date_range = df.index[start_index:]  # df의 인덱스가 날짜인 경우

# 3. Plotly 시계열 선 그래프
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=date_range, y=y_test_inv, mode='lines', name='Actual', line=dict(width=2)))
fig1.add_trace(go.Scatter(x=date_range, y=y_pred_inv, mode='lines', name='Predicted', line=dict(dash='dot')))
fig1.update_layout(title='USD/KRW Exchange Rate Prediction',
                   xaxis_title='Date', yaxis_title='Exchange Rate', template='plotly_white')
fig1.show()

# 4. 오차 분포 히스토그램 (Matplotlib)
errors = y_test_inv - y_pred_inv
plt.figure(figsize=(10, 4))
plt.hist(errors, bins=30, edgecolor='black')
plt.title('Prediction Error Distribution')
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()


import numpy as np
import plotly.graph_objects as go
import pandas as pd

# === 1. 미래 7일 예측 ===
n_future = 7
last_sequence = X_test_final[-1].copy()  # (seq_len, n_features)
future_preds_scaled = []


for _ in range(n_future):
    pred = model_final.predict(last_sequence[np.newaxis, :, :])[0][0]
    future_preds_scaled.append(pred)

    # 업데이트 시퀀스: 맨 앞 버리고 새 pred 추가
    new_step = last_sequence[-1].copy()
    new_step[-1] = pred  # target 위치에 새 예측값
    last_sequence = np.vstack([last_sequence[1:], new_step])

# === 2. 역정규화 ===
y_test_inv = scalers[target_col].inverse_transform(y_test_final.reshape(-1, 1)).flatten()
y_pred_inv = scalers[target_col].inverse_transform(y_pred.reshape(-1, 1)).flatten()
future_preds_inv = scalers[target_col].inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()

# === 3. 날짜 범위 구성 ===
start_index = len(scaled_data) - len(y_test_inv)
date_range = df.index[start_index:]

future_dates = pd.date_range(start=date_range[-1], periods=n_future+1, freq='B')[1:]  # 'B' = business days

# === 4. Plotly 시각화 ===
fig = go.Figure()

# 실제값
fig.add_trace(go.Scatter(x=date_range, y=y_test_inv, mode='lines', name='Actual', line=dict(width=2)))
# 예측값
fig.add_trace(go.Scatter(x=date_range, y=y_pred_inv, mode='lines', name='Predicted', line=dict(dash='dot')))
# 미래 예측
fig.add_trace(go.Scatter(x=future_dates, y=future_preds_inv, mode='lines+markers', name='Future Forecast', line=dict(dash='dash')))

# 인터랙션 & 확대 설정
fig.update_layout(
    title='USD/KRW Prediction (Including 7-day Forecast)',
    xaxis_title='Date', yaxis_title='Exchange Rate',
    template='plotly_white',
    xaxis=dict(
        rangeselector=dict(
            buttons=[
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(step="all")
            ]
        ),
        rangeslider=dict(visible=True),
        type="date"
    ),
    hovermode="x unified"
)

fig.show()


