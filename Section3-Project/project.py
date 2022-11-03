# DB 데이터로 데이터프레임을 반환
def to_dataframe(table_name):
  query = cur.execute(f"SELECT * FROM {table_name}")    
  cols = [column[0] for column in query.description]
  df = pd.DataFrame(data=query.fetchall(), columns=cols)
  return df  

# Feature Engineering 메서드
def feat_engineering(df):
  wl_lst = []
  for idx in range(1, df.shape[0]):
    if df['open'].iloc[idx] > df['close'].iloc[idx]:
      wl_lst.append(0)
    else:
      wl_lst.append(1)
  df.drop(df.tail(1).index, inplace=True)
  df['win_lose'] = wl_lst
  return df



from types import NoneType
import pyupbit
import sqlite3
import pandas as pd



# DB 연결
conn = sqlite3.connect('coin.db')
cur = conn.cursor()


# 원화시장의 암호화폐 목록
krw_tickers = pyupbit.get_tickers("KRW")


# 테이블 생성 & 데이터 삽입
for ticker in krw_tickers:

  txt = ticker.split('-')[1]
  if txt == '1INCH':
    txt = 'ONEINCH'

#   # 일봉차트 데이터
  data = pyupbit.get_ohlcv(ticker=ticker, interval="day", count=2000)

  cur.execute(f"DROP TABLE IF EXISTS {txt};")

  if type(data) is NoneType:
    continue
  else:
    cur.execute(f"""
      CREATE TABLE IF NOT EXISTS {txt} (
        date DATETIME PRIMARY KEY,
        open INTEGER,
        high INTEGER,
        low INTEGER,
        close INTEGER,
        volume INTEGER
      );
    """)
    for idx in range(data.shape[0]):
      val = (str(data.index[idx]), data['open'][idx], data['high'][idx], data['low'][idx], data['close'][idx], data['volume'][idx])
      cur.execute(f"INSERT INTO {txt} (date,open,high,low,close,volume) VALUES (?,?,?,?,?,?);", val)


df = to_dataframe('BTC')

conn.commit()
conn.close()



# ML 모델 제작
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df = feat_engineering(df)

feature = ['open', 'high', 'low', 'close', 'volume']
target = 'win_lose'

X, y = df[feature], df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 기준모델
baseline = y_train.mode()[0]
y_pred_base = [baseline] * len(y_test)
print("기준모델의 정확도: ", accuracy_score(y_test, y_pred_base).round(2))

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 인덱스 정렬
y_train.reset_index(drop=True, inplace=True)


# 교차검증
model = LogisticRegression()
kf = KFold(n_splits=5)

acc_lst, prec_lst, recall_lst, f1_lst = [], [], [], []

for train_idx, val_idx in kf.split(X_train):
  X_train_cv, X_val_cv = X_train_scaled[train_idx], X_train_scaled[val_idx]
  y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
  model.fit(X_train_cv, y_train_cv)
  y_pred_cv = model.predict(X_val_cv)
  acc_lst.append(accuracy_score(y_val_cv, y_pred_cv)) # 정확도
  prec_lst.append(precision_score(y_val_cv, y_pred_cv)) # 정밀도
  recall_lst.append(recall_score(y_val_cv, y_pred_cv))  # 재현율
  f1_lst.append(f1_score(y_val_cv, y_pred_cv))  # f1-Score
  
y_pred_test = model.predict(X_test_scaled)

result = pd.DataFrame({
  'Accuracy' : np.round([np.mean(acc_lst), accuracy_score(y_test, y_pred_test)], 2),
  'Precision' : np.round([np.mean(prec_lst), precision_score(y_test, y_pred_test)], 2),
  'Recall' : np.round([np.mean(recall_lst), recall_score(y_test, y_pred_test)], 2),
  'F1_Score' : np.round([np.mean(f1_lst), f1_score(y_test, y_pred_test)], 2)
}, index=['Train', 'Test'])

print(result)
