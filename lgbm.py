import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load datasets
trans = pd.read_csv('train/transactions.csv', parse_dates=['doj', 'doi'])
train = pd.read_csv('train/train.csv', parse_dates=['doj'])
submission = pd.read_csv('data/test.csv')

trans_15 = trans[trans['dbd'] == 15].copy()
train_data = trans_15.merge(train, on=['doj', 'srcid', 'destid'], how='inner')

def add_date_features(df):
    df['day_of_week'] = df['doj'].dt.dayofweek
    df['month'] = df['doj'].dt.month
    return df

train_data = add_date_features(train_data)
for col in ['srcid_region', 'destid_region', 'srcid_tier', 'destid_tier']:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col].astype(str))

# --- Select Features ---
features = [
    'srcid', 'destid', 'srcid_region', 'destid_region',
    'srcid_tier', 'destid_tier', 'cumsum_seatcount', 'cumsum_searchcount',
    'day_of_week', 'month'
]
target = 'final_seatcount'

# --- Train/Validation Split ---
X_train, X_val, y_train, y_val = train_test_split(train_data[features], train_data[target], test_size=0.2, random_state=42)
# y_train_log = np.log1p(y_train)
# y_val_log = np.log1p(y_val)

# LightGBM dataset
train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": 8,
    "min_child_samples": 30,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "random_state": 42
}


model = lgb.train(
    params,
    train_set,
    num_boost_round=1000,
    valid_sets=[train_set, val_set]
)

# Predict and invert log
y_pred = model.predict(X_val)
# y_pred = np.expm1(y_pred_log)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"✅ LightGBM RMSE: {rmse:.2f}")

# Test data preparation
submission['doj'] = pd.to_datetime(submission['doj'])
test_15 = trans_15.merge(submission[['doj', 'srcid', 'destid']], on=['doj', 'srcid', 'destid'], how='inner')
test_15 = add_date_features(test_15)


for col in ['srcid_region', 'destid_region', 'srcid_tier', 'destid_tier']:
    le = LabelEncoder()
    test_15[col] = le.fit_transform(test_15[col].astype(str))

test_preds = model.predict(test_15[features])
test_15['final_seatcount'] = test_preds.round().astype(int)
submission = submission.merge(test_15[['doj', 'srcid', 'destid', 'final_seatcount']], on=['doj', 'srcid', 'destid'], how='left')

fallback = int(train['final_seatcount'].median())
submission['final_seatcount'] = submission['final_seatcount'].fillna(fallback).astype(int)

submission[['route_key', 'final_seatcount']].to_csv('submission_ml.csv', index=False)
print("✅ Submission with test data generated: submission_ml.csv")