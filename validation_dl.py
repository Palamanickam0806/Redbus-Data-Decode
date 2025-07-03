import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder

# --- Load Required Data ---
test = pd.read_csv('data/test.csv', parse_dates=['doj'])
trans = pd.read_csv('train/transactions.csv', parse_dates=["doj", "doi"])
train = pd.read_csv('train/train.csv', parse_dates=["doj"])
trans = trans[(trans['dbd'] >= 15) & (trans['dbd'] <= 30)]
trans_15_to_30 = trans.copy()

# --- Add Date Features ---
def add_date_features(df):
    df['day_of_week'] = df['doj'].dt.dayofweek
    df['month'] = df['doj'].dt.month
    return df

test = add_date_features(test)
test_data = trans_15_to_30.merge(test[['doj', 'srcid', 'destid', 'day_of_week', 'month']], on=['doj', 'srcid', 'destid'], how='inner')

# --- Encode Categorical Features ---
cat_cols = ['srcid_region', 'destid_region', 'srcid_tier', 'destid_tier']
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    test_data[col] = le.fit_transform(test_data[col].astype(str))
    label_encoders[col] = le

# --- Compute additional features ---
test_data["search_to_book"] = test_data["cumsum_searchcount"] / (test_data["cumsum_seatcount"] + 1)

# --- Group into sequences for model ---
static_cols = ['srcid_region', 'destid_region', 'srcid_tier', 'destid_tier', 'day_of_week', 'month']
feature_cols = ['cumsum_seatcount', 'cumsum_searchcount', 'search_to_book', 'dbd', 'day_of_week', 'month']

sequences = []
statics = []
meta = []

for (doj, srcid, destid), group in test_data.groupby(['doj', 'srcid', 'destid']):
    if group.shape[0] != 16:
        continue
    group = group.sort_values("dbd", ascending=False)
    seq = group[feature_cols].values
    static = group[static_cols].iloc[0].values
    sequences.append(seq)
    statics.append(static)
    meta.append((doj, srcid, destid))

# --- Model Definition (Must match training) ---
class BookingGRU(nn.Module):
    def __init__(self, input_dim, static_dim, hidden_size=128, num_layers=4):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + static_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_seq, x_static):
        _, h = self.gru(x_seq)
        h = h[-1]
        x = torch.cat([h, x_static], dim=1)
        return self.fc(x).squeeze(1)

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
static_dim = len(static_cols)
model = BookingGRU(6, static_dim).to(device)
model.load_state_dict(torch.load("best_booking_model.pt", map_location=device))
model.eval()

# --- Predict ---
X_seq = torch.tensor(np.array(sequences), dtype=torch.float32).to(device)
X_static = torch.tensor(np.array(statics), dtype=torch.float32).to(device)

with torch.no_grad():
    preds = model(X_seq, X_static).cpu().numpy()

# ✅ Clip predictions to prevent negatives
pred_df = pd.DataFrame(meta, columns=['doj', 'srcid', 'destid'])
pred_df['final_seatcount'] = np.round(np.clip(preds, 0, None)).astype(int)

# --- Merge into submission format ---
submission = test.copy()
submission = submission.merge(pred_df, on=['doj', 'srcid', 'destid'], how='left')

fallback = int(train['final_seatcount'].median())
submission['final_seatcount'] = submission['final_seatcount'].fillna(fallback).astype(int)

submission[['route_key', 'final_seatcount']].to_csv("submission_ml.csv", index=False)
print("✅ Submission generated: submission_ml.csv")
