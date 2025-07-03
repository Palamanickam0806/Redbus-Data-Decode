import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

trans = pd.read_csv("train/transactions.csv", parse_dates=["doj", "doi"])
train = pd.read_csv("train/train.csv", parse_dates=["doj"])

trans = trans[(trans['dbd'] >= 15) & (trans['dbd'] <= 30)].copy()
data = trans.merge(train, on=["doj", "srcid", "destid"], how="inner")

data["day_of_week"] = data["doi"].dt.dayofweek  # use DOI here
data["month"] = data["doi"].dt.month
data["search_to_book"] = data["cumsum_searchcount"] / (data["cumsum_seatcount"] + 1)

cat_cols = ['srcid_region', 'destid_region', 'srcid_tier', 'destid_tier']
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

sequence_features = [
    'dbd', 'cumsum_seatcount', 'cumsum_searchcount', 'search_to_book', 'day_of_week', 'month'
]
static_features = [
    'srcid', 'destid', 'srcid_region', 'destid_region', 'srcid_tier', 'destid_tier'
]

# --- Grouping ---
grouped = []
targets = []

for (doj, srcid, destid), group in data.groupby(['doj', 'srcid', 'destid']):
    if group.shape[0] != 16:
        continue
    group = group.sort_values("dbd", ascending=False)
    seq = group[sequence_features].values.astype(np.float32)
    static = group[static_features].iloc[0].values.astype(np.float32)
    target = group["final_seatcount"].iloc[0]
    grouped.append((seq, static))
    targets.append(target)

print(f"✅ Prepared {len(grouped)} training sequences.")

# --- Dataset ---
class BookingDataset(Dataset):
    def __init__(self, sequences, statics, targets):
        self.sequences = sequences
        self.statics = statics
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.statics[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

# --- Train/Val Split ---
train_idx, val_idx = train_test_split(np.arange(len(grouped)), test_size=0.2, random_state=42)
train_seq = [grouped[i][0] for i in train_idx]
train_static = [grouped[i][1] for i in train_idx]
train_targets = [targets[i] for i in train_idx]

val_seq = [grouped[i][0] for i in val_idx]
val_static = [grouped[i][1] for i in val_idx]
val_targets = [targets[i] for i in val_idx]

train_ds = BookingDataset(train_seq, train_static, train_targets)
val_ds = BookingDataset(val_seq, val_static, val_targets)

train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=128)

# --- Model ---
class BookingGRU(nn.Module):
    def __init__(self, input_dim, static_dim, hidden_size=128, num_layers=5):
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
        h = h[-1]  # use last layer output
        x = torch.cat([h, x_static], dim=1)
        return self.fc(x).squeeze(1)

# --- Train ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = len(sequence_features)
static_dim = len(static_features)

model = BookingGRU(input_dim, static_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

best_rmse = float("inf")
for epoch in range(80):  # ⏳ Adjust epochs here
    model.train()
    train_losses = []
    for xb, xs, yb in train_dl:
        xb, xs, yb = xb.to(device), xs.to(device), yb.to(device)
        pred = model(xb, xs)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for xb, xs, yb in val_dl:
            xb, xs, yb = xb.to(device), xs.to(device), yb.to(device)
            pred = model(xb, xs)
            val_preds.append(pred.cpu().numpy())
            val_targets.append(yb.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
    print(f"Epoch {epoch+1:03d} | Train Loss: {np.mean(train_losses):.3f} | Val RMSE: {rmse:.2f}")

    if rmse < best_rmse:
        torch.save(model.state_dict(), "best_booking_model_2.pt")
        best_rmse = rmse
        print("✅ Saved new best model")
