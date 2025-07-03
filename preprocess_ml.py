import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# --- Load Data ---
trans = pd.read_csv("train/transactions.csv", parse_dates=["doj", "doi"])
train = pd.read_csv("train/train.csv", parse_dates=["doj"])

# --- Filter dbd in [15, 30] ---
trans = trans[(trans["dbd"] >= 15) & (trans["dbd"] <= 30)].copy()

# --- Merge with Train Data ---
data = trans.merge(train, on=["doj", "srcid", "destid"], how="inner")

# --- Feature Engineering ---
data["day_of_week"] = data["doj"].dt.dayofweek
data["month"] = data["doj"].dt.month
data["search_to_book"] = data["cumsum_searchcount"] / (data["cumsum_seatcount"] + 1)

cat_cols = ["srcid_region", "destid_region", "srcid_tier", "destid_tier"]
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# --- Static and Sequence Features ---
sequence_features = ['cumsum_seatcount', 'cumsum_searchcount', 'search_to_book']
static_features = ["srcid_region", "destid_region", "srcid_tier", "destid_tier", "day_of_week", "month"]
expected_dbds = list(range(15, 31))  # 16 dbd days

flattened_rows = []
missing_groups = 0

# --- Group and Pad ---
for (doj, srcid, destid), group in tqdm(data.groupby(["doj", "srcid", "destid"])):
    group = group.set_index("dbd").reindex(expected_dbds)

    if group["cumsum_seatcount"].isnull().sum() > 4:
        missing_groups += 1
        continue

    group[sequence_features] = group[sequence_features].fillna(0)
    
    try:
        row = {
            "doj": doj,
            "srcid": srcid,
            "destid": destid,
            "final_seatcount": group["final_seatcount"].iloc[0],
        }

        for feature in sequence_features:
            for dbd in expected_dbds:
                row[f"{feature}_dbd{dbd}"] = group.loc[dbd, feature]

        for feat in static_features:
            row[feat] = group[feat].iloc[0]
        flattened_rows.append(row)
    except Exception:
        continue  # Skip if something's wrong

print(f"✅ Flattened sequences: {len(flattened_rows)}")
print(f"❌ Skipped due to missing values: {missing_groups}")

# --- Save to CSV ---
output_df = pd.DataFrame(flattened_rows)
output_df.to_csv("data/padded_training_data.csv", index=False)
print("📁 Saved padded_training_data.csv")
