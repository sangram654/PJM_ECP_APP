import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# =========================
# LOAD DATASET
# =========================
df = pd.read_excel("PJMW_MW_Hourly.xlsx")

df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.set_index("Datetime")

# =========================
# REDUCE MEMORY USAGE
# =========================
df["PJMW_MW"] = df["PJMW_MW"].astype("float32")

# =========================
# FEATURE ENGINEERING
# =========================
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month

# Lag features
df["lag24"] = df["PJMW_MW"].shift(24)
df["lag168"] = df["PJMW_MW"].shift(168)

df = df.dropna()

# =========================
# TRAIN TEST SPLIT
# =========================
split_date = df.index.max() - pd.DateOffset(years=1)

train = df[df.index < split_date]
test = df[df.index >= split_date]

features = ["hour","dayofweek","month","lag24","lag168"]

X_train = train[features]
y_train = train["PJMW_MW"]

X_test = test[features]
y_test = test["PJMW_MW"]

# =========================
# LIGHTWEIGHT MODEL
# =========================
model = RandomForestRegressor(
    n_estimators=80,      # reduced trees
    max_depth=10,         # limit depth
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# MODEL EVALUATION
# =========================
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
print("MAE:", mae)

# =========================
# SAVE MODEL (COMPRESSED)
# =========================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")