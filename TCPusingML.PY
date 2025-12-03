# ==========================================
# Traffic Congestion Prediction
# Random Forest (Snow Feature Removed)
# ==========================================

# 0. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import joblib

%matplotlib inline

# --------------------------
# 1. Load Data
# --------------------------

csv_path = "/content/Metro_Interstate_Traffic_Volume.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print("Loaded:", csv_path)
else:
    raise FileNotFoundError("Upload 'Metro_Interstate_Traffic_Volume.csv' to /content.")

print("Rows, Columns:", df.shape)
print(df.head())

# --------------------------
# 2. Basic Preprocessing
# --------------------------

df["date_time"] = pd.to_datetime(df["date_time"])
df = df.sort_values("date_time").reset_index(drop=True)

df["hour"] = df["date_time"].dt.hour
df["weekday"] = df["date_time"].dt.weekday
df["month"] = df["date_time"].dt.month
df["year"] = df["date_time"].dt.year
df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

# --------------------------
# 3. Holiday Feature (festival-aware)
# --------------------------

if "holiday" in df.columns:
    df["is_holiday"] = (df["holiday"].astype(str) != "None").astype(int)
else:
    df["is_holiday"] = 0  # fallback

# --------------------------
# 4. Encode weather_main
# --------------------------

if "weather_main" in df.columns:
    le = LabelEncoder()
    df["weather_main_enc"] = le.fit_transform(df["weather_main"].astype(str))
else:
    df["weather_main_enc"] = 0

# --------------------------
# 5. Numeric Missing Handling
# --------------------------

num_cols = df.select_dtypes(include=np.number).columns.tolist()
df[num_cols] = df[num_cols].fillna(method="ffill").fillna(method="bfill")

# --------------------------
# 6. Lag + Rolling Features
# --------------------------

df["traffic_lag1"] = df["traffic_volume"].shift(1)
df["traffic_roll3"] = (
    df["traffic_volume"].shift(1).rolling(3, min_periods=1).mean()
)

df["traffic_lag1"] = df["traffic_lag1"].fillna(method="bfill")
df["traffic_roll3"] = df["traffic_roll3"].fillna(method="bfill")

print("\nSample rows after feature engineering:")
print(
    df[
        [
            "date_time",
            "traffic_volume",
            "hour",
            "weekday",
            "is_weekend",
            "is_holiday",
            "weather_main_enc",
            "traffic_lag1",
            "traffic_roll3",
        ]
    ].head(10)
)

# --------------------------
# 7. Feature List (Snow Removed)
# --------------------------

target_col = "traffic_volume"

candidate_features = [
    "hour",
    "weekday",
    "is_weekend",
    "is_holiday",
    "month",
    "temp",
    "rain_1h",
    "clouds_all",
    "weather_main_enc",
    "traffic_lag1",
    "traffic_roll3",
]

features = [f for f in candidate_features if f in df.columns]

print("\nUsing features:")
print(features)

X = df[features]
y = df[target_col]

# --------------------------
# 8. Time-based Train/Test Split
# --------------------------

split_idx = int(0.8 * len(df))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print("\nTrain size:", X_train.shape[0], "Test size:", X_test.shape[0])

# --------------------------
# 9. Train Random Forest
# --------------------------

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

rf.fit(X_train, y_train)
print("\nRandom Forest trained successfully.")

# --------------------------
# 10. Evaluation (Regression)
# --------------------------

y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nRegression Metrics:")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"R²   = {r2:.3f}")

# --------------------------
# 11. Feature Importance Plot
# --------------------------

importances = rf.feature_importances_
imp_series = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(8, 4))
imp_series.plot(kind="bar", color="green")
plt.title("Feature Importances (Snow Removed)")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# --------------------------
# 12. Actual vs Predicted Plot
# --------------------------

plt.figure(figsize=(12, 4))
plt.plot(
    df["date_time"].iloc[split_idx : split_idx + 200],
    y_test.iloc[:200],
    label="Actual",
)
plt.plot(
    df["date_time"].iloc[split_idx : split_idx + 200],
    y_pred[:200],
    label="Predicted",
)
plt.title("Actual vs Predicted (First 200 Test Samples)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------
# 13. Convert to Congestion Categories
# --------------------------

low_thr = df[target_col].quantile(0.33)
high_thr = df[target_col].quantile(0.66)

def congestion_label(x):
    if x < low_thr:
        return 0
    if x < high_thr:
        return 1
    return 2

y_test_bin = y_test.apply(congestion_label)
y_pred_bin = pd.Series(y_pred, index=y_test.index).apply(congestion_label)

print("\nClassification Report:")
print(classification_report(y_test_bin, y_pred_bin, target_names=["Low", "Medium", "High"]))

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test_bin, y_pred_bin)
print("Accuracy =", acc)


cm = confusion_matrix(y_test_bin, y_pred_bin)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low", "Medium", "High"],
    yticklabels=["Low", "Medium", "High"],
)
plt.title("Confusion Matrix (Congestion Class)")
plt.tight_layout()
plt.show()

# --------------------------
# 14. Save Model
# --------------------------

joblib.dump(rf, "traffic_rf_model_no_snow.joblib")
print("\nModel saved as 'traffic_rf_model_no_snow.joblib'.")
