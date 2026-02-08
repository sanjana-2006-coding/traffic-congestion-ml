import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load data
data = pd.read_csv("data/processed_data.csv")

# Feature engineering
data['IsPeakHour'] = data['Hour'].apply(
    lambda x: 1 if (8 <= x <= 10 or 17 <= x <= 19) else 0
)
data['Hour_Squared'] = data['Hour'] ** 2

features = [
    'Junction',
    'Hour',
    'Hour_Squared',
    'TimeBlock',
    'IsPeakHour',
    'IsWeekend',
    'Season'
]

X = data[features]
y = data['Vehicles']

# Train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- BASELINE MODEL ----------------
baseline_pred = np.full(len(y_val), y_train.mean())
baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))

# ---------------- XGBOOST MODEL ----------------
model = XGBRegressor(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    objective='reg:squarederror',
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f"Baseline RMSE: {baseline_rmse:.2f}")
print(f"XGBoost RMSE: {rmse:.2f}")

# ---------------- FEATURE IMPORTANCE ----------------
importance = model.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

feature_importance.to_csv("model/feature_importance.csv", index=False)
print("Feature importance saved")

# Save model
joblib.dump(model, "model/traffic_model.pkl")
print("Model trained and saved")
