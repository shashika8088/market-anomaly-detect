import os
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load dataset
# -----------------------------
DATA_PATH = "data/FinancialMarketData.csv"

data = pd.read_csv(DATA_PATH)
print("Dataset loaded:", data.shape)

# -----------------------------
# Preprocessing
# -----------------------------
data = data.ffill()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# -----------------------------
# Train Isolation Forest
# -----------------------------
model = IsolationForest(
    n_estimators=100,
    contamination=0.2,   # expected % of anomalies
    random_state=42
)

model.fit(scaled_data)

# -----------------------------
# Predict anomalies
# -----------------------------
data["anomaly"] = model.predict(scaled_data)
data["anomaly"] = data["anomaly"].map({1: "Normal", -1: "Anomaly"})

print("\nAnomaly Detection Results:")
print(data)

# -----------------------------
# Save model
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/isolation_forest.pkl")

print("\nModel saved to models/isolation_forest.pkl")
