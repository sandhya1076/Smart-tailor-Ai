import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

input_file = "dataset/cleaned/cleaned_cloth_size.csv"
output_file = "dataset/processed/scaled_cloth_size.csv"
scaler_file = "dataset/processed/cloth_size_scaler.pkl"

Path("dataset/processed").mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(input_file)

features = [
    "weight",
    "height",
    "age"
]

target = "size"

X = df[features]
y = df[target]

# Create scaler
scaler = StandardScaler()

# Scale features
X_scaled = scaler.fit_transform(X)

# Convert to DataFrame
X_scaled = pd.DataFrame(
    X_scaled,
    columns=features
)

# Add target
X_scaled[target] = y.values

# Save scaled dataset
X_scaled.to_csv(
    output_file,
    index=False
)

# Save scaler for later prediction
joblib.dump(
    scaler,
    scaler_file
)

print("========== SCALING COMPLETE ==========")

print("\nOriginal features:")
print(X.head())

print("\nScaled features:")
print(X_scaled.head())

print("\nScaled dataset shape:")
print(X_scaled.shape)

print("\nScaled dataset saved to:")
print(output_file)

print("\nScaler saved to:")
print(scaler_file)