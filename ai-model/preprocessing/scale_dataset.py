import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

input_file = "dataset/cleaned/cleaned_clothing_dataset.csv"
output_file = "dataset/processed/scaled_clothing_dataset.csv"
scaler_file = "dataset/processed/scaler.pkl"

Path("dataset/processed").mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(input_file)

features = [
    "Height_cm",
    "Weight_kg",
    "Chest_cm",
    "Waist_cm",
    "Hip_cm",
    "ShoulderWidth_cm",
    "ArmLength_cm",
    "LegLength_cm"
]

target = "RecommendedSize"

# Separate features and target
X = df[features]
y = df[target]

# Create scaler
scaler = StandardScaler()

# Scale features
X_scaled = scaler.fit_transform(X)

# Convert scaled data back to DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=features)

# Add target column
X_scaled[target] = y.values

# Save scaled dataset
X_scaled.to_csv(output_file, index=False)

# Save scaler for later use
joblib.dump(scaler, scaler_file)

print("Original feature values:")
print(X.head())

print("\nScaled feature values:")
print(X_scaled.head())

print("\nScaled dataset saved to:")
print(output_file)

print("\nScaler saved to:")
print(scaler_file)