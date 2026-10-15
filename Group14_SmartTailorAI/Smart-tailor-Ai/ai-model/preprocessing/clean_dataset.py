import pandas as pd
from pathlib import Path

# ----------------------------------
# File paths
# ----------------------------------

input_file = "dataset/raw/personalized_clothing_dataset.csv"
output_file = "dataset/cleaned/cleaned_clothing_dataset.csv"

# Create output directory if needed
Path("dataset/cleaned").mkdir(parents=True, exist_ok=True)

# ----------------------------------
# Load dataset
# ----------------------------------

df = pd.read_csv(input_file)

print("Original shape:", df.shape)

# ----------------------------------
# Select required columns
# ----------------------------------

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

df_clean = df[features + [target]].copy()

# ----------------------------------
# Remove missing values
# ----------------------------------

df_clean = df_clean.dropna()

# ----------------------------------
# Remove duplicate rows
# ----------------------------------

df_clean = df_clean.drop_duplicates()

# ----------------------------------
# Display cleaned information
# ----------------------------------

print("\nCleaned shape:", df_clean.shape)

print("\nMissing values:")
print(df_clean.isnull().sum())

print("\nSize categories:")
print(df_clean[target].value_counts())

# ----------------------------------
# Save cleaned dataset
# ----------------------------------

df_clean.to_csv(output_file, index=False)

print("\nCleaned dataset saved to:")
print(output_file)