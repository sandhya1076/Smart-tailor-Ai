import pandas as pd
from pathlib import Path

input_file = "dataset/raw/clothSize.csv"
output_file = "dataset/cleaned/cleaned_cloth_size.csv"

Path("dataset/cleaned").mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(input_file)

print("Original shape:", df.shape)

# Keep required columns
features = [
    "weight",
    "height",
    "age"
]

target = "size"

df_clean = df[features + [target]].copy()

# Remove missing values
df_clean = df_clean.dropna()

# Remove duplicate rows
df_clean = df_clean.drop_duplicates()

# Basic validity checks
df_clean = df_clean[
    (df_clean["weight"] > 0) &
    (df_clean["height"] > 0) &
    (df_clean["age"] > 0)
]

print("\nCleaned shape:", df_clean.shape)

print("\nMissing values:")
print(df_clean.isnull().sum())

print("\nSize distribution:")
print(df_clean["size"].value_counts())

print("\nCleaned data preview:")
print(df_clean.head())

# Save
df_clean.to_csv(output_file, index=False)

print("\nCleaned dataset saved to:")
print(output_file)