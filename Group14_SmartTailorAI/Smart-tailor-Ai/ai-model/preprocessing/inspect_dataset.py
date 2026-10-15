import pandas as pd

# Dataset path
file_path = "dataset/raw/personalized_clothing_dataset.csv"

# Load dataset
df = pd.read_csv(file_path)

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== UNIQUE VALUES ==========")
for column in df.columns:
    print(f"\n{column}:")
    print(df[column].unique()[:20])