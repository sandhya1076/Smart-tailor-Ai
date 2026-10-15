import pandas as pd

file_path = "dataset/raw/clothSize.csv"

df = pd.read_csv(file_path)

print("========== DATASET SHAPE ==========")
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
    if df[column].dtype == "object":
        print(f"\n{column}:")
        print(df[column].unique())

print("\n========== NUMERICAL SUMMARY ==========")
print(df.describe())