import pandas as pd

file_path = "dataset/raw/clothSize.csv"

df = pd.read_csv(file_path)

print("========== SIZE COUNTS ==========")
print(df["size"].value_counts())

print("\n========== AVERAGE MEASUREMENTS BY SIZE ==========")

print(
    df.groupby("size")[["weight", "height", "age"]]
      .mean()
      .round(2)
)

print("\n========== MEDIAN MEASUREMENTS BY SIZE ==========")

print(
    df.groupby("size")[["weight", "height", "age"]]
      .median()
      .round(2)
)

print("\n========== MINIMUM BY SIZE ==========")

print(
    df.groupby("size")[["weight", "height"]]
      .min()
)

print("\n========== MAXIMUM BY SIZE ==========")

print(
    df.groupby("size")[["weight", "height"]]
      .max()
)

print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())