import pandas as pd

file_path = "dataset/cleaned/cleaned_clothing_dataset.csv"

df = pd.read_csv(file_path)

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

print("========== AVERAGE MEASUREMENTS BY SIZE ==========")

print(
    df.groupby("RecommendedSize")[features]
      .mean()
      .round(2)
)

print("\n========== MEDIAN MEASUREMENTS BY SIZE ==========")

print(
    df.groupby("RecommendedSize")[features]
      .median()
      .round(2)
)

print("\n========== SIZE COUNTS ==========")

print(df["RecommendedSize"].value_counts().sort_index())

print("\n========== CORRELATION WITH NUMERICAL SIZE ENCODING ==========")

size_mapping = {
    "XS": 1,
    "S": 2,
    "M": 3,
    "L": 4,
    "XL": 5,
    "XXL": 6
}

df["SizeNumber"] = df["RecommendedSize"].map(size_mapping)

print(
    df[features + ["SizeNumber"]]
      .corr()["SizeNumber"]
      .sort_values(ascending=False)
)