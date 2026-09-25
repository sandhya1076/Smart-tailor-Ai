import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

# Load scaled dataset
input_file = "dataset/processed/scaled_clothing_dataset.csv"

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

X = df[features]
y = df["RecommendedSize"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("========== RANDOM FOREST RESULTS ==========")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Accuracy:", accuracy)

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

# Save model
Path("ai-model/size-prediction").mkdir(parents=True, exist_ok=True)

model_file = "ai-model/size-prediction/random_forest_model.pkl"

joblib.dump(model, model_file)

print("\nModel saved to:")
print(model_file)