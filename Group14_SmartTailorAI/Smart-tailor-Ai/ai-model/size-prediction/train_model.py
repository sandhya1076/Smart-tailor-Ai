import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

# Load scaled dataset
input_file = "dataset/processed/scaled_clothing_dataset.csv"

df = pd.read_csv(input_file)

# Features
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

# Input and target
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

# Create model
model = LogisticRegression(
    max_iter=1000
)

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("========== MODEL RESULTS ==========")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Accuracy:", accuracy)

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

# Save model
Path("ai-model/size-prediction").mkdir(parents=True, exist_ok=True)

model_file = "ai-model/size-prediction/size_model.pkl"
joblib.dump(model, model_file)

print("\nModel saved to:")
print(model_file)