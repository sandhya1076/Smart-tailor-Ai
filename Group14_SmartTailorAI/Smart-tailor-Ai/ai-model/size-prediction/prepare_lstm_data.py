import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

# ==========================================
# 1. LOAD DATA
# ==========================================

input_file = "dataset/processed/scaled_cloth_size.csv"

df = pd.read_csv(input_file)

features = [
    "weight",
    "height",
    "age"
]

target = "size"

X = df[features].values
y = df[target].values


# ==========================================
# 2. ENCODE SIZE LABELS
# ==========================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("Size classes:")
print(label_encoder.classes_)

print("\nEncoded classes:")
for i, label in enumerate(label_encoder.classes_):
    print(label, "->", i)


# ==========================================
# 3. RESHAPE FOR LSTM
# ==========================================

# LSTM expects:
# (samples, time_steps, features)

X_lstm = X.reshape(
    X.shape[0],
    X.shape[1],
    1
)

print("\nLSTM input shape:")
print(X_lstm.shape)


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_lstm,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 5. SAVE DATA
# ==========================================

Path("dataset/processed").mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    "dataset/processed/X_train_lstm.npy",
    X_train
)

np.save(
    "dataset/processed/X_test_lstm.npy",
    X_test
)

np.save(
    "dataset/processed/y_train_lstm.npy",
    y_train
)

np.save(
    "dataset/processed/y_test_lstm.npy",
    y_test
)


# Save label encoder
joblib.dump(
    label_encoder,
    "dataset/processed/size_label_encoder.pkl"
)


print("\nLSTM data preparation complete.")