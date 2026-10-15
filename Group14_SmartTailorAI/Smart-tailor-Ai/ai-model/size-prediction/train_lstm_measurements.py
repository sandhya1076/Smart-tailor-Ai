import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. LOAD DATASET
# ============================================================

data_path = "../../dataset/raw/personalized_clothing_dataset.csv"

df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)


# ============================================================
# 2. SELECT FEATURES
# ============================================================

features = [
    "Height_cm",
    "ShoulderWidth_cm",
    "Hip_cm",
    "ArmLength_cm",
    "LegLength_cm"
]

target = "RecommendedSize"

df = df[features + [target]].dropna()

print("Clean dataset shape:", df.shape)


# ============================================================
# 3. INPUT AND TARGET
# ============================================================

X = df[features].values
y = df[target].values


# ============================================================
# 4. ENCODE CLOTHING SIZES
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nClothing sizes:")
print(label_encoder.classes_)


# ============================================================
# 5. SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


# ============================================================
# 7. CONVERT TO PYTORCH TENSORS
# ============================================================

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# ============================================================
# 8. RESHAPE FOR LSTM
# ============================================================
#
# 5 measurements become 5 time steps
# Each time step contains 1 feature.
#
# Shape:
# (samples, 5, 1)
#

X_train = X_train.reshape(X_train.shape[0], 5, 1)
X_test = X_test.reshape(X_test.shape[0], 5, 1)

print("\nTraining input shape:", X_train.shape)
print("Testing input shape:", X_test.shape)


# ============================================================
# 9. LSTM MODEL
# ============================================================

class ClothingMeasurementLSTM(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        num_layers=1,
        num_classes=6
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        last_hidden = hidden[-1]

        return self.fc(last_hidden)


# Number of clothing sizes
num_classes = len(label_encoder.classes_)

model = ClothingMeasurementLSTM(
    input_size=1,
    hidden_size=32,
    num_layers=1,
    num_classes=num_classes
)


# ============================================================
# 10. LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 11. TRAINING
# ============================================================

epochs = 100

print("\n========== TRAINING LSTM ==========")

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(
        outputs,
        y_train
    )

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {loss.item():.4f}"
        )


# ============================================================
# 12. EVALUATION
# ============================================================

model.eval()

with torch.no_grad():

    outputs = model(X_test)

    predictions = torch.argmax(
        outputs,
        dim=1
    )


accuracy = accuracy_score(
    y_test.numpy(),
    predictions.numpy()
)

print("\n========== MODEL RESULTS ==========")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test.numpy(),
        predictions.numpy(),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# 13. SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "clothing_measurement_lstm.pth"
)


# ============================================================
# 14. SAVE SCALER
# ============================================================

import joblib

joblib.dump(
    scaler,
    "measurement_scaler.pkl"
)


# ============================================================
# 15. SAVE LABEL ENCODER
# ============================================================

joblib.dump(
    label_encoder,
    "size_label_encoder.pkl"
)


print("\nFiles saved:")
print("clothing_measurement_lstm.pth")
print("measurement_scaler.pkl")
print("size_label_encoder.pkl")