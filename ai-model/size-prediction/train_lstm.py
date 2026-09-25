import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path


# ==========================================
# 1. LOAD DATA
# ==========================================

X_train = np.load(
    "dataset/processed/X_train_lstm.npy"
)

X_test = np.load(
    "dataset/processed/X_test_lstm.npy"
)

y_train = np.load(
    "dataset/processed/y_train_lstm.npy"
)

y_test = np.load(
    "dataset/processed/y_test_lstm.npy"
)

label_encoder = joblib.load(
    "dataset/processed/size_label_encoder.pkl"
)


print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ==========================================
# 2. CONVERT TO PYTORCH TENSORS
# ==========================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)


# ==========================================
# 3. LSTM MODEL
# ==========================================

class ClothingSizeLSTM(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        num_layers=1,
        num_classes=7
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

        # Use final hidden state
        last_hidden = hidden[-1]

        output = self.fc(last_hidden)

        return output


# ==========================================
# 4. CREATE MODEL
# ==========================================

device = torch.device("cpu")

model = ClothingSizeLSTM(
    input_size=1,
    hidden_size=32,
    num_layers=1,
    num_classes=len(label_encoder.classes_)
).to(device)


print("\n========== MODEL ==========")
print(model)


# ==========================================
# 5. LOSS + OPTIMIZER
# ==========================================

# Calculate class weights
class_counts = np.bincount(y_train.numpy())

class_weights = (
    len(y_train) /
    (len(class_counts) * class_counts)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 6. TRAINING
# ==========================================

epochs = 20

batch_size = 64

print("\n========== TRAINING LSTM ==========")

for epoch in range(epochs):

    model.train()

    total_loss = 0

    # Shuffle training data
    indices = torch.randperm(
        len(X_train)
    )

    for start in range(
        0,
        len(X_train),
        batch_size
    ):

        batch_indices = indices[
            start:start + batch_size
        ]

        batch_X = X_train[
            batch_indices
        ].to(device)

        batch_y = y_train[
            batch_indices
        ].to(device)

        # Forward pass
        outputs = model(batch_X)

        loss = criterion(
            outputs,
            batch_y
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss /
        ((len(X_train) + batch_size - 1)
         // batch_size)
    )

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {average_loss:.6f}"
    )


# ==========================================
# 7. EVALUATION
# ==========================================

model.eval()

with torch.no_grad():

    outputs = model(
        X_test.to(device)
    )

    predictions = torch.argmax(
        outputs,
        dim=1
    )

accuracy = accuracy_score(
    y_test.numpy(),
    predictions.cpu().numpy()
)

print("\n========== LSTM RESULTS ==========")

print("Accuracy:", accuracy)

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test.numpy(),
        predictions.cpu().numpy(),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ==========================================
# 8. SAVE MODEL
# ==========================================

model_path = (
    "ai-model/size-prediction/"
    "clothing_size_lstm.pth"
)

Path(
    "ai-model/size-prediction"
).mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    model.state_dict(),
    model_path
)

print("\nLSTM model saved to:")
print(model_path)