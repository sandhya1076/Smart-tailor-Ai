import torch
import torch.nn as nn
import joblib
import numpy as np

from ultralytics import YOLO 
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)
from measurement.feature_extractor import extract_measurement_features
from measurement.calibration import pixels_to_cm


# ============================================================
# 1. LSTM MODEL
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


# ============================================================
# 2. LOAD YOLO
# ============================================================

pose_model = YOLO("yolo26n-pose.pt")


# ============================================================
# 3. IMAGE PATH
# ============================================================

image_path = "measurement/test_person.jpg"


# ============================================================
# 4. DETECT BODY
# ============================================================

results = pose_model(image_path)

keypoints = results[0].keypoints

if keypoints is None or len(keypoints.xy) == 0:

    print("No person detected.")

    exit()


# ============================================================
# 5. EXTRACT KEYPOINTS
# ============================================================

points_tensor = keypoints.xy[0]

points = [
    (point[0].item(), point[1].item())
    for point in points_tensor
]


# ============================================================
# 6. EXTRACT PIXEL MEASUREMENTS
# ============================================================

features = extract_measurement_features(points)


# ============================================================
# 7. HEIGHT CALIBRATION
# ============================================================

user_height_cm = 165.0

scale_factor = (
    user_height_cm /
    features["height_pixels"]
)


# ============================================================
# 8. CONVERT TO CM
# ============================================================

height_cm = user_height_cm

shoulder_cm = pixels_to_cm(
    features["shoulder_width_pixels"],
    scale_factor
)

hip_cm = pixels_to_cm(
    features["hip_width_pixels"],
    scale_factor
)

arm_cm = pixels_to_cm(
    features["arm_length_pixels"],
    scale_factor
)

leg_cm = pixels_to_cm(
    features["leg_length_pixels"],
    scale_factor
)


# ============================================================
# 9. CREATE FEATURE VECTOR
# ============================================================

measurement_vector = np.array([
    height_cm,
    shoulder_cm,
    hip_cm,
    arm_cm,
    leg_cm
], dtype=np.float32)


# ============================================================
# 10. LOAD SCALER
# ============================================================

scaler = joblib.load(
    "size-prediction/measurement_scaler.pkl"
)

scaled_features = scaler.transform(
    measurement_vector.reshape(1, -1)
)


# ============================================================
# 11. CONVERT TO LSTM FORMAT
# ============================================================

input_tensor = torch.tensor(
    scaled_features,
    dtype=torch.float32
)

input_tensor = input_tensor.reshape(
    1, 5, 1
)


# ============================================================
# 12. LOAD LABEL ENCODER
# ============================================================

label_encoder = joblib.load(
    "size-prediction/size_label_encoder.pkl"
)


# ============================================================
# 13. LOAD LSTM
# ============================================================

model = ClothingMeasurementLSTM(
    input_size=1,
    hidden_size=32,
    num_layers=1,
    num_classes=len(label_encoder.classes_)
)

model.load_state_dict(
    torch.load(
    "size-prediction/clothing_measurement_lstm.pth",
    map_location="cpu"
)
)

model.eval()


# ============================================================
# 14. PREDICT SIZE
# ============================================================

with torch.no_grad():

    output = model(input_tensor)

    predicted_class = torch.argmax(
        output,
        dim=1
    ).item()


predicted_size = label_encoder.inverse_transform(
    [predicted_class]
)[0]


# ============================================================
# 15. DISPLAY RESULTS
# ============================================================

print("\n")
print("==============================================")
print("       AI BODY MEASUREMENT SYSTEM")
print("==============================================")

print(f"Height          : {height_cm:.2f} cm")
print(f"Shoulder Width  : {shoulder_cm:.2f} cm")
print(f"Hip Width       : {hip_cm:.2f} cm")
print(f"Arm Length      : {arm_cm:.2f} cm")
print(f"Leg Length      : {leg_cm:.2f} cm")

print("----------------------------------------------")

print(f"Recommended Size: {predicted_size}")

print("==============================================")