import sys
import os

# Allow imports from the AI project
AI_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if AI_MODEL_DIR not in sys.path:
    sys.path.append(AI_MODEL_DIR)


import torch
import torch.nn as nn
import joblib
import numpy as np

from ultralytics import YOLO

from measurement.feature_extractor import (
    extract_measurement_features
)

from measurement.calibration import (
    pixels_to_cm
)


# ============================================================
# LSTM MODEL
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
# LOAD MODELS
# ============================================================

pose_model = YOLO("yolo26n-pose.pt")


scaler = joblib.load(
    os.path.join(
        AI_MODEL_DIR,
        "size-prediction",
        "measurement_scaler.pkl"
    )
)


label_encoder = joblib.load(
    os.path.join(
        AI_MODEL_DIR,
        "size-prediction",
        "size_label_encoder.pkl"
    )
)


size_model = ClothingMeasurementLSTM(
    input_size=1,
    hidden_size=32,
    num_layers=1,
    num_classes=len(label_encoder.classes_)
)


size_model.load_state_dict(
    torch.load(
        os.path.join(
            AI_MODEL_DIR,
            "size-prediction",
            "clothing_measurement_lstm.pth"
        ),
        map_location="cpu"
    )
)

size_model.eval()


# ============================================================
# MAIN AI FUNCTION
# ============================================================

def analyze_body(image_path, user_height_cm):
    """
    Analyze a person's full-body image.

    Returns:
        Dictionary containing estimated measurements
        and experimental clothing-size prediction.
    """

    # --------------------------------------------------------
    # YOLO POSE
    # --------------------------------------------------------

    results = pose_model(image_path)

    keypoints = results[0].keypoints

    if keypoints is None or len(keypoints.xy) == 0:
        raise ValueError("No person detected in the image.")


    # --------------------------------------------------------
    # GET KEYPOINTS
    # --------------------------------------------------------

    points_tensor = keypoints.xy[0]

    points = [
        (
            point[0].item(),
            point[1].item()
        )
        for point in points_tensor
    ]


    # --------------------------------------------------------
    # EXTRACT PIXEL MEASUREMENTS
    # --------------------------------------------------------

    features = extract_measurement_features(points)


    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    scale_factor = (
        user_height_cm /
        features["height_pixels"]
    )


    # --------------------------------------------------------
    # PIXELS → CM
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # CREATE MODEL INPUT
    # --------------------------------------------------------

    measurement_vector = np.array(
        [
            height_cm,
            shoulder_cm,
            hip_cm,
            arm_cm,
            leg_cm
        ],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------

    scaled_features = scaler.transform(
        measurement_vector.reshape(1, -1)
    )


    # --------------------------------------------------------
    # LSTM INPUT
    # --------------------------------------------------------

    input_tensor = torch.tensor(
        scaled_features,
        dtype=torch.float32
    )

    input_tensor = input_tensor.reshape(
        1,
        5,
        1
    )


    # --------------------------------------------------------
    # SIZE PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        output = size_model(
            input_tensor
        )

        predicted_class = torch.argmax(
            output,
            dim=1
        ).item()


    predicted_size = label_encoder.inverse_transform(
        [predicted_class]
    )[0]


    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {

        "height_cm": round(
            float(height_cm), 2
        ),

        "shoulder_width_cm": round(
            float(shoulder_cm), 2
        ),

        "hip_width_cm": round(
            float(hip_cm), 2
        ),

        "arm_length_cm": round(
            float(arm_cm), 2
        ),

        "leg_length_cm": round(
            float(leg_cm), 2
        ),

        "recommended_size": str(
            predicted_size
        )

    }