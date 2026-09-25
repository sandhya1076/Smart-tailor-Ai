import os
import sys

# Allow imports from ai-model
AI_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if AI_MODEL_DIR not in sys.path:
    sys.path.append(AI_MODEL_DIR)


from ultralytics import YOLO

from measurement.feature_extractor import (
    extract_measurement_features
)


# ============================================================
# LOAD YOLO POSE MODEL
# ============================================================

model = YOLO("yolo26n-pose.pt")


# ============================================================
# IMAGE PATHS
# ============================================================

front_image = "test_person.jpg"
side_image = "test_person_side.jpg"


# ============================================================
# FUNCTION TO EXTRACT MEASUREMENTS
# ============================================================

def get_measurements(image_path):

    results = model(image_path)

    keypoints = results[0].keypoints

    if keypoints is None or len(keypoints.xy) == 0:
        raise ValueError(
            f"No person detected in {image_path}"
        )

    points_tensor = keypoints.xy[0]

    points = [
        (
            point[0].item(),
            point[1].item()
        )
        for point in points_tensor
    ]

    return extract_measurement_features(points)


# ============================================================
# FRONT VIEW
# ============================================================

front = get_measurements(front_image)


# ============================================================
# SIDE VIEW
# ============================================================

side = get_measurements(side_image)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n==============================================")
print("          MULTI-VIEW MEASUREMENT")
print("==============================================")


print("\n------------- FRONT VIEW ----------------")

print(
    f"Height pixels         : "
    f"{front['height_pixels']:.2f}"
)

print(
    f"Shoulder width pixels : "
    f"{front['shoulder_width_pixels']:.2f}"
)

print(
    f"Hip width pixels      : "
    f"{front['hip_width_pixels']:.2f}"
)

print(
    f"Arm length pixels     : "
    f"{front['arm_length_pixels']:.2f}"
)

print(
    f"Leg length pixels     : "
    f"{front['leg_length_pixels']:.2f}"
)


print("\n------------- SIDE VIEW -----------------")

print(
    f"Height pixels         : "
    f"{side['height_pixels']:.2f}"
)

print(
    f"Shoulder width pixels : "
    f"{side['shoulder_width_pixels']:.2f}"
)

print(
    f"Hip width pixels      : "
    f"{side['hip_width_pixels']:.2f}"
)

print(
    f"Arm length pixels     : "
    f"{side['arm_length_pixels']:.2f}"
)

print(
    f"Leg length pixels     : "
    f"{side['leg_length_pixels']:.2f}"
)


# ============================================================
# SIMPLE MULTI-VIEW FUSION
# ============================================================

height_pixels = (
    front["height_pixels"] +
    side["height_pixels"]
) / 2


shoulder_pixels = (
    front["shoulder_width_pixels"] +
    side["shoulder_width_pixels"]
) / 2


hip_pixels = (
    front["hip_width_pixels"] +
    side["hip_width_pixels"]
) / 2


arm_pixels = (
    front["arm_length_pixels"] +
    side["arm_length_pixels"]
) / 2


leg_pixels = (
    front["leg_length_pixels"] +
    side["leg_length_pixels"]
) / 2


print("\n------------- FUSED VIEW ----------------")

print(
    f"Average height pixels         : "
    f"{height_pixels:.2f}"
)

print(
    f"Average shoulder width pixels : "
    f"{shoulder_pixels:.2f}"
)

print(
    f"Average hip width pixels      : "
    f"{hip_pixels:.2f}"
)

print(
    f"Average arm length pixels     : "
    f"{arm_pixels:.2f}"
)

print(
    f"Average leg length pixels     : "
    f"{leg_pixels:.2f}"
)


print("\n==============================================")