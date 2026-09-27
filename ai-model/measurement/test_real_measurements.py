from feature_extractor import extract_measurement_features
from calibration import pixels_to_cm


# Example YOLO keypoints
points = [
    (141.22, 445.24),
    (147.96, 436.29),
    (131.24, 435.78),
    (151.61, 439.88),
    (108.42, 438.58),

    (159.46, 491.22),
    (87.03, 491.94),

    (184.53, 567.87),
    (117.12, 566.62),

    (162.14, 567.97),
    (164.41, 551.76),

    (152.63, 641.37),
    (97.83, 642.91),

    (167.84, 751.97),
    (85.97, 753.48),

    (189.35, 859.77),
    (72.88, 862.50)
]


# Extract pixel measurements
features = extract_measurement_features(points)


# Assumed user height for calibration
user_height_cm = 165

scale_factor = (
    user_height_cm /
    features["height_pixels"]
)


# Convert measurements to cm
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


print("========== REAL BODY MEASUREMENTS ==========")

print(
    f"Height: {user_height_cm:.2f} cm"
)

print(
    f"Shoulder width: {shoulder_cm:.2f} cm"
)

print(
    f"Hip width: {hip_cm:.2f} cm"
)

print(
    f"Arm length: {arm_cm:.2f} cm"
)

print(
    f"Leg length: {leg_cm:.2f} cm"
)