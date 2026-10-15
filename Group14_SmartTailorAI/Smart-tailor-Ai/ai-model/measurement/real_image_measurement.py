from ultralytics import YOLO
from feature_extractor import extract_measurement_features
from calibration import pixels_to_cm


# Load YOLO pose model
model = YOLO("yolo26n-pose.pt")

# Change this to your test image
image_path = "test_person.jpg"

# Run pose detection
results = model(image_path)

keypoints = results[0].keypoints

if keypoints is None or len(keypoints.xy) == 0:
    print("No person detected.")
    exit()

# Get first detected person's keypoints
points_tensor = keypoints.xy[0]

points = [
    (point[0].item(), point[1].item())
    for point in points_tensor
]

# Extract pixel measurements
features = extract_measurement_features(points)

# User's actual height
user_height_cm = 165

# Calibration
scale_factor = user_height_cm / features["height_pixels"]

# Convert pixels → cm
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


print("\n========== REAL IMAGE MEASUREMENTS ==========")

print(f"Height: {height_cm:.2f} cm")
print(f"Shoulder width: {shoulder_cm:.2f} cm")
print(f"Hip width: {hip_cm:.2f} cm")
print(f"Arm length: {arm_cm:.2f} cm")
print(f"Leg length: {leg_cm:.2f} cm")