from ultralytics import YOLO

# Load pretrained YOLO Pose model
model = YOLO("yolo26n-pose.pt")

# Run pose detection
results = model("https://ultralytics.com/images/bus.jpg", save=True)

# Get keypoints
keypoints = results[0].keypoints

# Names of YOLO's 17 body keypoints
keypoint_names = [
    "Nose",
    "Left Eye",
    "Right Eye",
    "Left Ear",
    "Right Ear",
    "Left Shoulder",
    "Right Shoulder",
    "Left Elbow",
    "Right Elbow",
    "Left Wrist",
    "Right Wrist",
    "Left Hip",
    "Right Hip",
    "Left Knee",
    "Right Knee",
    "Left Ankle",
    "Right Ankle"
]

if keypoints is not None:

    points = keypoints.xy[0]

    print("\n========== BODY KEYPOINTS ==========")

    for i, point in enumerate(points):

        x = point[0].item()
        y = point[1].item()

        print(f"{keypoint_names[i]:15} → X={x:.2f}, Y={y:.2f}")

else:
    print("No body keypoints detected.")