from ultralytics import YOLO
from math import sqrt


# -----------------------------
# Distance calculation
# -----------------------------
def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# -----------------------------
# Body height calculation
# -----------------------------
def calculate_body_height(points):

    top_y = points[0][1]

    left_ankle_y = points[15][1]
    right_ankle_y = points[16][1]

    bottom_y = max(left_ankle_y, right_ankle_y)

    return bottom_y - top_y


# -----------------------------
# Main measurement function
# -----------------------------
def calculate_measurements(points, height_cm):

    body_height_pixels = calculate_body_height(points)

    scale_factor = height_cm / body_height_pixels

    shoulder_width_pixels = distance(
        points[5],
        points[6]
    )

    hip_width_pixels = distance(
        points[11],
        points[12]
    )

    left_leg = (
        distance(points[11], points[13])
        + distance(points[13], points[15])
    )

    right_leg = (
        distance(points[12], points[14])
        + distance(points[14], points[16])
    )

    average_leg_pixels = (
        left_leg + right_leg
    ) / 2

    return {
        "body_height_cm": height_cm,

        "shoulder_width_cm":
            shoulder_width_pixels * scale_factor,

        "hip_width_cm":
            hip_width_pixels * scale_factor,

        "leg_length_cm":
            average_leg_pixels * scale_factor
    }


# -----------------------------
# YOLO Pose
# -----------------------------
model = YOLO("yolo26n-pose.pt")

results = model(
    "https://ultralytics.com/images/bus.jpg"
)

keypoints = results[0].keypoints


if keypoints is not None:

    # First detected person
    points_tensor = keypoints.xy[0]

    points = []

    for point in points_tensor:

        x = point[0].item()
        y = point[1].item()

        points.append((x, y))


    # User's known height
    # This will later come from the mobile app.
    user_height_cm = 165


    measurements = calculate_measurements(
        points,
        user_height_cm
    )


    print("\n========== AI BODY MEASUREMENTS ==========")

    for name, value in measurements.items():

        print(
            f"{name}: {value:.2f} cm"
        )

else:

    print("No person detected.")