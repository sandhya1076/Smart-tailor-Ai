from math import sqrt


def distance(point1, point2):
    """Calculate distance between two points."""
    x1, y1 = point1
    x2, y2 = point2

    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_measurements(points):
    """
    Calculate basic body measurements in pixels.

    YOLO Pose keypoint indexes:
    5  = Left Shoulder
    6  = Right Shoulder
    11 = Left Hip
    12 = Right Hip
    13 = Left Knee
    14 = Right Knee
    15 = Left Ankle
    16 = Right Ankle
    """

    # Shoulder width
    shoulder_width = distance(points[5], points[6])

    # Hip width
    hip_width = distance(points[11], points[12])

    # Left leg length
    left_thigh = distance(points[11], points[13])
    left_lower_leg = distance(points[13], points[15])
    left_leg_length = left_thigh + left_lower_leg

    # Right leg length
    right_thigh = distance(points[12], points[14])
    right_lower_leg = distance(points[14], points[16])
    right_leg_length = right_thigh + right_lower_leg

    # Average both legs
    average_leg_length = (left_leg_length + right_leg_length) / 2

    return {
        "shoulder_width_pixels": shoulder_width,
        "hip_width_pixels": hip_width,
        "leg_length_pixels": average_leg_length
    }