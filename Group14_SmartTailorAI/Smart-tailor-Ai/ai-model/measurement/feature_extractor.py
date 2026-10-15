from math import sqrt


# ==========================================
# DISTANCE BETWEEN TWO KEYPOINTS
# ==========================================

def distance(point1, point2):

    x1, y1 = point1
    x2, y2 = point2

    return sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


# ==========================================
# BODY MEASUREMENT FEATURES
# ==========================================

def extract_measurement_features(points):

    # Body height
    top_y = points[0][1]

    left_ankle_y = points[15][1]
    right_ankle_y = points[16][1]

    bottom_y = max(
        left_ankle_y,
        right_ankle_y
    )

    body_height = bottom_y - top_y


    # Shoulder width
    shoulder_width = distance(
        points[5],
        points[6]
    )


    # Hip width
    hip_width = distance(
        points[11],
        points[12]
    )


    # Left arm length
    left_upper_arm = distance(
        points[5],
        points[7]
    )

    left_forearm = distance(
        points[7],
        points[9]
    )

    left_arm_length = (
        left_upper_arm +
        left_forearm
    )


    # Right arm length
    right_upper_arm = distance(
        points[6],
        points[8]
    )

    right_forearm = distance(
        points[8],
        points[10]
    )

    right_arm_length = (
        right_upper_arm +
        right_forearm
    )


    # Average arm length
    arm_length = (
        left_arm_length +
        right_arm_length
    ) / 2


    # Left leg length
    left_thigh = distance(
        points[11],
        points[13]
    )

    left_lower_leg = distance(
        points[13],
        points[15]
    )

    left_leg_length = (
        left_thigh +
        left_lower_leg
    )


    # Right leg length
    right_thigh = distance(
        points[12],
        points[14]
    )

    right_lower_leg = distance(
        points[14],
        points[16]
    )

    right_leg_length = (
        right_thigh +
        right_lower_leg
    )


    # Average leg length
    leg_length = (
        left_leg_length +
        right_leg_length
    ) / 2


    # Return features
    return {
        "height_pixels": body_height,
        "shoulder_width_pixels": shoulder_width,
        "hip_width_pixels": hip_width,
        "arm_length_pixels": arm_length,
        "leg_length_pixels": leg_length
    }