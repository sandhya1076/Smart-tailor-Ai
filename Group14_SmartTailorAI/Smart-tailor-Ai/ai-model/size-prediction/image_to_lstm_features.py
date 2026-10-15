import numpy as np


def prepare_image_features(
    height_cm,
    shoulder_width_cm,
    hip_width_cm,
    arm_length_cm,
    leg_length_cm
):
    """
    Convert image-based body measurements into
    a feature vector for the size-prediction pipeline.
    """

    features = np.array([
        height_cm,
        shoulder_width_cm,
        hip_width_cm,
        arm_length_cm,
        leg_length_cm
    ], dtype=np.float32)

    return features


if __name__ == "__main__":

    # Example measurements from our YOLO image pipeline
    height = 165.00
    shoulder_width = 36.86
    hip_width = 23.92
    arm_length = 56.48
    leg_length = 84.00

    features = prepare_image_features(
        height,
        shoulder_width,
        hip_width,
        arm_length,
        leg_length
    )

    print("\n========== IMAGE → LSTM FEATURES ==========")
    print("Feature vector:")
    print(features)

    print("\nNumber of features:", len(features))