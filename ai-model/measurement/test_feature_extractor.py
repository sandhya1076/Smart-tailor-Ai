from feature_extractor import extract_measurement_features


# Example YOLO keypoints
points = [
    (141.22, 445.24),  # Nose
    (147.96, 436.29),  # Left Eye
    (131.24, 435.78),  # Right Eye
    (151.61, 439.88),  # Left Ear
    (108.42, 438.58),  # Right Ear

    (159.46, 491.22),  # Left Shoulder
    (87.03, 491.94),   # Right Shoulder

    (184.53, 567.87),  # Left Elbow
    (117.12, 566.62),  # Right Elbow

    (162.14, 567.97),  # Left Wrist
    (164.41, 551.76),  # Right Wrist

    (152.63, 641.37),  # Left Hip
    (97.83, 642.91),   # Right Hip

    (167.84, 751.97),  # Left Knee
    (85.97, 753.48),   # Right Knee

    (189.35, 859.77),  # Left Ankle
    (72.88, 862.50)    # Right Ankle
]


features = extract_measurement_features(points)


print("========== MEASUREMENT FEATURES ==========")

for name, value in features.items():

    print(
        f"{name}: {value:.2f}"
    )