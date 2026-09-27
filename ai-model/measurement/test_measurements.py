from measurements import calculate_measurements

# Body keypoints obtained from YOLO Pose
points = [
    (141.22, 445.24),  # 0 Nose
    (147.96, 436.29),  # 1 Left Eye
    (131.24, 435.78),  # 2 Right Eye
    (151.61, 439.88),  # 3 Left Ear
    (108.42, 438.58),  # 4 Right Ear
    (159.46, 491.22),  # 5 Left Shoulder
    (87.03, 491.94),   # 6 Right Shoulder
    (184.53, 567.87),  # 7 Left Elbow
    (117.12, 566.62),  # 8 Right Elbow
    (162.14, 567.97),  # 9 Left Wrist
    (164.41, 551.76),  # 10 Right Wrist
    (152.63, 641.37),  # 11 Left Hip
    (97.83, 642.91),   # 12 Right Hip
    (167.84, 751.97),  # 13 Left Knee
    (85.97, 753.48),   # 14 Right Knee
    (189.35, 859.77),  # 15 Left Ankle
    (72.88, 862.50)    # 16 Right Ankle
]

measurements = calculate_measurements(points)

print("\n========== BODY MEASUREMENTS ==========")

for name, value in measurements.items():
    print(f"{name}: {value:.2f} pixels")