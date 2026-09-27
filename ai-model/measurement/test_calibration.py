from calibration import calculate_scale_factor, pixels_to_cm


# Example: user enters their actual height
height_cm = 165

# Pixel measurements obtained from our YOLO pipeline
body_height_pixels = 417.26
shoulder_width_pixels = 72.43
hip_width_pixels = 54.82
leg_length_pixels = 221.29


# Calculate the scale factor
scale_factor = calculate_scale_factor(
    height_cm,
    body_height_pixels
)

# Convert pixel measurements to centimetres
shoulder_width_cm = pixels_to_cm(
    shoulder_width_pixels,
    scale_factor
)

hip_width_cm = pixels_to_cm(
    hip_width_pixels,
    scale_factor
)

leg_length_cm = pixels_to_cm(
    leg_length_pixels,
    scale_factor
)


print("\n========== CALIBRATION ==========")
print(f"Scale factor: {scale_factor:.4f} cm/pixel")

print("\n========== ESTIMATED MEASUREMENTS ==========")
print(f"Shoulder width: {shoulder_width_cm:.2f} cm")
print(f"Hip width:      {hip_width_cm:.2f} cm")
print(f"Leg length:     {leg_length_cm:.2f} cm")