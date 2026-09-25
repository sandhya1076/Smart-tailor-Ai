import torch
import cv2
import numpy as np

from torchvision import transforms
from ultralytics import YOLO

from autoencoder import DenoisingAutoencoder


# ==========================================
# SETTINGS
# ==========================================

image_path = "dataset/autoencoder/train/person_001.jpg"

autoencoder_path = (
    "ai-model/preprocessing/autoencoder/"
    "autoencoder_model.pth"
)

yolo_model_path = "yolo26n-pose.pt"


# ==========================================
# DEVICE
# ==========================================

device = torch.device("cpu")


# ==========================================
# LOAD AUTOENCODER
# ==========================================

autoencoder = DenoisingAutoencoder().to(device)

autoencoder.load_state_dict(
    torch.load(
        autoencoder_path,
        map_location=device
    )
)

autoencoder.eval()


# ==========================================
# LOAD YOLO POSE
# ==========================================

yolo = YOLO(yolo_model_path)


# ==========================================
# LOAD IMAGE
# ==========================================

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {image_path}"
    )


# ==========================================
# YOLO ON ORIGINAL IMAGE
# ==========================================

print("\n========== ORIGINAL IMAGE ==========")

original_results = yolo(image)

original_keypoints = original_results[0].keypoints

if original_keypoints is not None:

    original_points = original_keypoints.xy[0]

    print(
        "Keypoints detected:",
        len(original_points)
    )

else:

    print("No keypoints detected.")


# ==========================================
# PREPARE IMAGE FOR AUTOENCODER
# ==========================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# OpenCV uses BGR, convert to RGB
image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

input_tensor = transform(image_rgb)

input_tensor = input_tensor.unsqueeze(0).to(device)


# ==========================================
# AUTOENCODER RECONSTRUCTION
# ==========================================

with torch.no_grad():

    reconstructed = autoencoder(
        input_tensor
    )


# ==========================================
# CONVERT BACK TO OPENCV IMAGE
# ==========================================

reconstructed = (
    reconstructed.squeeze(0)
    .cpu()
    .permute(1, 2, 0)
    .numpy()
)

reconstructed = (
    reconstructed * 255
).clip(0, 255).astype(np.uint8)

reconstructed = cv2.cvtColor(
    reconstructed,
    cv2.COLOR_RGB2BGR
)


# ==========================================
# SAVE RECONSTRUCTED IMAGE
# ==========================================

output_path = (
    "results/autoencoder/"
    "reconstructed_for_yolo.jpg"
)

cv2.imwrite(
    output_path,
    reconstructed
)

print("\nReconstructed image saved to:")
print(output_path)


# ==========================================
# YOLO ON RECONSTRUCTED IMAGE
# ==========================================

print("\n========== RECONSTRUCTED IMAGE ==========")

reconstructed_results = yolo(
    reconstructed
)

reconstructed_keypoints = (
    reconstructed_results[0].keypoints
)

if (
    reconstructed_keypoints is not None
    and len(reconstructed_keypoints.xy) > 0
):

    reconstructed_points = (
        reconstructed_keypoints.xy[0]
    )

    print(
        "Keypoints detected:",
        len(reconstructed_points)
    )

else:

    reconstructed_points = None

    print(
        "No person/keypoints detected "
        "in reconstructed image."
    )


# ==========================================
# SUMMARY
# ==========================================

print("\n========== SUMMARY ==========")

if original_keypoints is not None:

    print(
        "Original image:",
        len(original_points),
        "keypoints"
    )

else:

    print(
        "Original image: No detection"
    )


if reconstructed_points is not None:

    print(
        "Reconstructed image:",
        len(reconstructed_points),
        "keypoints"
    )

else:

    print(
        "Reconstructed image: No detection"
    )