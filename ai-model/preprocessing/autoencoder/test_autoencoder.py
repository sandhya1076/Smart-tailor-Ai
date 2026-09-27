import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from train_autoencoder import DenoisingAutoencoder


# -----------------------------
# Paths
# -----------------------------

image_path = "dataset/autoencoder/train/person_001.jpg"

output_dir = Path("results/autoencoder")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "reconstructed_person_001.jpg"


# -----------------------------
# Device
# -----------------------------

device = torch.device("cpu")


# -----------------------------
# Load model
# -----------------------------

model = DenoisingAutoencoder().to(device)

model.load_state_dict(
    torch.load(
        "ai-model/preprocessing/autoencoder/autoencoder_model.pth",
        map_location=device
    )
)

model.eval()


# -----------------------------
# Prepare image
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

image = Image.open(image_path).convert("RGB")

original = transform(image).unsqueeze(0).to(device)


# -----------------------------
# Add noise
# -----------------------------

noise = torch.randn_like(original) * 0.1

noisy = original + noise

noisy = torch.clamp(noisy, 0, 1)


# -----------------------------
# Reconstruct
# -----------------------------

with torch.no_grad():

    reconstructed = model(noisy)


# -----------------------------
# Save comparison
# -----------------------------

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(original.squeeze().permute(1, 2, 0))
axes[0].set_title("Original")

axes[1].imshow(noisy.squeeze().permute(1, 2, 0))
axes[1].set_title("Noisy")

axes[2].imshow(reconstructed.squeeze().permute(1, 2, 0))
axes[2].set_title("Reconstructed")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig(output_path)

plt.close()

print("Comparison saved to:")
print(output_path)