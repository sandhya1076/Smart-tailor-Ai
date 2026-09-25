import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path


# ==========================================
# 1. AUTOENCODER MODEL
# ==========================================

class DenoisingAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 3,
                kernel_size=2,
                stride=2
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


# ==========================================
# 2. DATASET
# ==========================================

class ImageDataset(Dataset):

    def __init__(self, image_folder):

        self.image_files = list(
            Path(image_folder).glob("*.jpg")
        ) + list(
            Path(image_folder).glob("*.jpeg")
        ) + list(
            Path(image_folder).glob("*.png")
        )

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        image_path = self.image_files[index]

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        return image


# ==========================================
# 3. PREPARE DATA
# ==========================================

dataset_path = "dataset/autoencoder/train"

dataset = ImageDataset(dataset_path)

print("Total images:", len(dataset))

if len(dataset) == 0:
    raise RuntimeError("No images found in the training folder.")


# 80% training, 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)


print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ==========================================
# 4. DEVICE
# ==========================================

device = torch.device("cpu")

print("Using device:", device)


# ==========================================
# 5. MODEL
# ==========================================

model = DenoisingAutoencoder().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 6. TRAINING
# ==========================================

epochs = 20

print("\n========== TRAINING AUTOENCODER ==========")

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images in train_loader:

        images = images.to(device)

        # Add random noise
        noise = torch.randn_like(images) * 0.1

        noisy_images = images + noise

        noisy_images = torch.clamp(
            noisy_images,
            0,
            1
        )

        # Forward pass
        outputs = model(noisy_images)

        # Compare reconstructed image with original
        loss = criterion(outputs, images)

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Training Loss: {average_loss:.6f}"
    )


# ==========================================
# 7. VALIDATION
# ==========================================

model.eval()

validation_loss = 0

with torch.no_grad():

    for images in val_loader:

        images = images.to(device)

        noise = torch.randn_like(images) * 0.1

        noisy_images = images + noise

        noisy_images = torch.clamp(
            noisy_images,
            0,
            1
        )

        outputs = model(noisy_images)

        loss = criterion(outputs, images)

        validation_loss += loss.item()


validation_loss /= len(val_loader)

print("\n========== VALIDATION ==========")
print(f"Validation Loss: {validation_loss:.6f}")


# ==========================================
# 8. SAVE MODEL
# ==========================================

model_path = "ai-model/preprocessing/autoencoder/autoencoder_model.pth"

torch.save(model.state_dict(), model_path)

print("\nAutoencoder model saved to:")
print(model_path)