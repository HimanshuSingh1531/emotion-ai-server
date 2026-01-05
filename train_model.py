import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 🔧 SETTINGS
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001

DATASET_PATH = "dataset/train"
MODEL_SAVE_PATH = "emotion_model.pth"

# 🎨 IMAGE TRANSFORMS
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 📦 DATASET LOAD
dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

class_names = dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)

# 🧠 CNN MODEL
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 10 * 10, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = EmotionCNN(num_classes)

# ⚙️ LOSS + OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 🚀 TRAINING LOOP
print("\n🚀 Training started...\n")

for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {total_loss:.4f} | Accuracy: {accuracy:.2f}%")

# 💾 SAVE MODEL
torch.save({
    "model_state": model.state_dict(),
    "classes": class_names
}, MODEL_SAVE_PATH)

print(f"\n✅ Model saved as {MODEL_SAVE_PATH}")
