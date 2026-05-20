import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import timm
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# ✅ Set Random Seed for Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ✅ Force CPU usage
device = torch.device("cpu")
print("⚠️ Training is set to run on CPU (GPU disabled due to performance issues).")

# ✅ Directories
TRAIN_DIR = "archive/Training"
TEST_DIR = "archive/Testing"

# ✅ Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# ✅ Load datasets
print("Loading datasets...")
train_dataset_full = datasets.ImageFolder(TRAIN_DIR, transform=transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

# ✅ Split training into training + validation (90%/10%)
train_size = int(0.9 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

# Using small batch size as per original script
BATCH_SIZE = 4 # Slightly increased from 2 for a bit better gradient estimation
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ✅ Class mapping and sample count
print("Class-to-Index Mapping:", train_dataset_full.class_to_idx)
targets = [label for _, label in train_dataset_full.samples]
class_counts = Counter(targets)
print("Samples per class:")
for idx, count in class_counts.items():
    class_name = list(train_dataset_full.class_to_idx.keys())[list(train_dataset_full.class_to_idx.values()).index(idx)]
    print(f"  {class_name}: {count}")

# ✅ Compute class weights
total_samples = sum(class_counts.values())
weights = [total_samples / class_counts[i] for i in range(len(class_counts))]
weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)

# ✅ Load ViT model (tiny variant)
print("Initializing ViT model...")
model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=4)
model = model.to(device)

# ✅ Loss, Optimizer, and Scheduler
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# ✅ Training loop with Early Stopping
epochs = 20 # Increased epochs but with early stopping
patience = 5
best_val_loss = float('inf')
counter = 0

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

print("Starting training...")
for epoch in range(epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct_val, total_val = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = 100 * correct_val / total_val

    # Update history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

    # Scheduler and Early Stopping
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_vit_model.pth")
        print("✅ Best model saved.")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# ✅ Save history to CSV
pd.DataFrame(history).to_csv("training_history.csv", index=False)
print("✅ Training history saved to training_history.csv")

# ✅ Final Plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss History')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.title('Accuracy History')
plt.legend()

plt.savefig('training_plots.png')
print("✅ Training plots saved as training_plots.png")
