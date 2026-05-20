import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support
import pandas as pd

# ✅ Config
device = torch.device("cpu")
TEST_DIR = "archive/Testing"
MODEL_PATH = "best_vit_model.pth"
BATCH_SIZE = 4

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# ✅ Load Test Data
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_dataset.classes

# ✅ Load Model
model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ✅ Evaluation
all_preds = []
all_labels = []
all_probs = []

print("Running evaluation on test set...")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# ✅ Metrics
print("\n--- Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=class_names))

precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
accuracy = np.mean(all_preds == all_labels)

# ✅ Save metrics to file
with open("test_metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n\n")
    f.write(classification_report(all_labels, all_preds, target_names=class_names))

# ✅ Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
print("✅ Confusion Matrix saved as confusion_matrix.png")

# ✅ ROC-AUC Curve
plt.figure(figsize=(10, 8))
for i in range(len(class_names)):
    # One-vs-Rest ROC
    y_true = (all_labels == i).astype(int)
    y_prob = all_probs[:, i]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve')
plt.legend()
plt.savefig('roc_curve.png')
print("✅ ROC-AUC Curve saved as roc_curve.png")

# ✅ Class Distribution in Test Set
plt.figure(figsize=(8, 6))
sns.countplot(x=[class_names[i] for i in all_labels])
plt.title('Test Set Class Distribution')
plt.xticks(rotation=45)
plt.savefig('test_distribution.png')
print("✅ Test distribution plot saved.")

# ✅ Prediction Confidence Scores
confidences = np.max(all_probs, axis=1)
plt.figure(figsize=(10, 6))
sns.histplot(confidences, bins=20, kde=True)
plt.title('Prediction Confidence Distribution')
plt.xlabel('Confidence Score')
plt.savefig('confidence_scores.png')
print("✅ Confidence scores distribution saved.")
