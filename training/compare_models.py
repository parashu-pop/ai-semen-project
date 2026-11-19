import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= DATA =================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

test_data = datasets.ImageFolder("dataset/test", transform=transform)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

class_names = test_data.classes
num_classes = len(class_names)

# ================= MODEL PATHS =================
model_paths = {
    "ResNet (model.pth)": "models/model.pth",
    "ResNet (best_model.pth)": "models/best_model.pth",
    "ResNet (best_model2.pth)": "models/best_model2.pth",
    "EfficientNet (v1)": "models/efficientnet_best.pth",
    "EfficientNet (v2)": "models/efficientnet_best_v2.pth"
}

results = {}

# ================= EVALUATE FUNCTION =================
def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total


# ================= MAIN LOOP =================
for name, path in model_paths.items():

    if not os.path.exists(path):
        print(f"❌ {path} not found")
        continue

    print(f"\n🔍 Evaluating {name}")

    # ---------- RESNET ----------
    if "resnet" in name.lower():

        model = models.resnet50(weights=None)

        # 🔥 Correct architecture based on file
        if "best_model2" in path:
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.fc.in_features, num_classes)
            )
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    # ---------- EFFICIENTNET ----------
    else:
        model = models.efficientnet_b0(weights=None)

        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )

    # Load weights
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)

    # Evaluate
    acc = evaluate(model)
    results[name] = acc

    print(f"✅ Accuracy: {acc:.2f}%")

# ================= BEST MODEL =================
best_model = max(results, key=results.get)
print("\n🏆 BEST MODEL:", best_model)
print(f"🔥 BEST ACCURACY: {results[best_model]:.2f}%")

# ================= GRAPH =================
names = list(results.keys())
accuracies = list(results.values())

plt.figure(figsize=(10,5))
plt.bar(names, accuracies)
plt.xticks(rotation=30, ha='right')
plt.ylabel("Accuracy (%)")
plt.title("Model Comparison")

for i, v in enumerate(accuracies):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')

plt.tight_layout()
plt.show()