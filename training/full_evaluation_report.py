import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
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
    "ResNet_model": "models/model.pth",
    "ResNet_best": "models/best_model.pth",
    "ResNet_best2": "models/best_model2.pth",
    "EffNet_v1": "models/efficientnet_best.pth",
    "EffNet_v2": "models/efficientnet_best_v2.pth"
}

results = []

# ================= LOAD MODEL FUNCTION =================
def load_model(path, name):

    state_dict = torch.load(path, map_location=device)

    # ---------- RESNET ----------
    if "resnet" in name.lower():

        model = models.resnet50(weights=None)

        # auto-detect fc
        if any("fc.1" in k for k in state_dict.keys()):
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

    model.load_state_dict(state_dict)
    model = model.to(device)

    return model


# ================= EVALUATION =================
def evaluate_model(model):

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds


# ================= MAIN LOOP =================
for name, path in model_paths.items():

    if not os.path.exists(path):
        print(f"❌ {path} not found")
        continue

    print(f"\n🔍 Evaluating {name}")

    model = load_model(path, name)

    y_true, y_pred = evaluate_model(model)

    # Accuracy
    acc = np.mean(np.array(y_true) == np.array(y_pred)) * 100

    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)

    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    print(f"✅ Accuracy: {acc:.2f}%")

    # ================= CONFUSION MATRIX =================
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.savefig(f"{name}_cm.png")
    plt.close()


# ================= RESULTS TABLE =================
df = pd.DataFrame(results)
print("\n📊 FINAL RESULTS:\n")
print(df)

# Save CSV
df.to_csv("model_comparison.csv", index=False)

# ================= BEST MODEL =================
best_model = df.loc[df["Accuracy"].idxmax()]
print("\n🏆 BEST MODEL:")
print(best_model)

# ================= GRAPH =================
plt.figure(figsize=(10,5))
plt.bar(df["Model"], df["Accuracy"])
plt.xticks(rotation=30)
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")

for i, v in enumerate(df["Accuracy"]):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')

plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.show()

# ================= METRICS GRAPH =================
df.set_index("Model")[["Precision","Recall","F1-score"]].plot(kind="bar", figsize=(10,5))

plt.title("Precision / Recall / F1 Comparison")
plt.ylabel("Score")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("metrics_comparison.png")
plt.show()