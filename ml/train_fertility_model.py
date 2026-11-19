import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier
import joblib

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv(
    "outputs/final_fertility_dataset.csv"
)

# =========================
# BINARY FERTILITY LABELS
# =========================
# 1 = Normal
# 0 = Abnormal

def classify_fertility(progressive):

    if progressive >= 32:
        return 1

    else:
        return 0

df["fertility_label"] = df[
    "Progressive motility (%)"
].apply(classify_fertility)

# =========================
# CLEAN FEATURES
# =========================
X = df[[

    "total_sperm",

    "progressive_motility_pct",

    "average_speed",

    "Sperm vitality (%)",

    "Normal spermatozoa (%)",

    "DNA fragmentation index, DFI (%)",

    "Body mass index (kg/m²)",

    "Age (years)",

    "Serum total testosterone (nmol/L)",

    "Serum follicle-stimulating hormone, FSH (IU/L)",

    "Serum Luteinizing hormone, LH (IU/L)"
]]

# Target
y = df["fertility_label"]

# =========================
# FEATURE SCALING
# =========================
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# XGBOOST MODEL
# =========================
model = XGBClassifier(

    n_estimators=300,

    max_depth=4,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model.fit(X_train, y_train)

# =========================
# PREDICT
# =========================
y_pred = model.predict(X_test)

# =========================
# ACCURACY
# =========================
accuracy = accuracy_score(y_test, y_pred)

print("\n========== XGBOOST RESULTS ==========\n")

print(f"Accuracy: {accuracy * 100:.2f}%")

# =========================
# CLASSIFICATION REPORT
# =========================
print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# =========================
# CONFUSION MATRIX
# =========================
print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
joblib.dump(
    model,
    "backend/models/fertility_xgboost_model.pkl"
)

# Save scaler
joblib.dump(
    scaler,
    "backend/models/fertility_scaler.pkl"
)

print("\nModel saved successfully.")