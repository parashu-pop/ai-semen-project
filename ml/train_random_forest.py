import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv(
    "outputs/final_fertility_dataset.csv"
)

# =========================
# BINARY CLASSIFICATION
# =========================
# 1 = Normal
# 0 = Abnormal

def classify_fertility(progressive):

    if progressive >= 32:
        return 1

    else:
        return 0

# Create labels
df["fertility_label"] = df[
    "Progressive motility (%)"
].apply(classify_fertility)

# =========================
# SELECT CLEAN FEATURES
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
# RANDOM FOREST MODEL
# =========================
model = RandomForestClassifier(

    n_estimators=200,

    max_depth=6,

    min_samples_split=2,

    min_samples_leaf=1,

    random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================
y_pred = model.predict(X_test)

# =========================
# ACCURACY
# =========================
accuracy = accuracy_score(y_test, y_pred)

print("\n========== RANDOM FOREST RESULTS ==========")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# =========================
# SAVE MODEL
# =========================
joblib.dump(
    model,
    "backend/models/random_forest_fertility.pkl"
)

# Save scaler also
joblib.dump(
    scaler,
    "backend/models/random_forest_scaler.pkl"
)

print("\nRandom Forest model saved.")

print("Scaler saved.")