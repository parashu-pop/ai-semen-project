import pandas as pd
import joblib
from collections import Counter

# =========================
# LOAD TRAINED MODEL
# =========================
model = joblib.load(
    "backend/models/fertility_xgboost_model.pkl"
)

# =========================
# LOAD SCALER
# =========================
scaler = joblib.load(
    "backend/models/fertility_scaler.pkl"
)

# =========================
# LOAD GENERATED METRICS
# =========================
df = pd.read_csv(
    "outputs/final_fertility_dataset.csv"
)

# =========================
# SELECT FEATURES
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

# =========================
# SCALE FEATURES
# =========================
X_scaled = scaler.transform(X)

# =========================
# PREDICT
# =========================
predictions = model.predict(X_scaled)

# =========================
# MAP LABELS
# =========================
label_map = {

    1: "Normal Fertility",

    0: "Abnormal Fertility"
}

mapped_predictions = [
    label_map[p]
    for p in predictions
]

# =========================
# DISPLAY RESULTS
# =========================
print("\n========== FERTILITY PREDICTIONS ==========\n")

for idx, prediction in enumerate(
    mapped_predictions
):

    print(
        f"Participant {df.iloc[idx]['ID']}: "
        f"{prediction}"
    )

# =========================
# SUMMARY
# =========================
summary = Counter(mapped_predictions)

print("\n========== SUMMARY ==========\n")

for key, value in summary.items():

    print(f"{key}: {value}")