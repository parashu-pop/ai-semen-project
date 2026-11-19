import pandas as pd

# =========================
# LOAD TRACKING FEATURES
# =========================
tracking_df = pd.read_csv(
    "outputs/sperm_metrics.csv"
)

# Rename video_id -> ID
tracking_df.rename(
    columns={"video_id": "ID"},
    inplace=True
)

# =========================
# LOAD SEMEN ANALYSIS DATA
# =========================
semen_df = pd.read_csv(
    "dataset/visem_tracking/semen_analysis_data_Train.csv"
)

# =========================
# LOAD PARTICIPANT DATA
# =========================
participant_df = pd.read_csv(
    "dataset/visem_tracking/participant_related_data_Train.csv"
)

# =========================
# LOAD HORMONE DATA
# =========================
hormone_df = pd.read_csv(
    "dataset/visem_tracking/sex_hormones_Train.csv"
)

# =========================
# SELECT IMPORTANT COLUMNS
# =========================

# Clinical semen analysis
semen_df = semen_df[[

    "ID",

    "Sperm vitality (%)",

    "Normal spermatozoa (%)",

    "Progressive motility (%)",

    "Non progressive sperm motility (%)",

    "Immotile sperm (%)",

    "DNA fragmentation index, DFI (%)"
]]

# Participant/lifestyle data
participant_df = participant_df[[

    "ID",

    "Body mass index (kg/m²)",

    "Age (years)"
]]

# Hormone data
hormone_df = hormone_df[[

    "ID",

    "Serum total testosterone (nmol/L)",

    "Serum follicle-stimulating hormone, FSH (IU/L)",

    "Serum Luteinizing hormone, LH (IU/L)"
]]

# =========================
# MERGE ALL DATASETS
# =========================

# Merge tracking + semen analysis
merged_df = tracking_df.merge(
    semen_df,
    on="ID",
    how="left"
)

# Merge participant data
merged_df = merged_df.merge(
    participant_df,
    on="ID",
    how="left"
)

# Merge hormone data
merged_df = merged_df.merge(
    hormone_df,
    on="ID",
    how="left"
)

# =========================
# DISPLAY DATASET INFO
# =========================
print("\n========== FINAL MERGED DATASET ==========\n")

print(merged_df.head())

print("\n========== DATASET COLUMNS ==========\n")

print(merged_df.columns)

print("\n========== DATASET SHAPE ==========\n")

print(merged_df.shape)

# =========================
# SAVE FINAL DATASET
# =========================
merged_df.to_csv(
    "outputs/final_fertility_dataset.csv",
    index=False
)

print("\nSaved to outputs/final_fertility_dataset.csv")