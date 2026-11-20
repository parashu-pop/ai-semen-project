
import pandas as pd
import os

# ============================================
# READ FINAL TRACKING RESULTS
# ============================================

result_file = "outputs/final_tracking_results.txt"

if not os.path.exists(result_file):

    print("Tracking results file not found!")
    exit()

# ============================================
# DEFAULT VALUES
# ============================================

total_sperm = 0
progressive = 0
non_progressive = 0
immotile = 0

# ============================================
# READ FILE
# ============================================

with open(result_file, "r") as f:

    lines = f.readlines()

    for line in lines:

        if "Final Visible Sperm Count" in line:

            total_sperm = int(
                line.split(":")[1].strip()
            )

        elif line.startswith("Non-Progressive"):

            non_progressive = int(
                line.split(":")[1].strip()
            )

        elif line.startswith("Progressive"):

            progressive = int(
            line.split(":")[1].strip()
            )

        elif line.startswith("Immotile"):

            immotile = int(
            line.split(":")[1].strip()
            )
# ============================================
# CALCULATE PERCENTAGES
# ============================================

if total_sperm > 0:

    progressive_pct = (
        progressive / total_sperm
    ) * 100

    non_progressive_pct = (
        non_progressive / total_sperm
    ) * 100

    immotile_pct = (
        immotile / total_sperm
    ) * 100

else:

    progressive_pct = 0
    non_progressive_pct = 0
    immotile_pct = 0

# ============================================
# CREATE DATAFRAME
# ============================================

data = {

    "total_sperm": [total_sperm],

    "progressive_motility_pct": [
        progressive_pct
    ],

    "non_progressive_motility_pct": [
        non_progressive_pct
    ],

    "immotile_pct": [
        immotile_pct
    ],

    "average_speed": [5.0]
}

df = pd.DataFrame(data)

# ============================================
# SAVE CSV
# ============================================

os.makedirs("outputs", exist_ok=True)

csv_path = "outputs/sperm_metrics.csv"

df.to_csv(csv_path, index=False)

# ============================================
# FINAL OUTPUT
# ============================================

print("\n========== METRICS GENERATED ==========")

print(df)

print(f"\nSaved to {csv_path}")
