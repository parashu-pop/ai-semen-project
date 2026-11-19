import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

# ================= LOAD DATA =================
df = pd.read_csv("Final_Fertility_Lifestyle_Dataset.csv")

print("Columns:", df.columns)

# ================= ENCODE FEATURES =================

# Binary
df["Smoking"] = df["Smoking"].map({"No": 0, "Yes": 1})
df["Alcohol"] = df["Alcohol"].map({"No": 0, "Yes": 1})

# Ordinal
df["Physical_Activity"] = df["Physical_Activity"].map({
    "Low": 0,
    "Moderate": 1,
    "High": 2
})

# Gender
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

# Target encoding
df["Fertility"] = df["Fertility"].map({
    "Low": 0,
    "Medium": 1,
    "High": 2
})

# ================= FEATURES & TARGET =================
X = df.drop(["ID", "Fertility"], axis=1)
y = df["Fertility"]

# ================= SCALING =================
scaler = StandardScaler()
X[["Age", "Sleep_Hours", "BMI"]] = scaler.fit_transform(
    X[["Age", "Sleep_Hours", "BMI"]]
)

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # IMPORTANT
)

# ================= RANDOM FOREST =================
rf_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

print("\n🌲 Random Forest Accuracy:", rf_acc)
print(classification_report(y_test, rf_preds))

# ================= XGBOOST =================
xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss"
)

xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)

print("\n⚡ XGBoost Accuracy:", xgb_acc)
print(classification_report(y_test, xgb_preds))

# ================= SAVE BEST MODEL =================
if xgb_acc > rf_acc:
    joblib.dump(xgb_model, "models/lifestyle_model.pkl")
    print("\n🔥 XGBoost selected and saved!")
else:
    joblib.dump(rf_model, "models/lifestyle_model.pkl")
    print("\n🔥 Random Forest selected and saved!")