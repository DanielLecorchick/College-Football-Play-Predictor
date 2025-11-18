"""
College Football Play Prediction - Conditional Inference Forest Model
Author: Dominic Ullmer
Purpose: Predict Run vs Pass using ESPN 2024 play-by-play data
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

# === 1. Load and flatten JSON ===
INPUT_FILE = "all_plays_2024.json"
with open(INPUT_FILE, "r") as f:
    raw_data = json.load(f)

plays = []
for season, games in raw_data.items():
    for gid, gdata in games.items():
        for p in gdata.get("plays", []):
            p["game_id"] = gid
            p["home_team"] = gdata["home_team"]
            p["away_team"] = gdata["away_team"]
            plays.append(p)

df = pd.DataFrame(plays)
print(f"✅ Loaded {len(df)} total plays")

# === 2. Clean and preprocess ===
df = df.dropna(subset=["label_run_pass", "down", "distance", "yard_line"])
df = df[df["down"].between(1, 4)]
df = df[df["distance"] <= 30]

num_cols = [
    "score_diff", "yards_gained", "prev1_yards", "prev2_yards",
    "prev3_yards", "prev1_distance"
]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

for col in ["prev1_play_type", "prev2_play_type", "prev3_play_type"]:
    df[col] = df[col].fillna("None")

# === 3. Feature selection ===
feature_cols = [
    "down", "distance", "yard_line", "period", "score_diff",
    "prev1_play_type", "prev2_play_type", "prev3_play_type",
    "prev1_yards", "prev2_yards", "prev3_yards", "prev1_distance"
]
X = df[feature_cols].copy()
y = df["label_run_pass"]

# === 4. Encode categorical features ===
cat_cols = ["prev1_play_type", "prev2_play_type", "prev3_play_type"]
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# === 5. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 6. Prepare data for R ===
train_data = X_train.copy()
train_data['label_run_pass'] = y_train.values

# === 7. Train Conditional Inference Forest in R ===
print("\n🏈 Training Conditional Inference Forest (via R)...")

party = importr('party')
base = importr('base')

with localconverter(ro.default_converter + pandas2ri.converter):
    r_train_data = ro.conversion.py2rpy(train_data)

ro.r.assign('train_data', r_train_data)
ro.r('train_data$label_run_pass <- as.factor(train_data$label_run_pass)')

# Train the model directly in R's global environment
ro.r('''
    library(party)
    cif_model <- cforest(label_run_pass ~ ., 
                         data=train_data,
                         controls=cforest_unbiased(ntree=300, mtry=4))
''')

print("✅ Model training complete!")

# === 8. Make predictions ===
test_data = X_test.copy()
test_data['label_run_pass'] = y_test.values

with localconverter(ro.default_converter + pandas2ri.converter):
    r_test_data = ro.conversion.py2rpy(test_data)

ro.r.assign('test_data', r_test_data)
ro.r('test_data$label_run_pass <- as.factor(test_data$label_run_pass)')

r_predictions = ro.r('as.character(predict(cif_model, newdata=test_data, type="response"))')
y_pred = np.array(r_predictions)

# === 9. Evaluate ===
print("\n🏈 Conditional Inference Forest Results 🏈")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === 10. Variable Importance ===
var_imp = party.varimp(cif_model)
importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": np.array(var_imp)
}).sort_values("Importance", ascending=False)
print("\nFeature Importances:\n", importance_df)

# === 11. Save model ===
ro.r('save(cif_model, file="run_pass_cif.RData")')
joblib.dump(label_encoders, "encoders_cif.pkl")
print("\n✅ Saved Conditional Inference Forest model to 'run_pass_cif.RData'")