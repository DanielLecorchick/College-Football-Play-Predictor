"""
College Football Play Prediction - Gradient Boosting Model
Author: Daniel Lecorchick
Purpose: Predict Run vs Pass using ESPN 2024 play-by-play data using Gradient Boosting Model
"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

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
print(f"Loaded {len(df)} total plays")

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
    df[col].fillna(df[col].median(), inplace=True)

for col in ["prev1_play_type", "prev2_play_type", "prev3_play_type"]:
    df[col].fillna("None", inplace=True)

# === 3. Select features and labels ===
feature_cols = [
    "down", "distance", "yard_line", "period", "score_diff",
    "prev1_play_type", "prev2_play_type", "prev3_play_type",
    "prev1_yards", "prev2_yards", "prev3_yards", "prev1_distance"
]
X = df[feature_cols]
y = df["label_run_pass"]

# === 4. Encode categorical columns ===
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

# === 6. Gradient Boosting model ===
gb_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

gb_model.fit(X_train, y_train)

# === 7. Evaluate ===
y_pred = gb_model.predict(X_test)

print("\nGradient Boosting Results")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === 8. Feature Importance ===
importances = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": gb_model.feature_importances_
}).sort_values("Importance", ascending=False)
print("\nFeature Importances:\n", importances)

# === 9. Save model ===
joblib.dump(gb_model, "run_pass_gb.pkl")
joblib.dump(label_encoders, "encoders_gb.pkl")
print("\nSaved Gradient Boosting model to 'run_pass_gb.pkl'")
