"""
College Football Play Prediction - K-Nearest Neighbors Model
Author: Trenton Ottman
Purpose: Predict Run vs Pass using ESPN 2024 play-by-play data
"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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
    df[col].fillna(df[col].median(), inplace=True)

for col in ["prev1_play_type", "prev2_play_type", "prev3_play_type"]:
    df[col].fillna("None", inplace=True)

# === 3. Feature selection ===
feature_cols = [
    "down", "distance", "yard_line", "period", "score_diff",
    "prev1_play_type", "prev2_play_type", "prev3_play_type",
    "prev1_yards", "prev2_yards", "prev3_yards", "prev1_distance"
]
X = df[feature_cols]
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

# === 6. Scale features ===
# KNN is distance-based and requires feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === 7. Train K-Nearest Neighbors ===
knn_model = KNeighborsClassifier(
    n_neighbors=249,
    weights='distance',  # Weight by inverse distance
    algorithm='auto',
    metric='minkowski',
    p=2  # Euclidean distance
)
knn_model.fit(X_train, y_train)

# === 8. Evaluate ===
y_pred = knn_model.predict(X_test)

print("\n🏈 K-Nearest Neighbors Results 🏈")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === 9. Save model ===
joblib.dump(knn_model, "run_pass_knn.pkl")
joblib.dump(scaler, "scaler_knn.pkl")
joblib.dump(label_encoders, "encoders_knn.pkl")
print("\n✅ Saved K-Nearest Neighbors model to 'run_pass_knn.pkl'")