"""
College Football Play Prediction - Conditional Inference Forest Model
Author: Dominic Ullmer
Purpose: Predict Run vs Pass using ESPN 2024 play-by-play data
Using: Mondrian Forest (Conditional Inference Forest approximation)
"""
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skgarden.mondrian import MondrianForestClassifier
import joblib

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
feature_cols = [
    "down", "distance", "yard_line", "period", "score_diff",
    "prev1_play_type", "prev2_play_type", "prev3_play_type",
    "prev1_yards", "prev2_yards", "prev3_yards", "prev1_distance"
]
X = df[feature_cols]
y = df["label_run_pass"]
cat_cols = ["prev1_play_type", "prev2_play_type", "prev3_play_type"]
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
ci_model = MondrianForestClassifier(
    n_estimators=300,
    max_depth=None,     
    random_state=42
)
ci_model.fit(X_train, y_train)
y_pred = ci_model.predict(X_test)
print("\n🏈 Conditional Inference Forest Results 🏈")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
importances = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": ci_model.feature_importances_
}).sort_values("Importance", ascending=False)
print("\nFeature Importances:\n", importances)
joblib.dump(ci_model, "run_pass_ci_forest.pkl")
joblib.dump(label_encoders, "encoders_ci_forest.pkl")
print("\n✅ Saved Conditional Inference Forest model to 'run_pass_ci_forest.pkl'")
