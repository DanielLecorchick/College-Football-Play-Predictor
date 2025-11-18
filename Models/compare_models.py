"""
College Football Play Prediction - Model Comparison
Author: Dominic Ullmer
Purpose: Compare run/pass prediction models using ESPN 2024 play-by-play data
"""

import json
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, classification_report, confusion_matrix
from pathlib import Path
from tabulate import tabulate


def load_data(json_file="all_plays_2024.json"):
    with open(json_file, "r") as f:
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

    return df

def safe_transform(le, values):
    """Transform values using LabelEncoder, map unseen labels to 'None'"""
    safe_values = [v if v in le.classes_ else "None" for v in values]
    if "None" not in le.classes_:
        le.classes_ = np.concatenate([le.classes_, ["None"]])
    return le.transform(safe_values)

def bucket_score_diff(diff):
    if diff <= -14:
        return "Trailing big (≤ -14)"
    elif -13 <= diff <= -1:
        return "Trailing small (-13 to -1)"
    elif diff == 0:
        return "Tied"
    elif 1 <= diff <= 13:
        return "Leading small (1 to 13)"
    else:
        return "Leading big (≥ 14)"

def bucket_distance(dist):
    if dist <= 3:
        return "Short (≤ 3)"
    elif 4 <= dist <= 7:
        return "Medium (4–7)"
    elif 8 <= dist <= 15:
        return "Long (8–15)"
    else:
        return "Very long (> 15)"

def situational_accuracy(df, y_true, y_pred):
    df_eval = df.copy()
    df_eval["pred"] = y_pred

    def is_two_min(clock_str):
        try:
            mins, secs = map(int, clock_str.split(":"))
            return mins == 0 and secs <= 2
        except:
            return False

    df_eval["two_min"] = df_eval["clock"].apply(is_two_min)
    df_eval["score_bucket"] = df_eval["score_diff"].apply(bucket_score_diff)
    df_eval["distance_bucket"] = df_eval["distance"].apply(bucket_distance)

    results = {
        "down": df_eval.groupby("down").apply(lambda g: accuracy_score(g["label_run_pass"], g["pred"])).to_dict(),
        "period": df_eval.groupby("period").apply(lambda g: accuracy_score(g["label_run_pass"], g["pred"])).to_dict(),
        "score_diff": df_eval.groupby("score_bucket").apply(lambda g: accuracy_score(g["label_run_pass"], g["pred"])).to_dict(),
        "distance": df_eval.groupby("distance_bucket").apply(lambda g: accuracy_score(g["label_run_pass"], g["pred"])).to_dict(),
        "two_min": df_eval.groupby("two_min").apply(lambda g: accuracy_score(g["label_run_pass"], g["pred"])).to_dict(),
    }
    return results



def evaluate_single(name, prefix, X, y, df):
    model_file = f"run_pass_{prefix}.pkl"
    encoder_file = f"encoders_{prefix}.pkl"
    scaler_file = f"scaler_{prefix}.pkl"  # NEW: look for scaler

    if not Path(model_file).exists() or not Path(encoder_file).exists():
        print(f"Skipping {name} ({prefix}) — Missing model or encoder for prefix '{prefix}'\n")
        return None

    model = joblib.load(model_file)
    encoders = joblib.load(encoder_file)
    scaler = joblib.load(scaler_file) if Path(scaler_file).exists() else None

    X_eval = X.copy()
    cat_cols = encoders.keys()
    for col in cat_cols:
        if col in X_eval.columns:
            X_eval[col] = safe_transform(encoders[col], X_eval[col])

    # Apply scaler if present
    if scaler is not None:
        # Keep DataFrame structure so feature names are preserved
        X_eval = pd.DataFrame(scaler.transform(X_eval), columns=X_eval.columns)

    try:
        y_pred = model.predict(X_eval)
        y_pred_prob = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else None
    except Exception as e:
        print(f"Prediction error for {name}: {e}\n")
        return None

    # Confusion matrix & classification report
    cm = confusion_matrix(y, y_pred, labels=["Pass", "Run"])
    report = classification_report(y, y_pred, output_dict=True, labels=["Pass", "Run"])

    results = {
        "accuracy": round(accuracy_score(y, y_pred), 3),
        "f1_macro": round(report["macro avg"]["f1-score"], 3),
        "run_bias": round(cm[1].sum() / cm.sum(), 3),  # % predicted Run
        "situational": situational_accuracy(df, y, y_pred),
        "confusion_matrix": cm,
        "classification_report": report
    }

    if y_pred_prob is not None:
        results["cross_entropy"] = round(log_loss(y.map({"Run":0,"Pass":1}), y_pred_prob), 3)
        results["brier_score"] = round(brier_score_loss(y.map({"Run":0,"Pass":1}), y_pred_prob), 3)

    # Feature importance (tree-based or logistic regression)
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X_eval.columns)
        results["top_feature"] = importances.idxmax()
    elif hasattr(model, "coef_"):
        coef = pd.Series(model.coef_[0], index=X_eval.columns)
        results["top_feature"] = coef.abs().idxmax()
    else:
        results["top_feature"] = None

    return results



def print_evaluation(name, res):
    rows = [["Accuracy", res["accuracy"]],
            ["F1 (macro)", res["f1_macro"]],
            ["Run Bias", res["run_bias"]]]
    if "cross_entropy" in res:
        rows.append(["Cross-Entropy", res["cross_entropy"]])
        rows.append(["Brier Score", res["brier_score"]])
    if res["top_feature"]:
        rows.append(["Top Feature", res["top_feature"]])

    print(f"\nEvaluation Results for {name}")
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="github"))

    print("\nSituational Accuracy")
    for key, val in res["situational"].items():
        print(f"\nBy {key}:")
        print(tabulate(val.items(), headers=[key, "Accuracy"], tablefmt="github"))

def print_leaderboard(models_results):
    print("\nModel Leaderboard")
    rows = []
    for name, res in models_results.items():
        rows.append([
            name,
            res["accuracy"],
            res["f1_macro"],
            res["run_bias"],
            res["top_feature"] if res["top_feature"] else "-"
        ])
    rows.sort(key=lambda x: x[1], reverse=True)
    print(tabulate(
        [(i+1, *row) for i, row in enumerate(rows)],
        headers=["Rank", "Model", "Accuracy", "F1 (macro)", "Run Bias", "Top Feature"],
        tablefmt="github"
    ))

def collect_situational_results(models_results):
    situational_leaderboards = {}
    for model_name, res in models_results.items():
        for category, values in res["situational"].items():
            if category not in situational_leaderboards:
                situational_leaderboards[category] = {}
            for bucket, acc in values.items():
                situational_leaderboards[category].setdefault(bucket, [])
                situational_leaderboards[category][bucket].append((model_name, acc))
    return situational_leaderboards

def print_situational_leaderboards(situational_leaderboards):
    print("\nSituational Leaderboards")
    for category, buckets in situational_leaderboards.items():
        print(f"\n=== {category.upper()} ===")
        for bucket, results in buckets.items():
            results.sort(key=lambda x: x[1], reverse=True)
            print(f"\n{bucket}:")
            print(tabulate(
                [(i+1, name, acc) for i, (name, acc) in enumerate(results)],
                headers=["Rank", "Model", "Accuracy"],
                tablefmt="github"
            ))

def main():
    df = load_data()
    print(f"Data loaded: {len(df)} plays\n")

    feature_cols = [
        "down", "distance", "yard_line", "period", "score_diff",
        "prev1_play_type", "prev2_play_type", "prev3_play_type",
        "prev1_yards", "prev2_yards", "prev3_yards", "prev1_distance"
    ]
    X = df[feature_cols]
    y = df["label_run_pass"]

    models = [
        ("Random Forest", "rf"),
        ("KNN", "knn"),
        ("Naive Bayes", "nb"),
        ("Logistic Regression", "logreg"),
        ("Gradient Boosting", "gb"),
        ("Conditional Inference", "ctree")
    ]

    models_results = {}

    for name, prefix in models:
        print(f"\nEvaluating model: {name} (prefix='{prefix}')")
        res = evaluate_single(name, prefix, X, y, df)
        if res is None:
            continue
        models_results[name] = res
        print_evaluation(name, res)

    if models_results:
        print_leaderboard(models_results)
        situational_leaderboards = collect_situational_results(models_results)
        print_situational_leaderboards(situational_leaderboards)
    else:
        print("No models evaluated successfully.")


if __name__ == "__main__":
    main()
