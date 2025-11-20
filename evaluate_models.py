"""
College Football Play Prediction - Model Evaluation & Comparison
Purpose: Load and evaluate all trained models, comparing their performance
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Try to import R packages for Conditional Inference Forest
try:
    raise ImportError
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False
    print("Warning: rpy2 not available. Conditional Inference Forest will be skipped.")

# Model directory
MODEL_DIR = "Model_PKL"

def load_and_preprocess_data(input_file="all_plays_2024.json"):
    """Load and preprocess the play-by-play data"""
    print("Loading data...")
    with open(input_file, "r") as f:
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

    # Clean and preprocess
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

    # Feature selection
    feature_cols = [
        "down", "distance", "yard_line", "period", "score_diff",
        "prev1_play_type", "prev2_play_type", "prev3_play_type",
        "prev1_yards", "prev2_yards", "prev3_yards", "prev1_distance"
    ]
    X = df[feature_cols].copy()
    y = df["label_run_pass"]

    return X, y, feature_cols, df

def evaluate_model(name, y_test, y_pred, y_pred_proba=None):
    """Evaluate a single model and return metrics"""
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    # Calculate per-class metrics
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    metrics = {
        'Model': name,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4),
        'Pass_Precision': round(class_report.get('Pass', {}).get('precision', 0), 4),
        'Pass_Recall': round(class_report.get('Pass', {}).get('recall', 0), 4),
        'Pass_F1': round(class_report.get('Pass', {}).get('f1-score', 0), 4),
        'Run_Precision': round(class_report.get('Run', {}).get('precision', 0), 4),
        'Run_Recall': round(class_report.get('Run', {}).get('recall', 0), 4),
        'Run_F1': round(class_report.get('Run', {}).get('f1-score', 0), 4),
    }
    
    # Calculate AUC if probabilities available
    if y_pred_proba is not None:
        try:
            # Convert labels to binary
            y_test_binary = (y_test == 'Pass').astype(int)
            if len(np.unique(y_test_binary)) == 2:  # Need both classes
                auc = roc_auc_score(y_test_binary, y_pred_proba)
                metrics['AUC'] = round(auc, 4)
            else:
                metrics['AUC'] = None
        except:
            metrics['AUC'] = None
    else:
        metrics['AUC'] = None
    
    return metrics

def evaluate_situational_performance(models_dict, test_df, X_test, y_test):
    """Evaluate model performance in different game situations"""
    print("\n" + "="*70)
    print("SITUATIONAL PERFORMANCE ANALYSIS")
    print("="*70)
    
    situations = []
    
    # 1. By Down
    print("\nPerformance by Down:")
    for down in sorted(test_df['down'].unique()):
        mask = test_df['down'] == down
        if mask.sum() > 0:
            situation = {'Situation': f'Down {int(down)}', 'Count': mask.sum()}
            for name, model_data in models_dict.items():
                y_pred_situation = model_data['predictions'][mask]
                y_test_situation = y_test[mask]
                acc = accuracy_score(y_test_situation, y_pred_situation)
                situation[name] = round(acc, 4)
            situations.append(situation)
            print(f"  {situation['Situation']}: {situation['Count']} plays")
    
    # 2. By Distance (Short, Medium, Long)
    print("\nPerformance by Distance:")
    distance_bins = [
        ('Short (1-3 yds)', 1, 3),
        ('Medium (4-7 yds)', 4, 7),
        ('Long (8+ yds)', 8, 30)
    ]
    for label, min_dist, max_dist in distance_bins:
        mask = (test_df['distance'] >= min_dist) & (test_df['distance'] <= max_dist)
        if mask.sum() > 0:
            situation = {'Situation': label, 'Count': mask.sum()}
            for name, model_data in models_dict.items():
                y_pred_situation = model_data['predictions'][mask]
                y_test_situation = y_test[mask]
                acc = accuracy_score(y_test_situation, y_pred_situation)
                situation[name] = round(acc, 4)
            situations.append(situation)
            print(f"  {label}: {situation['Count']} plays")
    
    # 3. By Field Position
    print("\nPerformance by Field Position:")
    field_zones = [
        ('Own Territory (0-50)', 0, 50),
        ('Opponent Territory (51-100)', 51, 100)
    ]
    for label, min_yard, max_yard in field_zones:
        mask = (test_df['yard_line'] >= min_yard) & (test_df['yard_line'] <= max_yard)
        if mask.sum() > 0:
            situation = {'Situation': label, 'Count': mask.sum()}
            for name, model_data in models_dict.items():
                y_pred_situation = model_data['predictions'][mask]
                y_test_situation = y_test[mask]
                acc = accuracy_score(y_test_situation, y_pred_situation)
                situation[name] = round(acc, 4)
            situations.append(situation)
            print(f"  {label}: {situation['Count']} plays")
    
    # 4. By Period (Quarter)
    print("\nPerformance by Period:")
    for period in sorted(test_df['period'].unique()):
        mask = test_df['period'] == period
        if mask.sum() > 0:
            situation = {'Situation': f'Period {int(period)}', 'Count': mask.sum()}
            for name, model_data in models_dict.items():
                y_pred_situation = model_data['predictions'][mask]
                y_test_situation = y_test[mask]
                acc = accuracy_score(y_test_situation, y_pred_situation)
                situation[name] = round(acc, 4)
            situations.append(situation)
            print(f"  {situation['Situation']}: {situation['Count']} plays")
    
    # 5. By Score Differential
    print("\nPerformance by Score Differential:")
    score_bins = [
        ('Losing by 14+', -100, -14),
        ('Losing by 1-13', -13, -1),
        ('Tied', 0, 0),
        ('Winning by 1-13', 1, 13),
        ('Winning by 14+', 14, 100)
    ]
    for label, min_score, max_score in score_bins:
        mask = (test_df['score_diff'] >= min_score) & (test_df['score_diff'] <= max_score)
        if mask.sum() > 0:
            situation = {'Situation': label, 'Count': mask.sum()}
            for name, model_data in models_dict.items():
                y_pred_situation = model_data['predictions'][mask]
                y_test_situation = y_test[mask]
                acc = accuracy_score(y_test_situation, y_pred_situation)
                situation[name] = round(acc, 4)
            situations.append(situation)
            print(f"  {label}: {situation['Count']} plays")
    
    # 6. Critical Situations
    print("\nPerformance in Critical Situations:")
    
    # Third/Fourth down conversions
    mask = (test_df['down'] >= 3) & (test_df['distance'] <= 10)
    if mask.sum() > 0:
        situation = {'Situation': '3rd/4th & Short', 'Count': mask.sum()}
        for name, model_data in models_dict.items():
            y_pred_situation = model_data['predictions'][mask]
            y_test_situation = y_test[mask]
            acc = accuracy_score(y_test_situation, y_pred_situation)
            situation[name] = round(acc, 4)
        situations.append(situation)
        print(f"  3rd/4th & Short: {situation['Count']} plays")
    
    # Red Zone
    mask = (test_df['yard_line'] >= 80)
    if mask.sum() > 0:
        situation = {'Situation': 'Red Zone', 'Count': mask.sum()}
        for name, model_data in models_dict.items():
            y_pred_situation = model_data['predictions'][mask]
            y_test_situation = y_test[mask]
            acc = accuracy_score(y_test_situation, y_pred_situation)
            situation[name] = round(acc, 4)
        situations.append(situation)
        print(f"  Red Zone: {situation['Count']} plays")
    
    # Goal Line (within 5 yards)
    mask = (test_df['yard_line'] >= 95)
    if mask.sum() > 0:
        situation = {'Situation': 'Goal Line', 'Count': mask.sum()}
        for name, model_data in models_dict.items():
            y_pred_situation = model_data['predictions'][mask]
            y_test_situation = y_test[mask]
            acc = accuracy_score(y_test_situation, y_pred_situation)
            situation[name] = round(acc, 4)
        situations.append(situation)
        print(f"  Goal Line: {situation['Count']} plays")
    
    return pd.DataFrame(situations)

def plot_situational_performance(sit_df, output_dir="evaluation_results"):
    """Plot situational performance analysis"""
    Path(output_dir).mkdir(exist_ok=True)
    
    model_cols = [col for col in sit_df.columns if col not in ['Situation', 'Count']]
    
    # Create separate plots for each category
    categories = {
        'Down': sit_df[sit_df['Situation'].str.contains('Down')],
        'Distance': sit_df[sit_df['Situation'].str.contains('yds')],
        'Field Position': sit_df[sit_df['Situation'].str.contains('Territory')],
        'Period': sit_df[sit_df['Situation'].str.contains('Period')],
        'Score Differential': sit_df[sit_df['Situation'].str.contains('Losing|Winning|Tied')],
        'Critical Situations': sit_df[sit_df['Situation'].str.contains('Red Zone|Goal Line|3rd')]
    }
    
    n_categories = len([cat for cat in categories.values() if len(cat) > 0])
    fig, axes = plt.subplots(n_categories, 1, figsize=(14, 4*n_categories))
    
    if n_categories == 1:
        axes = [axes]
    
    ax_idx = 0
    for cat_name, cat_data in categories.items():
        if len(cat_data) == 0:
            continue
            
        ax = axes[ax_idx]
        
        # Prepare data for grouped bar chart
        x = np.arange(len(cat_data))
        width = 0.8 / len(model_cols)
        
        for i, model in enumerate(model_cols):
            offset = (i - len(model_cols)/2) * width + width/2
            bars = ax.bar(x + offset, cat_data[model], width, label=model)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Situation')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Model Performance: {cat_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(cat_data['Situation'], rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_ylim([0.4, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        ax_idx += 1
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/situational_performance.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved situational performance to {output_dir}/situational_performance.png")
    plt.close()
    
    # Create heatmap of all situations
    fig, ax = plt.subplots(figsize=(12, len(sit_df) * 0.5))
    
    heatmap_data = sit_df[model_cols].values
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                xticklabels=model_cols, yticklabels=sit_df['Situation'],
                vmin=0.4, vmax=1.0, ax=ax, cbar_kws={'label': 'Accuracy'})
    
    ax.set_title('Model Performance Across All Situations')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/situational_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved situational heatmap to {output_dir}/situational_heatmap.png")
    plt.close()

def plot_confusion_matrices(results, output_dir="evaluation_results"):
    """Plot confusion matrices for all models"""
    Path(output_dir).mkdir(exist_ok=True)
    
    n_models = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(results.items()):
        if idx >= len(axes):
            break
            
        cm = data['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Pass', 'Run'], yticklabels=['Pass', 'Run'])
        axes[idx].set_title(f'{name}\nAccuracy: {data["metrics"]["Accuracy"]:.3f}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrices to {output_dir}/confusion_matrices.png")
    plt.close()

def plot_metrics_comparison(metrics_df, output_dir="evaluation_results"):
    """Plot comparison of key metrics across models"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Plot overall metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        data = metrics_df.sort_values(metric, ascending=True)
        ax.barh(data['Model'], data[metric], color='steelblue')
        ax.set_xlabel(metric)
        ax.set_xlim([0.4, 1.0])
        ax.set_title(f'{metric} Comparison')
        
        # Add value labels
        for i, v in enumerate(data[metric]):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved metrics comparison to {output_dir}/metrics_comparison.png")
    plt.close()
    
    # Plot per-class metrics
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    class_metrics = [
        ('Pass_Precision', 'Pass Precision'),
        ('Pass_Recall', 'Pass Recall'),
        ('Pass_F1', 'Pass F1-Score'),
        ('Run_Precision', 'Run Precision'),
        ('Run_Recall', 'Run Recall'),
        ('Run_F1', 'Run F1-Score')
    ]
    
    for idx, (metric, title) in enumerate(class_metrics):
        ax = axes[idx // 3, idx % 3]
        data = metrics_df.sort_values(metric, ascending=True)
        ax.barh(data['Model'], data[metric], color='coral')
        ax.set_xlabel(title)
        ax.set_xlim([0.4, 1.0])
        ax.set_title(title)
        
        # Add value labels
        for i, v in enumerate(data[metric]):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/class_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved class metrics comparison to {output_dir}/class_metrics_comparison.png")
    plt.close()

def main():
    """Main evaluation function"""
    print("="*70)
    print("COLLEGE FOOTBALL PLAY PREDICTION - MODEL EVALUATION")
    print("="*70)
    
    # Load and preprocess data
    X, y, feature_cols, full_df = load_and_preprocess_data()
    
    # Prepare encoders for categorical features
    cat_cols = ["prev1_play_type", "prev2_play_type", "prev3_play_type"]
    X_encoded = X.copy()
    label_encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        label_encoders[col] = le
    
    # Train/test split (same as training scripts)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Get corresponding full dataframe rows for situational analysis
    test_indices = X_test.index
    test_df = full_df.loc[test_indices].copy()
    y_test = y_test.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    results = {}
    
    # 1. K-Nearest Neighbors
    print("\n" + "="*70)
    print("Evaluating K-Nearest Neighbors...")
    print("="*70)
    try:
        knn_model = joblib.load(f"{MODEL_DIR}/run_pass_knn.pkl")
        scaler_knn = joblib.load(f"{MODEL_DIR}/scaler_knn.pkl")
        X_test_scaled = scaler_knn.transform(X_test)
        y_pred = knn_model.predict(X_test_scaled)
        y_pred_proba = knn_model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = evaluate_model("KNN", y_test, y_pred, y_pred_proba)
        results['KNN'] = {
            'metrics': metrics,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'model': knn_model,
            'scaler': scaler_knn
        }
        print(f"Accuracy: {metrics['Accuracy']}")
        print(classification_report(y_test, y_pred))
    except FileNotFoundError as e:
        print(f"KNN model not found: {e}")
    
    # 2. Random Forest
    print("\n" + "="*70)
    print("Evaluating Random Forest...")
    print("="*70)
    try:
        rf_model = joblib.load(f"{MODEL_DIR}/run_pass_rf.pkl")
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_model("Random Forest", y_test, y_pred, y_pred_proba)
        results['Random Forest'] = {
            'metrics': metrics,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'model': rf_model
        }
        print(f"Accuracy: {metrics['Accuracy']}")
        print(classification_report(y_test, y_pred))
    except FileNotFoundError as e:
        print(f"Random Forest model not found: {e}")
    
    # 3. Gradient Boosting
    print("\n" + "="*70)
    print("Evaluating Gradient Boosting...")
    print("="*70)
    try:
        gb_model = joblib.load(f"{MODEL_DIR}/run_pass_gb.pkl")
        y_pred = gb_model.predict(X_test)
        y_pred_proba = gb_model.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_model("Gradient Boosting", y_test, y_pred, y_pred_proba)
        results['Gradient Boosting'] = {
            'metrics': metrics,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'model': gb_model
        }
        print(f"Accuracy: {metrics['Accuracy']}")
        print(classification_report(y_test, y_pred))
    except FileNotFoundError as e:
        print(f"Gradient Boosting model not found: {e}")
    
    # 4. Logistic Regression
    print("\n" + "="*70)
    print("Evaluating Logistic Regression...")
    print("="*70)
    try:
        logreg_model = joblib.load(f"{MODEL_DIR}/run_pass_logreg.pkl")
        scaler_logreg = joblib.load(f"{MODEL_DIR}/scaler_logreg.pkl")
        X_test_scaled = scaler_logreg.transform(X_test)
        y_pred = logreg_model.predict(X_test_scaled)
        y_pred_proba = logreg_model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = evaluate_model("Logistic Regression", y_test, y_pred, y_pred_proba)
        results['Logistic Regression'] = {
            'metrics': metrics,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'model': logreg_model,
            'scaler': scaler_logreg
        }
        print(f"Accuracy: {metrics['Accuracy']}")
        print(classification_report(y_test, y_pred))
    except FileNotFoundError as e:
        print(f"Logistic Regression model not found: {e}")
    
    # 5. Naive Bayes
    print("\n" + "="*70)
    print("Evaluating Naive Bayes...")
    print("="*70)
    try:
        nb_model = joblib.load(f"{MODEL_DIR}/run_pass_nb.pkl")
        scaler_nb = joblib.load(f"{MODEL_DIR}/scaler_nb.pkl")
        X_test_scaled = scaler_nb.transform(X_test)
        y_pred = nb_model.predict(X_test_scaled)
        y_pred_proba = nb_model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = evaluate_model("Naive Bayes", y_test, y_pred, y_pred_proba)
        results['Naive Bayes'] = {
            'metrics': metrics,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'model': nb_model,
            'scaler': scaler_nb
        }
        print(f"Accuracy: {metrics['Accuracy']}")
        print(classification_report(y_test, y_pred))
    except FileNotFoundError as e:
        print(f"Naive Bayes model not found: {e}")
    
    # 6. Conditional Inference Forest
    print("\n" + "="*70)
    print("Evaluating Conditional Inference Forest...")
    print("="*70)
    if R_AVAILABLE:
        try:
            ro.r(f'load("{MODEL_DIR}/run_pass_cif.RData")')
            
            test_data = X_test.copy()
            test_data['label_run_pass'] = y_test.values
            
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_test_data = ro.conversion.py2rpy(test_data)
            
            ro.r.assign('test_data', r_test_data)
            ro.r('test_data$label_run_pass <- as.factor(test_data$label_run_pass)')
            
            r_predictions = ro.r('as.character(predict(cif_model, newdata=test_data, type="response"))')
            y_pred = np.array(r_predictions)
            
            metrics = evaluate_model("Conditional Inference Forest", y_test, y_pred)
            results['Conditional Inference Forest'] = {
                'metrics': metrics,
                'predictions': y_pred,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            print(f"Accuracy: {metrics['Accuracy']}")
            print(classification_report(y_test, y_pred))
        except Exception as e:
            print(f"Conditional Inference Forest error: {e}")
    else:
        print("rpy2 not available. Skipping Conditional Inference Forest...")
    
    # Create summary comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    
    if results:
        metrics_df = pd.DataFrame([r['metrics'] for r in results.values()])
        metrics_df = metrics_df.sort_values('Accuracy', ascending=False)
        
        print("\nOverall Performance:")
        print(metrics_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']].to_string(index=False))
        
        print("\n\nPer-Class Performance (Pass):")
        print(metrics_df[['Model', 'Pass_Precision', 'Pass_Recall', 'Pass_F1']].to_string(index=False))
        
        print("\n\nPer-Class Performance (Run):")
        print(metrics_df[['Model', 'Run_Precision', 'Run_Recall', 'Run_F1']].to_string(index=False))
        
        # Save results
        output_dir = "evaluation_results"
        Path(output_dir).mkdir(exist_ok=True)
        metrics_df.to_csv(f'{output_dir}/model_comparison.csv', index=False)
        print(f"\nSaved comparison table to {output_dir}/model_comparison.csv")
        
        # Generate visualizations
        plot_confusion_matrices(results, output_dir)
        plot_metrics_comparison(metrics_df, output_dir)
        
        # Situational analysis
        situational_df = evaluate_situational_performance(results, test_df, X_test, y_test)
        situational_df.to_csv(f'{output_dir}/situational_performance.csv', index=False)
        print(f"\nSaved situational performance to {output_dir}/situational_performance.csv")
        
        plot_situational_performance(situational_df, output_dir)
        
        # Identify best models overall and by situation
        print("\n" + "="*70)
        print("BEST PERFORMING MODELS")
        print("="*70)
        
        best_overall = metrics_df.iloc[0]['Model']
        best_accuracy = metrics_df.iloc[0]['Accuracy']
        print(f"\nBest Overall: {best_overall} (Accuracy: {best_accuracy:.4f})")
        
        # Find best model for each situation
        print("\nBest Models by Situation:")
        model_cols = [col for col in situational_df.columns if col not in ['Situation', 'Count']]
        for _, row in situational_df.iterrows():
            best_model = max(model_cols, key=lambda x: row[x])
            best_acc = row[best_model]
            print(f"  {row['Situation']:30s}: {best_model:25s} ({best_acc:.4f})")
        
    else:
        print("No models were successfully evaluated.")

if __name__ == "__main__":
    main()