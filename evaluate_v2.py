"""
=============================================================================
PHASE 4 — POST-TRAINING EVALUATION & THRESHOLD OPTIMISATION
Edge-Intelligent Hybrid DDoS Detection System for 5G Networks
=============================================================================

Loads the saved models from models_v4 and produces:
  1. Per-class XGBoost classification report (precision, recall, F1)
  2. Fusion score threshold scan (Binary Acc, DR/TPR, FPR at 0.05 steps)
  3. Optimal threshold recommendation (best F1-beta, beta=0.5 to favour precision)
  4. Normalised confusion matrix saved as confusion_matrix_v4.png
  5. Full evaluation report saved as evaluate_v2_report.txt

Run after preprocess_phase1.py has completed successfully.
=============================================================================
"""

import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_DIR   = r"D:\DDos 1\models_v4"
TEST_PARQUET = r"D:\DDos 1\test.parquet"
REPORT_FILE = r"D:\DDos 1\evaluate_v2_report.txt"
CM_FILE     = r"D:\DDos 1\confusion_matrix_v4.png"
EPS         = 1e-9

# DPI weight (fixed per fusion formula)
W_IF  = 0.4
W_XGB = 0.4
W_DPI = 0.2

report_lines = []

def log(msg=""):
    report_lines.append(str(msg))
    print(msg)

# =============================================================================
# DPI SIGNATURE ENGINE
# =============================================================================
def evaluate_dpi(df):
    score = pd.Series(0.0, index=df.index, dtype=np.float32)

    # High-volume SYN flood
    score = np.where(
        (df['syn_count'] > 50) & (df['Rate'] > 5000) & (df['TCP'] > 0.9),
        np.maximum(score, 0.95), score)

    # TCP flood (pure ACK/data with high rate)
    score = np.where(
        (df['syn_count'] < 5) & (df['Rate'] > 8000) & (df['TCP'] > 0.9),
        np.maximum(score, 0.85), score)

    # UDP flood
    score = np.where(
        (df['UDP'] > 0.8) & (df['Rate'] > 5000),
        np.maximum(score, 0.90), score)

    # ICMP flood
    score = np.where(
        (df['ICMP'] > 0.5) & (df['Rate'] > 3000),
        np.maximum(score, 0.88), score)

    # Recon: low Number, moderate IAT
    score = np.where(
        (df['Number'] < 15) & (df['IAT_mean'] > 0.001),
        np.maximum(score, 0.70), score)

    # BENIGN-like: very low rate, normal IAT
    score = np.where(
        (df['Rate'] < 200) & (df['IAT_mean'] > 0.01),
        np.minimum(score, 0.1), score)

    return np.clip(score, 0.0, 1.0).astype(np.float32)

# =============================================================================
# LOAD MODELS
# =============================================================================
def load_models():
    log("Loading models from disk...")
    le     = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    iso_d  = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
    xgb    = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
    iso    = iso_d['model']
    if_min = iso_d['if_min']
    if_max = iso_d['if_max']
    log(f"  LabelEncoder  : {len(le.classes_)} classes")
    log(f"  StandardScaler: {len(scaler.feature_names_in_)} features")
    log(f"  IsolationForest: if_min={if_min:.4f} if_max={if_max:.4f}")
    log(f"  XGBoost best iteration: {getattr(xgb, 'best_iteration', 'N/A')}")
    return le, scaler, iso, if_min, if_max, xgb

# =============================================================================
# FEATURE PREPARATION
# =============================================================================
def prepare_features(df, scaler, le):
    X = df.drop(columns=['Label'])
    y = df['Label']
    y_enc = le.transform(y)

    # Scale
    present = [c for c in scaler.feature_names_in_ if c in X.columns]
    X_scaled = X.copy()
    X_scaled[present] = scaler.transform(X[present]).astype(np.float32)

    return X, X_scaled, y, y_enc

# =============================================================================
# THRESHOLD SCAN
# =============================================================================
def threshold_scan(fusion_scores, y_binary, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)

    results = []
    for t in thresholds:
        preds = (fusion_scores >= t).astype(int)
        tp = ((preds == 1) & (y_binary == 1)).sum()
        fp = ((preds == 1) & (y_binary == 0)).sum()
        tn = ((preds == 0) & (y_binary == 0)).sum()
        fn = ((preds == 0) & (y_binary == 1)).sum()
        dr  = tp / (tp + fn + EPS)
        fpr = fp / (fp + tn + EPS)
        acc = (tp + tn) / (len(y_binary) + EPS)
        # F-beta with beta=0.5 (favour precision — fewer false alarms in MEC)
        prec = tp / (tp + fp + EPS)
        fbeta = (1 + 0.5**2) * prec * dr / (0.5**2 * prec + dr + EPS)
        results.append({
            'threshold': round(t, 2),
            'binary_acc': acc * 100,
            'dr_tpr':    dr  * 100,
            'fpr':       fpr * 100,
            'precision': prec * 100,
            'fbeta_05':  fbeta * 100,
        })
    return pd.DataFrame(results)

# =============================================================================
# MAIN EVALUATION
# =============================================================================
def run():
    log("=" * 70)
    log("PHASE 4 — POST-TRAINING EVALUATION v2")
    log("=" * 70)

    # 1. Load
    le, scaler, iso, if_min, if_max, xgb = load_models()

    # 2. Test data
    log(f"\nLoading test dataset: {TEST_PARQUET}")
    test_df = pd.read_parquet(TEST_PARQUET, engine='pyarrow')
    log(f"  Shape: {test_df.shape[0]:,} rows x {test_df.shape[1]} columns")

    X_test, X_test_scaled, y_test, y_test_enc = prepare_features(test_df, scaler, le)

    # 3. XGBoost multi-class predictions
    log("\n[Step 1] XGBoost Multi-Class Classification...")
    xgb_proba = xgb.predict_proba(X_test.values)          # shape (N, n_classes)
    y_pred_enc = xgb_proba.argmax(axis=1)
    y_pred     = le.inverse_transform(y_pred_enc)

    xgb_acc = accuracy_score(y_test, y_pred)
    xgb_f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    log(f"  Multi-class Accuracy : {xgb_acc * 100:.4f}%")
    log(f"  Weighted F1-Score    : {xgb_f1:.4f}")

    # Detailed per-class report
    log("\n  Per-Class Classification Report:")
    report = classification_report(y_test, y_pred, zero_division=0, digits=4)
    for line in report.split('\n'):
        log(f"    {line}")

    # 4. Isolation Forest anomaly score
    log("\n[Step 2] Isolation Forest Anomaly Scoring...")
    ifs = iso.score_samples(X_test_scaled)                 # full scaled feature matrix
    ifs_clipped = np.clip(ifs, if_min, if_max)
    if_prob = 1.0 - (ifs_clipped - if_min) / (if_max - if_min + EPS)
    log(f"  IF score range: {ifs.min():.4f} to {ifs.max():.4f}")
    log(f"  IF prob range : {if_prob.min():.4f} to {if_prob.max():.4f}")

    # 5. DPI score
    log("\n[Step 3] DPI Signature Scoring...")
    dpi_scores = evaluate_dpi(X_test)
    log(f"  DPI mean score (all) : {dpi_scores.mean():.4f}")

    # 6. XGBoost attack probability (1 - benign class prob)
    benign_idx  = list(le.classes_).index('BENIGN')
    xgb_att_prob = 1.0 - xgb_proba[:, benign_idx]

    # 7. Score fusion
    log("\n[Step 4] Score Fusion: 0.4*IF + 0.4*XGB + 0.2*DPI")
    fusion = (W_IF  * if_prob +
              W_XGB * xgb_att_prob +
              W_DPI * dpi_scores)
    fusion = np.clip(fusion, 0.0, 1.0)

    # Binary ground truth: BENIGN=0, any attack=1
    y_binary = (y_test != 'BENIGN').astype(int).values

    # 8. Threshold scan
    log("\n[Step 5] Threshold Scan (Binary Detection):")
    log(f"  {'Threshold':>10} | {'Acc':>9} | {'DR/TPR':>9} | {'FPR':>8} | {'Precision':>10} | {'F-beta(0.5)':>12}")
    log(f"  {'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}-+-{'-'*10}-+-{'-'*12}")
    scan_df = threshold_scan(fusion, y_binary)
    for _, row in scan_df.iterrows():
        marker = " <-- OPTIMAL" if row['fbeta_05'] == scan_df['fbeta_05'].max() else ""
        log(f"  {row['threshold']:>10.2f} | {row['binary_acc']:>8.2f}% | "
            f"{row['dr_tpr']:>8.2f}% | {row['fpr']:>7.2f}% | "
            f"{row['precision']:>9.2f}% | {row['fbeta_05']:>11.2f}%{marker}")

    best = scan_df.loc[scan_df['fbeta_05'].idxmax()]
    log(f"\n  RECOMMENDED THRESHOLD : {best['threshold']:.2f}")
    log(f"    Binary Accuracy     : {best['binary_acc']:.4f}%")
    log(f"    Detection Rate (DR) : {best['dr_tpr']:.4f}%")
    log(f"    False Positive Rate : {best['fpr']:.4f}%")
    log(f"    Precision           : {best['precision']:.4f}%")

    # 9. Confusion matrix (multi-class, normalised)
    log("\n[Step 6] Generating Normalised Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred, labels=le.classes_, normalize='true')
    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_,
                linewidths=0.3, ax=ax, annot_kws={'size': 6})
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title(
        f'Normalised Confusion Matrix — XGBoost v2\n'
        f'Accuracy: {xgb_acc*100:.2f}%  |  Weighted F1: {xgb_f1:.4f}',
        fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(CM_FILE, dpi=150)
    plt.close()
    log(f"  Saved confusion matrix -> {CM_FILE}")

    # 10. Summary
    log(f"\n{'='*70}")
    log("EVALUATION SUMMARY")
    log(f"{'='*70}")
    log(f"  XGBoost Multi-Class Accuracy : {xgb_acc * 100:.4f}%")
    log(f"  XGBoost Weighted F1-Score    : {xgb_f1:.4f}")
    log(f"  Recommended Fusion Threshold : {best['threshold']:.2f}")
    log(f"  Fusion Detection Rate (DR)   : {best['dr_tpr']:.4f}%")
    log(f"  Fusion False Positive Rate   : {best['fpr']:.4f}%")
    log(f"  Fusion Binary Accuracy       : {best['binary_acc']:.4f}%")

    # Save report
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    log(f"\n  Report saved -> {REPORT_FILE}")


if __name__ == '__main__':
    run()
