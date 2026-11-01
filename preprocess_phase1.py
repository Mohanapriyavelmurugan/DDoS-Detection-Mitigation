"""
=============================================================================
PHASE 4 — MODEL TRAINING & SCORE FUSION PIPELINE
Edge-Intelligent Hybrid DDoS Detection System for 5G Networks
=============================================================================

This script implements model training and evaluation using:
  - Isolation Forest (trained on benign training rows) for anomaly detection.
  - XGBoost (trained on SMOTE-augmented data) for multi-class classification.
  - Signature-based DPI rules.
  - Score Fusion: Final Score = 0.4 * IF + 0.4 * XGBoost + 0.2 * DPI.

INPUT  : D:\\DDos 1\\balanced_train.parquet (preprocessed & SMOTE-augmented)
         D:\\DDos 1\\test.parquet           (preprocessed real rows only)
OUTPUT : D:\\DDos 1\\models_v4\\scaler.pkl
         D:\\DDos 1\\models_v4\\label_encoder.pkl
         D:\\DDos 1\\models_v4\\isolation_forest.pkl
         D:\\DDos 1\\models_v4\\xgboost_model.pkl
         D:\\DDos 1\\confusion_matrix_v4.png
         D:\\DDos 1\\phase4_report.txt
=============================================================================
"""

import os
import time
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.ensemble           import IsolationForest
from sklearn.model_selection    import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics            import (classification_report,
                                        confusion_matrix,
                                        accuracy_score, f1_score)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_PARQUET = r"D:\DDos 1\balanced_train.parquet"
TEST_PARQUET  = r"D:\DDos 1\test.parquet"
MODEL_DIR     = r"D:\DDos 1\models_v4"
REPORT_FILE   = r"D:\DDos 1\phase4_report.txt"
CM_FILE       = r"D:\DDos 1\confusion_matrix_v4.png"

EPS          = 1e-9
RANDOM_STATE = 42

# Set to True to always retrain even if saved models exist
# Must be True any time phase2_fix.py was re-run (new features or new data)
FORCE_RETRAIN = True

# Columns to be scaled for Isolation Forest & XGBoost
SCALE_COLS = [
    # Core network metrics
    'Header_Length', 'Time_To_Live', 'Rate',
    'ack_count', 'syn_count', 'fin_count', 'rst_count',
    # Protocol indicators
    'TCP', 'UDP', 'ICMP', 'ARP', 'HTTP', 'HTTPS', 'DNS', 'SSH',
    'proto_mix',
    # Size/payload stats
    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size',
    # Timing and packet stats
    'IAT', 'Number', 'Variance',
    'IAT_mean', 'IAT_per_byte', 'pkt_size_range', 'flow_rate_ratio',
    # Original 16 engineered features
    'log_ack_count', 'log_syn_count', 'log_rst_count',
    'syn_minus_ack', 'flag_entropy_norm', 'byte_per_pkt',
    'high_syn_ratio', 'log_rate', 'log_iat', 'ack_no_syn',
    'ddos_vs_dos_score', 'recon_iat_rate_ratio',
    'TTL_deviation', 'size_cv', 'payload_asymmetry', 'conn_teardown_rate',
    # 8 new discriminative features (DDoS vs DoS flood separation)
    'ack_to_syn_ratio', 'rst_to_syn_ratio', 'rate_per_pkt',
    'syn_flag_density', 'ack_teardown_interaction', 'flood_intensity_score',
    'tcp_ratio_sq', 'iat_rate_product',
]

# XGBoost Parameters — tuned for high multi-class accuracy across 25 attack classes
XGB_PARAMS = dict(
    n_estimators          = 3000,
    max_depth             = 9,
    learning_rate         = 0.02,
    subsample             = 0.85,
    colsample_bytree      = 0.70,
    colsample_bylevel     = 0.80,
    min_child_weight      = 3,
    gamma                 = 0.10,
    reg_alpha             = 0.10,
    reg_lambda            = 1.5,
    objective             = 'multi:softprob',
    eval_metric           = 'mlogloss',
    early_stopping_rounds = 200,
    tree_method           = 'hist',
    device                = 'cuda',   # GPU training — avoids CPU/GPU mismatch warning
    n_jobs                = -1,
    random_state          = RANDOM_STATE,
    verbosity             = 1,
)

report_lines = []

def log(msg="", also_print=True):
    report_lines.append(msg)
    if also_print:
        print(msg)

# =============================================================================
# STEP 1 — LOAD DATASETS
# =============================================================================

def load_datasets():
    log("="*65)
    log("PHASE 4 — MODEL TRAINING & SCORE FUSION")
    log("="*65)
    log("\n[Step 1] Loading preprocessed Parquet datasets...")
    t0 = time.time()

    train_df = pd.read_parquet(TRAIN_PARQUET, engine='pyarrow')
    test_df  = pd.read_parquet(TEST_PARQUET,  engine='pyarrow')
    
    log(f"  Train set: {len(train_df):,} rows x {train_df.shape[1]} columns")
    log(f"  Test set : {len(test_df):,} rows x {test_df.shape[1]} columns")
    log(f"  Loaded in {time.time()-t0:.1f}s")
    
    return train_df, test_df

# =============================================================================
# STEP 2 — ENCODE LABELS
# =============================================================================

def encode_labels(y_train, y_test):
    log("\n[Step 2] Encoding labels...")
    le = LabelEncoder()
    y_tr = le.fit_transform(y_train)
    y_te = le.transform(y_test)
    log(f"  {len(le.classes_)} classes detected and encoded.")
    
    path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    joblib.dump(le, path)
    log(f"  Saved LabelEncoder -> {path}")
    
    return le, y_tr, y_te

# =============================================================================
# STEP 3 — SCALE FEATURES
# =============================================================================

def fit_scaler(X_train):
    log("\n[Step 3] Fitting StandardScaler (for Isolation Forest)...")
    present = [c for c in SCALE_COLS if c in X_train.columns]
    
    scaler = StandardScaler()
    scaler.fit(X_train[present])
    
    path = os.path.join(MODEL_DIR, 'scaler.pkl')
    joblib.dump(scaler, path)
    log(f"  Fitted on {len(present)} numeric columns | Saved -> {path}")
    
    return scaler, present

def apply_scaler(X, scaler, scale_cols):
    X_out = X.copy()
    present = [c for c in scale_cols if c in X_out.columns]
    X_out[present] = scaler.transform(X[present]).astype(np.float32)
    return X_out

# =============================================================================
# STEP 4 — TRAIN ISOLATION FOREST
# =============================================================================

def train_isolation_forest(X_train, y_train, scaler, scale_cols):
    log("\n[Step 4] Training Isolation Forest (on BENIGN rows)...")
    
    X_benign = X_train[y_train == 'BENIGN'].reset_index(drop=True)
    log(f"  BENIGN training samples: {len(X_benign):,}")
    X_b_scaled = apply_scaler(X_benign, scaler, scale_cols)

    t0 = time.time()
    iso = IsolationForest(
        n_estimators  = 300,
        contamination = 0.05,
        max_samples   = 'auto',
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
    )
    iso.fit(X_b_scaled)
    log(f"  Training completed in {time.time()-t0:.1f}s")
    
    # Calculate benign training score boundaries for [0, 1] scaling
    train_benign_scores = iso.score_samples(X_b_scaled)
    if_min = train_benign_scores.min()
    if_max = train_benign_scores.max()
    log(f"  Benign training score range: {if_min:.4f} to {if_max:.4f}")
    
    # Save model alongside threshold parameters
    path = os.path.join(MODEL_DIR, 'isolation_forest.pkl')
    joblib.dump({'model': iso, 'if_min': if_min, 'if_max': if_max}, path)
    log(f"  Saved Isolation Forest components -> {path}")
    
    return iso, if_min, if_max

# =============================================================================
# STEP 5 — TRAIN XGBOOST CLASSIFIER
# =============================================================================

def train_xgboost(X_train, y_train_enc, le):
    log("\n[Step 5] Training XGBoost Classifier...")
    
    # GPU detection is handled via XGB_PARAMS device='cuda' directly
    # Removed subprocess detection: device is set at config level above

    # Calculate sample weights for imbalance handling (balanced strategy)
    sw = compute_sample_weight('balanced', y_train_enc).astype(np.float32)

    # 10% stratified validation split for early stopping
    X_tr, X_val, y_tr, y_val, sw_tr, _ = train_test_split(
        X_train.values, y_train_enc, sw,
        test_size    = 0.10,
        random_state = RANDOM_STATE,
        stratify     = y_train_enc,
    )
    log(f"  Training subset size: {len(X_tr):,}")
    log(f"  Validation subset size: {len(X_val):,}")

    params = {
        **XGB_PARAMS,
        'num_class' : len(le.classes_),
    }
    log(f"  XGBoost device: {params.get('device', 'cpu')}")

    xgb = XGBClassifier(**params)
    t0  = time.time()
    xgb.fit(
        X_tr, y_tr,
        sample_weight          = sw_tr,
        eval_set               = [(X_val, y_val)],
        verbose                = 100,
    )
    log(f"  Training completed in {(time.time()-t0)/60:.1f} minutes")
    log(f"  Best iteration: {xgb.best_iteration}")

    path = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
    joblib.dump(xgb, path)
    log(f"  Saved XGBoost model -> {path}")
    
    return xgb

# =============================================================================
# STEP 6 — DPI SIGNATURE EVALUATION
# =============================================================================

def evaluate_dpi(df):
    """
    Signature-based DPI match logic:
    Returns 1.0 for match (anomaly detected), 0.0 for normal.
    """
    # Rule 1: UDP volumetric signature
    udp_flood = (df['UDP'] == 1) & (df['Rate'] > 10000)
    
    # Rule 2: ICMP volumetric signature
    icmp_flood = (df['ICMP'] == 1) & (df['Rate'] > 5000)
    
    # Rule 3: SYN flood signature
    syn_flood = (df['syn_flag_number'] == 1) & (df['ack_flag_number'] == 0) & (df['Rate'] > 5000)
    
    # Rule 4: RST volumetric signature
    rst_flood = (df['rst_flag_number'] == 1) & (df['Rate'] > 5000)
    
    # Rule 5: DNS volumetric spoofing signature
    dns_spoof = (df['DNS'] == 1) & (df['Rate'] > 1000)
    
    # Rule 6: Web Attack signatures (HTTP/HTTPS, lower TTL, non-zero packet variance)
    web_attack = (df['HTTP'] == 1) & (df['Time_To_Live'] < 110) & (df['Variance'] > 0)
    
    # Rule 7: SSH brute force signature
    ssh_brute = (df['SSH'] == 1) & (df['Rate'] > 500)
    
    # Combine all rules
    dpi_match = udp_flood | icmp_flood | syn_flood | rst_flood | dns_spoof | web_attack | ssh_brute
    
    return dpi_match.astype(np.float32)

# =============================================================================
# STEP 7 — MODEL FUSION & EVALUATION
# =============================================================================

def evaluate(iso, if_min, if_max, xgb, X_test, y_test, y_test_enc, le, scaler, scale_cols, test_df):
    log("\n[Step 6] Evaluating models on held-out test set...")
    log(f"  Test rows: {len(X_test):,}")

    X_test_arr = X_test.values if hasattr(X_test, 'values') else X_test
    y_test_labels = y_test.values if hasattr(y_test, 'values') else y_test

    # ── 1. XGBoost Classification Metrics ────────────────────────────────────
    log("\n" + "-"*65)
    log("XGBoost Multi-Class Classification Performance")
    log("-" * 65)

    t0     = time.time()
    yp_enc = xgb.predict(X_test_arr)
    xgb_lat = (time.time() - t0) / len(X_test) * 1000
    yp     = le.inverse_transform(yp_enc)

    acc = accuracy_score(y_test_enc, yp_enc)
    wf1 = f1_score(y_test_enc, yp_enc, average='weighted')

    log(f"  Overall Classification Accuracy: {acc * 100:.4f}%")
    log(f"  Weighted F1-Score              : {wf1:.4f}")
    log(f"  Avg inference latency per row  : {xgb_lat:.4f} ms")

    log(f"\n  {'Class':<40} {'P':>6}  {'R':>6}  {'F1':>6}  {'N':>8}")
    log(f"  {'-'*66}")

    rd  = classification_report(y_test_labels, yp, output_dict=True, zero_division=0)
    low_f1 = []

    for cls in sorted(le.classes_):
        if cls in rd:
            r    = rd[cls]
            flag = " [!] F1<0.70" if r['f1-score'] < 0.70 else "  [OK]"
            log(f"  {cls:<40} {r['precision']:>6.3f}  "
                f"{r['recall']:>6.3f}  {r['f1-score']:>6.3f}  "
                f"{int(r['support']):>8}{flag}")
            if r['f1-score'] < 0.70:
                low_f1.append((cls, r['f1-score']))

    if low_f1:
        log(f"\n  Classes below F1=0.70 ({len(low_f1)}):")
        for cls, f in sorted(low_f1, key=lambda x: x[1]):
            log(f"    {cls:<40} F1={f:.3f}")
    else:
        log(f"\n  All {len(le.classes_)} classes achieved F1 >= 0.70 [OK]")

    # ── 2. Isolation Forest Anomaly Detection Metrics ────────────────────────
    log(f"\n{'-'*65}")
    log("Isolation Forest Anomaly Detection Performance")
    log("-" * 65)

    X_test_scaled = apply_scaler(X_test, scaler, scale_cols)  # returns full df with scale_cols standardized
    t0     = time.time()
    ifp    = iso.predict(X_test_scaled)   # full feature matrix
    ifs    = iso.score_samples(X_test_scaled)  # full feature matrix

    if_lat = (time.time() - t0) / len(X_test) * 1000

    # Binary evaluation: BENIGN is normal (1), attacks are anomalous (-1)
    ytb    = np.where(y_test_labels == 'BENIGN', 1, -1)
    tp     = ((ifp == -1) & (ytb == -1)).sum()
    tn     = ((ifp ==  1) & (ytb ==  1)).sum()
    fp     = ((ifp == -1) & (ytb ==  1)).sum()
    fn     = ((ifp ==  1) & (ytb == -1)).sum()

    dr     = tp / (tp + fn)   if (tp + fn) > 0 else 0
    fpr    = fp / (fp + tn)   if (fp + tn) > 0 else 0
    prec   = tp / (tp + fp)   if (tp + fp) > 0 else 0
    f1_if  = 2 * prec * dr / (prec + dr) if (prec + dr) > 0 else 0

    log(f"  Anomaly Detection Rate (TPR) : {dr * 100:.2f}%")
    log(f"  False Positive Rate (FPR)    : {fpr * 100:.2f}%")
    log(f"  Anomaly Precision            : {prec * 100:.2f}%")
    log(f"  Anomaly F1-score             : {f1_if:.4f}")
    log(f"  Avg inference latency per row: {if_lat:.4f} ms")
    log(f"  Counts: TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}")

    # ── 3. Score Fusion ──────────────────────────────────────────────────────
    log(f"\n{'-'*65}")
    log("Score Fusion Detection Performance (0.4*IF + 0.4*XGBoost + 0.2*DPI)")
    log("-" * 65)

    # Scale IF outlier score to [0, 1] anomaly score
    ifs_clipped = np.clip(ifs, if_min, if_max)
    if_prob = 1.0 - (ifs_clipped - if_min) / (if_max - if_min + 1e-9)

    # Scale XGBoost score to [0, 1] attack probability
    benign_idx = le.transform(['BENIGN'])[0]
    xgb_probs = xgb.predict_proba(X_test_arr)
    xgb_prob = 1.0 - xgb_probs[:, benign_idx]

    # Evaluate signature-based DPI rule score
    dpi_score = evaluate_dpi(test_df)

    # Calculate fusion score
    final_score = 0.4 * if_prob + 0.4 * xgb_prob + 0.2 * dpi_score

    # Binary prediction using a fusion threshold of 0.5 (block if score >= 0.5)
    threshold = 0.5
    fusion_pred = np.where(final_score >= threshold, -1, 1)

    tp_f   = ((fusion_pred == -1) & (ytb == -1)).sum()
    tn_f   = ((fusion_pred ==  1) & (ytb ==  1)).sum()
    fp_f   = ((fusion_pred == -1) & (ytb ==  1)).sum()
    fn_f   = ((fusion_pred ==  1) & (ytb == -1)).sum()

    dr_f   = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
    fpr_f  = fp_f / (fp_f + tn_f) if (fp_f + tn_f) > 0 else 0
    prec_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0
    f1_f   = 2 * prec_f * dr_f / (prec_f + dr_f) if (prec_f + dr_f) > 0 else 0

    binary_labels = np.where(y_test_labels == 'BENIGN', 0, 1)
    binary_preds  = np.where(final_score >= threshold, 1, 0)
    bin_acc       = accuracy_score(binary_labels, binary_preds)

    log(f"  Decision Threshold           : {threshold}")
    log(f"  Fusion Binary Accuracy       : {bin_acc * 100:.4f}%")
    log(f"  Fusion Detection Rate (TPR)  : {dr_f * 100:.2f}%")
    log(f"  Fusion False Positive Rate   : {fpr_f * 100:.2f}%")
    log(f"  Fusion Detection F1-score    : {f1_f:.4f}")
    log(f"  Fusion Counts: TP={tp_f:,}  TN={tn_f:,}  FP={fp_f:,}  FN={fn_f:,}")

    # Detection rate comparison per attack class
    log("\n  Detection Rate (TPR) by Attack Class:")
    log(f"  {'-'*75}")
    log(f"  {'Attack Class':<35} {'N':>8}   {'Fusion':>8}  {'XGBoost':>8}  {'IF':>8}")
    log(f"  {'-'*75}")
    for cls in sorted(le.classes_):
        if cls == 'BENIGN':
            continue
        mask = y_test_labels == cls
        n_samples = mask.sum()
        if n_samples > 0:
            dr_fusion_cls = (fusion_pred[mask] == -1).mean() * 100
            dr_xgb_cls = (yp[mask] == cls).mean() * 100
            dr_if_cls = (ifp[mask] == -1).mean() * 100
            log(f"  {cls:<35} {n_samples:>8,}   {dr_fusion_cls:>7.1f}%  {dr_xgb_cls:>7.1f}%  {dr_if_cls:>7.1f}%")

    return yp, acc, wf1, dr_f, fpr_f, bin_acc

# =============================================================================
# CONFUSION MATRIX PLOT
# =============================================================================

def save_cm(y_test, y_pred, le):
    classes = le.classes_
    y_arr   = y_test.values if hasattr(y_test, 'values') else y_test
    cm      = confusion_matrix(y_arr, y_pred, labels=classes)
    rs      = cm.sum(axis=1, keepdims=True)
    cn      = np.divide(cm.astype(float), rs, where=rs != 0,
                        out=np.zeros_like(cm, dtype=float))

    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(cn, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                ax=ax, linewidths=0.3, annot_kws={'size': 6})
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True',      fontsize=11)
    ax.set_title(f'XGBoost Classifier — Normalized Confusion Matrix\n'
                 f'{len(classes)} classes | SMOTE + 16 engineered features', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0,              fontsize=7)
    plt.tight_layout()
    plt.savefig(CM_FILE, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"\n  Confusion matrix plot saved -> {CM_FILE}")

# =============================================================================
# MAIN
# =============================================================================

def run():
    t_start = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load Preprocessed Parquet ───────────────────────────────────────────
    train_df, test_df = load_datasets()
    
    # Separate features and labels
    X_train = train_df.drop(columns=['Label'])
    y_train = train_df['Label']
    X_test  = test_df.drop(columns=['Label'])
    y_test  = test_df['Label']

    # Check if models are already trained and saved to skip training
    le_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    iso_path = os.path.join(MODEL_DIR, 'isolation_forest.pkl')
    xgb_path = os.path.join(MODEL_DIR, 'xgboost_model.pkl')

    if not FORCE_RETRAIN and os.path.exists(le_path) and os.path.exists(scaler_path) and os.path.exists(iso_path) and os.path.exists(xgb_path):
        log("\n[Step 2-5] All models found — loading from disk (FORCE_RETRAIN=False)...")
        le = joblib.load(le_path)
        y_tr_enc = le.transform(y_train)
        y_te_enc = le.transform(y_test)
        scaler = joblib.load(scaler_path)
        scale_cols = [c for c in SCALE_COLS if c in X_train.columns]
        iso_components = joblib.load(iso_path)
        iso = iso_components['model']
        if_min = iso_components['if_min']
        if_max = iso_components['if_max']
        xgb = joblib.load(xgb_path)
        log("  Loaded all models successfully [OK]")
    else:
        # Load LE, scaler, IF if they exist (survive interruption); only retrain XGBoost
        if os.path.exists(le_path) and os.path.exists(scaler_path) and os.path.exists(iso_path):
            log("\n[Step 2-4] Found LE + Scaler + IF on disk — loading to skip those steps...")
            le = joblib.load(le_path)
            y_tr_enc = le.transform(y_train)
            y_te_enc = le.transform(y_test)
            scaler = joblib.load(scaler_path)
            scale_cols = [c for c in SCALE_COLS if c in X_train.columns]
            iso_components = joblib.load(iso_path)
            iso = iso_components['model']
            if_min = iso_components['if_min']
            if_max = iso_components['if_max']
            log("  Loaded LE / Scaler / IF successfully [OK]")
        else:
            # ── Full fit from scratch ───────────────────────────────────────────
            le, y_tr_enc, y_te_enc = encode_labels(y_train, y_test)
            scaler, scale_cols = fit_scaler(X_train)
            iso, if_min, if_max = train_isolation_forest(X_train, y_train, scaler, scale_cols)

        # ── Always (re)train XGBoost ────────────────────────────────────────────
        xgb = train_xgboost(X_train, y_tr_enc, le)


    # ── Evaluate Score Fusion on Test Set ───────────────────────────────────
    y_pred, xgb_acc, xgb_f1, fus_dr, fus_fpr, fus_acc = evaluate(
        iso, if_min, if_max, xgb, X_test, y_test, y_te_enc, le, scaler, scale_cols, test_df
    )

    # Save multi-class classification confusion matrix
    save_cm(y_test, y_pred, le)

    # ── Final Execution Log ─────────────────────────────────────────────────
    log(f"\n{'='*65}")
    log("PHASE 4 COMPLETE")
    log(f"{'='*65}")
    total_min = (time.time() - t_start) / 60
    log(f"  Total execution time       : {total_min:.1f} minutes")
    log(f"  XGBoost Classifier Accuracy: {xgb_acc * 100:.4f}%")
    log(f"  XGBoost F1-Score           : {xgb_f1:.4f}")
    log(f"  Fusion Detection Rate (DR) : {fus_dr * 100:.2f}%")
    log(f"  Fusion False Positive Rate : {fus_fpr * 100:.2f}%")
    log(f"  Fusion Binary Accuracy     : {fus_acc * 100:.4f}%")
    log(f"\n  Saved model objects in: {MODEL_DIR}")
    log(f"    label_encoder.pkl  |  scaler.pkl")
    log(f"    isolation_forest.pkl (model + min/max parameters)  |  xgboost_model.pkl")

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\nPreprocessing & Training audit report saved -> {REPORT_FILE}")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    run()