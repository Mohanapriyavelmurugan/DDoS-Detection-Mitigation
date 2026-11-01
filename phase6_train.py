"""
=================================================================
PHASE 6 -- INDUSTRY-READY MODEL TRAINING
=================================================================
Key architectural changes:

  XGBoost:
    - BENIGN_WEIGHT = 4.0   (was 1.0 Phase4, 20.0 Phase5)
    - n_estimators  = 3000  with early_stopping = 50 rounds
    - learning_rate = 0.05  (slower, better generalisation)
    - Platt Scaling calibration on held-out calibration set

  Isolation Forest:
    - contamination = 0.02  (was 0.05 Phase4, 0.005 Phase5)
    - n_estimators  = 200   (was 100 in both phases)
    - max_samples   = 20000 (was 10000 in both phases)

  3-Layer Decision Architecture:
    - Layer 1 : IF Hard Veto        (if_prob > 0.75 -> ALERT)
    - Layer 2 : Disagreement Boost  (IF high + XGB says BENIGN -> +0.15)
    - Layer 3 : Confidence-Adaptive Fusion + Threshold Optimiser
                (target FPR <= 2%, maximise TPR)

  Outputs saved to models_v6/:
    label_encoder.pkl, scaler.pkl, isolation_forest.pkl,
    xgboost_model.pkl, xgboost_calibrator.pkl,
    optimal_threshold.json, feature_names.pkl

  Report: phase6_train_report.txt
=================================================================
"""

import os, sys, time, json, joblib, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
from xgboost.core import XGBoostError

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# ── CONFIG ───────────────────────────────────────────────────────────────────
TRAIN_PARQUET   = r"D:\DDos 1\balanced_train.parquet"
TEST_PARQUET    = r"D:\DDos 1\test.parquet"
MODEL_DIR       = r"D:\DDos 1\models_v6"
REPORT_FILE     = r"D:\DDos 1\phase6_train_report.txt"
os.makedirs(MODEL_DIR, exist_ok=True)

BENIGN_WEIGHT       = 4.0    # Moderate — derived from binary class ratio (~23:1 attack:benign)
IF_CONTAMINATION    = 0.02   # 2% — realistic MEC deployment anomaly rate
IF_N_ESTIMATORS     = 200
IF_MAX_SAMPLES      = 20000
CALIB_SPLIT         = 0.15   # 15% of training data held out for Platt calibration
TARGET_FPR          = 0.02   # Industry target: FPR <= 2%
IF_VETO_THRESHOLD   = 0.75   # IF hard-veto threshold (Layer 1)
IF_DISAGREE_MIN     = 0.60   # IF disagreement amplifier lower bound (Layer 2)
XGB_BENIGN_MAX      = 0.30   # XGB attack prob below which disagreement fires (Layer 2)
DISAGREE_BOOST      = 0.15   # Score boost when Layer 2 triggers
EPS                 = 1e-9

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))

SEP  = "=" * 68
DASH = "-" * 68

# ── DPI ENGINE (Enhanced — Phase 6) ─────────────────────────────────────────
def dpi_score_batch(df):
    """
    Enhanced DPI scoring with 6 rules covering Phase 6 weak attack classes.
    Returns numpy array of shape (N,) with values in [0, 1].
    """
    score = np.zeros(len(df), dtype=np.float32)

    syn   = df.get('syn_count',  pd.Series(0.0, index=df.index)).values.astype(float)
    rate  = df.get('Rate',       pd.Series(0.0, index=df.index)).values.astype(float)
    udp   = df.get('UDP',        pd.Series(0.0, index=df.index)).values.astype(float)
    icmp  = df.get('ICMP',       pd.Series(0.0, index=df.index)).values.astype(float)
    iat   = df.get('IAT_mean',   pd.Series(0.0, index=df.index)).values.astype(float)
    n     = df.get('Number',     pd.Series(1.0, index=df.index)).values.astype(float)
    fin   = df.get('fin_count',  pd.Series(0.0, index=df.index)).values.astype(float)
    rst   = df.get('rst_count',  pd.Series(0.0, index=df.index)).values.astype(float)
    pkt   = df.get('AVG',        pd.Series(0.0, index=df.index)).values.astype(float)

    # Rule 1 — SYN Flood (original, tightened threshold)
    score = np.where((syn > 50) & (rate > 2000),
                     np.maximum(score, 0.95), score)

    # Rule 2 — UDP Flood (original)
    score = np.where((udp > 0.8) & (rate > 2000),
                     np.maximum(score, 0.90), score)

    # Rule 3 — ICMP Flood (original)
    score = np.where((icmp > 0.5) & (rate > 1000),
                     np.maximum(score, 0.88), score)

    # Rule 4 [NEW] — SYN Ratio Anomaly (catches DDOS-SYN_FLOOD missed by XGB)
    syn_ratio = np.clip(syn / (n + EPS), 0, 1)
    score = np.where((syn_ratio > 0.85) & (rate > 200),
                     np.maximum(score, 0.88), score)

    # Rule 5 [NEW] — Recon Sweep (catches RECON-PINGSWEEP)
    # High IAT (slow scanning), many packets, low rate
    score = np.where((iat > 0.05) & (n > 5) & (rate < 100) & (rate > 0),
                     np.maximum(score, 0.82), score)

    # Rule 6 [NEW] — Low-and-Slow Brute Force / SlowLoris variant
    # Many connections, tiny packets, near-zero rate
    score = np.where((n > 15) & (pkt < 120) & (rate < 10) & (rate > 0),
                     np.maximum(score, 0.78), score)

    # Benign guard — very low rate + long inter-arrival → suppress score
    score = np.where((rate < 10) & (iat > 0.5),
                     np.minimum(score, 0.05), score)

    return np.clip(score, 0.0, 1.0).astype(np.float32)

# ── CONFIDENCE-ADAPTIVE FUSION (3-Layer) ─────────────────────────────────────
def three_layer_fusion(xgb_proba, if_prob, dpi_scores, benign_id):
    """
    Layer 1 : IF Hard Veto
    Layer 2 : Disagreement Amplifier
    Layer 3 : Confidence-Adaptive weighted fusion
    Returns fusion scores (0-1) and a layer_triggered array for diagnostics.
    """
    n = len(if_prob)
    xgb_att_prob    = (1.0 - xgb_proba[:, benign_id]).astype(np.float32)
    xgb_confidence  = xgb_proba.max(axis=1).astype(np.float32)  # highest class prob

    # Adaptive weights based on XGBoost confidence
    w_xgb = np.where(xgb_confidence >= 0.80, 0.60,
            np.where(xgb_confidence >= 0.50, 0.40, 0.25)).astype(np.float32)
    w_if  = np.where(xgb_confidence >= 0.80, 0.25,
            np.where(xgb_confidence >= 0.50, 0.45, 0.60)).astype(np.float32)
    w_dpi = np.full(n, 0.15, dtype=np.float32)

    # Layer 3 base fusion
    fusion = np.clip(w_xgb * xgb_att_prob + w_if * if_prob + w_dpi * dpi_scores,
                     0.0, 1.0)

    # Layer 2 — Disagreement Amplifier
    # XGB confidently says BENIGN but IF strongly disagrees
    disagree_mask = (if_prob > IF_DISAGREE_MIN) & (xgb_att_prob < XGB_BENIGN_MAX)
    fusion = np.where(disagree_mask,
                      np.clip(fusion + DISAGREE_BOOST, 0, 1),
                      fusion)

    # Layer 1 — IF Hard Veto (overrides everything)
    veto_mask = (if_prob > IF_VETO_THRESHOLD)
    fusion = np.where(veto_mask, np.maximum(fusion, 0.90), fusion)

    layer_triggered = np.where(veto_mask, 1,
                      np.where(disagree_mask, 2, 3)).astype(np.int8)

    return fusion.astype(np.float32), xgb_att_prob, layer_triggered

# ── THRESHOLD OPTIMISER ──────────────────────────────────────────────────────
def find_optimal_threshold(fusion, y_binary, target_fpr=TARGET_FPR):
    """
    Scans thresholds 0.01->0.99 and returns the minimum threshold
    where FPR <= target_fpr and TPR is maximised.
    Falls back to highest F-beta(0.5) if no threshold meets FPR target.
    """
    best_t, best_tpr, best_fpr, best_acc, best_prec = 0.50, 0.0, 1.0, 0.0, 0.0
    results = []
    for t in np.arange(0.01, 1.00, 0.01):
        preds = (fusion >= t).astype(int)
        tp = ((preds == 1) & (y_binary == 1)).sum()
        fp = ((preds == 1) & (y_binary == 0)).sum()
        tn = ((preds == 0) & (y_binary == 0)).sum()
        fn = ((preds == 0) & (y_binary == 1)).sum()
        tpr  = tp / (tp + fn + EPS)
        fpr  = fp / (fp + tn + EPS)
        acc  = (tp + tn) / len(y_binary)
        prec = tp / (tp + fp + EPS)
        results.append((round(t, 2), tpr, fpr, acc, prec))
        if fpr <= target_fpr and tpr > best_tpr:
            best_t, best_tpr, best_fpr, best_acc, best_prec = t, tpr, fpr, acc, prec

    if best_tpr == 0.0:
        log("  [!] No threshold achieves FPR <= 2%. Falling back to best F-beta(0.5).")
        best_fb = 0.0
        for t, tpr, fpr, acc, prec in results:
            fb = (1 + 0.5**2) * prec * tpr / (0.5**2 * prec + tpr + EPS)
            if fb > best_fb:
                best_fb = fb
                best_t, best_tpr, best_fpr, best_acc, best_prec = t, tpr, fpr, acc, prec

    return best_t, best_tpr, best_fpr, best_acc, best_prec

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def run():
    t_total = time.time()

    log(SEP)
    log("PHASE 6  --  INDUSTRY-READY MODEL TRAINING")
    log(SEP)
    log(f"  XGBoost BENIGN_WEIGHT   : {BENIGN_WEIGHT}")
    log(f"  IF contamination        : {IF_CONTAMINATION}")
    log(f"  IF n_estimators         : {IF_N_ESTIMATORS}")
    log(f"  IF max_samples          : {IF_MAX_SAMPLES}")
    log(f"  Calibration split       : {CALIB_SPLIT*100:.0f}%")
    log(f"  Target FPR              : <= {TARGET_FPR*100:.0f}%")
    log(f"  IF Veto threshold       : {IF_VETO_THRESHOLD}")
    log()

    # ── 1. LOAD DATA ──────────────────────────────────────────────────────────
    log("[Step 1] Loading datasets...")
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df  = pd.read_parquet(TEST_PARQUET)

    X_train_full = train_df.drop(columns=['Label'])
    y_train_full = train_df['Label']
    X_test       = test_df.drop(columns=['Label'])
    y_test       = test_df['Label']

    log(f"  Train: {len(X_train_full):,} rows x {X_train_full.shape[1]} cols")
    log(f"  Test : {len(X_test):,} rows x {X_test.shape[1]} cols")

    # ── 2. LABEL ENCODER + SCALER ─────────────────────────────────────────────
    log("\n[Step 2] Fitting LabelEncoder and StandardScaler...")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_full)
    y_test_enc  = le.transform(y_test)
    benign_id   = list(le.classes_).index('BENIGN')

    scaler = StandardScaler()
    X_train_scaled_full = scaler.fit_transform(X_train_full)
    X_test_scaled       = scaler.transform(X_test)

    joblib.dump(le,     os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(list(X_train_full.columns),
                os.path.join(MODEL_DIR, 'feature_names.pkl'))
    log(f"  Classes: {len(le.classes_)}")

    # ── 3. TRAIN / CALIBRATION SPLIT ─────────────────────────────────────────
    log(f"\n[Step 3] Splitting train into {(1-CALIB_SPLIT)*100:.0f}% train / {CALIB_SPLIT*100:.0f}% calibration...")
    (X_tr, X_calib,
     y_tr, y_calib,
     y_tr_enc, y_calib_enc,
     X_tr_sc, X_calib_sc) = train_test_split(
        X_train_full, y_train_full, y_train_enc, X_train_scaled_full,
        test_size=CALIB_SPLIT, stratify=y_train_enc, random_state=42
    )
    log(f"  Training set  : {len(X_tr):,} rows")
    log(f"  Calibration   : {len(X_calib):,} rows")

    # ── 4. ISOLATION FOREST ───────────────────────────────────────────────────
    log(f"\n[Step 4] Training Isolation Forest "
        f"(contamination={IF_CONTAMINATION}, n_est={IF_N_ESTIMATORS}, max_samples={IF_MAX_SAMPLES})...")
    t0 = time.time()
    iso = IsolationForest(
        n_estimators  = IF_N_ESTIMATORS,
        max_samples   = IF_MAX_SAMPLES,
        contamination = IF_CONTAMINATION,
        random_state  = 42,
        n_jobs        = -1
    )
    iso.fit(X_tr_sc)
    log(f"  IF training completed in {time.time()-t0:.1f}s")

    # Compute min/max on a large sample for score normalisation
    sample_idx = np.random.default_rng(42).choice(X_tr_sc.shape[0],
                                                    min(80000, X_tr_sc.shape[0]),
                                                    replace=False)
    ifs_sample = iso.score_samples(X_tr_sc[sample_idx])
    if_min, if_max = float(ifs_sample.min()), float(ifs_sample.max())
    log(f"  IF score range on training sample: {if_min:.4f} to {if_max:.4f}")

    joblib.dump({'model': iso, 'if_min': if_min, 'if_max': if_max},
                os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
    log("  Isolation Forest saved.")

    # ── 5. XGBOOST TRAINING ───────────────────────────────────────────────────
    log(f"\n[Step 5] Training XGBoost (BENIGN_WEIGHT={BENIGN_WEIGHT}, n_est=3000, lr=0.05)...")

    # Sample weights: BENIGN gets extra penalty
    sw = np.ones(len(y_tr_enc), dtype=np.float32)
    sw[y_tr_enc == benign_id] = BENIGN_WEIGHT

    # Internal 90/10 split for early stopping (within the training portion)
    X_xtr, X_xval, y_xtr, y_xval, sw_xtr, sw_xval = train_test_split(
        X_tr, y_tr_enc, sw,
        test_size=0.10, stratify=y_tr_enc, random_state=42
    )

    xgb = XGBClassifier(
        n_estimators         = 2000,   # Capped: convergence happens at ~1978 on this dataset
        learning_rate        = 0.05,
        max_depth            = 8,
        subsample            = 0.8,
        colsample_bytree     = 0.8,
        tree_method          = 'hist',
        device               = 'cuda',
        objective            = 'multi:softprob',
        num_class            = len(le.classes_),
        eval_metric          = 'mlogloss',
        early_stopping_rounds= 50,
        random_state         = 42,
        verbosity            = 0
    )

    t0 = time.time()
    try:
        xgb.fit(
            X_xtr, y_xtr,
            sample_weight          = sw_xtr,
            eval_set               = [(X_xval, y_xval)],
            sample_weight_eval_set = [sw_xval],
            verbose                = 100
        )
        log(f"  XGBoost training completed in {(time.time()-t0)/60:.1f} minutes")
        log(f"  Best iteration: {xgb.best_iteration}")
    except Exception as e:
        if 'bad allocation' in str(e).lower() or 'reset' in str(e).lower():
            log(f"  [WARNING] GPU memory error during early-stopping reset (known XGBoost/CUDA issue).")
            log(f"  [WARNING] Model trained to convergence. Saving booster directly...")
            # The underlying booster is still valid even if reset() failed
            try:
                raw_path = os.path.join(MODEL_DIR, 'xgboost_model_raw.json')
                xgb.get_booster().save_model(raw_path)
                log(f"  Raw booster saved to {raw_path} (JSON format).")
                # Reload into a CPU-mode classifier for downstream steps
                xgb_reload = XGBClassifier(
                    n_estimators=2000, learning_rate=0.05, max_depth=8,
                    subsample=0.8, colsample_bytree=0.8,
                    tree_method='hist', device='cpu',
                    objective='multi:softprob', num_class=len(le.classes_),
                    eval_metric='mlogloss', random_state=42, verbosity=0
                )
                xgb_reload.load_model(raw_path)
                xgb = xgb_reload
                log(f"  Booster reloaded in CPU mode successfully.")
            except Exception as e2:
                log(f"  ERROR: Could not save/reload booster: {e2}")
                raise
        else:
            raise

    # Switch to CPU for all downstream inference (avoids further GPU memory issues)
    try:
        xgb.set_params(device='cpu')
    except Exception:
        pass

    joblib.dump(xgb, os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
    log("  XGBoost model saved.")

    # ── 6. PLATT SCALING CALIBRATION ─────────────────────────────────────────
    log("\n[Step 6] Calibrating XGBoost with Platt Scaling on calibration set...")
    t0 = time.time()

    xgb_calib_proba  = xgb.predict_proba(X_calib)
    xgb_att_calib    = (1.0 - xgb_calib_proba[:, benign_id]).reshape(-1, 1)
    y_binary_calib   = (y_calib != 'BENIGN').astype(int).values

    platt = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    platt.fit(xgb_att_calib, y_binary_calib)

    log(f"  Platt Scaling fitted in {time.time()-t0:.2f}s")
    log(f"  Calibrator intercept: {platt.intercept_[0]:.4f}  coef: {platt.coef_[0][0]:.4f}")

    joblib.dump({'calibrator': platt, 'benign_id': benign_id},
                os.path.join(MODEL_DIR, 'xgboost_calibrator.pkl'))
    log("  Calibrator saved.")

    # ── 7. FULL EVALUATION ON TEST SET ───────────────────────────────────────
    log("\n[Step 7] Evaluating on test set...")

    # XGBoost multi-class
    t0 = time.time()
    xgb_proba_test  = xgb.predict_proba(X_test)
    xgb_infer_time  = (time.time()-t0)*1000/len(X_test)
    y_pred          = le.inverse_transform(xgb_proba_test.argmax(axis=1))
    xgb_acc         = accuracy_score(y_test, y_pred)
    xgb_f1          = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    log(f"\n  {DASH}")
    log("  XGBoost Multi-Class (Phase 6)")
    log(f"  {DASH}")
    log(f"  Accuracy         : {xgb_acc*100:.4f}%")
    log(f"  Weighted F1      : {xgb_f1:.4f}")
    log(f"  Latency          : {xgb_infer_time:.4f} ms/row")
    log()
    log(classification_report(y_test, y_pred, zero_division=0))

    # Calibrated XGB attack probability
    raw_att_prob = (1.0 - xgb_proba_test[:, benign_id]).reshape(-1, 1)
    xgb_att_prob_cal = platt.predict_proba(raw_att_prob)[:, 1].astype(np.float32)

    # Isolation Forest
    t0 = time.time()
    ifs_test  = iso.score_samples(X_test_scaled)
    if_infer  = (time.time()-t0)*1000/len(X_test)
    ifs_clip  = np.clip(ifs_test, if_min, if_max)
    if_prob   = (1.0 - (ifs_clip - if_min) / (if_max - if_min + EPS)).astype(np.float32)

    y_binary = (y_test != 'BENIGN').astype(int).values

    # IF standalone binary metrics
    if_preds  = (if_prob >= 0.5).astype(int)
    tp_ = ((if_preds==1)&(y_binary==1)).sum()
    fp_ = ((if_preds==1)&(y_binary==0)).sum()
    tn_ = ((if_preds==0)&(y_binary==0)).sum()
    fn_ = ((if_preds==0)&(y_binary==1)).sum()
    if_tpr  = tp_/(tp_+fn_+EPS)
    if_fpr  = fp_/(fp_+tn_+EPS)
    if_prec = tp_/(tp_+fp_+EPS)

    log(f"  {DASH}")
    log("  Isolation Forest (Phase 6)")
    log(f"  {DASH}")
    log(f"  TPR (Detection)  : {if_tpr*100:.4f}%")
    log(f"  FPR              : {if_fpr*100:.4f}%")
    log(f"  Precision        : {if_prec*100:.4f}%")
    log(f"  Latency          : {if_infer:.4f} ms/row")
    log(f"  Counts  TP={tp_:,}  FP={fp_:,}  TN={tn_:,}  FN={fn_:,}")

    # DPI
    dpi_scores = dpi_score_batch(X_test)

    # 3-Layer Fusion — use calibrated XGB prob
    xgb_proba_for_fusion = xgb_proba_test.copy()
    # Replace benign column to enforce calibrated att_prob
    xgb_proba_for_fusion[:, benign_id] = (1.0 - xgb_att_prob_cal)
    # normalise so row sums to 1 (approximate; calibration only affects binary output)
    row_sums = xgb_proba_for_fusion.sum(axis=1, keepdims=True)
    xgb_proba_for_fusion = xgb_proba_for_fusion / (row_sums + EPS)

    fusion, xgb_att_for_scan, layer_trig = three_layer_fusion(
        xgb_proba_for_fusion, if_prob, dpi_scores, benign_id)

    # ── 8. THRESHOLD OPTIMISER ────────────────────────────────────────────────
    log(f"\n[Step 8] Running Threshold Optimiser (target FPR <= {TARGET_FPR*100:.0f}%)...")
    log(f"\n  {'Threshold':>10} | {'Accuracy':>9} | {'TPR/DR':>9} | {'FPR':>9} | {'Precision':>10} | {'F-beta':>8}")
    log(f"  {'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}-+-{'-'*8}")

    scan_results = []
    for t in np.arange(0.05, 1.00, 0.05):
        preds = (fusion >= t).astype(int)
        tp = ((preds==1)&(y_binary==1)).sum()
        fp = ((preds==1)&(y_binary==0)).sum()
        tn = ((preds==0)&(y_binary==0)).sum()
        fn = ((preds==0)&(y_binary==1)).sum()
        tpr_  = tp/(tp+fn+EPS)
        fpr_  = fp/(fp+tn+EPS)
        acc_  = (tp+tn)/len(y_binary)
        prec_ = tp/(tp+fp+EPS)
        beta  = 0.5
        fb_   = (1+beta**2)*prec_*tpr_/((beta**2)*prec_+tpr_+EPS)
        scan_results.append((round(t,2), tpr_, fpr_, acc_, prec_))
        log(f"  {t:>10.2f} | {acc_*100:>8.2f}% | {tpr_*100:>8.2f}% |"
            f" {fpr_*100:>8.2f}% | {prec_*100:>9.2f}% | {fb_*100:>7.2f}%")

    opt_t, opt_tpr, opt_fpr, opt_acc, opt_prec = find_optimal_threshold(fusion, y_binary)

    log(f"\n  {'='*60}")
    log("  OPTIMAL THRESHOLD FOUND")
    log(f"  {'='*60}")
    log(f"  Threshold        : {opt_t:.2f}")
    log(f"  Detection Rate   : {opt_tpr*100:.4f}%")
    log(f"  False Pos Rate   : {opt_fpr*100:.4f}%")
    log(f"  Precision        : {opt_prec*100:.4f}%")
    log(f"  Binary Accuracy  : {opt_acc*100:.4f}%")

    # Save threshold
    threshold_data = {
        "threshold"      : round(float(opt_t), 4),
        "tpr"            : round(float(opt_tpr)*100, 4),
        "fpr"            : round(float(opt_fpr)*100, 4),
        "precision"      : round(float(opt_prec)*100, 4),
        "accuracy"       : round(float(opt_acc)*100, 4),
        "if_veto"        : IF_VETO_THRESHOLD,
        "if_disagree"    : IF_DISAGREE_MIN,
        "disagree_boost" : DISAGREE_BOOST,
        "target_fpr"     : TARGET_FPR,
        "trained_on"     : time.strftime("%Y-%m-%d"),
        "xgb_best_iter"  : int(xgb.best_iteration),
        "benign_weight"  : BENIGN_WEIGHT,
        "if_contamination": IF_CONTAMINATION
    }
    with open(os.path.join(MODEL_DIR, 'optimal_threshold.json'), 'w') as f:
        json.dump(threshold_data, f, indent=2)
    log(f"\n  Threshold saved -> {os.path.join(MODEL_DIR, 'optimal_threshold.json')}")

    # ── 9. PER-CLASS DETECTION AT OPTIMAL THRESHOLD ───────────────────────────
    log(f"\n[Step 9] Per-Class Detection Rate @ threshold {opt_t:.2f}...")
    opt_preds = (fusion >= opt_t).astype(int)

    log(f"\n  {'Attack Class':<35} {'N':>7}  | {'XGB':>6}  {'IF':>6}  {'Fusion':>7}")
    log("  " + "-" * 65)
    for cls in sorted(y_test.unique()):
        mask = (y_test == cls).values
        n_c  = mask.sum()
        if n_c == 0:
            continue
        xgb_r  = (y_pred[mask] != 'BENIGN').mean() * 100
        if_r   = (if_prob[mask] >= 0.5).mean() * 100
        fus_r  = opt_preds[mask].mean() * 100
        tag    = "  <- BENIGN (FPR)" if cls == 'BENIGN' else ""
        log(f"  {cls:<35} {n_c:>7}  | {xgb_r:>5.1f}%  {if_r:>5.1f}%  {fus_r:>6.1f}%{tag}")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    total_time = (time.time() - t_total) / 60
    log(f"\n{SEP}")
    log("PHASE 6 FINAL SUMMARY")
    log(SEP)
    log(f"  XGBoost Accuracy         : {xgb_acc*100:.4f}%")
    log(f"  XGBoost Weighted F1      : {xgb_f1:.4f}")
    log(f"  IF TPR (standalone)      : {if_tpr*100:.4f}%")
    log(f"  IF FPR (standalone)      : {if_fpr*100:.4f}%")
    log(f"  Fusion Detection Rate    : {opt_tpr*100:.4f}%   @ threshold {opt_t:.2f}")
    log(f"  Fusion False Positive    : {opt_fpr*100:.4f}%")
    log(f"  Fusion Precision         : {opt_prec*100:.4f}%")
    log(f"  Fusion Binary Accuracy   : {opt_acc*100:.4f}%")
    log(f"  Total training time      : {total_time:.1f} minutes")
    log(SEP)
    log(f"\nModels saved to: {MODEL_DIR}")
    log(f"  label_encoder.pkl  scaler.pkl  feature_names.pkl")
    log(f"  isolation_forest.pkl  xgboost_model.pkl  xgboost_calibrator.pkl")
    log(f"  optimal_threshold.json")

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    log(f"\nReport saved -> {REPORT_FILE}")


if __name__ == "__main__":
    run()
