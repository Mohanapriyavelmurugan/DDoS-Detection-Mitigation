"""
=================================================================
DEEP EVALUATION -- models_v5 (Phase 5 Cost-Sensitive Models)
=================================================================
Produces:
  1. XGBoost per-class precision / recall / F1
  2. Isolation Forest binary metrics
  3. Full fusion threshold scan  (0.05 -> 0.95)
  4. Per-attack-class fusion detection rate
=================================================================
"""

import os, sys, time, joblib, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score

warnings.filterwarnings('ignore')

# Force UTF-8 output so Unicode chars print correctly on Windows cp1252
sys.stdout.reconfigure(encoding='utf-8')

MODEL_DIR   = r"D:\DDos 1\models_v5"
TEST_FILE   = r"D:\DDos 1\test.parquet"
REPORT_OUT  = r"D:\DDos 1\deep_eval_v5_report.txt"
EPS         = 1e-9

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))

SEP  = "=" * 68
DASH = "-" * 68

# -- DPI rule (same as phase5_optimization.py) ----------------------
def evaluate_dpi(df):
    score = np.zeros(len(df), dtype=np.float32)
    syn   = df.get('syn_count', pd.Series(0, index=df.index)).values
    rate  = df.get('Rate',      pd.Series(0, index=df.index)).values
    udp   = df.get('UDP',       pd.Series(0, index=df.index)).values
    iat   = df.get('IAT_mean',  pd.Series(0, index=df.index)).values
    score = np.where((syn > 50)  & (rate > 5000),  np.maximum(score, 0.95), score)
    score = np.where((udp > 0.8) & (rate > 5000),  np.maximum(score, 0.90), score)
    score = np.where((rate < 200) & (iat  > 0.01), np.minimum(score, 0.10), score)
    return np.clip(score, 0.0, 1.0).astype(np.float32)

# -- LOAD ------------------------------------------------------------
log(SEP)
log("DEEP EVALUATION -- models_v5  (Phase 5 Cost-Sensitive)")
log(SEP)

log("\n[1] Loading test dataset...")
test_df = pd.read_parquet(TEST_FILE)
X_test  = test_df.drop(columns=['Label'])
y_test  = test_df['Label']
log(f"    Shape: {test_df.shape[0]:,} rows x {test_df.shape[1]} columns")

log("\n[2] Loading models...")
le      = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
scaler  = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
iso_d   = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
xgb     = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl'))

iso    = iso_d['model']
if_min = iso_d['if_min']
if_max = iso_d['if_max']

try:
    xgb.set_params(device='cpu')
except Exception:
    pass

log(f"    LabelEncoder : {len(le.classes_)} classes")
log(f"    IF min/max   : {if_min:.4f} / {if_max:.4f}")

# -- PREPROCESSING ---------------------------------------------------
log("\n[3] Preprocessing features...")
X_scaled = scaler.transform(X_test)

# -- XGBoost SCORING -------------------------------------------------
log("\n[4] XGBoost multi-class scoring...")
t0 = time.time()
xgb_proba     = xgb.predict_proba(X_test)
y_pred_enc    = xgb_proba.argmax(axis=1)
y_pred        = le.inverse_transform(y_pred_enc)
xgb_time      = (time.time() - t0) * 1000 / len(X_test)

benign_id     = list(le.classes_).index('BENIGN')
xgb_att_prob  = (1.0 - xgb_proba[:, benign_id]).astype(np.float32)

xgb_acc = accuracy_score(y_test, y_pred)
xgb_f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)

log("\n" + DASH)
log("XGBoost Multi-Class Classification (models_v5)")
log(DASH)
log(f"  Overall Accuracy : {xgb_acc*100:.4f}%")
log(f"  Weighted F1      : {xgb_f1:.4f}")
log(f"  Inference latency: {xgb_time:.4f} ms/row")
log()
log(classification_report(y_test, y_pred, zero_division=0))

# -- ISOLATION FOREST SCORING ----------------------------------------
log("\n" + DASH)
log("Isolation Forest Anomaly Detection (models_v5)")
log(DASH)

t0       = time.time()
ifs_raw  = iso.score_samples(X_scaled)
if_time  = (time.time() - t0) * 1000 / len(X_scaled)

ifs_clip = np.clip(ifs_raw, if_min, if_max)
if_prob  = (1.0 - (ifs_clip - if_min) / (if_max - if_min + EPS)).astype(np.float32)

y_binary   = (y_test != 'BENIGN').astype(int).values
if_preds   = (if_prob >= 0.5).astype(int)
tp = ((if_preds == 1) & (y_binary == 1)).sum()
fp = ((if_preds == 1) & (y_binary == 0)).sum()
tn = ((if_preds == 0) & (y_binary == 0)).sum()
fn = ((if_preds == 0) & (y_binary == 1)).sum()

if_tpr  = tp / (tp + fn + EPS)
if_fpr  = fp / (fp + tn + EPS)
if_prec = tp / (tp + fp + EPS)
if_f1   = 2 * if_prec * if_tpr / (if_prec + if_tpr + EPS)

log(f"  IF score range   : {ifs_raw.min():.4f}  to  {ifs_raw.max():.4f}")
log(f"  IF prob range    : {if_prob.min():.4f}  to  {if_prob.max():.4f}")
log(f"  TPR (Detection)  : {if_tpr*100:.4f}%")
log(f"  FPR              : {if_fpr*100:.4f}%")
log(f"  Precision        : {if_prec*100:.4f}%")
log(f"  F1-score         : {if_f1:.4f}")
log(f"  Counts  TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")
log(f"  Inference latency: {if_time:.4f} ms/row")

# -- DPI SCORING -----------------------------------------------------
log("\n" + DASH)
log("DPI Scoring (models_v5)")
log(DASH)
dpi_scores = evaluate_dpi(X_test)
log(f"  DPI mean score   : {dpi_scores.mean():.4f}")
log(f"  DPI score > 0    : {(dpi_scores > 0).sum():,} flows")

# -- FUSION THRESHOLD SCAN -------------------------------------------
log("\n" + DASH)
log("Fusion Score Scan  (0.4*IF + 0.4*XGB + 0.2*DPI)")
log(DASH)
fusion = np.clip(0.4 * if_prob + 0.4 * xgb_att_prob + 0.2 * dpi_scores, 0.0, 1.0)

log(f"\n  {'Threshold':>10} | {'Accuracy':>9} | {'TPR/DR':>9} | {'FPR':>9} | {'Precision':>10} | {'F-beta(0.5)':>12}")
log(f"  {'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}-+-{'-'*12}")

scan_results = []
for t in np.arange(0.05, 1.00, 0.05):
    preds = (fusion >= t).astype(int)
    tp_ = ((preds == 1) & (y_binary == 1)).sum()
    fp_ = ((preds == 1) & (y_binary == 0)).sum()
    tn_ = ((preds == 0) & (y_binary == 0)).sum()
    fn_ = ((preds == 0) & (y_binary == 1)).sum()
    tpr_  = tp_ / (tp_ + fn_ + EPS)
    fpr_  = fp_ / (fp_ + tn_ + EPS)
    acc_  = (tp_ + tn_) / len(y_binary)
    prec_ = tp_ / (tp_ + fp_ + EPS)
    beta  = 0.5
    fb_   = (1 + beta**2) * prec_ * tpr_ / (beta**2 * prec_ + tpr_ + EPS)
    scan_results.append({'t': t, 'acc': acc_, 'tpr': tpr_, 'fpr': fpr_,
                         'prec': prec_, 'fb': fb_,
                         'tp': tp_, 'fp': fp_, 'tn': tn_, 'fn': fn_})
    log(f"  {t:>10.2f} | {acc_*100:>8.2f}% | {tpr_*100:>8.2f}% | {fpr_*100:>8.2f}% | {prec_*100:>9.2f}% | {fb_*100:>11.2f}%")

# -- FIND OPTIMAL THRESHOLDS -----------------------------------------
df_scan = pd.DataFrame(scan_results)

best_fb  = df_scan.loc[df_scan['fb'].idxmax()]
cons_df  = df_scan[df_scan['fpr'] < 0.05]
best_cons = cons_df.sort_values('tpr', ascending=False).iloc[0] if len(cons_df) else best_fb
ultra_df  = df_scan[df_scan['fpr'] < 0.01]
best_ultra = ultra_df.sort_values('tpr', ascending=False).iloc[0] if len(ultra_df) else best_cons

t_balanced  = round(float(best_fb['t']),    2)
t_conserv   = round(float(best_cons['t']),  2)
t_ultra     = round(float(best_ultra['t']), 2)

log(f"\n  -- OPTIMAL THRESHOLDS --")
log(f"  [BALANCED]      t={t_balanced:.2f}  TPR={best_fb['tpr']*100:.2f}%  FPR={best_fb['fpr']*100:.2f}%  Acc={best_fb['acc']*100:.2f}%  Prec={best_fb['prec']*100:.2f}%")
log(f"  [CONSERVATIVE]  t={t_conserv:.2f}  TPR={best_cons['tpr']*100:.2f}%  FPR={best_cons['fpr']*100:.2f}%  Acc={best_cons['acc']*100:.2f}%  Prec={best_cons['prec']*100:.2f}%")
log(f"  [ULTRA LOW FPR] t={t_ultra:.2f}  TPR={best_ultra['tpr']*100:.2f}%  FPR={best_ultra['fpr']*100:.2f}%  Acc={best_ultra['acc']*100:.2f}%  Prec={best_ultra['prec']*100:.2f}%")

# -- PER-CLASS FUSION DETECTION RATE ---------------------------------
log("\n" + DASH)
log(f"Per-Class Fusion Detection Rate  (models_v5  -- 3 key thresholds)")
log(DASH)

preds_bal   = (fusion >= t_balanced).astype(int)
preds_cons  = (fusion >= t_conserv).astype(int)
preds_ultra = (fusion >= t_ultra).astype(int)
if_pred50   = (if_prob >= 0.5).astype(int)

hdr = (f"  {'Attack Class':<35} {'N':>7}  |"
       f" {'XGB':>6}  {'IF':>6} "
       f" {'Fus@'+str(t_balanced):>8} "
       f" {'Fus@'+str(t_conserv):>8} "
       f" {'Fus@'+str(t_ultra):>8}")
log("\n" + hdr)
log("  " + "-" * (len(hdr) - 2))

for cls in sorted(y_test.unique()):
    mask = (y_test == cls).values
    n    = mask.sum()
    if n == 0:
        continue
    xgb_r  = (y_pred[mask] != 'BENIGN').mean() * 100
    if_r   = if_pred50[mask].mean() * 100
    fus_b  = preds_bal[mask].mean()   * 100
    fus_c  = preds_cons[mask].mean()  * 100
    fus_u  = preds_ultra[mask].mean() * 100
    tag    = "  (BENIGN -- FPR)" if cls == 'BENIGN' else ""
    log(f"  {cls:<35} {n:>7}  |"
        f" {xgb_r:>5.1f}%  {if_r:>5.1f}%"
        f"  {fus_b:>7.1f}%  {fus_c:>7.1f}%  {fus_u:>7.1f}%{tag}")

# -- SUMMARY ---------------------------------------------------------
log("\n" + SEP)
log("FINAL SUMMARY -- models_v5")
log(SEP)
log(f"  XGBoost Multi-Class Accuracy     : {xgb_acc*100:.4f}%")
log(f"  XGBoost Weighted F1-Score        : {xgb_f1:.4f}")
log()
log(f"  Isolation Forest TPR             : {if_tpr*100:.4f}%")
log(f"  Isolation Forest FPR             : {if_fpr*100:.4f}%")
log(f"  Isolation Forest Precision       : {if_prec*100:.4f}%")
log()
log(f"  RECOMMENDED THRESHOLD (Balanced) : {t_balanced:.2f}")
log(f"    Binary Accuracy  : {best_fb['acc']*100:.4f}%")
log(f"    Detection Rate   : {best_fb['tpr']*100:.4f}%")
log(f"    False Positive   : {best_fb['fpr']*100:.4f}%")
log(f"    Precision        : {best_fb['prec']*100:.4f}%")
log()
log(f"  CONSERVATIVE THRESHOLD           : {t_conserv:.2f}")
log(f"    Binary Accuracy  : {best_cons['acc']*100:.4f}%")
log(f"    Detection Rate   : {best_cons['tpr']*100:.4f}%")
log(f"    False Positive   : {best_cons['fpr']*100:.4f}%")
log(f"    Precision        : {best_cons['prec']*100:.4f}%")
log()
log(f"  ULTRA LOW FPR THRESHOLD          : {t_ultra:.2f}")
log(f"    Binary Accuracy  : {best_ultra['acc']*100:.4f}%")
log(f"    Detection Rate   : {best_ultra['tpr']*100:.4f}%")
log(f"    False Positive   : {best_ultra['fpr']*100:.4f}%")
log(f"    Precision        : {best_ultra['prec']*100:.4f}%")
log(SEP)

with open(REPORT_OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
log(f"\nReport saved -> {REPORT_OUT}")
