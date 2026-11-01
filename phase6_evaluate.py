"""
=================================================================
PHASE 6 -- DEEP EVALUATION
=================================================================
Loads trained models_v6 and runs a comprehensive evaluation:
  1. XGBoost per-class precision/recall/F1
  2. Isolation Forest standalone binary metrics
  3. 3-Layer fusion threshold scan + layer contribution analysis
  4. Per-class fusion detection at optimal + two reference thresholds
  5. Layer-by-layer breakdown (how many alerts came from each layer)

Reads: models_v6/ + optimal_threshold.json
Reads: test.parquet
Writes: phase6_eval_report.txt
=================================================================
"""

import os, sys, json, time, joblib, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

MODEL_DIR   = r"D:\DDos 1\models_v6"
TEST_FILE   = r"D:\DDos 1\test.parquet"
REPORT_OUT  = r"D:\DDos 1\phase6_eval_report.txt"
EPS         = 1e-9

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))

SEP  = "=" * 72
DASH = "-" * 72

# ── DPI ENGINE (Phase 6 Enhanced) ─────────────────────────────────────────────
def dpi_score_batch(df):
    score = np.zeros(len(df), dtype=np.float32)
    syn  = df.get('syn_count', pd.Series(0.0, index=df.index)).values.astype(float)
    rate = df.get('Rate',      pd.Series(0.0, index=df.index)).values.astype(float)
    udp  = df.get('UDP',       pd.Series(0.0, index=df.index)).values.astype(float)
    icmp = df.get('ICMP',      pd.Series(0.0, index=df.index)).values.astype(float)
    iat  = df.get('IAT_mean',  pd.Series(0.0, index=df.index)).values.astype(float)
    n    = df.get('Number',    pd.Series(1.0, index=df.index)).values.astype(float)
    pkt  = df.get('AVG',       pd.Series(0.0, index=df.index)).values.astype(float)

    score = np.where((syn > 50) & (rate > 2000),              np.maximum(score, 0.95), score)
    score = np.where((udp > 0.8) & (rate > 2000),             np.maximum(score, 0.90), score)
    score = np.where((icmp > 0.5) & (rate > 1000),            np.maximum(score, 0.88), score)
    syn_ratio = np.clip(syn / (n + EPS), 0, 1)
    score = np.where((syn_ratio > 0.85) & (rate > 200),       np.maximum(score, 0.88), score)
    score = np.where((iat > 0.05) & (n > 5) & (rate < 100) & (rate > 0), np.maximum(score, 0.82), score)
    score = np.where((n > 15) & (pkt < 120) & (rate < 10) & (rate > 0),  np.maximum(score, 0.78), score)
    score = np.where((rate < 10) & (iat > 0.5),               np.minimum(score, 0.05), score)
    return np.clip(score, 0.0, 1.0).astype(np.float32)

# ── 3-LAYER FUSION ────────────────────────────────────────────────────────────
def three_layer_fusion(xgb_proba, if_prob, dpi_scores, benign_id,
                       if_veto=0.75, if_disagree=0.60, xgb_benign_max=0.30, boost=0.15):
    n = len(if_prob)
    xgb_att_prob   = (1.0 - xgb_proba[:, benign_id]).astype(np.float32)
    xgb_confidence = xgb_proba.max(axis=1).astype(np.float32)

    w_xgb = np.where(xgb_confidence >= 0.80, 0.60,
            np.where(xgb_confidence >= 0.50, 0.40, 0.25)).astype(np.float32)
    w_if  = np.where(xgb_confidence >= 0.80, 0.25,
            np.where(xgb_confidence >= 0.50, 0.45, 0.60)).astype(np.float32)
    w_dpi = np.full(n, 0.15, dtype=np.float32)

    fusion = np.clip(w_xgb * xgb_att_prob + w_if * if_prob + w_dpi * dpi_scores, 0, 1)

    disagree_mask = (if_prob > if_disagree) & (xgb_att_prob < xgb_benign_max)
    fusion = np.where(disagree_mask, np.clip(fusion + boost, 0, 1), fusion)

    veto_mask = (if_prob > if_veto)
    fusion = np.where(veto_mask, np.maximum(fusion, 0.90), fusion)

    layer_triggered = np.where(veto_mask, 1,
                      np.where(disagree_mask, 2, 3)).astype(np.int8)

    return fusion.astype(np.float32), xgb_att_prob, layer_triggered

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
log(SEP)
log("PHASE 6  --  DEEP EVALUATION")
log(SEP)

# ── LOAD THRESHOLD CONFIG ─────────────────────────────────────────────────────
thresh_file = os.path.join(MODEL_DIR, 'optimal_threshold.json')
with open(thresh_file) as f:
    thresh_cfg = json.load(f)

OPT_T         = thresh_cfg['threshold']
IF_VETO       = thresh_cfg.get('if_veto',        0.75)
IF_DISAGREE   = thresh_cfg.get('if_disagree',    0.60)
DISAGREE_BOOST= thresh_cfg.get('disagree_boost', 0.15)

log(f"\n  Loaded threshold config from: {thresh_file}")
log(f"  Optimal threshold : {OPT_T}")
log(f"  IF Veto threshold : {IF_VETO}")
log(f"  Target TPR        : {thresh_cfg['tpr']:.2f}%")
log(f"  Target FPR        : {thresh_cfg['fpr']:.2f}%")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
log("\n[1] Loading test dataset...")
test_df = pd.read_parquet(TEST_FILE)
X_test  = test_df.drop(columns=['Label'])
y_test  = test_df['Label']
log(f"    Shape: {test_df.shape[0]:,} rows x {test_df.shape[1]} cols")

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
log("\n[2] Loading models...")
le      = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
scaler  = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
iso_d   = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
xgb     = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
calib_d = joblib.load(os.path.join(MODEL_DIR, 'xgboost_calibrator.pkl'))

iso     = iso_d['model']
if_min  = iso_d['if_min']
if_max  = iso_d['if_max']
platt   = calib_d['calibrator']
benign_id = calib_d['benign_id']

try:
    xgb.set_params(device='cpu')
except Exception:
    pass

log(f"    LabelEncoder  : {len(le.classes_)} classes")
log(f"    IF min/max    : {if_min:.4f} / {if_max:.4f}")
log(f"    Benign ID     : {benign_id} ({le.classes_[benign_id]})")

# ── PREPROCESS ───────────────────────────────────────────────────────────────
log("\n[3] Scaling features...")
X_scaled = scaler.transform(X_test)

# ── XGBOOST SCORING ──────────────────────────────────────────────────────────
log("\n[4] XGBoost inference...")
t0 = time.time()
xgb_proba  = xgb.predict_proba(X_test)
xgb_time   = (time.time()-t0)*1000/len(X_test)
y_pred     = le.inverse_transform(xgb_proba.argmax(axis=1))
xgb_acc    = accuracy_score(y_test, y_pred)
xgb_f1     = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Apply calibration to binary attack probability
raw_att    = (1.0 - xgb_proba[:, benign_id]).reshape(-1, 1)
att_calib  = platt.predict_proba(raw_att)[:, 1].astype(np.float32)

log(f"\n{DASH}")
log("XGBoost Multi-Class Classification (Phase 6)")
log(DASH)
log(f"  Accuracy         : {xgb_acc*100:.4f}%")
log(f"  Weighted F1      : {xgb_f1:.4f}")
log(f"  Latency          : {xgb_time:.4f} ms/row")
log()
log(classification_report(y_test, y_pred, zero_division=0))

# ── ISOLATION FOREST SCORING ─────────────────────────────────────────────────
log(f"{DASH}")
log("Isolation Forest (Phase 6)")
log(DASH)
t0      = time.time()
ifs_raw = iso.score_samples(X_scaled)
if_time = (time.time()-t0)*1000/len(X_scaled)
ifs_c   = np.clip(ifs_raw, if_min, if_max)
if_prob = (1.0 - (ifs_c - if_min) / (if_max - if_min + EPS)).astype(np.float32)

y_binary = (y_test != 'BENIGN').astype(int).values
if50  = (if_prob >= 0.50).astype(int)
tp_ = ((if50==1)&(y_binary==1)).sum()
fp_ = ((if50==1)&(y_binary==0)).sum()
tn_ = ((if50==0)&(y_binary==0)).sum()
fn_ = ((if50==0)&(y_binary==1)).sum()
if_tpr  = tp_/(tp_+fn_+EPS)
if_fpr  = fp_/(fp_+tn_+EPS)
if_prec = tp_/(tp_+fp_+EPS)
if_f1   = 2*if_prec*if_tpr/(if_prec+if_tpr+EPS)

log(f"  IF score range   : {ifs_raw.min():.4f}  to  {ifs_raw.max():.4f}")
log(f"  TPR (at t=0.50)  : {if_tpr*100:.4f}%")
log(f"  FPR (at t=0.50)  : {if_fpr*100:.4f}%")
log(f"  Precision        : {if_prec*100:.4f}%")
log(f"  F1-score         : {if_f1:.4f}")
log(f"  Counts  TP={tp_:,}  FP={fp_:,}  TN={tn_:,}  FN={fn_:,}")
log(f"  Latency          : {if_time:.4f} ms/row")

# ── DPI ───────────────────────────────────────────────────────────────────────
dpi_scores = dpi_score_batch(X_test)
log(f"\n  DPI mean score   : {dpi_scores.mean():.4f}")
log(f"  DPI score > 0    : {(dpi_scores > 0).sum():,} flows")

# ── 3-LAYER FUSION ────────────────────────────────────────────────────────────
log(f"\n{DASH}")
log("3-Layer Fusion Score Scan")
log(DASH)

# Build modified proba with calibrated attack prob
xgb_proba_mod = xgb_proba.copy()
xgb_proba_mod[:, benign_id] = (1.0 - att_calib)
row_sums = xgb_proba_mod.sum(axis=1, keepdims=True)
xgb_proba_mod = xgb_proba_mod / (row_sums + EPS)

fusion, xgb_att_out, layer_trig = three_layer_fusion(
    xgb_proba_mod, if_prob, dpi_scores, benign_id,
    if_veto=IF_VETO, if_disagree=IF_DISAGREE, boost=DISAGREE_BOOST
)

log(f"\n  {'Threshold':>10} | {'Accuracy':>9} | {'TPR/DR':>9} | {'FPR':>9} | {'Precision':>10} | {'F-beta':>8}")
log(f"  {'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}-+-{'-'*8}")

for t in np.arange(0.05, 1.00, 0.05):
    preds = (fusion >= t).astype(int)
    tp = ((preds==1)&(y_binary==1)).sum()
    fp = ((preds==1)&(y_binary==0)).sum()
    tn = ((preds==0)&(y_binary==0)).sum()
    fn = ((preds==0)&(y_binary==1)).sum()
    tpr_  = tp/(tp+fn+EPS);  fpr_ = fp/(fp+tn+EPS)
    acc_  = (tp+tn)/len(y_binary)
    prec_ = tp/(tp+fp+EPS)
    beta  = 0.5
    fb_   = (1+beta**2)*prec_*tpr_/((beta**2)*prec_+tpr_+EPS)
    marker = " <-- TARGET" if abs(t - OPT_T) < 0.001 else ""
    log(f"  {t:>10.2f} | {acc_*100:>8.2f}% | {tpr_*100:>8.2f}% |"
        f" {fpr_*100:>8.2f}% | {prec_*100:>9.2f}% | {fb_*100:>7.2f}%{marker}")

# ── LAYER CONTRIBUTION BREAKDOWN ──────────────────────────────────────────────
opt_preds = (fusion >= OPT_T).astype(int)
total_alerts = opt_preds.sum()
layer1_alerts = ((layer_trig == 1) & (opt_preds == 1)).sum()
layer2_alerts = ((layer_trig == 2) & (opt_preds == 1)).sum()
layer3_alerts = ((layer_trig == 3) & (opt_preds == 1)).sum()

log(f"\n{DASH}")
log(f"Layer Contribution @ threshold {OPT_T:.2f}  (total alerts = {total_alerts:,})")
log(DASH)
log(f"  Layer 1 (IF Hard Veto)       : {layer1_alerts:>7,} alerts ({layer1_alerts/max(total_alerts,1)*100:.2f}%)")
log(f"  Layer 2 (Disagree Amplifier) : {layer2_alerts:>7,} alerts ({layer2_alerts/max(total_alerts,1)*100:.2f}%)")
log(f"  Layer 3 (Adaptive Fusion)    : {layer3_alerts:>7,} alerts ({layer3_alerts/max(total_alerts,1)*100:.2f}%)")

# How many attacks would XGB alone have missed but Layer 1/2 caught?
xgb_alone_preds = (xgb_att_out >= 0.5).astype(int)
missed_by_xgb  = ((xgb_alone_preds == 0) & (y_binary == 1)).sum()
caught_by_if   = ((layer_trig <= 2) & (opt_preds == 1) & (y_binary == 1)).sum()
log(f"\n  Attacks XGB alone would miss : {missed_by_xgb:,}")
log(f"  Of those, caught by IF layers: {caught_by_if:,}  (zero-day safety net)")

# ── PER-CLASS AT OPTIMAL + REFERENCE THRESHOLDS ───────────────────────────────
log(f"\n{DASH}")
log(f"Per-Class Detection Rate  (Phase 6)")
log(DASH)

preds_opt = (fusion >= OPT_T).astype(int)
preds_90  = (fusion >= 0.90).astype(int)   # ultra-strict reference

log(f"\n  {'Attack Class':<35} {'N':>7}  | {'XGB':>6}  {'IF':>6}  {'Fus@'+str(OPT_T):>8}  {'Fus@0.90':>9}")
log("  " + "-" * 72)

for cls in sorted(y_test.unique()):
    mask = (y_test == cls).values
    n_c  = mask.sum()
    if n_c == 0:
        continue
    xgb_r = (y_pred[mask] != 'BENIGN').mean() * 100
    if_r  = (if_prob[mask] >= 0.5).mean() * 100
    f_opt = preds_opt[mask].mean() * 100
    f_90  = preds_90[mask].mean() * 100
    tag   = "  <- BENIGN (FPR)" if cls == 'BENIGN' else ""
    log(f"  {cls:<35} {n_c:>7}  | {xgb_r:>5.1f}%  {if_r:>5.1f}%  {f_opt:>7.1f}%  {f_90:>8.1f}%{tag}")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
tp = ((preds_opt==1)&(y_binary==1)).sum()
fp = ((preds_opt==1)&(y_binary==0)).sum()
tn = ((preds_opt==0)&(y_binary==0)).sum()
fn = ((preds_opt==0)&(y_binary==1)).sum()
opt_tpr  = tp/(tp+fn+EPS)
opt_fpr  = fp/(fp+tn+EPS)
opt_prec = tp/(tp+fp+EPS)
opt_acc  = (tp+tn)/len(y_binary)

log(f"\n{SEP}")
log("PHASE 6 FINAL RESULTS")
log(SEP)
log(f"  XGBoost Multi-Class Accuracy  : {xgb_acc*100:.4f}%")
log(f"  XGBoost Weighted F1-Score     : {xgb_f1:.4f}")
log()
log(f"  Isolation Forest TPR          : {if_tpr*100:.4f}%")
log(f"  Isolation Forest FPR          : {if_fpr*100:.4f}%")
log(f"  Isolation Forest Precision    : {if_prec*100:.4f}%")
log()
log(f"  3-Layer Fusion @ t={OPT_T:.2f}")
log(f"    Detection Rate (TPR)        : {opt_tpr*100:.4f}%")
log(f"    False Positive Rate (FPR)   : {opt_fpr*100:.4f}%")
log(f"    Precision                   : {opt_prec*100:.4f}%")
log(f"    Binary Accuracy             : {opt_acc*100:.4f}%")
log(f"    TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")
log()

targets = {
    "XGBoost Accuracy >= 82%"       : xgb_acc >= 0.82,
    "XGBoost F1 >= 0.80"            : xgb_f1  >= 0.80,
    "IF TPR >= 40%"                 : if_tpr  >= 0.40,
    "IF FPR <= 3%"                  : if_fpr  <= 0.03,
    "Fusion TPR >= 95%"             : opt_tpr >= 0.95,
    "Fusion FPR <= 2%"              : opt_fpr <= 0.02,
    "Fusion Precision >= 98%"       : opt_prec>= 0.98,
    "Fusion Accuracy >= 95%"        : opt_acc >= 0.95,
}
log("  Verification Checklist:")
all_pass = True
for name, passed in targets.items():
    mark = "[PASS]" if passed else "[FAIL]"
    log(f"    {mark}  {name}")
    if not passed:
        all_pass = False
log()
log(f"  Overall: {'ALL TARGETS MET - INDUSTRY READY' if all_pass else 'SOME TARGETS MISSED - REVIEW ABOVE'}")
log(SEP)

with open(REPORT_OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
log(f"\nReport saved -> {REPORT_OUT}")
