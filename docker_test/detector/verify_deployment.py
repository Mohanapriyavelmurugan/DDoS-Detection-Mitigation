"""
===============================================================
DEPLOYMENT VERIFIER — Phase 6 DDoS Detection System
===============================================================
This script runs automatically during Docker build (RUN python verify_deployment.py).
It catches any model file issues BEFORE transferring to MEC.

Checks performed:
  1. All 6 required model files exist
  2. All model files load without errors
  3. Feature count matches expected (53 features)
  4. Label classes are present and correct
  5. Threshold JSON has all required keys
  6. A test inference runs end-to-end without crashing
  7. Inference speed benchmark (flows per second)
===============================================================
"""

import os
import sys
import time
import json
import joblib
import numpy as np

MODEL_DIR = "./models_v6"
EPS = 1e-9

# ── Colour output ──────────────────────────────────────────────────────
OK   = "\033[92m[OK]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[96m[INFO]\033[0m"

print("\n" + "="*60)
print("  PHASE 6 DEPLOYMENT VERIFICATION")
print("="*60)

errors = []

# ── Check 1: All required files exist ─────────────────────────────────
print(f"\n{INFO} Check 1: Model files present...")
REQUIRED_FILES = [
    "label_encoder.pkl",
    "scaler.pkl",
    "isolation_forest.pkl",
    "xgboost_model.pkl",
    "xgboost_calibrator.pkl",
    "optimal_threshold.json",
    "feature_names.pkl",
]

for fname in REQUIRED_FILES:
    fpath = os.path.join(MODEL_DIR, fname)
    size_mb = os.path.getsize(fpath) / (1024 * 1024) if os.path.exists(fpath) else 0
    if os.path.exists(fpath):
        print(f"  {OK}  {fname:<35} ({size_mb:.2f} MB)")
    else:
        print(f"  {FAIL} {fname:<35} NOT FOUND")
        errors.append(f"Missing file: {fname}")

# ── Check 2: Load all models ───────────────────────────────────────────
print(f"\n{INFO} Check 2: Loading all model files...")
try:
    le     = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print(f"  {OK}  LabelEncoder    → {len(le.classes_)} classes")
except Exception as e:
    errors.append(f"LabelEncoder load failed: {e}")

try:
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"  {OK}  StandardScaler  → {scaler.n_features_in_} features")
except Exception as e:
    errors.append(f"Scaler load failed: {e}")

try:
    iso_d  = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    iso    = iso_d["model"]
    if_min = iso_d["if_min"]
    if_max = iso_d["if_max"]
    print(f"  {OK}  IsolationForest → n_est={iso.n_estimators}  score range=[{if_min:.4f}, {if_max:.4f}]")
except Exception as e:
    errors.append(f"IsolationForest load failed: {e}")

try:
    xgb = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.pkl"))
    try:
        xgb.set_params(device="cpu")  # Force CPU for MEC (no GPU assumed)
    except Exception:
        pass
    print(f"  {OK}  XGBoostClassifier → best_iter={getattr(xgb, 'best_iteration', 'N/A')}")
except Exception as e:
    errors.append(f"XGBoost load failed: {e}")

try:
    calib_d = joblib.load(os.path.join(MODEL_DIR, "xgboost_calibrator.pkl"))
    platt   = calib_d["calibrator"]
    benign_id_stored = calib_d["benign_id"]
    print(f"  {OK}  Platt Calibrator  → benign_id={benign_id_stored}  coef={platt.coef_[0][0]:.4f}")
except Exception as e:
    errors.append(f"Calibrator load failed: {e}")

try:
    feats = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    print(f"  {OK}  Feature Names   → {len(feats)} features")
except Exception as e:
    errors.append(f"Feature names load failed: {e}")

try:
    with open(os.path.join(MODEL_DIR, "optimal_threshold.json")) as f:
        cfg = json.load(f)
    print(f"  {OK}  Threshold JSON  → threshold={cfg['threshold']}  target_fpr={cfg.get('target_fpr', 'N/A')}")
except Exception as e:
    errors.append(f"Threshold JSON load failed: {e}")

if errors:
    print(f"\n{FAIL} MODEL LOADING FAILED — cannot continue verification")
    for e in errors:
        print(f"  → {e}")
    sys.exit(1)

# ── Check 3: Feature count consistency ────────────────────────────────
print(f"\n{INFO} Check 3: Feature count consistency...")
expected_features = len(feats)
scaler_features   = scaler.n_features_in_

if expected_features == scaler_features:
    print(f"  {OK}  feature_names.pkl ({expected_features}) == scaler.n_features_in_ ({scaler_features})")
else:
    errors.append(f"Feature count mismatch: feature_names={expected_features} vs scaler={scaler_features}")
    print(f"  {FAIL} MISMATCH: feature_names={expected_features} vs scaler={scaler_features}")

# ── Check 4: BENIGN class present ─────────────────────────────────────
print(f"\n{INFO} Check 4: Label classes verification...")
if "BENIGN" in le.classes_:
    benign_id = list(le.classes_).index("BENIGN")
    print(f"  {OK}  BENIGN class found at index {benign_id}")
    print(f"  {OK}  Total attack classes: {len(le.classes_) - 1}")
    # Print all classes for log
    for i, cls in enumerate(sorted(le.classes_)):
        tag = " ← BENIGN" if cls == "BENIGN" else ""
        print(f"         [{i:>2}] {cls}{tag}")
else:
    errors.append("BENIGN class not found in LabelEncoder")
    print(f"  {FAIL} BENIGN class not found")

# ── Check 5: Threshold JSON has all required keys ─────────────────────
print(f"\n{INFO} Check 5: Threshold configuration validation...")
required_keys = ["threshold", "if_veto", "if_disagree", "disagree_boost", "target_fpr"]
for key in required_keys:
    if key in cfg:
        print(f"  {OK}  {key:<20} = {cfg[key]}")
    else:
        errors.append(f"Missing key in optimal_threshold.json: {key}")
        print(f"  {FAIL} Missing key: {key}")

# ── Check 6: End-to-end test inference ────────────────────────────────
print(f"\n{INFO} Check 6: End-to-end inference test...")
try:
    # Create a dummy flow with all zeros (simulates benign idle traffic)
    dummy = np.zeros((1, len(feats)), dtype=np.float32)
    
    # Run through full pipeline
    X_scaled      = scaler.transform(dummy)
    xgb_proba     = xgb.predict_proba(dummy)
    xgb_att_prob  = (1.0 - xgb_proba[:, benign_id]).astype(np.float32)
    ifs_raw       = iso.score_samples(X_scaled)
    ifs_c         = np.clip(ifs_raw, if_min, if_max)
    if_prob       = (1.0 - (ifs_c - if_min) / (if_max - if_min + EPS))
    
    # Platt calibration
    raw_att       = xgb_att_prob.reshape(-1, 1)
    cal_prob      = platt.predict_proba(raw_att)[:, 1]
    
    # Fusion
    fusion_score  = float(np.clip(0.40 * xgb_att_prob[0] + 0.45 * if_prob[0] + 0.15 * 0.0, 0, 1))
    is_attack     = fusion_score >= cfg["threshold"]
    
    print(f"  {OK}  Dummy inference successful")
    print(f"         XGB attack prob : {xgb_att_prob[0]:.4f}")
    print(f"         IF anomaly prob : {if_prob[0]:.4f}")
    print(f"         Calibrated prob : {cal_prob[0]:.4f}")
    print(f"         Fusion score    : {fusion_score:.4f}")
    print(f"         Decision        : {'ATTACK' if is_attack else 'BENIGN'} (threshold={cfg['threshold']})")
except Exception as e:
    errors.append(f"End-to-end inference failed: {e}")
    print(f"  {FAIL} Inference crashed: {e}")

# ── Check 7: Speed benchmark ───────────────────────────────────────────
print(f"\n{INFO} Check 7: Inference speed benchmark (100 flows)...")
try:
    dummy_batch   = np.zeros((100, len(feats)), dtype=np.float32)
    t0            = time.time()
    X_sc_batch    = scaler.transform(dummy_batch)
    xgb_proba_b   = xgb.predict_proba(dummy_batch)
    ifs_b         = iso.score_samples(X_sc_batch)
    elapsed_ms    = (time.time() - t0) * 1000
    flows_per_sec = 100 / (elapsed_ms / 1000)
    print(f"  {OK}  100 flows processed in {elapsed_ms:.1f}ms")
    print(f"         Throughput: {flows_per_sec:.0f} flows/second")
    print(f"         Latency per flow: {elapsed_ms/100:.2f}ms")
    
    if flows_per_sec < 100:
        print(f"  ⚠️  WARNING: Throughput below 100 flows/sec — check MEC load")
    else:
        print(f"  {OK}  Throughput is deployment-ready")
except Exception as e:
    errors.append(f"Speed benchmark failed: {e}")
    print(f"  {FAIL} Benchmark failed: {e}")

# ── Final result ────────────────────────────────────────────────────────
print("\n" + "="*60)
if errors:
    print(f"  {FAIL} VERIFICATION FAILED — {len(errors)} error(s) found:")
    for e in errors:
        print(f"    → {e}")
    print("="*60)
    sys.exit(1)
else:
    print(f"  {OK}  ALL CHECKS PASSED — Image is MEC-deployment ready")
    print(f"       Models baked in  : {MODEL_DIR}")
    print(f"       Feature count    : {len(feats)}")
    print(f"       Attack classes   : {len(le.classes_) - 1}")
    print(f"       Decision threshold: {cfg['threshold']}")
    print("="*60 + "\n")
    sys.exit(0)
