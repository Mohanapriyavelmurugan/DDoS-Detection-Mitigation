import joblib, sys, time, json
import numpy as np
import pandas as pd

print('=== MODEL DEEP INSPECTION ===')
print()

# ── XGBoost ──────────────────────────────────────────────────────────────────
xgb = joblib.load('./models_v6/xgboost_model.pkl')
p   = xgb.get_params()
print(f'XGBoost type         : {type(xgb).__name__}')
print(f'n_estimators         : {p.get("n_estimators")}')
print(f'max_depth            : {p.get("max_depth")}')
print(f'learning_rate        : {p.get("learning_rate")}')
print(f'num_class            : {p.get("num_class")}')
print(f'tree_method          : {p.get("tree_method")}')
print(f'n_jobs               : {p.get("n_jobs")}')
print(f'device               : {p.get("device")}')
print(f'subsample            : {p.get("subsample")}')
print(f'colsample_bytree     : {p.get("colsample_bytree")}')
print(f'objective            : {p.get("objective")}')

try:
    booster = xgb.get_booster()
    ntrees  = booster.num_boosted_rounds()
    print(f'Actual boosted rounds: {ntrees}')
except Exception as e:
    print(f'booster info err     : {e}')

print()

# ── Isolation Forest ─────────────────────────────────────────────────────────
iso_d = joblib.load('./models_v6/isolation_forest.pkl')
iso   = iso_d['model']
print(f'Isolation Forest')
print(f'  n_estimators       : {iso.n_estimators}')
print(f'  max_samples        : {iso.max_samples}')
print(f'  max_features       : {iso.max_features}')
print(f'  n_features_in      : {iso.n_features_in_}')

print()

# ── Scaler ───────────────────────────────────────────────────────────────────
scaler = joblib.load('./models_v6/scaler.pkl')
print(f'Scaler type          : {type(scaler).__name__}')
print(f'  n_features         : {scaler.n_features_in_}')

print()

# ── Versions ─────────────────────────────────────────────────────────────────
import xgboost
import sklearn
import platform
print(f'XGBoost version      : {xgboost.__version__}')
print(f'scikit-learn version : {sklearn.__version__}')
print(f'Python               : {sys.version.split()[0]}')
print(f'CPU                  : {platform.processor()}')

print()

# ── Component-Level Timing (100 single-flow runs) ─────────────────────────────
print('=== COMPONENT TIMING (100 single-flow runs, post warm-up) ===')

le    = joblib.load('./models_v6/label_encoder.pkl')
iso   = iso_d['model']
if_min, if_max = iso_d['if_min'], iso_d['if_max']
calib_path = './models_v6/xgboost_calibrator.pkl'
import os
calib_d = joblib.load(calib_path) if os.path.exists(calib_path) else None
cols = list(scaler.feature_names_in_)
EPS  = 1e-9

# Dummy single-flow DataFrame
np.random.seed(42)
dummy = pd.DataFrame([{c: np.random.rand() for c in cols}])
X_raw    = dummy[cols].values.astype(np.float32)
benign_id = list(le.classes_).index('BENIGN')

# WARM UP (3 runs)
for _ in range(3):
    X_s = scaler.transform(dummy[cols])
    xgb.predict_proba(X_raw)
    iso.score_samples(X_s)

# TIMED RUNS
N = 100
t_scaler, t_xgb_proba, t_xgb_pred, t_calib, t_iso, t_fusion, t_total = \
    [], [], [], [], [], [], []

for _ in range(N):
    t0 = time.perf_counter()

    t_a = time.perf_counter()
    X_scaled = scaler.transform(dummy[cols])
    t_b = time.perf_counter()

    raw_proba = xgb.predict_proba(X_raw)
    t_c = time.perf_counter()

    raw_pred  = xgb.predict(X_raw)
    t_d = time.perf_counter()

    raw_att = (1.0 - raw_proba[:, benign_id]).reshape(-1, 1)
    if calib_d is not None:
        xgb_att = calib_d['calibrator'].predict_proba(raw_att)[:, 1]
    else:
        xgb_att = raw_att.ravel()
    t_e = time.perf_counter()

    ifs_raw  = iso.score_samples(X_scaled)
    t_f = time.perf_counter()

    # Fusion math
    ifs_clip = np.clip(ifs_raw, if_min, if_max)
    if_prob  = 1.0 - (ifs_clip - if_min) / (if_max - if_min + EPS)
    fusion   = float(np.clip(0.60 * xgb_att[0] + 0.25 * if_prob[0], 0, 1))
    t_g = time.perf_counter()

    t_total.append((t_g - t_a) * 1000)
    t_scaler.append((t_b - t_a) * 1000)
    t_xgb_proba.append((t_c - t_b) * 1000)
    t_xgb_pred.append((t_d - t_c) * 1000)
    t_calib.append((t_e - t_d) * 1000)
    t_iso.append((t_f - t_e) * 1000)
    t_fusion.append((t_g - t_f) * 1000)

def s(arr):
    a = np.array(arr)
    return f'mean={np.mean(a):7.3f}ms  median={np.median(a):7.3f}ms  p95={np.percentile(a,95):7.3f}ms  min={np.min(a):7.3f}ms  max={np.max(a):7.3f}ms'

print(f'  Scaler.transform   : {s(t_scaler)}')
print(f'  XGB.predict_proba  : {s(t_xgb_proba)}   <-- MAIN COST')
print(f'  XGB.predict        : {s(t_xgb_pred)}')
print(f'  Platt Calibrator   : {s(t_calib)}')
print(f'  IsoForest.score    : {s(t_iso)}')
print(f'  Fusion math        : {s(t_fusion)}')
print(f'  TOTAL (all above)  : {s(t_total)}')

print()

# ── Batch vs Single comparison ─────────────────────────────────────────────
print('=== BATCH SIZE IMPACT ON XGBoost (predict_proba) ===')
for bs in [1, 5, 10, 20, 50, 100, 200]:
    batch_df  = pd.DataFrame([{c: np.random.rand() for c in cols} for _ in range(bs)])
    batch_X   = batch_df[cols].values.astype(np.float32)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        xgb.predict_proba(batch_X)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    mean_ms    = np.mean(times)
    per_flow   = mean_ms / bs
    print(f'  Batch={bs:>4}  total={mean_ms:8.3f}ms  per_flow={per_flow:7.3f}ms')

print()

# ── IsoForest batch ───────────────────────────────────────────────────────────
print('=== BATCH SIZE IMPACT ON Isolation Forest (score_samples) ===')
for bs in [1, 5, 10, 20, 50, 100, 200]:
    batch_df  = pd.DataFrame([{c: np.random.rand() for c in cols} for _ in range(bs)])
    batch_X   = scaler.transform(batch_df[cols])
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        iso.score_samples(batch_X)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    mean_ms  = np.mean(times)
    per_flow = mean_ms / bs
    print(f'  Batch={bs:>4}  total={mean_ms:8.3f}ms  per_flow={per_flow:7.3f}ms')
