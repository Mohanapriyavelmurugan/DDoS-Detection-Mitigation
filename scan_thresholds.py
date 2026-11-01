import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

MODEL_DIR = r"D:\DDos 1\models_v4"
print("Loading models...")
le = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")

iso_comp = joblib.load(f"{MODEL_DIR}/isolation_forest.pkl")
iso = iso_comp['model']
if_min, if_max = iso_comp['if_min'], iso_comp['if_max']

xgb = joblib.load(f"{MODEL_DIR}/xgboost_model.pkl")

print("Loading test dataset...")
test_df = pd.read_parquet('test.parquet')
X_test = test_df.drop(columns=['Label'])
y_test = test_df['Label']

# 1. Scale features for IF
print("Scaling features...")
X_ts = X_test.copy()
scale_cols = [c for c in scaler.feature_names_in_ if c in X_ts.columns]
X_ts[scale_cols] = scaler.transform(X_test[scale_cols])

# 2. Get IF anomaly probability - pass full feature matrix (all cols, with scale_cols scaled)
# IF was trained via apply_scaler which returns full df with scale_cols scaled in-place
print("Computing Isolation Forest scores...")
ifs = iso.score_samples(X_ts)  # all 55 features, scale_cols already standardized

ifs_clipped = np.clip(ifs, if_min, if_max)
if_prob = 1.0 - (ifs_clipped - if_min) / (if_max - if_min + 1e-9)

# 3. Get XGBoost attack probability
print("Computing XGBoost scores...")
benign_idx = le.transform(['BENIGN'])[0]
xgb_probs = xgb.predict_proba(X_test.values)
xgb_prob = 1.0 - xgb_probs[:, benign_idx]

# 4. Get DPI score
print("Computing DPI scores...")
def evaluate_dpi(df):
    udp_flood = (df['UDP'] == 1) & (df['Rate'] > 10000)
    icmp_flood = (df['ICMP'] == 1) & (df['Rate'] > 5000)
    syn_flood = (df['syn_flag_number'] == 1) & (df['ack_flag_number'] == 0) & (df['Rate'] > 5000)
    rst_flood = (df['rst_flag_number'] == 1) & (df['Rate'] > 5000)
    dns_spoof = (df['DNS'] == 1) & (df['Rate'] > 1000)
    web_attack = (df['HTTP'] == 1) & (df['Time_To_Live'] < 110) & (df['Variance'] > 0)
    ssh_brute = (df['SSH'] == 1) & (df['Rate'] > 500)
    dpi_match = udp_flood | icmp_flood | syn_flood | rst_flood | dns_spoof | web_attack | ssh_brute
    return dpi_match.astype(np.float32)

dpi_score = evaluate_dpi(test_df)

# 5. Scan thresholds
print("\n=== THRESHOLD SCAN RESULTS ===")
labels = np.where(y_test == 'BENIGN', 0, 1)

for th in np.arange(0.1, 1.0, 0.05):
    final_score = 0.4 * if_prob + 0.4 * xgb_prob + 0.2 * dpi_score
    preds = np.where(final_score >= th, 1, 0)
    acc = accuracy_score(labels, preds)
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"Threshold: {th:.2f} | Binary Acc: {acc*100:.4f}% | TPR (DR): {tpr*100:.2f}% | FPR: {fpr*100:.2f}%")
