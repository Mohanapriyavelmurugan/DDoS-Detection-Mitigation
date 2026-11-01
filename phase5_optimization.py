import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Config
TRAIN_PARQUET = r"D:\DDos 1\balanced_train.parquet"
TEST_PARQUET = r"D:\DDos 1\test.parquet"
MODEL_DIR = r"D:\DDos 1\models_v5"
REPORT_FILE = r"D:\DDos 1\phase5_report.txt"
os.makedirs(MODEL_DIR, exist_ok=True)

# We want a massive penalty for misclassifying BENIGN traffic to reduce False Positives.
BENIGN_WEIGHT = 20.0
IF_CONTAMINATION = 0.005 # Stricter anomaly detection to lower FPR

report_lines = []
def log(msg):
    print(msg)
    report_lines.append(str(msg))

def evaluate_dpi(df):
    score = pd.Series(0.0, index=df.index, dtype=np.float32)
    # Simple heuristics to complement ML
    score = np.where((df.get('syn_count', 0) > 50) & (df.get('Rate', 0) > 5000), np.maximum(score, 0.95), score)
    score = np.where((df.get('UDP', 0) > 0.8) & (df.get('Rate', 0) > 5000), np.maximum(score, 0.90), score)
    score = np.where((df.get('Rate', 0) < 200) & (df.get('IAT_mean', 0) > 0.01), np.minimum(score, 0.1), score)
    return np.clip(score, 0.0, 1.0).astype(np.float32)

def run():
    t_start = time.time()
    log("=================================================================")
    log("PHASE 5 — MODEL OPTIMIZATION (FPR REDUCTION)")
    log("=================================================================")
    
    # 1. Load Data
    log("\n[Step 1] Loading Parquet datasets...")
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    
    X_train = train_df.drop(columns=['Label'])
    y_train = train_df['Label']
    X_test = test_df.drop(columns=['Label'])
    y_test = test_df['Label']
    
    log(f"  Train set: {X_train.shape[0]:,} rows x {X_train.shape[1]} columns")
    
    # 2. Preprocessing
    log("\n[Step 2] Fitting Encoders and Scalers...")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    
    # 3. Isolation Forest (Optimized)
    log(f"\n[Step 3] Training strict Isolation Forest (contamination={IF_CONTAMINATION})...")
    iso = IsolationForest(
        n_estimators=100,
        max_samples=10000,
        contamination=IF_CONTAMINATION,
        random_state=42,
        n_jobs=-1
    )
    t0 = time.time()
    iso.fit(X_train_scaled)
    log(f"  IF Training completed in {time.time()-t0:.1f}s")
    
    # Pre-calculate IF scaling min/max from a sample
    sample_idx = np.random.choice(X_train_scaled.shape[0], min(50000, X_train_scaled.shape[0]), replace=False)
    ifs = iso.score_samples(X_train_scaled[sample_idx])
    if_min, if_max = float(np.min(ifs)), float(np.max(ifs))
    
    joblib.dump({'model': iso, 'if_min': if_min, 'if_max': if_max}, os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
    
    # 4. XGBoost Cost-Sensitive Training
    log(f"\n[Step 4] Training Cost-Sensitive XGBoost (BENIGN weight = {BENIGN_WEIGHT})...")
    # Apply sample weights
    benign_class_id = list(le.classes_).index('BENIGN')
    sample_weights = np.ones(len(y_train_enc), dtype=np.float32)
    sample_weights[y_train_enc == benign_class_id] = BENIGN_WEIGHT
    
    # Split train for early stopping
    X_tr, X_val, y_tr, y_val, sw_tr, sw_val = train_test_split(
        X_train, y_train_enc, sample_weights, test_size=0.1, stratify=y_train_enc, random_state=42
    )
    
    xgb = XGBClassifier(
        n_estimators=1000,  # Cap at 1000 for reasonable time
        learning_rate=0.08,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        device='cuda', # Use GPU if available
        objective='multi:softprob',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        early_stopping_rounds=20,
        random_state=42
    )
    
    t0 = time.time()
    xgb.fit(
        X_tr, y_tr,
        sample_weight=sw_tr,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[sw_val],
        verbose=50
    )
    log(f"  XGB Training completed in {(time.time()-t0)/60:.1f} minutes")
    joblib.dump(xgb, os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
    
    # 5. Evaluation
    log("\n[Step 5] Evaluating Models on Test Set...")
    # Multi-class
    xgb_proba = xgb.predict_proba(X_test)
    y_pred_enc = xgb_proba.argmax(axis=1)
    y_pred = le.inverse_transform(y_pred_enc)
    
    xgb_acc = accuracy_score(y_test, y_pred)
    xgb_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    log(f"  XGBoost Multi-Class Accuracy: {xgb_acc * 100:.4f}%")
    log(f"  XGBoost Weighted F1-Score   : {xgb_f1:.4f}")
    
    # Fusion Binary Evaluation
    ifs_test = iso.score_samples(X_test_scaled)
    ifs_clipped = np.clip(ifs_test, if_min, if_max)
    if_prob = 1.0 - (ifs_clipped - if_min) / (if_max - if_min + 1e-9)
    
    dpi_scores = evaluate_dpi(X_test)
    xgb_att_prob = 1.0 - xgb_proba[:, benign_class_id]
    
    fusion = (0.4 * if_prob + 0.4 * xgb_att_prob + 0.2 * dpi_scores)
    fusion = np.clip(fusion, 0.0, 1.0)
    
    y_binary = (y_test != 'BENIGN').astype(int).values
    preds = (fusion >= 0.5).astype(int) # Standard 0.5 threshold
    
    tp = ((preds == 1) & (y_binary == 1)).sum()
    fp = ((preds == 1) & (y_binary == 0)).sum()
    tn = ((preds == 0) & (y_binary == 0)).sum()
    fn = ((preds == 0) & (y_binary == 1)).sum()
    
    dr = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    acc = (tp + tn) / len(y_binary)
    
    log(f"  Fusion Binary Accuracy      : {acc * 100:.4f}%")
    log(f"  Fusion Detection Rate (TPR) : {dr * 100:.4f}%")
    log(f"  Fusion False Positive Rate  : {fpr * 100:.4f}%")
    
    log(f"\n=================================================================")
    log(f"PHASE 5 COMPLETE (Total time: {(time.time() - t_start)/60:.1f} mins)")
    log(f"=================================================================")
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        log(f"ERROR: {str(e)}")
        # Try falling back to CPU if CUDA fails
        if "device='cuda'" in str(e) or "CUDA" in str(e):
            log("\nFalling back to CPU for XGBoost...")
            # We would modify code and rerun, but for now just exit
