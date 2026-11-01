import os
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

warnings.filterwarnings('ignore')

MODEL_DIR = r"D:\DDos 1\models_v5"
TEST_PARQUET = r"D:\DDos 1\test.parquet"

def run():
    # Load Data
    test_df = pd.read_parquet(TEST_PARQUET)
    X_test = test_df.drop(columns=['Label'])
    y_test = test_df['Label']
    y_binary = (y_test != 'BENIGN').astype(int).values
    
    # Load Models
    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    iso_d = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
    xgb = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
    
    iso = iso_d['model']
    X_test_scaled = scaler.transform(X_test)
    
    benign_class_id = list(le.classes_).index('BENIGN')
    
    # ---------------------------------------------------------
    # 1. XGBoost Metrics
    # ---------------------------------------------------------
    xgb_proba = xgb.predict_proba(X_test)
    # Multi-class accuracy
    y_pred_enc = xgb_proba.argmax(axis=1)
    xgb_mc_acc = accuracy_score(le.transform(y_test), y_pred_enc)
    
    # Binary metrics (Attack vs Benign)
    # If the max probability is anything OTHER than BENIGN, it's an attack.
    xgb_binary_preds = (y_pred_enc != benign_class_id).astype(int)
    xgb_bin_acc = accuracy_score(y_binary, xgb_binary_preds)
    xgb_bin_prec = precision_score(y_binary, xgb_binary_preds, zero_division=0)
    xgb_bin_rec = recall_score(y_binary, xgb_binary_preds, zero_division=0)
    
    print("\n--- XGBoost Model (Binary Classification) ---")
    print(f"Multi-Class Accuracy: {xgb_mc_acc*100:.2f}%")
    print(f"Binary Accuracy     : {xgb_bin_acc*100:.2f}%")
    print(f"Precision           : {xgb_bin_prec*100:.2f}%")
    print(f"Recall              : {xgb_bin_rec*100:.2f}%")
    
    # ---------------------------------------------------------
    # 2. Isolation Forest Metrics
    # ---------------------------------------------------------
    # IF returns -1 for anomaly, 1 for normal
    iso_preds_raw = iso.predict(X_test_scaled)
    # Convert to 1 = Attack (Anomaly), 0 = Benign (Normal)
    iso_binary_preds = (iso_preds_raw == -1).astype(int)
    
    iso_bin_acc = accuracy_score(y_binary, iso_binary_preds)
    iso_bin_prec = precision_score(y_binary, iso_binary_preds, zero_division=0)
    iso_bin_rec = recall_score(y_binary, iso_binary_preds, zero_division=0)
    
    print("\n--- Isolation Forest Model ---")
    print(f"Accuracy : {iso_bin_acc*100:.2f}%")
    print(f"Precision: {iso_bin_prec*100:.2f}%")
    print(f"Recall   : {iso_bin_rec*100:.2f}%")
    
    # ---------------------------------------------------------
    # 3. Overall Fusion Model Metrics (Threshold = 0.40)
    # ---------------------------------------------------------
    if_min, if_max = iso_d['if_min'], iso_d['if_max']
    ifs_test = iso.score_samples(X_test_scaled)
    ifs_clipped = np.clip(ifs_test, if_min, if_max)
    if_prob = 1.0 - (ifs_clipped - if_min) / (if_max - if_min + 1e-9)
    
    xgb_att_prob = 1.0 - xgb_proba[:, benign_class_id]
    
    # Simple DPI mock for script
    dpi_scores = np.zeros_like(if_prob)
    dpi_scores = np.where((X_test.get('syn_count', 0) > 50) & (X_test.get('Rate', 0) > 5000), np.maximum(dpi_scores, 0.95), dpi_scores)
    dpi_scores = np.where((X_test.get('UDP', 0) > 0.8) & (X_test.get('Rate', 0) > 5000), np.maximum(dpi_scores, 0.90), dpi_scores)
    dpi_scores = np.where((X_test.get('Rate', 0) < 200) & (X_test.get('IAT_mean', 0) > 0.01), np.minimum(dpi_scores, 0.1), dpi_scores)
    dpi_scores = np.clip(dpi_scores, 0.0, 1.0).astype(np.float32)
    
    fusion = (0.4 * if_prob + 0.4 * xgb_att_prob + 0.2 * dpi_scores)
    fusion = np.clip(fusion, 0.0, 1.0)
    
    fusion_preds = (fusion >= 0.40).astype(int)
    
    fus_bin_acc = accuracy_score(y_binary, fusion_preds)
    fus_bin_prec = precision_score(y_binary, fusion_preds, zero_division=0)
    fus_bin_rec = recall_score(y_binary, fusion_preds, zero_division=0)
    
    print("\n--- OVERALL FUSION MODEL (Threshold = 0.40) ---")
    print(f"Accuracy : {fus_bin_acc*100:.2f}%")
    print(f"Precision: {fus_bin_prec*100:.2f}%")
    print(f"Recall   : {fus_bin_rec*100:.2f}%")

if __name__ == "__main__":
    run()
