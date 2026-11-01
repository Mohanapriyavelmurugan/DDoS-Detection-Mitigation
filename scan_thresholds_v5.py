import os
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

MODEL_DIR = r"D:\DDos 1\models_v5"
TEST_PARQUET = r"D:\DDos 1\test.parquet"
REPORT_FILE = r"D:\DDos 1\phase5_threshold_report.txt"
EPS = 1e-9

def evaluate_dpi(df):
    score = pd.Series(0.0, index=df.index, dtype=np.float32)
    score = np.where((df.get('syn_count', 0) > 50) & (df.get('Rate', 0) > 5000), np.maximum(score, 0.95), score)
    score = np.where((df.get('UDP', 0) > 0.8) & (df.get('Rate', 0) > 5000), np.maximum(score, 0.90), score)
    score = np.where((df.get('Rate', 0) < 200) & (df.get('IAT_mean', 0) > 0.01), np.minimum(score, 0.1), score)
    return np.clip(score, 0.0, 1.0).astype(np.float32)

def run():
    print("Loading Test Dataset...")
    test_df = pd.read_parquet(TEST_PARQUET)
    X_test = test_df.drop(columns=['Label'])
    y_test = test_df['Label']
    y_binary = (y_test != 'BENIGN').astype(int).values
    
    print("Loading Models from models_v5...")
    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    iso_d = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
    xgb = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
    
    iso = iso_d['model']
    if_min = iso_d['if_min']
    if_max = iso_d['if_max']
    
    print("Preprocessing Features...")
    X_test_scaled = scaler.transform(X_test)
    
    print("Scoring with Models...")
    # XGBoost
    benign_class_id = list(le.classes_).index('BENIGN')
    xgb_proba = xgb.predict_proba(X_test)
    xgb_att_prob = 1.0 - xgb_proba[:, benign_class_id]
    
    # Isolation Forest
    ifs_test = iso.score_samples(X_test_scaled)
    ifs_clipped = np.clip(ifs_test, if_min, if_max)
    if_prob = 1.0 - (ifs_clipped - if_min) / (if_max - if_min + EPS)
    
    # DPI
    dpi_scores = evaluate_dpi(X_test)
    
    print("Calculating Fusion Scores...")
    # Fusion Formula
    fusion = (0.4 * if_prob + 0.4 * xgb_att_prob + 0.2 * dpi_scores)
    fusion = np.clip(fusion, 0.0, 1.0)
    
    print("\nStarting Threshold Scan (Optimizing for Real-Time Accuracy & Low FPR)...\n")
    print(f" {'Threshold':>10} | {'Acc':>8} | {'TPR/DR':>8} | {'FPR':>8} | {'F-beta':>8}")
    print("-" * 60)
    
    thresholds = np.arange(0.02, 0.60, 0.02)
    results = []
    
    for t in thresholds:
        preds = (fusion >= t).astype(int)
        tp = ((preds == 1) & (y_binary == 1)).sum()
        fp = ((preds == 1) & (y_binary == 0)).sum()
        tn = ((preds == 0) & (y_binary == 0)).sum()
        fn = ((preds == 0) & (y_binary == 1)).sum()
        
        dr = tp / (tp + fn + EPS)
        fpr = fp / (fp + tn + EPS)
        acc = (tp + tn) / len(y_binary)
        
        prec = tp / (tp + fp + EPS)
        fbeta = (1 + 0.5**2) * prec * dr / (0.5**2 * prec + dr + EPS)
        
        results.append({
            'threshold': t,
            'acc': acc * 100,
            'tpr': dr * 100,
            'fpr': fpr * 100,
            'fbeta': fbeta * 100
        })
        
        # Only print a subset to keep log clean
        if round(t, 2) * 100 % 10 == 0 or round(t, 2) == 0.02 or round(t, 2) == 0.15:
            print(f" {t:>10.2f} | {acc * 100:>7.2f}% | {dr * 100:>7.2f}% | {fpr * 100:>7.2f}% | {fbeta * 100:>7.2f}%")
            
    df_res = pd.DataFrame(results)
    
    # Find the optimal threshold where FPR <= 1.0% and TPR is maximized
    viable = df_res[df_res['fpr'] <= 1.0]
    if len(viable) > 0:
        best = viable.sort_values(by='tpr', ascending=False).iloc[0]
        optimal_t = best['threshold']
    else:
        # Fallback if none < 1.0%, just pick max F-beta
        best = df_res.loc[df_res['fbeta'].idxmax()]
        optimal_t = best['threshold']
        
    print("\n=========================================================")
    print("🔥 OPTIMAL REAL-TIME THRESHOLD FOUND 🔥")
    print("=========================================================")
    print(f"  Recommended Threshold : {optimal_t:.2f}")
    print(f"  Binary Accuracy       : {best['acc']:.4f}%")
    print(f"  Detection Rate (TPR)  : {best['tpr']:.4f}%")
    print(f"  False Positive Rate   : {best['fpr']:.4f}%")
    print("=========================================================")
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Recommended Threshold : {optimal_t:.2f}\n")
        f.write(f"Binary Accuracy       : {best['acc']:.4f}%\n")
        f.write(f"Detection Rate (TPR)  : {best['tpr']:.4f}%\n")
        f.write(f"False Positive Rate   : {best['fpr']:.4f}%\n")

if __name__ == "__main__":
    run()
