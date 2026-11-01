import os
import joblib
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

MODEL_DIR = r"D:\DDos 1\models_v5"
NIDD_FILE = r"D:\DDos 1\Combined.csv"
REPORT_FILE = r"D:\DDos 1\5g_nidd_validation_report.txt"
EPS = 1e-9

def evaluate_dpi(df):
    score = pd.Series(0.0, index=df.index, dtype=np.float32)
    score = np.where((df.get('syn_count', 0) > 50) & (df.get('Rate', 0) > 5000), np.maximum(score, 0.95), score)
    score = np.where((df.get('UDP', 0) > 0.8) & (df.get('Rate', 0) > 5000), np.maximum(score, 0.90), score)
    score = np.where((df.get('Rate', 0) < 200) & (df.get('IAT_mean', 0) > 0.01), np.minimum(score, 0.1), score)
    return np.clip(score, 0.0, 1.0).astype(np.float32)

def run():
    print("="*60)
    print("5G-NIDD CROSS-DATASET VALIDATION")
    print("="*60)
    
    print("\n1. Loading Models...")
    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    iso_d = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
    xgb = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
    iso = iso_d['model']
    if_min, if_max = iso_d['if_min'], iso_d['if_max']
    
    expected_cols = list(scaler.feature_names_in_)
    
    print("\n2. Loading 5G-NIDD Dataset...")
    # Load just enough data to get a solid test without running out of memory
    df_raw = pd.read_csv(NIDD_FILE, nrows=500000)
    print(f"   Loaded {df_raw.shape[0]} rows.")
    
    # Extract binary ground truth
    # In 5G-NIDD, 'Label' column is 'Benign' or attack name.
    # If the column is called something else, we try to find it.
    label_col = 'Label' if 'Label' in df_raw.columns else 'Attack Type'
    if label_col not in df_raw.columns:
        print(f"Error: Could not find label column in {df_raw.columns}")
        return
        
    y_raw = df_raw[label_col].astype(str).str.lower()
    y_binary = (~y_raw.str.contains('benign|normal')).astype(int).values
    
    print("\n3. Mapping Features (Argus -> CICFlowMeter logic)...")
    # Initialize dataframe with expected columns set to 0
    df = pd.DataFrame(0, index=df_raw.index, columns=expected_cols, dtype=np.float32)
    
    # Direct mappings
    if 'Rate' in df_raw.columns: df['Rate'] = df_raw['Rate']
    if 'TotPkts' in df_raw.columns: df['Number'] = df_raw['TotPkts']
    if 'TotBytes' in df_raw.columns: df['Tot sum'] = df_raw['TotBytes']
    if 'TotBytes' in df_raw.columns: df['Tot size'] = df_raw['TotBytes']
    if 'Min' in df_raw.columns: df['Min'] = df_raw['Min']
    if 'Max' in df_raw.columns: df['Max'] = df_raw['Max']
    if 'Mean' in df_raw.columns: df['AVG'] = df_raw['Mean']
    if 'sTtl' in df_raw.columns: df['Time_To_Live'] = df_raw['sTtl']
    
    # Protocol Mapping
    if 'Proto' in df_raw.columns:
        proto = df_raw['Proto'].astype(str).str.lower()
        df['TCP'] = (proto == 'tcp').astype(np.float32)
        df['UDP'] = (proto == 'udp').astype(np.float32)
        df['ICMP'] = (proto == 'icmp').astype(np.float32)
        df['Proto_6'] = df['TCP']
        df['Proto_17'] = df['UDP']
        df['Proto_1'] = df['ICMP']
        
    # Time Mapping
    if 'Dur' in df_raw.columns and 'TotPkts' in df_raw.columns:
        df['IAT_mean'] = df_raw['Dur'] / (df_raw['TotPkts'] + EPS)
        
    # Flag Imputation (Argus doesn't give raw counts, we estimate based on state/protocol)
    if 'State' in df_raw.columns:
        state = df_raw['State'].astype(str).str.upper()
        # SYN approximation
        syn_mask = state.str.startswith('S') | state.str.contains('SYN')
        df.loc[syn_mask, 'syn_count'] = df.loc[syn_mask, 'Number'] * 0.8
        df.loc[syn_mask, 'syn_flag_number'] = 1.0
        
        # ACK approximation
        ack_mask = state.str.contains('CON') | state.str.contains('EST')
        df.loc[ack_mask, 'ack_count'] = df.loc[ack_mask, 'Number'] * 0.9
        df.loc[ack_mask, 'ack_flag_number'] = 1.0
        
        # RST/FIN approximation
        rst_mask = state.str.contains('RST') | state.str.contains('FIN')
        df.loc[rst_mask, 'rst_count'] = df.loc[rst_mask, 'Number'] * 0.2
        df.loc[rst_mask, 'fin_count'] = df.loc[rst_mask, 'Number'] * 0.2
        df.loc[rst_mask, 'rst_flag_number'] = 1.0
        df.loc[rst_mask, 'fin_flag_number'] = 1.0
        
    # Re-apply Engineered Features from Phase 2
    print("   Engineering derived features...")
    df['log_ack_count'] = np.log1p(df['ack_count'])
    df['log_syn_count'] = np.log1p(df['syn_count'])
    df['log_rst_count'] = np.log1p(df['rst_count'])
    df['syn_minus_ack'] = df['syn_count'] - df['ack_count']
    
    flag_cols = ['syn_flag_number', 'ack_flag_number', 'rst_flag_number', 'fin_flag_number', 'psh_flag_number']
    total_flags = sum(df[c] for c in flag_cols if c in df.columns) + EPS
    entropy = 0.0
    for f in flag_cols:
        if f in df.columns:
            p = (df[f] + EPS) / total_flags
            entropy -= p * np.log(p + EPS)
    df['flag_entropy_norm'] = (entropy / np.log(5)).astype(np.float32)
    
    df['byte_per_pkt'] = (df['Tot sum'] / (df['Number'] + EPS)).astype(np.float32)
    df['high_syn_ratio'] = (df['syn_count'] / (df['Number'] + EPS)).clip(0, 1).astype(np.float32)
    df['log_rate'] = np.log1p(df['Rate']).astype(np.float32)
    df['log_iat'] = np.log1p(df['IAT_mean'] * 1e6).astype(np.float32)
    df['ack_no_syn'] = (df.get('ack_flag_number', 0) * (1 - df.get('syn_flag_number', 0))).astype(np.float32)
    df['ddos_vs_dos_score'] = (df['syn_count'] / (df['ack_count'] + 1.0)).astype(np.float32)
    df['recon_iat_rate_ratio'] = (np.log1p(df['IAT_mean'] * 1e6) / (np.log1p(df['Rate']) + EPS)).astype(np.float32)
    df['TTL_deviation'] = np.minimum(np.abs(df['Time_To_Live'] - 64), np.abs(df['Time_To_Live'] - 128)).astype(np.float32)
    df['size_cv'] = (df.get('Std', 0) / (df['AVG'] + EPS)).astype(np.float32)
    df['payload_asymmetry'] = (df['Max'] / (df['AVG'] + EPS)).astype(np.float32)
    df['conn_teardown_rate'] = ((df['fin_count'] + df['rst_count']) / (df['Number'] + EPS)).astype(np.float32)
    df['ack_to_syn_ratio'] = (df['ack_count'] / (df['syn_count'] + EPS)).clip(0, 1).astype(np.float32)
    df['rst_to_syn_ratio'] = (df['rst_count'] / (df['syn_count'] + EPS)).clip(0, 5).astype(np.float32)
    df['rate_per_pkt'] = (df['Rate'] / (df['Number'] + EPS)).astype(np.float32)
    
    all_flags = df.get('syn_flag_number',0) + df.get('ack_flag_number',0) + df.get('rst_flag_number',0) + df.get('fin_flag_number',0) + df.get('psh_flag_number',0) + EPS
    df['syn_flag_density'] = (df.get('syn_flag_number',0) / all_flags).astype(np.float32)
    
    ack_norm = (df['ack_count'] / (df['Number'] + EPS)).clip(0, 1)
    td_rate  = ((df['fin_count'] + df['rst_count']) / (df['Number'] + EPS)).clip(0, 1)
    df['ack_teardown_interaction'] = (ack_norm * td_rate).astype(np.float32)
    df['flood_intensity_score'] = (np.log1p(df['Rate']) * df['high_syn_ratio']).astype(np.float32)
    df['tcp_ratio_sq'] = (df.get('TCP',0) ** 2).astype(np.float32)
    df['iat_rate_product'] = (np.log1p(df['IAT_mean'] * 1e6) * np.log1p(df['Rate'])).astype(np.float32)
    
    # Ensure columns match scaler exactly
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    X = df[expected_cols]
    
    print("\n4. Performing Inference...")
    X_scaled = scaler.transform(X)
    
    benign_class_id = list(le.classes_).index('BENIGN')
    xgb_proba = xgb.predict_proba(X)
    xgb_att_prob = 1.0 - xgb_proba[:, benign_class_id]
    
    ifs = iso.score_samples(X_scaled)
    ifs_clipped = np.clip(ifs, if_min, if_max)
    if_prob = 1.0 - (ifs_clipped - if_min) / (if_max - if_min + EPS)
    
    dpi_scores = evaluate_dpi(X)
    
    # Fusion at threshold 0.40
    fusion = (0.4 * if_prob + 0.4 * xgb_att_prob + 0.2 * dpi_scores)
    fusion = np.clip(fusion, 0.0, 1.0)
    preds = (fusion >= 0.40).astype(int)
    
    print("\n5. Calculating Cross-Dataset Metrics...")
    tp = ((preds == 1) & (y_binary == 1)).sum()
    fp = ((preds == 1) & (y_binary == 0)).sum()
    tn = ((preds == 0) & (y_binary == 0)).sum()
    fn = ((preds == 0) & (y_binary == 1)).sum()
    
    dr = tp / (tp + fn + EPS)
    fpr = fp / (fp + tn + EPS)
    acc = (tp + tn) / len(y_binary)
    
    print("="*60)
    print("FINAL 5G-NIDD VALIDATION RESULTS")
    print("="*60)
    print(f"Total Rows Tested   : {len(y_binary):,}")
    print(f"Actual Attacks      : {sum(y_binary):,}")
    print(f"Actual Benign       : {len(y_binary) - sum(y_binary):,}")
    print(f"Accuracy            : {acc*100:.2f}%")
    print(f"Detection Rate (TPR): {dr*100:.2f}%")
    print(f"False Positive Rate : {fpr*100:.2f}%")
    
    with open(REPORT_FILE, 'w') as f:
        f.write(f"Accuracy: {acc*100:.2f}%\nTPR: {dr*100:.2f}%\nFPR: {fpr*100:.2f}%\n")

if __name__ == "__main__":
    run()
