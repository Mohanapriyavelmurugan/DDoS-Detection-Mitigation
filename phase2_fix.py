"""
=============================================================================
PHASE 2 — COMPREHENSIVE PREPROCESSING & BALANCING PIPELINE
Edge-Intelligent Hybrid DDoS Detection System for 5G Networks
=============================================================================

This script implements a scientifically rigorous, memory-efficient 10-step 
preprocessing pipeline that prepares the raw dataset directly for model training.

Steps:
  1. Load processed_dataset.parquet and validate 49-column schema
  2. Drop 9 low-variance columns (ece_flag_number, cwr_flag_number, etc.)
  3. Remove NaN rows + Global Duplicate Removal (cross-file deduplication)
  4. Apply Rate-stratified capping at 100,000 rows per class
  5. Engineer 24 features (log transforms, asymmetry, entropy, flood discriminators, etc.)
  6. Verify data integrity (zero NaN/Inf, replace boundaries)
  7. Stratified 80/20 Train/Test split
  8. Apply SMOTE to training set only (target 20,000 for minority classes)
  9. Save balanced_train.parquet and test.parquet
  10. Generate and save phase2_report.txt

INPUT  : D:\\DDos 1\\processed_dataset.parquet
OUTPUT : D:\\DDos 1\\balanced_train.parquet (SMOTE-augmented training set)
         D:\\DDos 1\\test.parquet           (Real rows only test set)
         D:\\DDos 1\\phase2_report.txt      (Full audit log)
=============================================================================
"""

import os
import time
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_PARQUET  = r"D:\DDos 1\processed_dataset.parquet"
TRAIN_PARQUET  = r"D:\DDos 1\balanced_train.parquet"
TEST_PARQUET   = r"D:\DDos 1\test.parquet"
REPORT_FILE    = r"D:\DDos 1\phase2_report.txt"

# Capping and SMOTE Parameters
GLOBAL_CAP   = 100_000
SMOTE_TARGET = 20_000

# Split Ratio
TEST_SIZE    = 0.20
RANDOM_STATE = 42
EPS          = 1e-9

# Low variance features to drop
DROP_FEATURES = [
    'ece_flag_number',
    'cwr_flag_number',
    'Telnet',
    'SMTP',
    'IRC',
    'DHCP',
    'IGMP',
    'IPv',
    'LLC'
]

# Expected raw columns
EXPECTED_COLUMNS = [
    'Header_Length', 'Time_To_Live', 'Rate', 'fin_flag_number', 'syn_flag_number',
    'rst_flag_number', 'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'rst_count',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP',
    'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max',
    'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Variance', 'Label',
    'IAT_mean', 'IAT_per_byte', 'pkt_size_range', 'flow_rate_ratio',
    'proto_mix', 'Proto_0', 'Proto_1', 'Proto_6', 'Proto_17', 'Proto_47'
]

report_lines = []

def log(msg="", also_print=True):
    report_lines.append(msg)
    if also_print:
        print(msg)

def print_class_distribution(df, title):
    """Print formatted class distribution table."""
    log(f"\n{'='*65}")
    log(f"{title}")
    log(f"{'='*65}")
    log(f"  Total rows : {len(df):,}")
    log(f"  Classes    : {df['Label'].nunique()}")
    vc = df['Label'].value_counts()
    log(f"\n  {'Class':<42} {'Rows':>10}  {'%':>6}")
    log(f"  {'-'*62}")
    for label, count in vc.items():
        pct = count / len(df) * 100
        bar = '*' * int(pct / 2)
        log(f"  {label:<42} {count:>10,}  {pct:>5.2f}%  {bar}")
    log(f"\n  Imbalance ratio : {vc.max():,} : {vc.min():,}"
        f"  =  {vc.max()/vc.min():.1f} : 1")
    log(f"  Majority class  : {vc.index[0]}  ({vc.iloc[0]:,})")
    log(f"  Minority class  : {vc.index[-1]}  ({vc.iloc[-1]:,})")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    """Engineers 24 features on the dataset."""
    log(f"  Engineering 24 features on dataset...")
    
    # 1-3. Log-compressed counts
    df['log_ack_count'] = np.log1p(df['ack_count'])
    df['log_syn_count'] = np.log1p(df['syn_count'])
    df['log_rst_count'] = np.log1p(df['rst_count'])

    # 4. SYN - ACK signed difference
    df['syn_minus_ack'] = df['syn_count'] - df['ack_count']

    # 5. Shannon entropy of TCP flags (normalized)
    flag_cols = ['syn_flag_number', 'ack_flag_number', 'rst_flag_number',
                 'fin_flag_number', 'psh_flag_number']
    total_flags = sum(df[f] for f in flag_cols) + EPS
    entropy = 0.0
    for f in flag_cols:
        p = (df[f] + EPS) / total_flags
        entropy -= p * np.log(p + EPS)
    df['flag_entropy_norm'] = (entropy / np.log(len(flag_cols))).astype(np.float32)

    # 6. Bytes per packet
    df['byte_per_pkt'] = (df['Tot sum'] / (df['Number'] + EPS)).astype(np.float32)

    # 7. High SYN ratio
    df['high_syn_ratio'] = (df['syn_count'] / (df['Number'] + EPS)).clip(0, 1).astype(np.float32)

    # 8. Log rate
    df['log_rate'] = np.log1p(df['Rate']).astype(np.float32)

    # 9. Log IAT
    df['log_iat'] = np.log1p(df['IAT_mean'] * 1e6).astype(np.float32)

    # 10. ACK without SYN flag set
    df['ack_no_syn'] = (df['ack_flag_number'] * (1 - df['syn_flag_number'])).astype(np.float32)

    # 11. DDoS vs DoS score (asymmetric flow ratio)
    df['ddos_vs_dos_score'] = (df['syn_count'] / (df['ack_count'] + 1.0)).astype(np.float32)

    # 12. Reconnaissance IAT-to-rate ratio
    df['recon_iat_rate_ratio'] = (np.log1p(df['IAT_mean'] * 1e6) / (np.log1p(df['Rate']) + EPS)).astype(np.float32)

    # 13. TTL deviation from OS standards (64/128)
    ttl = df['Time_To_Live']
    df['TTL_deviation'] = np.minimum(np.abs(ttl - 64), np.abs(ttl - 128)).astype(np.float32)

    # 14. Packet size CV
    df['size_cv'] = (df['Std'] / (df['AVG'] + EPS)).astype(np.float32)

    # 15. Payload asymmetry (Max / AVG)
    df['payload_asymmetry'] = (df['Max'] / (df['AVG'] + EPS)).astype(np.float32)

    # 16. Connection teardown rate (fin+rst per packet)
    df['conn_teardown_rate'] = ((df['fin_count'] + df['rst_count']) / (df['Number'] + EPS)).astype(np.float32)

    # ---- 8 NEW DISCRIMINATIVE FEATURES (DDOS vs DOS flood separation) -------

    # 17. ACK-to-SYN ratio — DDOS-SYN has very low ack (~3) vs syn (~97)
    #     DOS-SYN has more balanced ack (~7) vs syn (~92)  => strong separator
    df['ack_to_syn_ratio'] = (df['ack_count'] / (df['syn_count'] + EPS)).clip(0, 1).astype(np.float32)

    # 18. RST-to-SYN ratio — DDOS-TCP teardown is near zero; DOS-TCP has more RSTs
    df['rst_to_syn_ratio'] = (df['rst_count'] / (df['syn_count'] + EPS)).clip(0, 5).astype(np.float32)

    # 19. Rate per packet — floods with high pkt count & high rate signal DDoS scale
    df['rate_per_pkt'] = (df['Rate'] / (df['Number'] + EPS)).astype(np.float32)

    # 20. SYN flag density — proportion of SYN flags relative to all flag counts
    all_flags = (df['syn_flag_number'] + df['ack_flag_number'] +
                 df['rst_flag_number'] + df['fin_flag_number'] +
                 df['psh_flag_number'] + EPS)
    df['syn_flag_density'] = (df['syn_flag_number'] / all_flags).astype(np.float32)

    # 21. ACK * teardown interaction — captures established-but-torn-down sessions
    #     DOS-TCP: higher ack_count AND higher conn_teardown than DDOS-TCP
    ack_norm = (df['ack_count'] / (df['Number'] + EPS)).clip(0, 1)
    td_rate  = ((df['fin_count'] + df['rst_count']) / (df['Number'] + EPS)).clip(0, 1)
    df['ack_teardown_interaction'] = (ack_norm * td_rate).astype(np.float32)

    # 22. Flood intensity score — log(Rate) * high_syn_ratio (attacks specific to syn floods)
    df['flood_intensity_score'] = (np.log1p(df['Rate']) * df['high_syn_ratio']).astype(np.float32)

    # 23. TCP ratio squared — amplifies small TCP/UDP differences across flood types
    df['tcp_ratio_sq'] = (df['TCP'] ** 2).astype(np.float32)

    # 24. IAT * rate product (log-space) — recon has high IAT & moderate rate
    #     floods have extremely low IAT & very high rate => log-product near zero
    df['iat_rate_product'] = (np.log1p(df['IAT_mean'] * 1e6) *
                              np.log1p(df['Rate'])).astype(np.float32)

    return df

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run():
    t_start = time.time()
    
    log("="*65)
    log("PHASE 2 PREPROCESSING & BALANCING")
    log("="*65)
    
    # ── Step 1: Load Parquet & Validate Schema ───────────────────────────────
    log("\n[Step 1] Loading Parquet file & validating schema...")
    if not os.path.exists(INPUT_PARQUET):
        log(f"ERROR: File not found: {INPUT_PARQUET}")
        raise FileNotFoundError(INPUT_PARQUET)
        
    df = pd.read_parquet(INPUT_PARQUET, engine='pyarrow')
    log(f"  Loaded shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    # Validate columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        log(f"WARNING: Missing columns from source schema: {missing_cols}")
    else:
        log("  Schema validation: PASS [OK] (All 49 columns present)")

    # ── Step 2: Drop Low-Variance Columns ────────────────────────────────────
    log("\n[Step 2] Dropping 9 low-variance columns...")
    drop_present = [c for c in DROP_FEATURES if c in df.columns]
    df.drop(columns=drop_present, inplace=True)
    log(f"  Dropped features: {drop_present}")
    log(f"  Shape after drop: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Downcast datatypes to save memory
    log("  Downcasting data types for memory optimization...")
    for col in df.select_dtypes(include=[np.float64]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=[np.int64]).columns:
        df[col] = df[col].astype(np.int32)

    # ── Step 3: Remove NaN Rows + Global Duplicate Removal ───────────────────
    log("\n[Step 3] Cleaning NaNs and performing global deduplication...")
    before_nan = len(df)
    df.dropna(inplace=True)
    nans_removed = before_nan - len(df)
    log(f"  NaN rows removed: {nans_removed}")
    
    before_dup = len(df)
    df.drop_duplicates(inplace=True)
    dups_removed = before_dup - len(df)
    log(f"  Global duplicates removed: {dups_removed:,} ({(dups_removed/before_dup)*100:.2f}% reduction)")
    log(f"  Remaining unique rows: {len(df):,}")

    # Print class distribution before capping
    print_class_distribution(df, "CLASS DISTRIBUTION — After Global Deduplication")

    # ── Step 4: Stratified Capping ───────────────────────────────────────────
    log(f"\n[Step 4] Applying stratified capping (max {GLOBAL_CAP:,} rows per class)...")
    frames = []
    vc = df['Label'].value_counts()
    capped_classes_count = 0
    
    for label, count in vc.items():
        class_df = df[df['Label'] == label]
        if count > GLOBAL_CAP:
            # Stratified by 'Rate' to preserve distribution of attack intensities
            class_df_sorted = class_df.sort_values('Rate').reset_index(drop=True)
            n = len(class_df_sorted)
            third = n // 3
            per_group = GLOBAL_CAP // 3
            remainder = GLOBAL_CAP - (per_group * 3)

            low = class_df_sorted.iloc[:third].sample(
                n=min(per_group, third), random_state=RANDOM_STATE
            )
            mid = class_df_sorted.iloc[third:2*third].sample(
                n=min(per_group, third), random_state=RANDOM_STATE
            )
            high = class_df_sorted.iloc[2*third:].sample(
                n=min(per_group + remainder, n - 2*third), random_state=RANDOM_STATE
            )

            sampled = pd.concat([low, mid, high], ignore_index=True)
            frames.append(sampled)
            log(f"  CAPPED  {label:42s}: {count:>10,} -> {len(sampled):,}")
            capped_classes_count += 1
        else:
            frames.append(class_df)
            log(f"  KEPT    {label:42s}: {count:>10,} (below cap)")
            
    df = pd.concat(frames, ignore_index=True)
    # Shuffle
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    log(f"  Total classes capped: {capped_classes_count}")
    log(f"  Total rows after capping: {len(df):,}")

    # ── Step 5: Feature Engineering ──────────────────────────────────────────
    log("\n[Step 5] Applying row-level feature engineering (24 features)...")
    t0 = time.time()
    df = engineer_features(df)
    log(f"  Features engineered in {time.time()-t0:.1f}s. Total columns now: {df.shape[1]}")

    # ── Step 6: Verify Data Integrity ────────────────────────────────────────
    log("\n[Step 6] Verifying data integrity...")
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    n_nan = df[num_cols].isna().sum().sum()
    n_inf = df[num_cols].isin([np.inf, -np.inf]).sum().sum()
    log(f"  Initial checks: NaN count = {n_nan}, Inf count = {n_inf}")
    
    if n_inf > 0:
        log("  Replacing infinite values with column maximums...")
        for col in num_cols:
            col_inf_mask = df[col].isin([np.inf, -np.inf])
            if col_inf_mask.any():
                col_max = df[col].replace([np.inf, -np.inf], np.nan).max()
                if pd.isna(col_max):
                    col_max = 0.0
                df[col] = df[col].replace([np.inf, -np.inf], col_max)
                
    if n_nan > 0 or df.isna().any().any():
        log("  Filling missing values with 0...")
        df.fillna(0, inplace=True)
        
    log("  Integrity checks completed: clean of NaNs and Infs [OK]")

    # ── Step 7: Train/Test Split (Before SMOTE) ──────────────────────────────
    log(f"\n[Step 7] Splitting dataset 80/20 stratified (random_state={RANDOM_STATE})...")
    from sklearn.model_selection import train_test_split
    
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    test_df  = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
    log(f"  Training set size: {len(train_df):,} rows")
    log(f"  Test set size:     {len(test_df):,} rows (strictly real data)")

    # ── Step 8: Apply SMOTE on Training Set Only ─────────────────────────────
    log(f"\n[Step 8] Applying SMOTE on training set (Target: {SMOTE_TARGET:,} per class)...")
    from imblearn.over_sampling import SMOTE
    
    vc_train = train_df['Label'].value_counts()
    needs_smote = vc_train[vc_train < SMOTE_TARGET].index.tolist()
    
    if needs_smote:
        log(f"  Classes requiring SMOTE upsampling ({len(needs_smote)} total):")
        for label in needs_smote:
            log(f"    {label:42s}: {vc_train[label]:,} -> {SMOTE_TARGET:,}")
            
        min_samples = min(vc_train[label] for label in needs_smote)
        k_neighbors = min(5, min_samples - 1)
        log(f"  Using SMOTE with k_neighbors={k_neighbors}")
        
        sampling_strategy = {label: SMOTE_TARGET for label in needs_smote}
        
        X_train_res = train_df.drop(columns=['Label'])
        y_train_res = train_df['Label']
        
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=RANDOM_STATE
        )
        
        t0 = time.time()
        X_resampled, y_resampled = smote.fit_resample(X_train_res, y_train_res)
        log(f"  SMOTE completed in {time.time()-t0:.1f}s")
        
        train_df = pd.concat([X_resampled, y_resampled], axis=1)
        # Shuffle resampled training data
        train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        log("  No classes require SMOTE upsampling.")

    print_class_distribution(train_df, "FINAL TRAINING SET — After SMOTE Upsampling")
    print_class_distribution(test_df,  "FINAL TEST SET — Real Rows Only (Untouched)")

    # ── Step 9: Save Parquet Outputs ─────────────────────────────────────────
    log("\n[Step 9] Saving outputs to Parquet...")
    
    t0 = time.time()
    train_df.to_parquet(TRAIN_PARQUET, engine='pyarrow', compression='snappy', index=False)
    train_size_mb = os.path.getsize(TRAIN_PARQUET) / (1024*1024)
    log(f"  Saved training set -> {TRAIN_PARQUET} ({len(train_df):,} rows, {train_size_mb:.1f} MB, took {time.time()-t0:.1f}s)")
    
    t0 = time.time()
    test_df.to_parquet(TEST_PARQUET, engine='pyarrow', compression='snappy', index=False)
    test_size_mb = os.path.getsize(TEST_PARQUET) / (1024*1024)
    log(f"  Saved test set     -> {TEST_PARQUET} ({len(test_df):,} rows, {test_size_mb:.1f} MB, took {time.time()-t0:.1f}s)")

    # ── Step 10: Generate Report ─────────────────────────────────────────────
    total_time = time.time() - t_start
    log(f"\n{'='*65}")
    log(f"PHASE 2 COMPLETED SUCCESSFUL IN {total_time/60:.1f} MINUTES")
    log(f"  Training set ready -> {TRAIN_PARQUET}")
    log(f"  Evaluation set ready -> {TEST_PARQUET}")
    log(f"="*65)
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\nPreprocessing report saved to {REPORT_FILE}")

if __name__ == '__main__':
    run()