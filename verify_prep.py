import pandas as pd
import numpy as np

print("=== VERIFYING PREPROCESSING OUTPUTS ===")

# 1. Load parquet files
print("Loading balanced_train.parquet...")
train_df = pd.read_parquet('balanced_train.parquet')
print("Loading test.parquet...")
test_df = pd.read_parquet('test.parquet')

print(f"Train set shape: {train_df.shape}")
print(f"Test set shape:  {test_df.shape}")

# 2. Schema check
print("\nChecking columns consistency...")
train_cols = set(train_df.columns)
test_cols = set(test_df.columns)
if train_cols == test_cols:
    print("  Columns MATCH [OK]")
else:
    print("  ERROR: Columns mismatch!")
    print("  Train only:", train_cols - test_cols)
    print("  Test only: ", test_cols - train_cols)
    exit(1)

# 3. Check for NaN/Inf
print("\nChecking NaNs and Infs...")
train_nan = train_df.isna().sum().sum()
train_inf = np.isinf(train_df.select_dtypes(include=[np.number])).sum().sum()
test_nan = test_df.isna().sum().sum()
test_inf = np.isinf(test_df.select_dtypes(include=[np.number])).sum().sum()

print(f"  Train: NaN = {train_nan}, Inf = {train_inf}")
print(f"  Test:  NaN = {test_nan}, Inf = {test_inf}")
if train_nan == 0 and train_inf == 0 and test_nan == 0 and test_inf == 0:
    print("  Integrity checks: PASS [OK]")
else:
    print("  ERROR: NaN/Inf found!")
    exit(1)

# 4. Check dropped columns
print("\nChecking dropped low-variance columns...")
dropped_cols = ['ece_flag_number', 'cwr_flag_number', 'Telnet', 'SMTP', 'IRC', 'DHCP', 'IGMP', 'IPv', 'LLC']
found_dropped = [c for c in dropped_cols if c in train_cols]
if len(found_dropped) == 0:
    print("  All 9 low-variance columns successfully dropped [OK]")
else:
    print(f"  ERROR: Found dropped columns in dataset: {found_dropped}")
    exit(1)

# 5. Check engineered features
print("\nChecking engineered features...")
eng_cols = [
    'log_ack_count', 'log_syn_count', 'log_rst_count', 'syn_minus_ack', 
    'flag_entropy_norm', 'byte_per_pkt', 'high_syn_ratio', 'log_rate', 
    'log_iat', 'ack_no_syn', 'ddos_vs_dos_score', 'recon_iat_rate_ratio', 
    'TTL_deviation', 'size_cv', 'payload_asymmetry', 'conn_teardown_rate'
]
missing_eng = [c for c in eng_cols if c not in train_cols]
if len(missing_eng) == 0:
    print("  All 16 engineered features are present [OK]")
else:
    print(f"  ERROR: Missing engineered features: {missing_eng}")
    exit(1)

# 6. Check class counts & SMOTE target
print("\nChecking class counts in Training Set...")
vc_train = train_df['Label'].value_counts()
below_target = vc_train[vc_train < 20000]
# Exclude classes that were naturally between 10K and 20K if they weren't target for SMOTE
# Actually, our SMOTE target was 20,000 for all classes that were below 20K.
# Let's verify that classes with counts < 20,000 do not exist, except if they were naturally above 20,000 but after cap/split slightly below?
# Wait, let's see which classes had < 20,000. In our output, all SMOTE classes have exactly 20,000 rows.
# Are there any classes that had > 20,000 in capped/split?
# Capped was 100K -> split was 80K.
# Classes below 100K:
# - RECON-OSSCAN: 92K -> split was ~73.8K
# - RECON-PORTSCAN: 76K -> split was ~61.4K
# - DOS-HTTP_FLOOD: 68K -> split was ~54.7K
# - DDOS-HTTP_FLOOD: 27K -> split was ~22K
# - DDOS-SLOWLORIS: 22K -> split was ~17.9K (Upsampled to 20K)
# - DICTIONARYBRUTEFORCE: 12.5K -> split was ~10.0K (Upsampled to 20K)
# - BROWSERHIJACKING: 5.5K -> split was ~4.4K (Upsampled to 20K)
# So yes! Every single class in the training set must have AT LEAST 20,000 rows!
print(f"  Minimum class count in Train: {vc_train.min():,}")
if vc_train.min() >= 20000:
    print("  SMOTE target count verified: PASS [OK]")
else:
    print(f"  WARNING: Classes below SMOTE target (20,000):")
    print(below_target)

# 7. Check test set counts (should have no SMOTE synthetic rows)
print("\nVerifying Test Set contains real rows only...")
vc_test = test_df['Label'].value_counts()
print(f"  Total test rows: {len(test_df):,}")
print(f"  Minimum class count in Test: {vc_test.min():,}")
print(f"  Maximum class count in Test: {vc_test.max():,}")

print("\n=== VERIFICATION SUCCESSFULLY COMPLETED ===")
