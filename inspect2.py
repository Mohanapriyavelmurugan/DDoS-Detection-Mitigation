"""Final inspection: raw data columns + duplicate rows + low-variance features."""
import pandas as pd
import numpy as np

# --- balanced_train (already loaded from Phase 2 output) ---
print("Loading balanced_train.parquet...")
df = pd.read_parquet('balanced_train.parquet', engine='pyarrow')

print(f"\n=== DUPLICATE ROWS IN balanced_train ===")
n_dup = df.duplicated().sum()
print(f"Exact duplicate rows: {n_dup:,} out of {len(df):,} ({n_dup/len(df)*100:.2f}%)")

print(f"\n=== LOW VARIANCE FEATURES ===")
num_cols = df.select_dtypes(include=[np.number]).columns
for c in num_cols:
    var = df[c].var()
    nuniq = df[c].nunique()
    cv = df[c].std() / (abs(df[c].mean()) + 1e-9)
    if var < 0.001 or cv < 0.05:
        print(f"  {c:30s}  var={var:.8f}  unique={nuniq:>6}  cv={cv:.4f}")

print(f"\n=== NEGATIVE VALUE CHECK ===")
for c in num_cols:
    n_neg = (df[c] < 0).sum()
    if n_neg > 0:
        print(f"  {c:30s}  {n_neg:,} negative values (min={df[c].min():.4f})")

print(f"\n=== COLUMNS WITH CONSTANT VALUE ===")
for c in df.columns:
    if df[c].nunique() <= 1:
        print(f"  {c}: constant = {df[c].iloc[0]}")

# Check the proto_mix feature
print(f"\n=== proto_mix FEATURE ===")
print(f"  dtype: {df['proto_mix'].dtype}")
print(f"  range: {df['proto_mix'].min():.4f} to {df['proto_mix'].max():.4f}")
print(f"  unique: {df['proto_mix'].nunique()}")
print(f"  value_counts (top 10):")
print(df['proto_mix'].value_counts().head(10).to_string())

# Check raw processed_dataset columns for Label_encoded
print(f"\n\n=== CHECKING processed_dataset.parquet columns ===")
import pyarrow.parquet as pq
pf = pq.ParquetFile('processed_dataset.parquet')
schema = pf.schema_arrow
print(f"Columns ({len(schema)}):")
for i, field in enumerate(schema):
    print(f"  {i:2d}. {field.name:30s}  {field.type}")

print("\nDone.")
