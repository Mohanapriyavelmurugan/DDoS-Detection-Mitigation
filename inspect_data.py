"""Quick inspection of the raw processed_dataset.parquet"""
import pandas as pd
import numpy as np

print("Loading processed_dataset.parquet...")
df = pd.read_parquet('processed_dataset.parquet', engine='pyarrow')
print(f"Shape: {df.shape}")

print("\n=== COLUMNS ===")
for i, c in enumerate(df.columns):
    print(f"  {i:2d}. {c} ({df[c].dtype})")

print(f"\n=== NaN PER COLUMN ===")
nans = df.isnull().sum()
has_nan = nans[nans > 0]
if len(has_nan) > 0:
    print(has_nan.to_string())
else:
    print("No NaN found")

print(f"\n=== Inf PER COLUMN ===")
num_cols = df.select_dtypes(include=[np.number]).columns
for c in num_cols:
    n_inf = np.isinf(df[c]).sum()
    if n_inf > 0:
        print(f"  {c}: {n_inf} Inf values")
if all(np.isinf(df[c]).sum() == 0 for c in num_cols):
    print("No Inf found")

print(f"\n=== LABEL DISTRIBUTION ===")
vc = df['Label'].value_counts()
for label, count in vc.items():
    pct = count / len(df) * 100
    print(f"  {label:<42s} {count:>10,}  ({pct:.2f}%)")
print(f"\n  Total classes: {df['Label'].nunique()}")
print(f"  Imbalance ratio: {vc.max():,} : {vc.min():,} = {vc.max()/vc.min():.1f}:1")

print(f"\n=== NUMERIC RANGE SUMMARY ===")
for c in num_cols:
    mn, mx = df[c].min(), df[c].max()
    mean = df[c].mean()
    std = df[c].std()
    print(f"  {c:<30s}  min={mn:>12.2f}  max={mx:>14.2f}  mean={mean:>12.2f}  std={std:>12.2f}")

print(f"\n=== DUPLICATE CHECK ===")
n_dup = df.duplicated().sum()
print(f"  Exact duplicates remaining: {n_dup:,}")

# Check balanced_train too
print("\n\n" + "="*65)
print("Loading balanced_train.parquet...")
bt = pd.read_parquet('balanced_train.parquet', engine='pyarrow')
print(f"Shape: {bt.shape}")
print(f"Columns: {list(bt.columns)}")
vc2 = bt['Label'].value_counts()
print(f"\nLabel distribution:")
for label, count in vc2.items():
    print(f"  {label:<42s} {count:>10,}")

print("\n\nLoading test.parquet...")
te = pd.read_parquet('test.parquet', engine='pyarrow')
print(f"Shape: {te.shape}")

# Check if columns match
if list(bt.columns) == list(te.columns):
    print("Train/test columns MATCH ✓")
else:
    print("WARNING: Train/test columns DIFFER!")
    print(f"  Train only: {set(bt.columns) - set(te.columns)}")
    print(f"  Test only: {set(te.columns) - set(bt.columns)}")

print("\nDone.")
