"""Check column issues in the current data."""
import pandas as pd
import numpy as np

df = pd.read_parquet('balanced_train.parquet', engine='pyarrow')
cols = list(df.columns)

print("=== COLUMN LIST ===")
for i, c in enumerate(cols):
    print(f"  {i:2d}. {c:30s}  dtype={df.iloc[:,i].dtype}")

dups = [c for c in set(cols) if cols.count(c) > 1]
print(f"\nDuplicate column names: {dups}")
print(f"Total columns: {len(cols)}")
print(f"Unique column names: {len(set(cols))}")

# Check if Header_Length columns are identical or different
if 'Header_Length' in dups:
    hl_indices = [i for i, c in enumerate(cols) if c == 'Header_Length']
    print(f"\nHeader_Length appears at indices: {hl_indices}")
    for idx in hl_indices:
        col_data = df.iloc[:, idx]
        print(f"  Index {idx}: dtype={col_data.dtype}, min={col_data.min():.2f}, max={col_data.max():.2f}, mean={col_data.mean():.2f}")
    # Check correlation
    if len(hl_indices) == 2:
        corr = df.iloc[:, hl_indices[0]].corr(df.iloc[:, hl_indices[1]])
        is_identical = (df.iloc[:, hl_indices[0]] == df.iloc[:, hl_indices[1]]).all()
        print(f"  Correlation: {corr:.6f}")
        print(f"  Identical: {is_identical}")

# Check Label_encoded
print(f"\nLabel_encoded dtype: {df['Label_encoded'].dtype}")
print(f"Label_encoded range: {df['Label_encoded'].min()} to {df['Label_encoded'].max()}")
print(f"Label_encoded unique count: {df['Label_encoded'].nunique()}")

# Check for the 4.6M duplicates issue in raw data
print("\n=== CHECKING balanced_train FOR DUPLICATES ===")
n_dup = df.duplicated().sum()
print(f"Exact duplicate rows in balanced_train: {n_dup:,}")

# Check variance of each column
print("\n=== LOW VARIANCE FEATURES ===")
num_cols = df.select_dtypes(include=[np.number]).columns
for c in num_cols:
    var = df[c].var()
    if var < 0.001:
        nuniq = df[c].nunique()
        print(f"  {c:30s}  var={var:.8f}  unique={nuniq}")
