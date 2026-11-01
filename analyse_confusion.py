import pandas as pd
import numpy as np

train = pd.read_parquet('balanced_train.parquet')
print("=== FEATURE STATISTICS FOR CONFUSED CLASSES ===")
confused_classes = [
    'DDOS-SYN_FLOOD', 'DOS-SYN_FLOOD',
    'DDOS-TCP_FLOOD', 'DOS-TCP_FLOOD',
    'DDOS-SYNONYMOUSIP_FLOOD',
    'RECON-OSSCAN', 'RECON-PORTSCAN',
    'VULNERABILITYSCAN', 'DICTIONARYBRUTEFORCE'
]

for cls in confused_classes:
    df = train[train['Label'] == cls]
    print(f"\n{cls} (n={len(df):,}):")
    print(f"  syn_count        = {df['syn_count'].mean():.2f}")
    print(f"  ack_count        = {df['ack_count'].mean():.2f}")
    print(f"  fin_count        = {df['fin_count'].mean():.2f}")
    print(f"  rst_count        = {df['rst_count'].mean():.2f}")
    print(f"  Rate             = {df['Rate'].mean():.0f}")
    print(f"  TCP              = {df['TCP'].mean():.4f}")
    print(f"  UDP              = {df['UDP'].mean():.4f}")
    print(f"  ddos_vs_dos_score= {df['ddos_vs_dos_score'].mean():.4f}")
    print(f"  conn_teardown_rate= {df['conn_teardown_rate'].mean():.4f}")
    print(f"  proto_mix        = {df['proto_mix'].mean():.4f}")
    print(f"  Number           = {df['Number'].mean():.2f}")
    print(f"  IAT_mean         = {df['IAT_mean'].mean():.6f}")

# Print feature correlation for confused pairs
print("\n=== COHEN'S d BETWEEN DDOS-SYN vs DOS-SYN ===")
a = train[train['Label'] == 'DDOS-SYN_FLOOD']
b = train[train['Label'] == 'DOS-SYN_FLOOD']
for col in ['syn_count', 'ack_count', 'fin_count', 'rst_count', 'Rate', 'ddos_vs_dos_score', 'conn_teardown_rate', 'log_rate', 'flag_entropy_norm']:
    pooled_std = np.sqrt((a[col].std()**2 + b[col].std()**2) / 2) + 1e-9
    d = abs(a[col].mean() - b[col].mean()) / pooled_std
    print(f"  {col:<30}: d={d:.3f}  A={a[col].mean():.4f}  B={b[col].mean():.4f}")
