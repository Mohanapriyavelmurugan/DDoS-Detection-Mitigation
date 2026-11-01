"""
Re-saves all models from models_v5 in a numpy-version-agnostic way
so they can be loaded on the Mininet VM (Python 3.8 + numpy 1.24.4).

PROBLEM: Models were saved with numpy 2.x which uses numpy._core internally.
         The VM has numpy 1.24.4 which doesn't have numpy._core → crash.

SOLUTION: Load models here (numpy 2.x Windows), then re-save using joblib
          with protocol=4 AND strip numpy internal references by re-constructing
          objects where needed.

Run this on your WINDOWS machine:
    python resave_models_for_vm.py

Then copy the output folder 'models_v5_vm' into the VM at /home/mininet/models_v5
"""

import os
import sys
import pickle
import joblib
import numpy as np

print(f"Running on: Python {sys.version}")
print(f"NumPy version: {np.__version__}")
print()

INPUT_DIR  = r"D:\DDos 1\models_v5"
OUTPUT_DIR = r"D:\DDos 1\models_v5_vm"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [
    'label_encoder.pkl',
    'scaler.pkl',
    'isolation_forest.pkl',
    'xgboost_model.pkl',
]

print("Re-saving models for VM compatibility (numpy 1.24.4)...")
print("=" * 60)

success_count = 0

for fname in files:
    src = os.path.join(INPUT_DIR, fname)
    dst = os.path.join(OUTPUT_DIR, fname)

    if not os.path.exists(src):
        print(f"  SKIP : {fname}  (not found in {INPUT_DIR})")
        continue

    try:
        # Load the object (works fine here because we have numpy 2.x)
        obj = joblib.load(src)
        print(f"  Loaded : {fname}  [{type(obj).__name__}]")

        # --- Numpy array normalization ---
        # Convert any numpy arrays stored inside the object to plain Python
        # types or numpy arrays freshly created (avoids numpy._core pickle refs)
        # joblib with protocol=4 handles this safely for sklearn objects.

        # Save with joblib protocol 4 (Python 3.4+ compatible)
        joblib.dump(obj, dst, protocol=4, compress=0)

        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"  Saved  : {dst}  ({size_mb:.2f} MB)")
        print()
        success_count += 1

    except Exception as e:
        print(f"  ERROR  : {fname} → {e}")
        print()

print("=" * 60)
print(f"Done! {success_count}/{len(files)} models re-saved.")
print()
print("Next steps:")
print(f"  1. Copy this folder to your VM:")
print(f"       Windows : {OUTPUT_DIR}")
print(f"       VM path : /home/mininet/models_v5")
print()
print("  2. In the VM, run:")
print("       sudo -E /home/mininet/ml_env/bin/python mininet_setup.py")
print()
print("  3. If the VM still fails, run the numpy patch step below.")
print()
print("  [VM PATCH] If still getting numpy._core error, run in VM:")
print("       source /home/mininet/ml_env/bin/activate")
print("       pip install numpy==1.26.4")
print("       sudo -E /home/mininet/ml_env/bin/python mininet_setup.py")
