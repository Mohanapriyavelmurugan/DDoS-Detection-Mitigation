# -*- coding: utf-8 -*-
"""
===============================================================
MEC EXPORT SCRIPT — One-command build and package for MEC node
===============================================================
Run this on your LAPTOP to build and export the entire system.

Usage:
    python build_for_mec.py

What it does (step by step):
    Step 1 - Copy models_v6/ into the detector build context
    Step 2 - Build the Docker image (runs verify_deployment.py inside)
    Step 3 - Run a quick smoke test inside the built image
    Step 4 - Export image to a .tar file
    Step 5 - Print transfer instructions for MEC node

Output:
    ddos_detector_v6_mec.tar   (~300-500 MB, self-contained)

Transfer to MEC:
    scp ddos_detector_v6_mec.tar user@192.168.10.12:~/
    OR copy to USB drive
===============================================================
"""

import os
import sys
import io
import time
import shutil
import subprocess

# Force UTF-8 output on Windows (fixes UnicodeEncodeError for → ✓ etc.)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCKER_DIR      = os.path.dirname(__file__)
DETECTOR_DIR    = os.path.join(DOCKER_DIR, "detector")
MODELS_SRC      = os.path.join(PROJECT_ROOT, "models_v6")
MODELS_DST      = os.path.join(DETECTOR_DIR, "models_v6")
OUTPUT_TAR      = os.path.join(PROJECT_ROOT, "ddos_detector_v6_mec.tar")
IMAGE_NAME      = "ddos_detector:v6"
MEC_IP          = "192.168.10.12"

OK   = "[OK] "
FAIL = "[FAIL]"
INFO = "[->] "
BOLD = ""
RST  = ""

def run(cmd, check=True, capture=False):
    """Run a shell command and optionally capture output."""
    print(f"  {INFO} {cmd}")
    result = subprocess.run(
        cmd, shell=True,
        capture_output=capture,
        text=True
    )
    if check and result.returncode != 0:
        print(f"\n  {FAIL} Command failed with exit code {result.returncode}")
        if result.stderr:
            print(result.stderr)
        sys.exit(1)
    return result

def main():
    t_start = time.time()
    print(f"\n{BOLD}{'='*60}{RST}")
    print(f"{BOLD}  DDoS Detector — MEC Export Pipeline{RST}")
    print(f"{BOLD}  Phase 6 | Target MEC: {MEC_IP}{RST}")
    print(f"{BOLD}{'='*60}{RST}\n")

    # ── Step 1: Verify models exist ────────────────────────────────────
    print(f"{BOLD}[Step 1] Verifying model files on laptop...{RST}")
    required_models = [
        "label_encoder.pkl",
        "scaler.pkl",
        "isolation_forest.pkl",
        "xgboost_model.pkl",
        "xgboost_calibrator.pkl",
        "optimal_threshold.json",
        "feature_names.pkl",
    ]
    all_found = True
    for fname in required_models:
        fpath = os.path.join(MODELS_SRC, fname)
        size  = os.path.getsize(fpath) / (1024*1024) if os.path.exists(fpath) else 0
        if os.path.exists(fpath):
            print(f"  {OK}  {fname:<40} {size:.2f} MB")
        else:
            print(f"  {FAIL} {fname:<40} NOT FOUND")
            all_found = False

    if not all_found:
        print(f"\n  {FAIL} Missing model files. Run Phase 6 training first.")
        sys.exit(1)

    # ── Step 2: Copy models into detector build context ────────────────
    print(f"\n{BOLD}[Step 2] Copying models into Docker build context...{RST}")
    if os.path.exists(MODELS_DST):
        shutil.rmtree(MODELS_DST)
    shutil.copytree(MODELS_SRC, MODELS_DST)
    total_size = sum(
        os.path.getsize(os.path.join(MODELS_DST, f))
        for f in os.listdir(MODELS_DST)
    ) / (1024 * 1024)
    print(f"  {OK}  Copied models_v6/ -> {MODELS_DST}  ({total_size:.1f} MB total)")

    # ── Step 3: Build Docker image ─────────────────────────────────────
    print(f"\n{BOLD}[Step 3] Building Docker image (this runs verify_deployment.py inside)...{RST}")
    print(f"  Image name: {IMAGE_NAME}")
    print(f"  This may take 3-5 minutes on first build (pip installs)...\n")
    run(f'docker build --build-arg DEV_MODE=false -t {IMAGE_NAME} "{DETECTOR_DIR}"')
    print(f"\n  {OK}  Image built successfully: {IMAGE_NAME}")

    # ── Step 4: Smoke test inside built image ──────────────────────────
    print(f"\n{BOLD}[Step 4] Running smoke test inside built image...{RST}")
    result = run(
        f'docker run --rm {IMAGE_NAME} python verify_deployment.py',
        capture=True
    )
    if "ALL CHECKS PASSED" in result.stdout:
        print(f"  {OK}  Smoke test PASSED — image is deployment ready")
    else:
        print(f"  {FAIL} Smoke test output unexpected:")
        print(result.stdout)
        sys.exit(1)

    # ── Step 5: Export image to .tar ──────────────────────────────────
    print(f"\n{BOLD}[Step 5] Exporting image to portable .tar file...{RST}")
    print(f"  Output: {OUTPUT_TAR}")
    print(f"  This may take 1-2 minutes...")
    run(f'docker save -o "{OUTPUT_TAR}" {IMAGE_NAME}')
    tar_size = os.path.getsize(OUTPUT_TAR) / (1024 * 1024)
    print(f"  {OK}  Export complete: ddos_detector_v6_mec.tar ({tar_size:.0f} MB)")

    # ── Step 6: Cleanup build context ─────────────────────────────────
    print(f"\n{BOLD}[Step 6] Cleaning up build context...{RST}")
    shutil.rmtree(MODELS_DST)
    print(f"  {OK}  Removed temporary models_v6/ from build context")

    # ── Done ───────────────────────────────────────────────────────────
    elapsed = (time.time() - t_start) / 60
    print(f"\n{'='*60}")
    print(f"{BOLD}  BUILD COMPLETE in {elapsed:.1f} minutes{RST}")
    print(f"{'='*60}")
    print(f"""
  TRANSFER TO MEC NODE:
  ─────────────────────────────────────────────────────────

  Option A — Network (if MEC is on same network):
    scp ddos_detector_v6_mec.tar user@{MEC_IP}:~/

  Option B — USB Drive:
    Copy ddos_detector_v6_mec.tar to USB, plug into MEC

  LOAD ON MEC NODE:
  ─────────────────────────────────────────────────────────
    docker load -i ddos_detector_v6_mec.tar

  RUN ON MEC NODE (simulation mode):
    docker run --rm -it \\
      --cap-add NET_ADMIN --cap-add NET_RAW \\
      {IMAGE_NAME} \\
      python realtime_demo.py --samples 100

  RUN ON MEC NODE (live capture mode):
    docker run --rm -it \\
      --cap-add NET_ADMIN --cap-add NET_RAW \\
      --network host \\
      {IMAGE_NAME} \\
      python realtime_demo.py --live --iface eth0

  IMAGE INFO:
    Name      : {IMAGE_NAME}
    File size : {tar_size:.0f} MB
    Target MEC: {MEC_IP}
    Arch      : x86_64
    Python    : 3.10-slim
""")

if __name__ == "__main__":
    main()
