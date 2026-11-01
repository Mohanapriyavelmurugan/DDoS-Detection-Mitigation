"""
=============================================================================
MEASUREMENT 2 — Real-Time Throughput (Flows per Second)
MEC Advantage Demo | Phase 7
=============================================================================

WHAT THIS MEASURES:
  Maximum number of network flows the MEC detection engine can analyze
  per second — at increasing load levels (UE counts).

  The test simulates N concurrent UEs each generating a burst of attack
  flows simultaneously. The system measures:
    - Total flows processed per second
    - Mean inference time per flow at each load level
    - Whether detection rate degrades under load

  MEC ADVANTAGE CLAIM:
    A dedicated MEC node processes far more flows/sec than an
    overloaded centralized server serving many sites simultaneously.

HOW TO RUN:
  python3 /app/measure_throughput.py --model-dir /app/models_v6
  python3 /app/measure_throughput.py --model-dir /app/models_v6 --max-ues 200
=============================================================================
"""

import argparse
import os
import sys
import time
import warnings
import json
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import sys
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")

RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

ap = argparse.ArgumentParser(description="M2 — Throughput Measurement")
ap.add_argument("--model-dir", default="./models_v6")
ap.add_argument("--max-ues",   type=int, default=100, help="Max UEs to simulate (default 100)")
ap.add_argument("--repeats",   type=int, default=5,   help="Repeat each UE count N times")
ap.add_argument("--output",    default="./m2_throughput_results.txt")
args = ap.parse_args()

EPS = 1e-9

# -- Load models ---------------------------------------------------------------
def load():
    print(f"{CYAN}[M2] Loading models...{RESET}")
    le     = joblib.load(os.path.join(args.model_dir, "label_encoder.pkl"))
    scaler = joblib.load(os.path.join(args.model_dir, "scaler.pkl"))
    iso_d  = joblib.load(os.path.join(args.model_dir, "isolation_forest.pkl"))
    xgb    = joblib.load(os.path.join(args.model_dir, "xgboost_model.pkl"))
    calib_p= os.path.join(args.model_dir, "xgboost_calibrator.pkl")
    calib  = joblib.load(calib_p) if os.path.exists(calib_p) else None

    try:   xgb.set_params(n_jobs=-1)
    except Exception: pass
    iso_d["model"].n_jobs = -1

    bid   = list(le.classes_).index("BENIGN")
    smean = scaler.mean_.astype(np.float32)
    sscal = scaler.scale_.astype(np.float32)
    cols  = list(scaler.feature_names_in_)
    thresh= 0.40
    tp    = os.path.join(args.model_dir, "optimal_threshold.json")
    if os.path.exists(tp):
        thresh = json.load(open(tp))["threshold"]

    print(f"{GREEN}[M2] Ready. Threshold={thresh:.2f}{RESET}\n")
    return le, xgb, iso_d["model"], iso_d["if_min"], iso_d["if_max"], \
           calib, bid, smean, sscal, cols, thresh

# -- Synthetic flow batch ------------------------------------------------------
RNG = np.random.default_rng(42)

def make_batch(n_flows, cols):
    """Generate n_flows synthetic attack flow rows as a numpy float32 array."""
    data = []
    for i in range(n_flows):
        syn  = RNG.uniform(50, 600)
        rate = RNG.uniform(500, 5000)
        n_   = RNG.uniform(100, 600)
        avg  = RNG.uniform(60, 512)
        iat  = RNG.uniform(0.001, 0.01)
        tot  = n_ * avg
        row  = {c: 0.0 for c in cols}
        row.update({
            "Number": n_, "Rate": rate, "Tot sum": tot, "Tot size": tot,
            "Min": avg*0.5, "Max": avg*1.5, "AVG": avg, "Std": avg*0.2,
            "Variance": (avg*0.2)**2, "IAT": iat, "IAT_mean": iat,
            "TCP": 1.0, "syn_count": syn, "ack_count": syn*0.1,
            "syn_flag_number": 1.0, "ack_flag_number": 0.0,
            "high_syn_ratio": float(np.clip(syn/n_, 0, 1)),
            "log_syn_count": float(np.log1p(syn)),
            "log_ack_count": float(np.log1p(syn*0.1)),
            "log_rate": float(np.log1p(rate)),
            "log_iat":  float(np.log1p(iat*1e6)),
            "byte_per_pkt": float(tot/(n_+EPS)),
            "ddos_vs_dos_score": float(syn/(syn*0.1+1.0)),
            "flood_intensity_score": float(np.log1p(rate)*np.clip(syn/n_, 0, 1)),
            "rate_per_pkt": float(rate/(n_+EPS)),
            "syn_minus_ack": float(syn - syn*0.1),
        })
        data.append(row)
    df = pd.DataFrame(data)
    return df

# -- Single batch inference ----------------------------------------------------
def run_batch(df, xgb, iso, if_min, if_max, calib, bid, smean, sscal, cols, thresh):
    X    = df[cols].to_numpy(dtype=np.float32)
    Xs   = (X - smean) / sscal
    prob = xgb.predict_proba(X)
    raw  = (1.0 - prob[:, bid]).reshape(-1, 1)
    att  = calib["calibrator"].predict_proba(raw)[:, 1] if calib else raw.ravel()
    ifs  = iso.score_samples(Xs)
    ifc  = np.clip(ifs, if_min, if_max)
    ifp  = 1.0 - (ifc - if_min) / (if_max - if_min + EPS)
    conf = att
    w_xgb= np.where(conf>=0.80, 0.60, np.where(conf>=0.40, 0.40, 0.25))
    w_if = np.where(conf>=0.80, 0.25, np.where(conf>=0.40, 0.45, 0.60))
    fus  = np.clip(w_xgb*att + w_if*ifp, 0, 1)
    n_det= int((fus >= thresh).sum())
    return n_det, len(df)

# -- Main ----------------------------------------------------------------------
def run():
    le, xgb, iso, if_min, if_max, calib, bid, smean, sscal, cols, thresh = load()

    # UE counts to test
    ue_counts = []
    c = 1
    while c <= args.max_ues:
        ue_counts.append(c)
        if c < 10:   c += 1
        elif c < 50: c += 5
        else:        c += 10

    # Progressive warm-up — pre-heats XGBoost caches and numpy memory pools
    # for small, medium, and large batch sizes to eliminate cold-start spikes.
    print(f"{CYAN}[M2] Warming up model caches (3 passes)...{RESET}")
    for warmup_n in [50, 150, 300]:
        w_df = make_batch(warmup_n, cols)
        run_batch(w_df, xgb, iso, if_min, if_max, calib, bid, smean, sscal, cols, thresh)
    print(f"{GREEN}[M2] Warm-up complete. Starting benchmark...{RESET}\n")

    print(f"{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  MEASUREMENT 2 — REAL-TIME THROUGHPUT (FLOWS / SECOND){RESET}")
    print(f"{BOLD}  Simulating 1 → {args.max_ues} concurrent UEs attacking simultaneously{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")
    print(f"\n  {'UEs':>5} {'Flows/window':>13} {'Batch time':>11} "
          f"{'Flows/sec':>10} {'Detect rate':>12} {'Status'}")
    print(f"  {'-'*68}")

    summary_rows = []

    for n_ues in ue_counts:
        # Each UE contributes ~10 flows per 2-second window
        flows_per_window = n_ues * 10
        df = make_batch(flows_per_window, cols)

        times = []
        det_total = 0
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            nd, nt = run_batch(df, xgb, iso, if_min, if_max, calib,
                               bid, smean, sscal, cols, thresh)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            det_total += nd

        mean_ms     = np.mean(times)
        flows_sec   = flows_per_window / (mean_ms / 1000)
        det_rate    = det_total / (args.repeats * flows_per_window) * 100
        per_flow_ms = mean_ms / flows_per_window

        if flows_sec >= 5000:
            status = f"{GREEN}EXCELLENT{RESET}"
        elif flows_sec >= 1000:
            status = f"{GREEN}GOOD{RESET}"
        elif flows_sec >= 500:
            status = f"{YELLOW}ADEQUATE{RESET}"
        else:
            status = f"{RED}DEGRADED{RESET}"

        print(f"  {n_ues:>5} {flows_per_window:>13} {mean_ms:>9.1f}ms "
              f"{flows_sec:>10,.0f} {det_rate:>10.1f}%   {status}")

        summary_rows.append({
            "n_ues": n_ues,
            "flows_per_window": flows_per_window,
            "batch_ms": round(mean_ms, 2),
            "flows_per_sec": round(flows_sec, 0),
            "per_flow_ms": round(per_flow_ms, 4),
            "detect_rate_pct": round(det_rate, 1),
        })

    # Find max sustainable UE count (flows/sec >= 1000 = 1K flows/sec)
    sustainable = [r for r in summary_rows if r["flows_per_sec"] >= 1000]
    max_ue_ok   = max((r["n_ues"] for r in sustainable), default=0)
    peak_fps    = max(r["flows_per_sec"] for r in summary_rows)

    lines = [
        "",
        "="*70,
        "  MEASUREMENT 2 — THROUGHPUT SUMMARY",
        "="*70,
        f"  Peak throughput          : {peak_fps:,.0f} flows/second",
        f"  Max UEs at >1000 fps     : {max_ue_ok} concurrent UEs",
        f"  Flows per UE per window  : 10 flows / 2-second window",
        "",
        "  MEC ADVANTAGE:",
        "  A dedicated MEC node handles all 5G UEs attached to its gNB.",
        "  A centralized cloud server handles traffic from MANY gNBs,",
        "  sharing compute across sites → far lower per-site throughput.",
        "",
        f"  {'UEs':>5} | {'Flows/win':>10} | {'Batch ms':>9} | "
        f"{'Flows/sec':>10} | {'Det rate':>9}",
        f"  {'-'*60}",
    ]
    for r in summary_rows:
        lines.append(
            f"  {r['n_ues']:>5} | {r['flows_per_window']:>10} | "
            f"{r['batch_ms']:>9.1f} | {r['flows_per_sec']:>10,.0f} | "
            f"{r['detect_rate_pct']:>8.1f}%"
        )
    lines += [
        "="*70,
        f"  Full results saved to: {args.output}",
        "="*70,
    ]

    print()
    for line in lines:
        print(line)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n{GREEN}[M2] Done. Results saved to {args.output}{RESET}\n")

if __name__ == "__main__":
    run()
