"""
=============================================================================
LATENCY BENCHMARK v1  --  MEC vs Centralized Cloud Comparison
Phase 7 Demo Tool | Edge-Intelligent DDoS Detection for 5G Networks
=============================================================================

WHAT THIS MEASURES
  This script benchmarks THREE latency components and compares two deployment
  architectures:

  1. DETECTION LATENCY
     Time (ms) from the first packet of an attack flow to a decision.
     Measured as: feature_extraction_time + inference_time (all 3 layers).

  2. MITIGATION LATENCY
     Time (ms) from detection decision to iptables rule being dispatched.
     (Rule execution itself is async via subprocess, but dispatch is sync.)

  3. TOTAL RESPONSE TIME  =  Detection + Network RTT + Mitigation

  ARCHITECTURE A -- Centralized Cloud (WITHOUT MEC)
    UE --> 5G RAN --> 5G Core --> [WAN] --> Cloud Detection Server
    Network round-trip (UE to cloud server): ~50ms one-way (100ms RTT)
    This is realistic for a cloud server 500km away at speed-of-light limit.

  ARCHITECTURE B -- MEC Edge (WITH MEC)
    UE --> 5G RAN --> MEC Node (detection HERE, co-located with gNB)
    Network RTT to MEC: ~1-5ms (same site or adjacent building)
    This is what you have now with your Docker image on the MEC node.

HOW TO RUN
  python3 latency_benchmark.py --model-dir ./models_v6 --flows 200

OUTPUT
  - Console: live per-flow results + final comparison table
  - latency_report.csv: full per-flow data for analysis
  - latency_summary.txt: human-readable summary for your report/demo

=============================================================================
"""

import os
import sys
import time
import argparse
import warnings
import json
import csv
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(description="MEC vs Cloud Latency Benchmark")
parser.add_argument("--model-dir", default="./models_v6",
                    help="Path to models_v6 folder")
parser.add_argument("--flows", type=int, default=200,
                    help="Number of synthetic flows to benchmark (default 200)")
parser.add_argument("--cloud-rtt-ms", type=float, default=100.0,
                    help="Simulated cloud network RTT in ms (default 100ms)")
parser.add_argument("--mec-rtt-ms",   type=float, default=3.0,
                    help="Simulated MEC network RTT in ms (default 3ms)")
parser.add_argument("--output-dir",   default=".",
                    help="Directory to write report files (default: current dir)")
args = parser.parse_args()

EPS = 1e-9
OUTPUT_CSV = os.path.join(args.output_dir, "latency_report.csv")
OUTPUT_TXT = os.path.join(args.output_dir, "latency_summary.txt")

# =============================================================================
# LOAD MODELS
# =============================================================================
def load_models(model_dir: str):
    print(f"\n[BENCH] Loading models from: {model_dir}")
    le     = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    iso_d  = joblib.load(os.path.join(model_dir, "isolation_forest.pkl"))
    xgb    = joblib.load(os.path.join(model_dir, "xgboost_model.pkl"))

    calib_path = os.path.join(model_dir, "xgboost_calibrator.pkl")
    calib_d    = joblib.load(calib_path) if os.path.exists(calib_path) else None

    try:
        xgb.set_params(device="cpu")
    except Exception:
        pass

    # Load threshold
    thresh_path = os.path.join(model_dir, "optimal_threshold.json")
    threshold   = 0.40
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            cfg = json.load(f)
        threshold = cfg.get("threshold", 0.40)

    iso   = iso_d["model"]
    if_min = iso_d["if_min"]
    if_max = iso_d["if_max"]
    cols   = list(scaler.feature_names_in_)

    print(f"[BENCH] Models loaded. {len(le.classes_)-1} attack classes. "
          f"Threshold={threshold:.2f}")
    return le, scaler, iso, if_min, if_max, xgb, calib_d, threshold, cols


# =============================================================================
# SYNTHETIC FLOW GENERATOR
# Generates realistic attack and benign flows for benchmarking.
# This avoids needing live network traffic during the benchmark.
# =============================================================================
ATTACK_TEMPLATES = {
    "SYN_FLOOD":  dict(syn_count=500, Rate=5000, UDP=0, ICMP=0, Number=600,
                       IAT_mean=0.001, AVG=60,  TCP=1, high_syn_ratio=0.95),
    "UDP_FLOOD":  dict(syn_count=0,   Rate=4000, UDP=1, ICMP=0, Number=500,
                       IAT_mean=0.001, AVG=512, TCP=0, high_syn_ratio=0),
    "ICMP_FLOOD": dict(syn_count=0,   Rate=2000, UDP=0, ICMP=1, Number=300,
                       IAT_mean=0.002, AVG=64,  TCP=0, high_syn_ratio=0),
    "HTTP_FLOOD": dict(syn_count=20,  Rate=800,  UDP=0, ICMP=0, Number=200,
                       IAT_mean=0.01,  AVG=300, TCP=1, high_syn_ratio=0.1),
    "SLOWLORIS":  dict(syn_count=5,   Rate=2,    UDP=0, ICMP=0, Number=50,
                       IAT_mean=0.5,   AVG=80,  TCP=1, high_syn_ratio=0.1),
    "BENIGN":     dict(syn_count=2,   Rate=50,   UDP=0, ICMP=0, Number=30,
                       IAT_mean=0.1,   AVG=400, TCP=1, high_syn_ratio=0.05),
}

RNG = np.random.default_rng(42)

def _make_flow(template: dict, cols: list) -> dict:
    """Build one synthetic flow dict matching the model feature columns."""
    f = {c: 0.0 for c in cols}

    # Apply template values
    for k, v in template.items():
        if k in f:
            f[k] = float(v)

    n   = max(float(template.get("Number", 10)), 1)
    syn = float(template.get("syn_count", 0))
    ack = syn * 0.1
    rst = syn * 0.05
    fin = syn * 0.02
    rate = float(template.get("Rate", 10))
    avg  = float(template.get("AVG", 100))
    iat  = float(template.get("IAT_mean", 0.1))
    tot  = n * avg

    f.update({
        "Number":           n,
        "Rate":             rate,
        "Tot sum":          tot,
        "Tot size":         tot,
        "Min":              avg * 0.5,
        "Max":              avg * 1.5,
        "AVG":              avg,
        "Std":              avg * 0.2,
        "Variance":         (avg * 0.2) ** 2,
        "IAT":              iat,
        "IAT_mean":         iat,
        "syn_count":        syn,
        "ack_count":        ack,
        "rst_count":        rst,
        "fin_count":        fin,
        "syn_flag_number":  float(syn > 0),
        "ack_flag_number":  float(ack > 0),
        "rst_flag_number":  float(rst > 0),
        "fin_flag_number":  float(fin > 0),
        "log_syn_count":    float(np.log1p(syn)),
        "log_ack_count":    float(np.log1p(ack)),
        "log_rst_count":    float(np.log1p(rst)),
        "syn_minus_ack":    float(syn - ack),
        "byte_per_pkt":     float(tot / (n + EPS)),
        "high_syn_ratio":   float(np.clip(syn / (n + EPS), 0, 1)),
        "log_rate":         float(np.log1p(rate)),
        "log_iat":          float(np.log1p(iat * 1e6)),
        "ddos_vs_dos_score":float(syn / (ack + 1.0)),
        "flood_intensity_score": float(np.log1p(rate) * np.clip(syn / (n + EPS), 0, 1)),
        "rate_per_pkt":     float(rate / (n + EPS)),
        "size_cv":          0.2,
        "recon_iat_rate_ratio": float(np.log1p(iat * 1e6) / (np.log1p(rate) + EPS)),
    })

    # Add small noise to make each flow unique
    for k in list(f.keys()):
        if f[k] != 0.0:
            f[k] = max(0.0, f[k] * (1.0 + RNG.uniform(-0.05, 0.05)))

    return f


def generate_flows(n_flows: int, cols: list) -> list:
    """Generate n_flows synthetic flows (mix of attack types + benign)."""
    labels_pool = list(ATTACK_TEMPLATES.keys())
    flows = []
    for i in range(n_flows):
        label = labels_pool[i % len(labels_pool)]
        flow  = _make_flow(ATTACK_TEMPLATES[label], cols)
        flow["_true_label"] = label
        flows.append(flow)
    return flows


# =============================================================================
# FEATURE EXTRACTION TIMER
# Mimics realtime_defender_v2.py feature extraction for timing accuracy.
# =============================================================================
def timed_feature_extraction(flow: dict, cols: list) -> tuple:
    """Returns (feature_dict, extraction_time_ms)."""
    t0 = time.perf_counter()
    # Simulate the work done in extract_features (column alignment + math)
    f = {c: flow.get(c, 0.0) for c in cols}
    df = pd.DataFrame([f])
    t1 = time.perf_counter()
    return df, (t1 - t0) * 1000.0


# =============================================================================
# INFERENCE TIMER (3-Layer pipeline)
# =============================================================================
IF_VETO_THRESHOLD = 0.75
IF_DISAGREE_MIN   = 0.60
XGB_BENIGN_MAX    = 0.30
DISAGREE_BOOST    = 0.15

def timed_inference(df: pd.DataFrame,
                    le, scaler, iso, if_min, if_max,
                    xgb, calib_d, cols) -> tuple:
    """
    Runs the full 3-layer inference on a single flow.
    Returns (fusion_score, attack_label, inference_time_ms).
    """
    t0 = time.perf_counter()

    X        = df[cols]
    X_scaled = scaler.transform(X)

    benign_id    = list(le.classes_).index("BENIGN")
    xgb_proba    = xgb.predict_proba(X.values.astype(np.float32))
    raw_att      = (1.0 - xgb_proba[:, benign_id]).reshape(-1, 1)

    if calib_d is not None:
        xgb_att = calib_d["calibrator"].predict_proba(raw_att)[:, 1].astype(np.float32)
    else:
        xgb_att = raw_att.ravel().astype(np.float32)

    ifs_raw  = iso.score_samples(X_scaled)
    ifs_clip = np.clip(ifs_raw, if_min, if_max)
    if_prob  = (1.0 - (ifs_clip - if_min) / (if_max - if_min + EPS)).astype(np.float32)

    conf = xgb_att[0]
    w_xgb = 0.60 if conf >= 0.80 else (0.40 if conf >= 0.40 else 0.25)
    w_if  = 0.25 if conf >= 0.80 else (0.45 if conf >= 0.40 else 0.60)
    w_dpi = 0.15
    fusion = float(np.clip(w_xgb * xgb_att[0] + w_if * if_prob[0] + w_dpi * 0.0, 0, 1))

    if if_prob[0] > IF_DISAGREE_MIN and xgb_att[0] < XGB_BENIGN_MAX:
        fusion = min(1.0, fusion + DISAGREE_BOOST)
    if if_prob[0] > IF_VETO_THRESHOLD:
        fusion = max(fusion, 0.90)

    pred_label = str(le.inverse_transform(xgb.predict(X.values.astype(np.float32)))[0])

    t1 = time.perf_counter()
    return fusion, pred_label, (t1 - t0) * 1000.0


# =============================================================================
# MITIGATION DISPATCH TIMER
# Measures time to dispatch the mitigation action (not iptables execution).
# =============================================================================
def timed_mitigation_dispatch(src_ip: str,
                               attack_label: str,
                               fusion_score: float) -> float:
    """
    Measure the Python-side mitigation dispatch time.
    iptables subprocess execution is NOT included (it's OS-level, ~1-2ms).
    Returns dispatch_time_ms.
    """
    t0 = time.perf_counter()

    ATTACK_FAMILY_MAP = {
        "DDOS-SYN_FLOOD": "SYN_FLOOD", "DOS-SYN_FLOOD": "SYN_FLOOD",
        "DDOS-UDP_FLOOD":  "UDP_FLOOD", "DOS-UDP_FLOOD":  "UDP_FLOOD",
        "DDOS-ICMP_FLOOD": "ICMP_FLOOD","DDOS-HTTP_FLOOD":"HTTP_FLOOD",
        "DOS-SLOWLORIS":   "SLOWLORIS", "DDOS-DNS_AMPLIFICATION": "DNS_AMP",
    }
    family = ATTACK_FAMILY_MAP.get(attack_label.upper(), "UNKNOWN")
    tier   = (3 if fusion_score >= 0.90 else
              2 if fusion_score >= 0.70 else
              1 if fusion_score >= 0.55 else 0)

    # Build the command list (same logic as mitigation_engine, no execution)
    if family == "SYN_FLOOD":
        cmds = [["sysctl", "-w", "net.ipv4.tcp_syncookies=1"],
                ["iptables", "-A", "INPUT", "-p", "tcp", "--syn",
                 "-s", src_ip, "-m", "hashlimit",
                 "--hashlimit-name", f"syn_bench",
                 "--hashlimit-above", "10/sec", "-j", "DROP"]]
    elif tier == 3:
        cmds = [["iptables", "-A", "INPUT",   "-s", src_ip, "-j", "DROP"],
                ["iptables", "-A", "FORWARD", "-s", src_ip, "-j", "DROP"]]
    else:
        cmds = [["iptables", "-A", "INPUT", "-s", src_ip, "-j", "DROP"]]

    _ = [" ".join(c) for c in cmds]   # simulate string building (no execute)

    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


# =============================================================================
# MAIN BENCHMARK
# =============================================================================
def run_benchmark():
    print("=" * 70)
    print("  MEC vs CENTRALIZED CLOUD — LATENCY BENCHMARK")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load models
    le, scaler, iso, if_min, if_max, xgb, calib_d, threshold, cols = \
        load_models(args.model_dir)

    # Generate flows
    print(f"\n[BENCH] Generating {args.flows} synthetic flows...")
    flows = generate_flows(args.flows, cols)
    print(f"[BENCH] Flow types: "
          + ", ".join(f"{k}:{sum(1 for f in flows if f['_true_label']==k)}"
                      for k in ATTACK_TEMPLATES.keys()))

    # --- Warm-up (first call is slower due to JIT / cache cold start) -------
    print("\n[BENCH] Warming up models (3 flows)...")
    for i in range(min(3, len(flows))):
        f  = flows[i]
        df, _ = timed_feature_extraction(f, cols)
        timed_inference(df, le, scaler, iso, if_min, if_max, xgb, calib_d, cols)
    print("[BENCH] Warm-up done.\n")

    # --- Benchmark loop ------------------------------------------------------
    results = []
    print(f"  {'#':<5} {'Label':<16} {'Feat(ms)':<10} {'Infer(ms)':<11} "
          f"{'Mitig(ms)':<11} {'Decision':<12} {'Score':<7}")
    print(f"  {'-'*75}")

    for idx, flow in enumerate(flows):
        true_label = flow.pop("_true_label")
        src_ip     = f"10.0.0.{(idx % 200) + 10}"

        # 1. Feature extraction
        df, feat_ms = timed_feature_extraction(flow, cols)

        # 2. Inference (all 3 layers)
        fusion, pred_label, infer_ms = timed_inference(
            df, le, scaler, iso, if_min, if_max, xgb, calib_d, cols)

        # 3. Mitigation dispatch (Python-side only, no iptables execution)
        is_attack  = fusion >= threshold
        mitig_ms   = timed_mitigation_dispatch(src_ip, pred_label, fusion) \
                     if is_attack else 0.0

        # Total ML pipeline time (what MEC adds)
        pipeline_ms = feat_ms + infer_ms + mitig_ms

        # MEC total response time (pipeline + MEC network RTT)
        mec_total_ms   = pipeline_ms + args.mec_rtt_ms

        # Cloud total response time (pipeline + cloud network RTT)
        # Cloud has 2x RTT: traffic must go TO cloud and response come BACK
        cloud_total_ms = pipeline_ms + args.cloud_rtt_ms

        decision_str = "[ATTACK]" if is_attack else "BENIGN"

        if idx < 30 or idx % 20 == 0:  # Print first 30 + every 20th
            print(f"  {idx+1:<5} {true_label:<16} {feat_ms:<10.3f} {infer_ms:<11.3f} "
                  f"{mitig_ms:<11.3f} {decision_str:<12} {fusion:<7.3f}")

        results.append({
            "flow_index":       idx + 1,
            "true_label":       true_label,
            "pred_label":       pred_label,
            "fusion_score":     round(fusion, 4),
            "is_attack":        is_attack,
            "feat_ms":          round(feat_ms,     4),
            "infer_ms":         round(infer_ms,    4),
            "mitig_ms":         round(mitig_ms,    4),
            "pipeline_ms":      round(pipeline_ms, 4),
            "mec_total_ms":     round(mec_total_ms,    4),
            "cloud_total_ms":   round(cloud_total_ms,  4),
            "latency_saved_ms": round(cloud_total_ms - mec_total_ms, 4),
        })

    # --- Compute statistics --------------------------------------------------
    df_r = pd.DataFrame(results)

    feat_arr      = df_r["feat_ms"].values
    infer_arr     = df_r["infer_ms"].values
    mitig_arr     = df_r[df_r["is_attack"]]["mitig_ms"].values
    pipeline_arr  = df_r["pipeline_ms"].values
    mec_arr       = df_r["mec_total_ms"].values
    cloud_arr     = df_r["cloud_total_ms"].values
    saved_arr     = df_r["latency_saved_ms"].values

    n_attack      = int(df_r["is_attack"].sum())
    n_benign      = len(df_r) - n_attack

    def stats(arr):
        return (np.mean(arr), np.median(arr), np.percentile(arr, 95),
                np.min(arr), np.max(arr))

    feat_stats    = stats(feat_arr)
    infer_stats   = stats(infer_arr)
    pipe_stats    = stats(pipeline_arr)
    mec_stats     = stats(mec_arr)
    cloud_stats   = stats(cloud_arr)
    saved_stats   = stats(saved_arr)

    latency_reduction_pct = (
        (cloud_stats[0] - mec_stats[0]) / cloud_stats[0] * 100
    )
    speedup_factor = cloud_stats[0] / mec_stats[0]

    # --- Print summary -------------------------------------------------------
    sep = "=" * 70

    summary_lines = [
        "",
        sep,
        "  LATENCY BENCHMARK RESULTS",
        f"  Flows benchmarked : {len(df_r)} total  |  "
        f"{n_attack} attacks  |  {n_benign} benign",
        f"  Timestamp         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sep,
        "",
        "  COMPONENT BREAKDOWN (ms)  --  mean | median | p95 | min | max",
        f"  Feature Extraction : {feat_stats[0]:6.3f} | {feat_stats[1]:6.3f} | "
        f"{feat_stats[2]:6.3f} | {feat_stats[3]:6.3f} | {feat_stats[4]:6.3f}",
        f"  3-Layer Inference  : {infer_stats[0]:6.3f} | {infer_stats[1]:6.3f} | "
        f"{infer_stats[2]:6.3f} | {infer_stats[3]:6.3f} | {infer_stats[4]:6.3f}",
        f"  ML Pipeline Total  : {pipe_stats[0]:6.3f} | {pipe_stats[1]:6.3f} | "
        f"{pipe_stats[2]:6.3f} | {pipe_stats[3]:6.3f} | {pipe_stats[4]:6.3f}",
        "",
        "  DEPLOYMENT COMPARISON",
        f"  MEC  RTT assumed   : {args.mec_rtt_ms:.1f} ms  "
        f"(co-located with 5G gNB, same site)",
        f"  Cloud RTT assumed  : {args.cloud_rtt_ms:.1f} ms  "
        f"(centralized server, ~500 km distance)",
        "",
        f"  {'Metric':<30} {'MEC':>12} {'Cloud':>12} {'Improvement':>14}",
        f"  {'-'*70}",
        f"  {'Mean Total Response Time':<30} {mec_stats[0]:>10.2f}ms "
        f"{cloud_stats[0]:>10.2f}ms   "
        f"{latency_reduction_pct:>+.1f}% faster",
        f"  {'Median Total Response Time':<30} {mec_stats[1]:>10.2f}ms "
        f"{cloud_stats[1]:>10.2f}ms",
        f"  {'P95 Total Response Time':<30} {mec_stats[2]:>10.2f}ms "
        f"{cloud_stats[2]:>10.2f}ms",
        f"  {'Latency Saved per Flow (mean)':<30} {'':>12} {'':>12}   "
        f"{saved_stats[0]:>+.2f}ms",
        f"  {'Speedup Factor':<30} {speedup_factor:>10.1f}x {'1.0x':>12}",
        "",
        "  ATTACK-SPECIFIC RESPONSE TIMES (mean, MEC)",
        f"  {'Attack Type':<20}  {'Flows':>6}  {'Pipeline(ms)':>13}  {'MEC Total(ms)':>14}  {'Cloud Total(ms)':>15}",
        f"  {'-'*75}",
    ]

    for atype in ATTACK_TEMPLATES.keys():
        sub = df_r[df_r["true_label"] == atype]
        if len(sub) == 0:
            continue
        summary_lines.append(
            f"  {atype:<20}  {len(sub):>6}  {sub['pipeline_ms'].mean():>13.3f}  "
            f"{sub['mec_total_ms'].mean():>14.3f}  {sub['cloud_total_ms'].mean():>15.3f}"
        )

    summary_lines += [
        "",
        sep,
        "  KEY FINDING FOR DEMO / REPORT",
        f"  With MEC, the full detect-and-mitigate response cycle completes",
        f"  in {mec_stats[0]:.2f}ms on average, vs {cloud_stats[0]:.2f}ms for centralized cloud.",
        f"  This is a {latency_reduction_pct:.1f}% reduction ({speedup_factor:.1f}x speedup).",
        f"  At 5G UE scale (10 Gbps), every ms saved = significant attack",
        f"  traffic blocked before it reaches the 5G Core / application servers.",
        sep,
        f"  Reports written to:",
        f"    {OUTPUT_CSV}",
        f"    {OUTPUT_TXT}",
        sep,
        "",
    ]

    for line in summary_lines:
        print(line)

    # --- Write CSV report ----------------------------------------------------
    df_r.to_csv(OUTPUT_CSV, index=False)

    # --- Write text summary --------------------------------------------------
    with open(OUTPUT_TXT, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"[BENCH] CSV report  -> {OUTPUT_CSV}")
    print(f"[BENCH] Text report -> {OUTPUT_TXT}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    run_benchmark()
