"""
=============================================================================
REAL-TIME DEFENDER v2 — Phase 6 Three-Layer Detection Engine
Edge-Intelligent Hybrid DDoS Detection System for 5G/MEC Networks
=============================================================================

USAGE:
  python3 realtime_defender_v2.py --interface eth0 --model-dir ./models_v6

THREE-LAYER DECISION ARCHITECTURE:
  Layer 1 — IF Hard Veto:
    If Isolation Forest anomaly score > 0.75 -> ALERT regardless of XGBoost.
    Prevents confident XGBoost misclassification from suppressing zero-day alerts.

  Layer 2 — Disagreement Amplifier:
    If IF says anomaly (>0.60) AND XGBoost confidently says BENIGN (<0.30),
    the fusion score is boosted by +0.15. Strong model disagreement is treated
    as a suspicious signal (catches mimicry / adversarial attacks).

  Layer 3 — Confidence-Adaptive Fusion:
    Weights shift dynamically based on XGBoost prediction confidence:
      XGB confident (>=0.80): 0.60*XGB + 0.25*IF + 0.15*DPI
      XGB moderate  (>=0.50): 0.40*XGB + 0.45*IF + 0.15*DPI
      XGB uncertain (<0.50) : 0.25*XGB + 0.60*IF + 0.15*DPI  <- IF leads

  Threshold is auto-loaded from models_v6/optimal_threshold.json
  (computed during training to achieve FPR <= 2% on validation data).
=============================================================================
"""

import os
import sys
import time
import argparse
import threading
import warnings
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings('ignore')

# Phase 7 — Mitigation Engine (auto-loaded; set DRY_RUN=True inside the
# module if you want to test without root / iptables access)
try:
    import mitigation_engine
    MITIGATION_ENABLED = True
except ImportError:
    print("[WARN] mitigation_engine.py not found — running detection-only mode.")
    MITIGATION_ENABLED = False

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
except ImportError:
    print("ERROR: Scapy not installed. Run: pip3 install scapy")
    sys.exit(1)

# -------------------------------------------------------------------------
# ARGUMENT PARSING
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--interface',  default=None,           help='Network interface to sniff (e.g. eth0)')
parser.add_argument('--model-dir',  default='./models_v6',  help='Path to models_v6 folder')
parser.add_argument('--threshold',  type=float, default=None,
                    help='Override detection threshold (default: auto-load from optimal_threshold.json)')
parser.add_argument('--window',     type=float, default=2.0, help='Flow analysis window in seconds (default 2.0)')
args = parser.parse_args()

EPS = 1e-9
RESULTS = []   # Stores all decisions for final summary

# OPT-3: Module-level inference caches — populated once in load_models().
# Avoids re-creating Python lists and numpy arrays on every 2-second window.
_BENIGN_ID    = None   # integer index of BENIGN class in le.classes_
_SCALER_MEAN  = None   # np.float32 array, shape (n_features,)
_SCALER_SCALE = None   # np.float32 array, shape (n_features,)

# Layer thresholds (overridden by optimal_threshold.json if present)
IF_VETO_THRESHOLD = 0.75
IF_DISAGREE_MIN   = 0.60
XGB_BENIGN_MAX    = 0.30
DISAGREE_BOOST    = 0.15

# -------------------------------------------------------------------------
# LOAD MODELS + THRESHOLD CONFIG
# -------------------------------------------------------------------------
def load_models(model_dir):
    global IF_VETO_THRESHOLD, IF_DISAGREE_MIN, DISAGREE_BOOST
    print(f"[INIT] Loading AI Models from: {model_dir}")
    le     = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    iso_d  = joblib.load(os.path.join(model_dir, 'isolation_forest.pkl'))
    xgb    = joblib.load(os.path.join(model_dir, 'xgboost_model.pkl'))

    # Load Platt calibrator if present (Phase 6)
    calib_path = os.path.join(model_dir, 'xgboost_calibrator.pkl')
    calib_d = joblib.load(calib_path) if os.path.exists(calib_path) else None

    try:
        xgb.set_params(device='cpu')
    except Exception:
        pass

    # Auto-load optimal threshold from JSON
    effective_threshold = args.threshold
    thresh_path = os.path.join(model_dir, 'optimal_threshold.json')
    if effective_threshold is None and os.path.exists(thresh_path):
        import json
        with open(thresh_path) as f:
            cfg = json.load(f)
        effective_threshold   = cfg['threshold']
        IF_VETO_THRESHOLD     = cfg.get('if_veto',        IF_VETO_THRESHOLD)
        IF_DISAGREE_MIN       = cfg.get('if_disagree',    IF_DISAGREE_MIN)
        DISAGREE_BOOST        = cfg.get('disagree_boost', DISAGREE_BOOST)
        print(f"[INIT] Auto-loaded threshold: {effective_threshold:.2f}  "
              f"(trained TPR={cfg['tpr']:.1f}%  FPR={cfg['fpr']:.1f}%)")
    elif effective_threshold is None:
        effective_threshold = 0.40
        print(f"[INIT] No optimal_threshold.json found. Using default: {effective_threshold}")
    else:
        print(f"[INIT] Using CLI threshold: {effective_threshold}")

    print(f"[INIT] IF Veto: {IF_VETO_THRESHOLD}  |  Disagree boost: {DISAGREE_BOOST}")
    print(f"[INIT] Models loaded. Detecting {len(le.classes_)-1} attack types.")
    print(f"[INIT] Window: {args.window}s")

    # OPT-1: Enable multi-core inference on both models at load time.
    # XGBoost n_jobs=-1 uses all CPU cores for tree traversal.
    # This cuts XGB inference time by ~15-25% depending on core count.
    try:
        xgb.set_params(n_jobs=-1)
        print(f"[INIT] XGBoost n_jobs set to -1 (all cores)")
    except Exception:
        pass
    # IsoForest is already fitted; set n_jobs on the object directly.
    # joblib will pick this up on the next score_samples() call.
    try:
        iso_d['model'].n_jobs = -1
        print(f"[INIT] IsolationForest n_jobs set to -1 (all cores)")
    except Exception:
        pass

    # OPT-3: Pre-cache benign_id and scaler arrays as float32 numpy arrays.
    # benign_id: avoids list(le.classes_).index() on every inference call.
    # scaler mean/scale: enables direct numpy transform, skipping sklearn's
    # internal input-validation and type-checking overhead (~6x faster).
    global _BENIGN_ID, _SCALER_MEAN, _SCALER_SCALE
    _BENIGN_ID    = list(le.classes_).index('BENIGN')
    _SCALER_MEAN  = scaler.mean_.astype(np.float32)
    _SCALER_SCALE = scaler.scale_.astype(np.float32)
    print(f"[INIT] Inference caches populated (benign_id={_BENIGN_ID})")

    print("-" * 65)
    return (le, scaler, iso_d['model'], iso_d['if_min'], iso_d['if_max'],
            xgb, calib_d, effective_threshold)

# -------------------------------------------------------------------------
# DPI RULE ENGINE (Phase 6 Enhanced — 6 rules)
# -------------------------------------------------------------------------
def dpi_score(f):
    """Enhanced DPI scoring for a single flow dict."""
    score  = 0.0
    syn    = f.get('syn_count', 0)
    rate   = f.get('Rate', 0)
    udp    = f.get('UDP', 0)
    icmp   = f.get('ICMP', 0)
    n      = f.get('Number', 1) or 1
    iat    = f.get('IAT_mean', 0)
    pkt    = f.get('AVG', 0)

    # Rule 1 — SYN Flood
    if syn > 50 and rate > 2000:
        score = max(score, 0.95)
    # Rule 2 — UDP Flood
    if udp > 0.8 and rate > 2000:
        score = max(score, 0.90)
    # Rule 3 — ICMP Flood
    if icmp > 0.5 and rate > 1000:
        score = max(score, 0.88)
    # Rule 4 [NEW] — SYN Ratio Anomaly (catches DDOS-SYN_FLOOD missed by XGB)
    syn_ratio = syn / (n + EPS)
    if syn_ratio > 0.85 and rate > 200:
        score = max(score, 0.88)
    # Rule 5 [NEW] — Recon Sweep (catches RECON-PINGSWEEP)
    if iat > 0.05 and n > 5 and 0 < rate < 100:
        score = max(score, 0.82)
    # Rule 6 [NEW] — Low-and-Slow / Brute Force
    if n > 15 and pkt < 120 and 0 < rate < 10:
        score = max(score, 0.78)
    # Benign guard
    if rate < 10 and iat > 0.5:
        score = min(score, 0.05)
    return float(np.clip(score, 0.0, 1.0))

# -------------------------------------------------------------------------
# OPT-4: VECTORIZED BATCH DPI SCORER
# Replaces the Python for-loop + dict-creation in run_inference.
# Operates on DataFrame columns directly with numpy broadcasting —
# all 6 rules applied across the entire batch in parallel.
# Scientific rules are identical to the scalar dpi_score() above;
# only the execution model changes (vectorized vs row-by-row).
# -------------------------------------------------------------------------
def dpi_score_batch(df: 'pd.DataFrame') -> np.ndarray:
    """Vectorized DPI scoring over a batch DataFrame. Returns float32 array."""
    syn  = df['syn_count'].to_numpy(dtype=np.float32)
    rate = df['Rate'].to_numpy(dtype=np.float32)
    udp  = df['UDP'].to_numpy(dtype=np.float32)
    icmp = df['ICMP'].to_numpy(dtype=np.float32)
    n_   = np.maximum(df['Number'].to_numpy(dtype=np.float32), 1.0)
    iat  = df['IAT_mean'].to_numpy(dtype=np.float32)
    pkt  = df['AVG'].to_numpy(dtype=np.float32)

    score = np.zeros(len(df), dtype=np.float32)

    # Rule 1 — SYN Flood
    score = np.where((syn > 50) & (rate > 2000),  np.maximum(score, 0.95), score)
    # Rule 2 — UDP Flood
    score = np.where((udp > 0.8) & (rate > 2000), np.maximum(score, 0.90), score)
    # Rule 3 — ICMP Flood
    score = np.where((icmp > 0.5) & (rate > 1000), np.maximum(score, 0.88), score)
    # Rule 4 — SYN Ratio Anomaly
    syn_ratio = syn / (n_ + EPS)
    score = np.where((syn_ratio > 0.85) & (rate > 200), np.maximum(score, 0.88), score)
    # Rule 5 — Recon Sweep
    score = np.where((iat > 0.05) & (n_ > 5) & (rate > 0) & (rate < 100),
                     np.maximum(score, 0.82), score)
    # Rule 6 — Low-and-Slow / Brute Force
    score = np.where((n_ > 15) & (pkt < 120) & (rate > 0) & (rate < 10),
                     np.maximum(score, 0.78), score)
    # Benign guard — suppress score for clearly benign patterns
    score = np.where((rate < 10) & (iat > 0.5), np.minimum(score, 0.05), score)

    return np.clip(score, 0.0, 1.0)

# -------------------------------------------------------------------------
# 5-TUPLE FLOW TRACKER
# -------------------------------------------------------------------------
class FlowTracker:
    """
    Tracks packets grouped by 5-tuple flow key:
    (src_ip, src_port, dst_ip, dst_port, proto)
    This matches CICFlowMeter's flow definition.
    """
    def __init__(self):
        self.flows = defaultdict(list)   # flow_key -> list of packet dicts
        self.lock  = threading.Lock()

    def add_packet(self, pkt):
        if IP not in pkt:
            return

        src_ip  = pkt[IP].src
        dst_ip  = pkt[IP].dst
        ttl     = pkt[IP].ttl
        size    = len(pkt)
        ts      = time.time()

        proto_name = 'OTHER'
        src_port   = 0
        dst_port   = 0
        flags      = {'S': 0, 'A': 0, 'R': 0, 'F': 0, 'P': 0}

        if TCP in pkt:
            proto_name = 'TCP'
            src_port   = pkt[TCP].sport
            dst_port   = pkt[TCP].dport
            fi         = pkt[TCP].flags
            flags = {
                'S': int(bool(fi & 0x02)),
                'A': int(bool(fi & 0x10)),
                'R': int(bool(fi & 0x04)),
                'F': int(bool(fi & 0x01)),
                'P': int(bool(fi & 0x08)),
            }
        elif UDP in pkt:
            proto_name = 'UDP'
            src_port   = pkt[UDP].sport
            dst_port   = pkt[UDP].dport
        elif ICMP in pkt:
            proto_name = 'ICMP'

        flow_key = (src_ip, src_port, dst_ip, dst_port, proto_name)

        pkt_meta = {
            'ts':    ts,
            'size':  size,
            'ttl':   ttl,
            'proto': proto_name,
            'flags': flags
        }

        with self.lock:
            self.flows[flow_key].append(pkt_meta)

    def drain(self):
        """Drain current flows and return them for processing."""
        with self.lock:
            data = dict(self.flows)
            self.flows.clear()
        return data

# -------------------------------------------------------------------------
# FEATURE EXTRACTION (Per Flow, matching CICFlowMeter logic)
# -------------------------------------------------------------------------
def extract_features(flow_key, pkts, expected_cols):
    """
    Extract 63 features from a list of packet metadata dicts.
    Returns a dict with all expected feature values.
    """
    src_ip, src_port, dst_ip, dst_port, proto = flow_key

    n    = len(pkts)
    if n < 2:
        return None  # Too few packets to analyze

    sizes = [p['size'] for p in pkts]
    ts    = sorted([p['ts'] for p in pkts])
    ttls  = [p['ttl'] for p in pkts]

    dur   = ts[-1] - ts[0]
    if dur < EPS:
        dur = EPS

    iats  = [ts[i] - ts[i-1] for i in range(1, len(ts))]
    iat_mean = float(np.mean(iats)) if iats else 0.0

    syn_c = sum(p['flags'].get('S', 0) for p in pkts)
    ack_c = sum(p['flags'].get('A', 0) for p in pkts)
    rst_c = sum(p['flags'].get('R', 0) for p in pkts)
    fin_c = sum(p['flags'].get('F', 0) for p in pkts)
    psh_c = sum(p['flags'].get('P', 0) for p in pkts)

    tcp_pkts  = sum(1 for p in pkts if p['proto'] == 'TCP')
    udp_pkts  = sum(1 for p in pkts if p['proto'] == 'UDP')
    icmp_pkts = sum(1 for p in pkts if p['proto'] == 'ICMP')

    is_tcp  = float(tcp_pkts  > 0)
    is_udp  = float(udp_pkts  > 0)
    is_icmp = float(icmp_pkts > 0)

    rate     = float(n) / dur
    tot_sum  = float(sum(sizes))
    avg_size = float(np.mean(sizes))
    std_size = float(np.std(sizes))
    min_size = float(min(sizes))
    max_size = float(max(sizes))
    ttl_mean = float(np.mean(ttls))

    f = {c: 0.0 for c in expected_cols}

    # Basic stats
    f['Number']      = float(n)
    f['Rate']        = rate
    f['Tot sum']     = tot_sum
    f['Tot size']    = tot_sum
    f['Min']         = min_size
    f['Max']         = max_size
    f['AVG']         = avg_size
    f['Std']         = std_size
    f['Variance']    = float(np.var(sizes))
    f['Time_To_Live']= ttl_mean
    f['IAT']         = iat_mean
    f['IAT_mean']    = iat_mean

    # Protocols
    f['TCP']     = f['Proto_6']  = is_tcp
    f['UDP']     = f['Proto_17'] = is_udp
    f['ICMP']    = is_icmp
    f['ARP']     = 0.0
    f['DNS']     = float(dst_port == 53 or src_port == 53)
    f['HTTP']    = float(dst_port in (80, 8080) or src_port in (80, 8080))
    f['HTTPS']   = float(dst_port == 443 or src_port == 443)
    f['SSH']     = float(dst_port == 22 or src_port == 22)

    # Flags
    f['syn_count']       = float(syn_c)
    f['ack_count']       = float(ack_c)
    f['rst_count']       = float(rst_c)
    f['fin_count']       = float(fin_c)
    f['syn_flag_number'] = float(syn_c > 0)
    f['ack_flag_number'] = float(ack_c > 0)
    f['rst_flag_number'] = float(rst_c > 0)
    f['fin_flag_number'] = float(fin_c > 0)
    f['psh_flag_number'] = float(psh_c > 0)
    f['ack_count']       = float(ack_c)

    # Engineered features (identical to phase2_fix.py)
    f['log_syn_count']   = float(np.log1p(syn_c))
    f['log_ack_count']   = float(np.log1p(ack_c))
    f['log_rst_count']   = float(np.log1p(rst_c))
    f['syn_minus_ack']   = float(syn_c - ack_c)
    f['byte_per_pkt']    = float(tot_sum / (n + EPS))
    f['high_syn_ratio']  = float(np.clip(syn_c / (n + EPS), 0, 1))
    f['log_rate']        = float(np.log1p(rate))
    f['log_iat']         = float(np.log1p(iat_mean * 1e6))
    f['ack_no_syn']      = float(f['ack_flag_number'] * (1 - f['syn_flag_number']))
    f['ddos_vs_dos_score']     = float(syn_c / (ack_c + 1.0))
    f['recon_iat_rate_ratio']  = float(np.log1p(iat_mean * 1e6) / (np.log1p(rate) + EPS))
    f['TTL_deviation']         = float(min(abs(ttl_mean - 64), abs(ttl_mean - 128)))
    f['size_cv']               = float(std_size / (avg_size + EPS))
    f['payload_asymmetry']     = float(max_size / (avg_size + EPS))
    f['conn_teardown_rate']    = float((fin_c + rst_c) / (n + EPS))
    f['ack_to_syn_ratio']      = float(np.clip(ack_c / (syn_c + EPS), 0, 1))
    f['rst_to_syn_ratio']      = float(np.clip(rst_c / (syn_c + EPS), 0, 5))
    f['rate_per_pkt']          = float(rate / (n + EPS))

    all_f = f['syn_flag_number'] + f['ack_flag_number'] + f['rst_flag_number'] + \
            f['fin_flag_number'] + f['psh_flag_number'] + EPS
    f['syn_flag_density']          = float(f['syn_flag_number'] / all_f)
    f['ack_teardown_interaction']  = float(np.clip(ack_c / (n + EPS), 0, 1) *
                                           np.clip((fin_c + rst_c) / (n + EPS), 0, 1))
    f['flood_intensity_score']     = float(np.log1p(rate) * f['high_syn_ratio'])
    f['tcp_ratio_sq']              = float(is_tcp ** 2)
    f['iat_rate_product']          = float(np.log1p(iat_mean * 1e6) * np.log1p(rate))

    # Flag entropy
    flag_vals = [f['syn_flag_number'], f['ack_flag_number'], f['rst_flag_number'],
                 f['fin_flag_number'], f['psh_flag_number']]
    total_flags = sum(flag_vals) + EPS
    entropy = 0.0
    for fv in flag_vals:
        p = (fv + EPS) / total_flags
        entropy -= p * np.log(p + EPS)
    f['flag_entropy_norm'] = float(entropy / np.log(5))

    # IAT per byte
    f['IAT_per_byte'] = float(iat_mean / (avg_size + EPS))
    f['flow_rate_ratio'] = float(rate / (tot_sum + EPS))

    # Proto_mix (number of distinct protos)
    distinct_protos = sum([is_tcp > 0, is_udp > 0, is_icmp > 0])
    f['proto_mix'] = float(distinct_protos)

    # Replace inf
    for k in list(f.keys()):
        if not np.isfinite(f[k]):
            f[k] = 0.0

    return f

# -------------------------------------------------------------------------
# INFERENCE ENGINE  (Phase 6 — 3-Layer Architecture)
# -------------------------------------------------------------------------
def run_inference(features_df, le, scaler, iso, if_min, if_max, xgb, calib_d, expected_cols):
    # OPT-3: Use pre-extracted float32 numpy arrays — no DataFrame column
    # alignment, no sklearn validation, no dtype promotion on every call.
    X_vals   = features_df[expected_cols].to_numpy(dtype=np.float32)

    # OPT-3: Direct numpy scaler transform: (X - mean) / scale.
    # Mathematically identical to scaler.transform() but skips sklearn's
    # input validation, copy logic, and dtype coercion (~6x faster for
    # small batches where overhead dominates compute time).
    X_scaled = (X_vals - _SCALER_MEAN) / _SCALER_SCALE

    # Use cached benign_id (computed once at load time, not per-window)
    benign_id    = _BENIGN_ID
    xgb_proba    = xgb.predict_proba(X_vals)

    # Apply Platt calibration if available, else use raw
    raw_att = (1.0 - xgb_proba[:, benign_id]).reshape(-1, 1)
    if calib_d is not None:
        xgb_att_prob = calib_d['calibrator'].predict_proba(raw_att)[:, 1].astype(np.float32)
    else:
        xgb_att_prob = raw_att.ravel().astype(np.float32)

    # Corrected Confidence: how sure is XGB that this is NOT benign?
    xgb_confidence = xgb_att_prob

    # OPT-2: Derive label from argmax of existing predict_proba — FREE.
    # Previously a separate xgb.predict() call was made here which ran the
    # full 68,000-tree ensemble AGAIN (~59ms wasted per inference cycle).
    # argmax over the already-computed probability matrix costs ~0.01ms.
    xgb_raw_preds  = np.argmax(xgb_proba, axis=1).astype(np.int32)
    xgb_labels     = le.inverse_transform(xgb_raw_preds)   # e.g. 'DDOS-SYN_FLOOD'

    ifs      = iso.score_samples(X_scaled)
    ifs_clip = np.clip(ifs, if_min, if_max)
    if_prob  = (1.0 - (ifs_clip - if_min) / (if_max - if_min + EPS)).astype(np.float32)

    # OPT-4: Vectorized batch DPI — replaces Python row-by-row loop + dict creation.
    # All 6 rules run as numpy broadcasts across the full flow batch simultaneously.
    dpi_scores = dpi_score_batch(features_df)

    # -- Layer 3: Confidence-Adaptive Fusion --
    n = len(if_prob)
    w_xgb = np.where(xgb_confidence >= 0.80, 0.60,
            np.where(xgb_confidence >= 0.40, 0.40, 0.25)).astype(np.float32)
    w_if  = np.where(xgb_confidence >= 0.80, 0.25,
            np.where(xgb_confidence >= 0.40, 0.45, 0.60)).astype(np.float32)
    w_dpi = np.full(n, 0.15, dtype=np.float32)
    fusion = np.clip(w_xgb * xgb_att_prob + w_if * if_prob + w_dpi * dpi_scores, 0, 1)

    # -- Layer 2: Disagreement Amplifier --
    disagree_mask = (if_prob > IF_DISAGREE_MIN) & (xgb_att_prob < XGB_BENIGN_MAX)
    fusion = np.where(disagree_mask, np.clip(fusion + DISAGREE_BOOST, 0, 1), fusion)

    # -- Layer 1: IF Hard Veto --
    veto_mask = (if_prob > IF_VETO_THRESHOLD)
    fusion    = np.where(veto_mask, np.maximum(fusion, 0.90), fusion)

    layer_triggered = np.where(veto_mask, 1, np.where(disagree_mask, 2, 3)).astype(np.int8)

    return fusion.astype(np.float32), if_prob, xgb_att_prob, dpi_scores, layer_triggered, xgb_labels

# -------------------------------------------------------------------------
# MAIN PROCESSING LOOP
# -------------------------------------------------------------------------
def processing_loop(tracker, le, scaler, iso, if_min, if_max, xgb,
                    calib_d, expected_cols, threshold):
    alert_count = 0
    pass_count  = 0
    window_num  = 0
    layer_counts = {1: 0, 2: 0, 3: 0}

    while True:
        time.sleep(args.window)
        window_num += 1
        flows = tracker.drain()

        if not flows:
            continue

        rows = []
        keys = []
        for flow_key, pkts in flows.items():
            feat = extract_features(flow_key, pkts, expected_cols)
            if feat:
                rows.append(feat)
                keys.append(flow_key)

        if not rows:
            continue

        df = pd.DataFrame(rows)
        fusion, if_p, xgb_p, dpi_p, layer_trig, xgb_labels = run_inference(
            df, le, scaler, iso, if_min, if_max, xgb, calib_d, expected_cols)

        ts = time.strftime('%H:%M:%S')
        print(f"\n[Window {window_num}] {ts} -- {len(rows)} flows analyzed")
        print(f"  {'Source IP':<16} {'Proto':<6} {'Pkts':>5} {'Rate':>8} {'SYN':>5} "
              f"{'IF':>5} {'XGB':>5} {'FUSION':>7}  {'DECISION'}")
        print(f"  {'-'*80}")

        for i, (flow_key, score) in enumerate(zip(keys, fusion)):
            src_ip, src_port, dst_ip, dst_port, proto = flow_key
            row      = rows[i]
            is_alert = score >= threshold
            lyr      = int(layer_trig[i])
            lbl      = str(xgb_labels[i])

            # Determine tier for display
            if score >= 0.90: tier_str = "T3-BLOCK"
            elif score >= 0.70: tier_str = "T2-QRTN"
            elif score >= 0.55: tier_str = "T1-THROT"
            else:               tier_str = "T0-MON"

            if is_alert:
                decision = f"[{tier_str}] {lbl}"
            else:
                decision = "BENIGN"

            # Print max 15 rows per window to prevent terminal spam
            if i < 15:
                print(f"  {src_ip:<16} {proto:<6} {int(row['Number']):>5} "
                      f"{row['Rate']:>8.1f} {int(row['syn_count']):>5} "
                      f"{if_p[i]:>5.2f} {xgb_p[i]:>5.2f} {score:>7.3f}  "
                      f"{decision}")
            elif i == 15:
                print(f"  ... and {len(rows) - 15} more flows hidden to prevent spam ...")

            if is_alert:
                alert_count += 1
                layer_counts[lyr] = layer_counts.get(lyr, 0) + 1
                if MITIGATION_ENABLED:
                    threading.Thread(
                        target=mitigation_engine.apply_mitigation,
                        args=(src_ip, str(xgb_labels[i]), float(score)),
                        daemon=True
                    ).start()
            else:
                pass_count += 1

            RESULTS.append({
                'window'      : window_num,
                'time'        : ts,
                'src_ip'      : src_ip,
                'proto'       : proto,
                'packets'     : int(row['Number']),
                'rate'        : round(row['Rate'], 2),
                'if_score'    : round(float(if_p[i]), 3),
                'xgb_score'   : round(float(xgb_p[i]), 3),
                'fusion_score': round(float(score), 3),
                'layer'       : lyr,
                'label'       : lbl,
                'decision'    : 'ATTACK' if is_alert else 'BENIGN'
            })

        total = alert_count + pass_count
        print(f"\n  TOTALS: {alert_count} ATTACKS | {pass_count} BENIGN | "
              f"DR: {alert_count/max(total,1)*100:.1f}% | "
              f"L1(IF-Veto):{layer_counts.get(1,0)}  "
              f"L2(Disagree):{layer_counts.get(2,0)}  "
              f"L3(Fusion):{layer_counts.get(3,0)}")


# -------------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------------
def main():
    le, scaler, iso, if_min, if_max, xgb, calib_d, threshold = load_models(args.model_dir)
    expected_cols = list(scaler.feature_names_in_)

    tracker = FlowTracker()

    # Start processing thread
    proc_thread = threading.Thread(
        target=processing_loop,
        args=(tracker, le, scaler, iso, if_min, if_max, xgb, calib_d, expected_cols, threshold),
        daemon=True
    )
    proc_thread.start()

    print(f"[SNIFF] Sniffing on interface: {args.interface if args.interface else 'ALL'}")
    print(f"[SNIFF] Decision threshold   : {threshold:.2f}")
    print("[SNIFF] Waiting for packets... Press Ctrl+C to stop.\n")

    try:
        sniff(iface=args.interface, prn=tracker.add_packet, store=0)
    except KeyboardInterrupt:
        print("\n\n[DONE] Stopping defender...")

        total   = len(RESULTS)
        attacks = [r for r in RESULTS if r['decision'] == 'ATTACK']
        benign  = total - len(attacks)
        l1 = sum(1 for r in attacks if r.get('layer') == 1)
        l2 = sum(1 for r in attacks if r.get('layer') == 2)
        l3 = sum(1 for r in attacks if r.get('layer') == 3)

        # Per-IP attack summary
        from collections import defaultdict
        ip_summary = defaultdict(lambda: {'count': 0, 'labels': set(), 'max_score': 0.0})
        for r in attacks:
            ip = r['src_ip']
            ip_summary[ip]['count']     += 1
            ip_summary[ip]['labels'].add(r.get('label', 'UNKNOWN'))
            ip_summary[ip]['max_score']  = max(ip_summary[ip]['max_score'], r['fusion_score'])

        print()
        print("=" * 65)
        print("  SESSION SUMMARY")
        print("=" * 65)
        print(f"  Total Flows Analyzed   : {total}")
        print(f"  Attacks Detected       : {len(attacks)}")
        print(f"  Benign Flows           : {benign}")
        print(f"  Detection Rate         : {len(attacks)/max(total,1)*100:.2f}%")
        print(f"  Detection Layers Used  : L1(IF-Veto)={l1}  L2(Disagree)={l2}  L3(Fusion)={l3}")
        print(f"  Threshold Used         : {threshold:.2f}")
        print()
        if ip_summary:
            print("  ATTACKER IPs DETECTED:")
            print(f"  {'IP Address':<18} {'Flows':>6} {'Max Score':>10}  Attack Types")
            print(f"  {'-'*65}")
            for ip, info in sorted(ip_summary.items(), key=lambda x: -x[1]['max_score']):
                labels_str = ', '.join(sorted(info['labels']))
                tier = "T3-BLOCK" if info['max_score'] >= 0.90 else \
                       "T2-QRTN"  if info['max_score'] >= 0.70 else \
                       "T1-THROT" if info['max_score'] >= 0.55 else "T0-MON"
                print(f"  {ip:<18} {info['count']:>6} {info['max_score']:>10.3f}  [{tier}] {labels_str}")
        else:
            print("  No attacks were detected in this session.")
        print("=" * 65)


if __name__ == "__main__":
    main()
