"""
=================================================================
REAL-TIME DDOS DETECTOR  —  Phase 6  (Docker Testing)
=================================================================
TWO MODES:
  python realtime_demo.py              -> Simulation (replays test.parquet)
  python realtime_demo.py --live       -> Live capture from Docker network
=================================================================
"""

import os, sys, json, time, joblib, argparse, warnings, math
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

try:
    import telemetry as _telemetry
except ImportError:
    _telemetry = None

try:
    import mitigation_engine as _mitigation
    MITIGATION_ENABLED = True
except ImportError:
    _mitigation = None
    MITIGATION_ENABLED = False

# ── Colours ───────────────────────────────────────────────────────────
try:
    from colorama import Fore, Style, init as cinit
    cinit(autoreset=True)
    RED     = Fore.RED    + Style.BRIGHT
    GREEN   = Fore.GREEN  + Style.BRIGHT
    YELLOW  = Fore.YELLOW + Style.BRIGHT
    CYAN    = Fore.CYAN
    MAGENTA = Fore.MAGENTA + Style.BRIGHT
    RESET   = Style.RESET_ALL
except ImportError:
    RED = GREEN = YELLOW = CYAN = MAGENTA = RESET = ""

MODEL_DIR = "./models_v6"
DATA_FILE = "./test.parquet"
EPS       = 1e-9

# ═════════════════════════════════════════════════════════════════════
# 1. LOAD MODELS
# ═════════════════════════════════════════════════════════════════════
def load_models():
    print(f"\n{CYAN}[*] Loading Phase 6 models...{RESET}")
    le     = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    iso_d  = joblib.load(f"{MODEL_DIR}/isolation_forest.pkl")
    xgb    = joblib.load(f"{MODEL_DIR}/xgboost_model.pkl")
    feats  = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
    with open(f"{MODEL_DIR}/optimal_threshold.json") as f:
        cfg = json.load(f)
    try:
        xgb.set_params(device='cpu')
    except Exception:
        pass
    benign_id = list(le.classes_).index('BENIGN')
    print(f"{GREEN}[+] Models loaded.{RESET}")
    print(f"    Classes   : {len(le.classes_)}")
    print(f"    Features  : {len(feats)}")
    print(f"    Threshold : {cfg['threshold']}  IF veto: {cfg['if_veto']}")
    return le, scaler, iso_d, xgb, feats, cfg, benign_id

# ═════════════════════════════════════════════════════════════════════
# 2. FULL 63-FEATURE EXTRACTOR  (from a list of raw Scapy packets)
# ═════════════════════════════════════════════════════════════════════
def extract_features(packets, timestamps, raw_buf, feats):
    """
    Compute all 63 training features.
    raw_buf: dict with pre-counted {tcp, udp, icmp, syn, ack, ...}
             These are guaranteed correct because they were counted
             directly in on_packet() as each packet arrived.
    """
    from scapy.all import IP, TCP, UDP, ICMP, ARP

    n = len(packets)
    if n == 0:
        return {f: 0.0 for f in feats}

    # ── Use the pre-counted raw values (guaranteed correct) ───────────
    tcp_count  = raw_buf['tcp']
    udp_count  = raw_buf['udp']
    icmp_count = raw_buf['icmp']
    arp_count  = raw_buf['arp']
    syn_count  = raw_buf['syn']
    ack_count  = raw_buf['ack']
    fin_count  = raw_buf['fin']
    rst_count  = raw_buf['rst']
    psh_count  = raw_buf['psh']

    # ── Packet-size stats (still need to loop packets for sizes/TTL) ──
    pkt_sizes   = [len(p) for p in packets]
    ttl_vals    = []
    header_lens = []
    protos      = []
    http_count = https_count = dns_count = ssh_count = 0

    for pkt in packets:
        if IP in pkt:
            ttl_vals.append(float(pkt[IP].ttl))
            header_lens.append(float(pkt[IP].ihl * 4))
            protos.append(pkt[IP].proto)
        else:
            ttl_vals.append(0.0)
            header_lens.append(0.0)
            protos.append(0)
        if TCP in pkt:
            dp = pkt[TCP].dport; sp = pkt[TCP].sport
            if dp == 80  or sp == 80:   http_count  += 1
            if dp == 443 or sp == 443:  https_count += 1
            if dp == 22  or sp == 22:   ssh_count   += 1
        elif UDP in pkt:
            dp = pkt[UDP].dport; sp = pkt[UDP].sport
            if dp == 53 or sp == 53: dns_count += 1

    sizes_arr = np.array(pkt_sizes, dtype=np.float32)
    tot_sum   = float(np.sum(sizes_arr))
    pkt_min   = float(np.min(sizes_arr))
    pkt_max   = float(np.max(sizes_arr))
    pkt_avg   = float(np.mean(sizes_arr))
    pkt_std   = float(np.std(sizes_arr))
    pkt_var   = float(np.var(sizes_arr))
    pkt_range = pkt_max - pkt_min

    if len(timestamps) >= 2:
        ts_arr   = np.array(sorted(timestamps))
        iats     = np.diff(ts_arr)
        duration = float(ts_arr[-1] - ts_arr[0])
        iat_sum  = float(np.sum(iats))
        iat_mean = float(np.mean(iats))
    else:
        duration = 1.0; iat_sum = 0.0; iat_mean = 0.0
    duration = max(duration, EPS)

    rate      = float(n) / duration
    tcp_r     = tcp_count  / n
    udp_r     = udp_count  / n
    icmp_r    = icmp_count / n
    arp_r     = arp_count  / n
    http_r    = http_count / n
    https_r   = https_count/ n
    dns_r     = dns_count  / n
    ssh_r     = ssh_count  / n
    syn_r     = syn_count  / n    # high_syn_ratio
    iat_per_b = iat_mean / (pkt_avg + EPS)

    proto_counts = defaultdict(int)
    for p in protos:
        proto_counts[p] += 1
    proto_mix = float(len(set(protos)))
    proto_0   = proto_counts.get(0,  0) / n
    proto_1   = proto_counts.get(1,  0) / n
    proto_6   = proto_counts.get(6,  0) / n
    proto_17  = proto_counts.get(17, 0) / n
    proto_47  = proto_counts.get(47, 0) / n

    avg_hdr = float(np.mean(header_lens)) if header_lens else 0.0
    avg_ttl = float(np.mean(ttl_vals))    if ttl_vals    else 0.0
    ttl_dev = float(np.std(ttl_vals))     if ttl_vals    else 0.0

    log_ack = math.log(ack_count + 1)
    log_syn = math.log(syn_count + 1)
    log_rst = math.log(rst_count + 1)

    flag_arr = np.array([syn_count, ack_count, fin_count, rst_count, psh_count], dtype=np.float32)
    flag_sum = flag_arr.sum()
    if flag_sum > 0:
        fp = flag_arr / flag_sum; fp = fp[fp > 0]
        flag_ent = -np.sum(fp * np.log2(fp + EPS)) / math.log2(5 + EPS)
    else:
        flag_ent = 0.0

    size_cv    = pkt_std / (pkt_avg + EPS)
    byte_per_p = tot_sum / (n + EPS)
    log_rate   = math.log(rate + 1)
    log_iat    = math.log(iat_mean + 1)
    ack_no_syn = max(0.0, float(ack_count - syn_count))
    ddos_dos   = rate * (1.0 / (pkt_avg + EPS)) * (1 + syn_count)
    recon_r    = (iat_mean * 1000) / (rate + EPS)
    teardown   = (fin_count + rst_count) / (n + EPS)
    ack_syn_r  = ack_count / (syn_count + EPS)
    rst_syn_r  = rst_count / (syn_count + EPS)
    rate_pkt   = rate / (n + EPS)
    syn_dens   = syn_count / duration
    ack_tear   = ack_count * (fin_count + rst_count)
    flood_int  = rate * syn_r
    tcp_sq     = tcp_r ** 2
    iat_rate   = iat_mean * rate
    flow_r_r   = rate / (n / max(duration, EPS) + EPS)

    fd = {
        'Header_Length'           : avg_hdr,
        'Time_To_Live'            : avg_ttl,
        'Rate'                    : rate,
        'fin_flag_number'         : fin_count / (n + EPS),
        'syn_flag_number'         : syn_count / (n + EPS),
        'rst_flag_number'         : rst_count / (n + EPS),
        'psh_flag_number'         : psh_count / (n + EPS),
        'ack_flag_number'         : ack_count / (n + EPS),
        'ack_count'               : float(ack_count),
        'syn_count'               : float(syn_count),
        'fin_count'               : float(fin_count),
        'rst_count'               : float(rst_count),
        'HTTP'                    : http_r,
        'HTTPS'                   : https_r,
        'DNS'                     : dns_r,
        'SSH'                     : ssh_r,
        'TCP'                     : tcp_r,
        'UDP'                     : udp_r,
        'ARP'                     : arp_r,
        'ICMP'                    : icmp_r,
        'Tot sum'                 : tot_sum,
        'Min'                     : pkt_min,
        'Max'                     : pkt_max,
        'AVG'                     : pkt_avg,
        'Std'                     : pkt_std,
        'Tot size'                : tot_sum,
        'IAT'                     : iat_sum,
        'Number'                  : float(n),
        'Variance'                : pkt_var,
        'IAT_mean'                : iat_mean,
        'IAT_per_byte'            : iat_per_b,
        'pkt_size_range'          : pkt_range,
        'flow_rate_ratio'         : flow_r_r,
        'proto_mix'               : proto_mix,
        'Proto_0'                 : proto_0,
        'Proto_1'                 : proto_1,
        'Proto_6'                 : proto_6,
        'Proto_17'                : proto_17,
        'Proto_47'                : proto_47,
        'log_ack_count'           : log_ack,
        'log_syn_count'           : log_syn,
        'log_rst_count'           : log_rst,
        'syn_minus_ack'           : float(syn_count - ack_count),
        'flag_entropy_norm'       : flag_ent,
        'byte_per_pkt'            : byte_per_p,
        'high_syn_ratio'          : syn_r,
        'log_rate'                : log_rate,
        'log_iat'                 : log_iat,
        'ack_no_syn'              : ack_no_syn,
        'ddos_vs_dos_score'       : ddos_dos,
        'recon_iat_rate_ratio'    : recon_r,
        'TTL_deviation'           : ttl_dev,
        'size_cv'                 : size_cv,
        'payload_asymmetry'       : size_cv,
        'conn_teardown_rate'      : teardown,
        'ack_to_syn_ratio'        : ack_syn_r,
        'rst_to_syn_ratio'        : rst_syn_r,
        'rate_per_pkt'            : rate_pkt,
        'syn_flag_density'        : syn_dens,
        'ack_teardown_interaction' : float(ack_tear),
        'flood_intensity_score'   : flood_int,
        'tcp_ratio_sq'            : tcp_sq,
        'iat_rate_product'        : iat_rate,
    }
    return {f: fd.get(f, 0.0) for f in feats}

# ═════════════════════════════════════════════════════════════════════
# 3. DPI ENGINE
# ═════════════════════════════════════════════════════════════════════
def dpi_score(row):
    score  = 0.0
    syn    = float(row.get('syn_count',       0))
    rate   = float(row.get('Rate',            0))
    udp_r  = float(row.get('UDP',             0))
    icmp_r = float(row.get('ICMP',            0))
    iat    = float(row.get('IAT_mean',        0))
    n      = float(row.get('Number',          1))
    pkt    = float(row.get('AVG',             0))
    syn_r  = float(row.get('high_syn_ratio',  0))
    fi     = float(row.get('flood_intensity_score', 0))

    if syn > 50   and rate > 2000:               score = max(score, 0.95)
    if udp_r > 0.8 and rate > 2000:              score = max(score, 0.92)
    if icmp_r > 0.5 and rate > 1000:             score = max(score, 0.90)
    if syn_r  > 0.85 and rate > 200:             score = max(score, 0.88)
    if iat > 0.05 and n > 5 and 0 < rate < 100: score = max(score, 0.82)
    if fi > 1000:                                score = max(score, 0.85)
    if rate > 5000:                              score = max(score, 0.80)
    return min(score, 1.0)

# ═════════════════════════════════════════════════════════════════════
# 4. RAW-COUNT CLASSIFIER  (uses guaranteed-correct per-packet counts)
# ═════════════════════════════════════════════════════════════════════
def raw_classify(buf, window_elapsed):
    """
    Exact attack-type classifier from per-packet counters.
    Volumetric floods are decided here so the dashboard shows the
    correct type (SYN / UDP / ICMP / …), not a mismatched XGB label.
    Returns: (attack_class or None, evidence_string)
    """
    n      = max(buf['n'], 1)
    rate   = n / max(window_elapsed, EPS)
    tcp_r  = buf['tcp']  / n
    udp_r  = buf['udp']  / n
    icmp_r = buf['icmp'] / n
    syn_r  = buf['syn']  / n
    ack_r  = buf['ack']  / n
    rst_r  = buf['rst']  / n
    fin_r  = buf['fin']  / n
    psh_r  = buf['psh']  / n

    # Order matters: most specific volumetric signatures first

    # HTTP flood BEFORE pure SYN: completed handshakes + request data (PSH/ACK)
    if (
        tcp_r > 0.70
        and rate > 60
        and n > 40
        and ack_r > 0.15
        and (psh_r > 0.08 or ack_r > 0.30)
        and syn_r < 0.75
    ):
        lbl = "DDOS-HTTP_FLOOD" if rate > 400 else "DOS-HTTP_FLOOD"
        return lbl, f"tcp={tcp_r:.2f} syn={syn_r:.2f} ack={ack_r:.2f} psh={psh_r:.2f} rate={rate:.0f}"

    if tcp_r > 0.55 and syn_r > 0.50 and rate > 100:
        lbl = 'DDOS-SYN_FLOOD' if rate > 5000 else 'DOS-SYN_FLOOD'
        return lbl, f"tcp={tcp_r:.2f} syn={syn_r:.2f} rate={rate:.0f}"

    if udp_r > 0.55 and rate > 100:
        lbl = 'DDOS-UDP_FLOOD' if rate > 5000 else 'DOS-UDP_FLOOD'
        return lbl, f"udp={udp_r:.2f} rate={rate:.0f}"

    if icmp_r > 0.40 and rate > 50:
        return 'DDOS-ICMP_FLOOD', f"icmp={icmp_r:.2f} rate={rate:.0f}"

    if (rst_r + fin_r) > 0.55 and rate > 100:
        return 'DDOS-RSTFINFLOOD', f"rst={rst_r:.2f} fin={fin_r:.2f}"

    if tcp_r > 0.55 and psh_r > 0.40 and ack_r > 0.40 and rate > 200:
        return 'DDOS-PSHACK_FLOOD', f"psh={psh_r:.2f} ack={ack_r:.2f} rate={rate:.0f}"

    # Slowloris-like: many TCP SYNs/partial conns at low rate
    if tcp_r > 0.7 and syn_r > 0.3 and 5 < rate < 80 and n > 30:
        return 'DOS-SLOWLORIS', f"syn={syn_r:.2f} rate={rate:.1f} n={n}"

    # Recon: very low rate + incomplete handshakes only (avoid normal TCP chatter)
    if tcp_r > 0.8 and syn_r > 0.6 and ack_r < 0.2 and 1 < rate < 40 and n > 20:
        return 'RECON-PORTSCAN', f"syn={syn_r:.2f} ack={ack_r:.2f} rate={rate:.1f}"

    return None, ""

# ═════════════════════════════════════════════════════════════════════
# 5. 3-LAYER FUSION ENGINE
# ═════════════════════════════════════════════════════════════════════
def detect(flow_df, le, scaler, iso_d, xgb, feats, cfg, benign_id):
    iso    = iso_d['model']
    if_min = iso_d['if_min']
    if_max = iso_d['if_max']
    thresh = cfg['threshold']
    if_veto= cfg['if_veto']

    X        = flow_df[feats]
    X_scaled = scaler.transform(X)

    xgb_proba    = xgb.predict_proba(X)
    xgb_att_prob = (1.0 - xgb_proba[:, benign_id]).astype(np.float32)
    xgb_idx      = xgb_proba.argmax(axis=1)[0]
    xgb_class    = le.classes_[xgb_idx]

    ifs_raw  = iso.score_samples(X_scaled)
    ifs_c    = np.clip(ifs_raw, if_min, if_max)
    if_prob  = (1.0 - (ifs_c - if_min) / (if_max - if_min + EPS)).astype(np.float32)

    row_dict = flow_df.iloc[0].to_dict()
    dpi_s    = np.array([dpi_score(row_dict)], dtype=np.float32)

    conf = float(xgb_att_prob[0])
    if conf >= 0.80:
        w_xgb, w_if, w_dpi = 0.60, 0.25, 0.15; mode = "HIGH-CONF"
    elif conf >= 0.40:
        w_xgb, w_if, w_dpi = 0.40, 0.45, 0.15; mode = "MED-CONF"
    else:
        w_xgb, w_if, w_dpi = 0.25, 0.60, 0.15; mode = "LOW-CONF"

    fusion = float(np.clip(w_xgb*xgb_att_prob[0] + w_if*if_prob[0] + w_dpi*dpi_s[0], 0, 1))

    layer = 3
    if if_prob[0] > if_veto:
        fusion = max(fusion, 0.90); layer = 1
    elif if_prob[0] > cfg.get('if_disagree', 0.60) and xgb_att_prob[0] < 0.30:
        fusion = min(fusion + cfg.get('disagree_boost', 0.15), 1.0); layer = 2

    return {
        'is_attack'      : fusion >= thresh,
        'fusion_score'   : round(fusion, 4),
        'xgb_class'      : xgb_class,
        'xgb_raw_class'  : xgb_class,
        'was_overridden' : False,
        'override_reason': '',
        'xgb_prob'       : round(float(xgb_att_prob[0]), 4),
        'xgb_conf_pct'   : round(float(xgb_proba[0, xgb_idx]) * 100, 1),
        'if_prob'        : round(float(if_prob[0]), 4),
        'dpi_score'      : round(float(dpi_s[0]), 4),
        'layer_used'     : layer,
        'conf_mode'      : mode,
    }

# ═════════════════════════════════════════════════════════════════════
# 6. PRINT RESULT
# ═════════════════════════════════════════════════════════════════════
def _parse_src_ip(true_label):
    if not true_label:
        return ""
    if isinstance(true_label, str) and true_label.startswith("src="):
        return true_label.split("=", 1)[1]
    return ""


def _run_mitigation(src_ip: str, attack_label: str, fusion_score: float) -> dict:
    """Apply type-specific mitigation and return details for the dashboard."""
    if not MITIGATION_ENABLED:
        return {
            "mitigated": False,
            "status": "unavailable",
            "family": "",
            "action": "",
            "tier": 0,
            "tier_name": "",
            "expires_in": 0,
            "commands": [],
            "summary": "mitigation_engine not loaded",
        }
    try:
        t0 = time.perf_counter()
        info = _mitigation.apply_mitigation(src_ip, attack_label, float(fusion_score)) or {}
        info = dict(info)
        info["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        return info
    except Exception as exc:
        return {
            "mitigated": False,
            "status": "error",
            "family": "",
            "action": "",
            "tier": 0,
            "tier_name": "",
            "expires_in": 0,
            "commands": [],
            "summary": f"mitigation error: {exc}",
        }


def emit_dashboard(idx, result, true_label=None, src_ip="", packets=0, rate=0.0,
                   proto="TCP", latency_ms=None, mitigation=None, notify_phase=""):
    """Push a detection or mitigation event to the dashboard."""
    if _telemetry is None:
        return
    ip = src_ip or _parse_src_ip(true_label) or "unknown"
    mit = mitigation or {}
    try:
        _telemetry.emit_from_result(
            result,
            flow_id=idx,
            src_ip=ip,
            dst_ip=os.environ.get("DETECTOR_IP", "172.20.0.10"),
            proto=proto,
            packets=int(packets),
            rate=float(rate),
            latency_ms=latency_ms,
            true_label=true_label or "",
            mitigation_action=mit.get("action") or "",
            mitigation=mit,
            notify_phase=notify_phase,
        )
    except Exception as exc:
        print(f"[telemetry] emit failed: {exc}", flush=True)


def print_result(idx, result, true_label=None, src_ip="", packets=0, rate=0.0,
                 proto="TCP", latency_ms=None, mitigation=None, notify_phase=""):
    ts  = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    fs  = result['fusion_score']
    cls = result['xgb_class']
    pct = result['xgb_conf_pct']
    mit = mitigation or {}
    phase = (notify_phase or "").lower()

    if phase == "mitigated":
        status = f"{YELLOW}ATTACK MITIGATED{RESET}"
    elif result['is_attack']:
        status = f"{RED}ATTACK DETECTED{RESET}"
    else:
        status = f"{GREEN}BENIGN{RESET}"
    cls_str = f"{MAGENTA}{cls} ({pct}%){RESET}" if result['is_attack'] else f"{GREEN}{cls}{RESET}"

    layer_map = {1:"L1-IF-VETO", 2:"L2-DISAGREE", 3:"L3-FUSION"}
    layer = layer_map.get(result['layer_used'], "?")

    override_str = ""
    if result.get('was_overridden') and phase != "mitigated":
        raw = result.get('xgb_raw_class', '?')
        rsn = result.get('override_reason', '')
        override_str = f"\n           {YELLOW}[TYPE-LOCK: XGB={raw} -> {cls} | {rsn}]{RESET}"

    true_str = f" | src={true_label}" if true_label else ""
    print(f"[{ts}] Flow#{idx:>5}  {status}{true_str}")
    print(f"           Type={cls_str}  Score={fs:.3f}  XGB={result['xgb_prob']:.3f}"
          f"  IF={result['if_prob']:.3f}  DPI={result['dpi_score']:.3f}"
          f"  [{layer}]{override_str}")
    if phase == "mitigated" and mit:
        print(f"           {YELLOW}MITIGATION: {mit.get('summary') or mit.get('action') or mit.get('status')}{RESET}")

    emit_dashboard(
        idx, result, true_label=true_label, src_ip=src_ip,
        packets=packets, rate=rate, proto=proto, latency_ms=latency_ms,
        mitigation=mit, notify_phase=phase or ("detected" if result.get("is_attack") else "benign"),
    )

# ═════════════════════════════════════════════════════════════════════
# 7. SIMULATION MODE  (replays test.parquet with all 63 features)
# ═════════════════════════════════════════════════════════════════════
def run_simulation(le, scaler, iso_d, xgb, feats, cfg, benign_id, delay=0.1, n_samples=200):
    print(f"\n{CYAN}{'='*65}{RESET}")
    print(f"{CYAN} MODE 1: SIMULATION — Replaying {n_samples} pre-extracted test flows{RESET}")
    print(f"{CYAN}{'='*65}{RESET}\n")

    df = pd.read_parquet(DATA_FILE)
    benign_rows = df[df['Label'] == 'BENIGN'].sample(min(30, (df['Label']=='BENIGN').sum()))
    attack_rows = df[df['Label'] != 'BENIGN'].sample(min(n_samples-30, (df['Label']!='BENIGN').sum()))
    sample_df   = pd.concat([benign_rows, attack_rows]).sample(frac=1).reset_index(drop=True)
    print(f"[*] Running on {len(sample_df)} flows...\n")
    time.sleep(1.0)

    tp = fp = tn = fn = 0
    for i, (_, row) in enumerate(sample_df.iterrows()):
        true_label = row['Label']
        flow_df    = pd.DataFrame([row.drop('Label')])
        t0 = time.perf_counter()
        result     = detect(flow_df, le, scaler, iso_d, xgb, feats, cfg, benign_id)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        # Synthetic but stable source IPs for simulation history
        src = f"10.0.{(i % 5) + 1}.{(i % 40) + 20}"
        rate = float(row.get('Rate', 0) or 0)
        pkts = int(float(row.get('Number', 0) or 0))
        print_result(
            i+1, result, true_label,
            src_ip=src, packets=pkts, rate=rate,
            latency_ms={"feature": 0.0, "inference": round(infer_ms, 3), "mitigation": 0.0},
        )
        a = (true_label != 'BENIGN')
        if result['is_attack'] and a:      tp += 1
        elif result['is_attack']:          fp += 1
        elif a:                            fn += 1
        else:                              tn += 1
        time.sleep(delay)

    tpr  = tp / (tp+fn+EPS)
    fpr  = fp / (fp+tn+EPS)
    prec = tp / (tp+fp+EPS)
    print(f"\n{CYAN}{'='*65}{RESET}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Detection Rate : {tpr*100:.1f}%")
    print(f"  False Pos Rate : {fpr*100:.1f}%")
    print(f"  Precision      : {prec*100:.1f}%")
    print(f"{CYAN}{'='*65}{RESET}\n")

# ═════════════════════════════════════════════════════════════════════
# 8. LIVE CAPTURE MODE  — Full 63-Feature + Raw-Count Override
# ═════════════════════════════════════════════════════════════════════
def _build_ignore_ips():
    """Ignore Docker/self/dashboard traffic so telemetry HTTP is not classified as attacks."""
    ignore = {
        "127.0.0.1",
        "172.20.0.1",   # bridge gateway
        "172.20.0.10",  # detector
        "172.20.0.30",  # dashboard
    }
    extra = os.environ.get("IGNORE_IPS", "")
    for ip in extra.split(","):
        ip = ip.strip()
        if ip:
            ignore.add(ip)
    try:
        import socket
        ignore.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except Exception:
        pass
    try:
        import subprocess as _sp
        out = _sp.run(["hostname", "-I"], capture_output=True, text=True).stdout
        for ip in out.split():
            ignore.add(ip.strip())
    except Exception:
        pass
    return ignore


def run_live(le, scaler, iso_d, xgb, feats, cfg, benign_id,
             interface="eth0", window_sec=5.0, idle_timeout=2.0):

    print(f"\n{CYAN}{'='*65}{RESET}")
    print(f"{CYAN} MODE 2: LIVE CAPTURE  —  Full 63-Feature Extractor{RESET}")
    print(f"{CYAN} Interface: [{interface}]   Window: {window_sec}s{RESET}")
    print(f"{CYAN} Waiting for traffic from attacker container...{RESET}")
    print(f"{CYAN}{'='*65}{RESET}\n")

    try:
        from scapy.all import sniff, IP, TCP, UDP as SUDP, ICMP, ARP
    except ImportError:
        print(f"{RED}[!] Scapy not installed.{RESET}"); sys.exit(1)

    ignore_ips = _build_ignore_ips()
    print(f"[*] Ignoring internal IPs: {', '.join(sorted(ignore_ips))}")
    print(f"[*] Mitigation engine: {'ACTIVE' if MITIGATION_ENABLED else 'NOT LOADED'}\n")

    # ── Flow buffer ────────────────────────────────────────────────────
    def new_buf():
        return {
            'packets': [], 'timestamps': [], 'start': None,
            'n': 0,
            'tcp': 0, 'udp': 0, 'icmp': 0, 'arp': 0,
            'syn': 0, 'ack': 0, 'fin': 0, 'rst': 0, 'psh': 0,
        }

    flow_buf   = defaultdict(new_buf)
    flow_count = [0]
    _blocked_notify_at = {}  # src_ip -> last dashboard notify while already blocked

    def flush(src, buf):
        if buf['n'] < 2:
            return
        if src in ignore_ips:
            return
        elapsed = time.time() - (buf['start'] or time.time())
        elapsed = max(elapsed, EPS)

        raw_class, raw_ev = raw_classify(buf, elapsed)

        print(f"{CYAN}  [DBG {src}] "
              f"n={buf['n']} tcp={buf['tcp']} udp={buf['udp']} "
              f"icmp={buf['icmp']} syn={buf['syn']} ack={buf['ack']} "
              f"rate={buf['n']/elapsed:.0f}/s  type={raw_class}{RESET}")

        t_feat = time.perf_counter()
        feat_d  = extract_features(buf['packets'], buf['timestamps'], buf, feats)
        feat_ms = (time.perf_counter() - t_feat) * 1000.0
        flow_df = pd.DataFrame([feat_d])
        t_inf = time.perf_counter()
        result  = detect(flow_df, le, scaler, iso_d, xgb, feats, cfg, benign_id)
        infer_ms = (time.perf_counter() - t_inf) * 1000.0

        # Exact type lock: packet counters win for volumetric floods
        if raw_class:
            if result['xgb_class'] != raw_class:
                result['xgb_raw_class']  = result['xgb_class']
                result['was_overridden'] = True
                result['override_reason'] = raw_ev
            result['xgb_class'] = raw_class
            # Force attack when a clear flood signature is present
            if any(k in raw_class for k in ("FLOOD", "SLOWLORIS", "RSTFIN")):
                result['is_attack'] = True
                result['fusion_score'] = max(float(result['fusion_score']), 0.85)
        else:
            # ── Isolation Forest novel / zero-day gate ──────────────────
            # No DPI flood signature means this is NOT a known volumetric
            # class we can type-lock. If IF is elevated, trust IF and expose
            # UNKNOWN-ANOMALY (even when XGB hallucinates a trained label).
            ifp = float(result.get("if_prob") or 0.0)
            xgbp = float(result.get("xgb_prob") or 0.0)
            if_veto = float(cfg.get("if_veto", 0.75))
            if_unknown = float(cfg.get("if_unknown", 0.48))
            if ifp >= if_unknown:
                raw_xgb = str(result.get("xgb_class") or "")
                result["xgb_raw_class"] = raw_xgb
                result["xgb_class"] = "UNKNOWN-ANOMALY"
                result["was_overridden"] = True
                result["override_reason"] = (
                    f"IF-led novel anomaly if={ifp:.3f} "
                    f"xgb_was={raw_xgb} xgb_att={xgbp:.3f}"
                )
                result["is_attack"] = True
                if ifp >= if_veto:
                    result["layer_used"] = 1
                    result["fusion_score"] = max(float(result["fusion_score"]), 0.90)
                else:
                    result["layer_used"] = 2
                    result["fusion_score"] = max(
                        float(result["fusion_score"]), min(0.68 + 0.3 * ifp, 0.92)
                    )
                result["conf_mode"] = "IF-UNKNOWN"

        print(
            f"{CYAN}  [SCORES {src}] "
            f"XGB={result.get('xgb_prob', 0):.3f}/{result.get('xgb_class')} "
            f"IF={result.get('if_prob', 0):.3f} "
            f"DPI={result.get('dpi_score', 0):.3f} "
            f"fusion={result.get('fusion_score', 0):.3f} "
            f"attack={result.get('is_attack')} "
            f"layer={result.get('layer_used')} "
            f"dpi_type={raw_class}{RESET}"
        )

        flow_count[0] += 1
        proto = "TCP" if buf['tcp'] >= buf['udp'] and buf['tcp'] >= buf['icmp'] else (
            "UDP" if buf['udp'] >= buf['icmp'] else "ICMP"
        )
        lat_detect = {
            "feature": round(feat_ms, 3),
            "inference": round(infer_ms, 3),
            "mitigation": 0.0,
        }

        # Skip dashboard spam for clear benign (still printed above via SCORES)
        if not result["is_attack"]:
            return

        already_blocked = False
        try:
            already_blocked = bool(
                MITIGATION_ENABLED and _mitigation.is_ip_mitigated(src)
            )
        except Exception:
            already_blocked = False

        # Already mitigated: keep scoring quiet most of the time, but
        # rate-limit a dashboard ping so the UI still shows a toast while
        # the attacker keeps firing (no second history row — upsert only).
        if already_blocked:
            now_t = time.time()
            last = float(_blocked_notify_at.get(src, 0.0) or 0.0)
            if now_t - last < 12.0:
                return
            _blocked_notify_at[src] = now_t
            mitigation = _run_mitigation(src, result["xgb_class"], result["fusion_score"])
            mit_ms = float(mitigation.get("latency_ms") or 0.0)
            lat = {
                "feature": round(feat_ms, 3),
                "inference": round(infer_ms, 3),
                "mitigation": mit_ms,
            }
            print_result(
                flow_count[0],
                result,
                true_label=src,
                src_ip=src,
                packets=buf["n"],
                rate=buf["n"] / elapsed,
                proto=proto,
                latency_ms=lat,
                mitigation=mitigation,
                notify_phase="mitigated",
            )
            return

        # Single dashboard notification after mitigation (detect+mitigate
        # used to emit two events → two toasts / two history rows).
        mitigation = _run_mitigation(src, result["xgb_class"], result["fusion_score"])
        mit_ms = float(mitigation.get("latency_ms") or 0.0)
        status = str(mitigation.get("status") or "")
        lat = {
            "feature": round(feat_ms, 3),
            "inference": round(infer_ms, 3),
            "mitigation": mit_ms,
        }
        if status in ("applied", "partial", "monitor", "already_active"):
            _blocked_notify_at[src] = time.time()
            print_result(
                flow_count[0],
                result,
                true_label=src,
                src_ip=src,
                packets=buf["n"],
                rate=buf["n"] / elapsed,
                proto=proto,
                latency_ms=lat,
                mitigation=mitigation,
                notify_phase="mitigated",
            )
        else:
            # Mitigation unavailable — still report the detection once
            print_result(
                flow_count[0],
                result,
                true_label=src,
                src_ip=src,
                packets=buf["n"],
                rate=buf["n"] / elapsed,
                proto=proto,
                latency_ms=lat_detect,
                mitigation=mitigation or {},
                notify_phase="detected",
            )

    def on_packet(pkt):
        if IP not in pkt:
            return
        src = pkt[IP].src
        # Skip traffic ORIGINATING from detector/dashboard/gateway (not attack targets).
        # Attacker → detector (dst=172.20.0.10) must still be analyzed.
        if src in ignore_ips:
            return

        now = time.time()
        buf = flow_buf[src]
        if buf['start'] is None:
            buf['start'] = now
        # Cap buffer so a long flood cannot exhaust memory
        if len(buf['packets']) < 4000:
            buf['packets'].append(pkt)
            buf['timestamps'].append(now)
        buf['n'] += 1

        if TCP in pkt:
            buf['tcp'] += 1
            try:
                f = int(pkt[TCP].flags)
            except Exception:
                f = 0
            if f & 0x01: buf['fin'] += 1
            if f & 0x02: buf['syn'] += 1
            if f & 0x04: buf['rst'] += 1
            if f & 0x08: buf['psh'] += 1
            if f & 0x10: buf['ack'] += 1
        elif SUDP in pkt:
            buf['udp'] += 1
        elif ICMP in pkt:
            buf['icmp'] += 1
        elif ARP in pkt:
            buf['arp'] += 1

        if now - buf['start'] >= window_sec:
            flush(src, buf)
            flow_buf[src] = new_buf()

    import threading
    def idle_flusher():
        while True:
            time.sleep(idle_timeout)
            now = time.time()
            for src in list(flow_buf.keys()):
                buf = flow_buf[src]
                if buf['start'] and (now - buf['start']) >= idle_timeout and buf['n'] > 1:
                    flush(src, buf)
                    flow_buf[src] = new_buf()

    threading.Thread(target=idle_flusher, daemon=True).start()
    print(f"[*] Sniffing on {interface}... (Ctrl+C to stop)\n")
    sniff(iface=interface, prn=on_packet, store=False)

def _start_lab_http_server(port: int = 80):
    """Tiny victim web server so HTTP floods can complete TCP + send GETs."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _ok(self):
            try:
                body = b"ok"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                # Client hung up mid-flood — ignore (do not spam detector logs)
                return

        def do_GET(self):
            self._ok()

        def do_POST(self):
            self._ok()

        def handle_one_request(self):
            try:
                super().handle_one_request()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass

    try:
        srv = ThreadingHTTPServer(("0.0.0.0", port), _H)
        srv.daemon_threads = True
        srv.request_queue_size = 128
    except OSError as exc:
        print(f"[!] Lab HTTP server not started on :{port}: {exc}", flush=True)
        return

    threading.Thread(target=srv.serve_forever, daemon=True, name="lab-http").start()
    print(f"[*] Lab HTTP victim listening on 0.0.0.0:{port}", flush=True)


# ═════════════════════════════════════════════════════════════════════
# 9. MAIN
# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6 DDoS Detector")
    parser.add_argument('--live',    action='store_true',    help='Live capture mode')
    parser.add_argument('--iface',   default='eth0',         help='Network interface')
    parser.add_argument('--samples', type=int,   default=200, help='Simulation flows')
    parser.add_argument('--delay',   type=float, default=0.05,help='Sim delay (sec)')
    parser.add_argument('--window',  type=float, default=5.0, help='Flow window (sec)')
    args = parser.parse_args()

    le, scaler, iso_d, xgb, feats, cfg, benign_id = load_models()

    if _telemetry is not None:
        try:
            _telemetry.start_heartbeat(20.0)
        except Exception as exc:
            print(f"[telemetry] heartbeat not started: {exc}", flush=True)

    if args.live:
        _start_lab_http_server(80)
        run_live(le, scaler, iso_d, xgb, feats, cfg, benign_id,
                 interface=args.iface, window_sec=args.window)
    else:
        run_simulation(le, scaler, iso_d, xgb, feats, cfg, benign_id,
                       delay=args.delay, n_samples=args.samples)
