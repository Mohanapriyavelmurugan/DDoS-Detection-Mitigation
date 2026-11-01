# DDoS Detection & Mitigation

**Edge-intelligent, real-time DDoS detection and automated mitigation** for Docker lab and 5G/MEC-style edge deployment.

The system fuses **XGBoost** classification, **Isolation Forest** anomaly detection, and **DPI** heuristics into a confidence-adaptive three-layer engine, then applies **type-specific mitigation** (iptables / rate limits / blocks).

> **Lab use only.** Run attacks only against systems you own or have explicit permission to test.

---

## Features

- **34-class** traffic classification (DDoS / DoS / recon / app-layer + benign)
- **3-layer fusion:** IF veto → disagree amplifier → adaptive XGB + IF + DPI weights
- **Zero-day path:** unknown / out-of-distribution probes via Isolation Forest
- **Automated mitigation:** SYN cookies, hashlimit, connlimit, hard DROP, and more
- **Live dashboard:** WebSocket feed, toast alerts, history, latency / throughput views
- **Docker lab:** detector + attacker + dashboard on one compose network

---

## Architecture (short)

```
Attacker ──packets──► Detector (sniff + fuse + mitigate)
                           │
                           └──HTTP/WS──► Dashboard (:5000)
```

| Layer | Role |
|-------|------|
| **L1 – IF veto** | Extreme anomalies override confident XGB mistakes |
| **L2 – Disagree** | Boosts score when XGB and IF strongly disagree |
| **L3 – Fusion** | Confidence-adaptive mix of XGB, IF, and DPI |

Decision threshold (lab default): **0.61** (see `models_v6/optimal_threshold.json`).

---

## Repository layout

```
.
├── docker_test/                 # Main runnable lab
│   ├── docker-compose.yml       # LOCAL stack
│   ├── docker-compose.mec.yml   # MEC overlay
│   ├── detector/                # Live sniff + fusion + mitigation
│   ├── attacker/                # Known + unknown attack scripts
│   ├── dashboard/               # FastAPI + static UI
│   └── TEAM_RUN_GUIDE.md        # Teammate command cheat-sheet
├── models_v6/                   # Place trained .pkl models here
├── paper_results/               # Evaluation tables / figures
├── mitigation_engine.py         # Standalone mitigation module (also in detector)
├── realtime_defender_v2.py      # Earlier real-time defender script
└── phase6_*.py                  # Training / evaluation scripts
```

---

## Requirements

- **Docker Desktop** (Windows / macOS / Linux)
- Trained **Phase-6 models** under `models_v6/` (see [Models](#models))
- Browser for the dashboard

Optional (offline training / analysis): Python 3.10+, pandas, scikit-learn, xgboost, scapy.

---

## Quick start (Docker lab)

### 1. Clone

```bash
git clone https://github.com/Mohanapriyavelmurugan/DDoS-Detection-Mitigation.git
cd DDoS-Detection-Mitigation
```

### 2. Add models

Copy the Phase-6 `.pkl` files into `models_v6/` (they are **not** stored in git — files are too large for GitHub).  
Keep `models_v6/optimal_threshold.json` (already in the repo).

### 3. Start the stack

```bash
cd docker_test
docker compose up --build
```

### 4. Open the dashboard

- URL: **http://localhost:5000**
- Hard refresh once (**Ctrl+F5**)
- Open **Live Monitor** and confirm **Detector Online**

### 5. Launch attacks (second terminal)

```bash
cd docker_test
docker exec -it ddos_attacker bash
```

Example (SYN flood):

```bash
python3 /attack_scripts/run_attack.py --target 172.20.0.10 --type syn --duration 20
```

Full attack list (known + unknown): see [`docker_test/TEAM_RUN_GUIDE.md`](docker_test/TEAM_RUN_GUIDE.md).

### 6. Stop

```bash
# Ctrl+C in the compose terminal, then:
docker compose down
```

---

## Known attacks (summary)

Run inside `ddos_attacker`:

| Attack | Command |
|--------|---------|
| SYN Flood | `python3 /attack_scripts/run_attack.py --target 172.20.0.10 --type syn --duration 20` |
| UDP Flood | `python3 /attack_scripts/run_attack.py --target 172.20.0.10 --type udp --duration 20` |
| ICMP Flood | `python3 /attack_scripts/run_attack.py --target 172.20.0.10 --type icmp --duration 20` |
| HTTP Flood | `python3 /attack_scripts/run_attack.py --target 172.20.0.10 --type http --duration 25` |
| Slowloris | `python3 /attack_scripts/run_attack.py --target 172.20.0.10 --type slowloris --duration 25` |
| ACK / PSH+ACK / RST+FIN / XMAS | `--type ack` · `pshack` · `rstfin` · `xmas` |
| Port scan | `--type scan` |
| Fragmentation | `--type frag` |
| All (sequence) | `--type all` |

Target IP is always the detector: **`172.20.0.10`**.

---

## Unknown / anomaly probes

```bash
python3 /attack_scripts/unknown_if_probe.py --target 172.20.0.10 --mode zeroday --duration 30
python3 /attack_scripts/unknown_if_probe.py --target 172.20.0.10 --mode nullpulse --duration 25
python3 /attack_scripts/unknown_if_probe.py --target 172.20.0.10 --mode sizestorm --duration 25
python3 /attack_scripts/unknown_if_probe.py --target 172.20.0.10 --mode protomix --duration 25
```

Expect IF-led labels such as **UNKNOWN-ANOMALY** on the dashboard.

---

## Models

| File | Role |
|------|------|
| `xgboost_model.pkl` | Multi-class attack classifier |
| `isolation_forest.pkl` | Anomaly / zero-day layer |
| `scaler.pkl` | Feature scaling |
| `label_encoder.pkl` | Class labels |
| `feature_names.pkl` | Feature column order |
| `optimal_threshold.json` | Fusion threshold + IF settings |

**Why models are not in git:** `xgboost_model.pkl` is hundreds of MB (GitHub file limit is 100 MB). Share models via a private drive, USB, or a GitHub **Release** asset, then place them in `models_v6/` before `docker compose up`.

Compose mounts `../models_v6` into the detector container as `/app/models_v6`.

---

## Example metrics (Phase 6 evaluation)

Reported on a large held-out flow set (see project docs / `paper_results/`):

| Metric | Value |
|--------|-------|
| Detection rate (TPR) | ~80% overall (high on major volumetric floods) |
| False positive rate | ~1.6% (target &lt; 2%) |
| Precision | ~99.9% |

Lab mitigation windows (LOCAL): block ~90s, throttle ~60s (configurable via env).

---

## MEC / edge overlay

```bash
cd docker_test
docker compose -f docker-compose.yml -f docker-compose.mec.yml up --build
```

Use the MEC host IP for the dashboard as documented in your deployment notes.

---

## Safety & ethics

- Use **only** in isolated lab networks or with written authorization.
- Do not point attack scripts at public or third-party systems.
- Mitigation changes firewall rules inside the lab containers.

---

## Team docs

- **[Team run guide](docker_test/TEAM_RUN_GUIDE.md)** — terminal-by-terminal commands for demos
- `project_document.md` — academic / scope write-up
- `Mitigation.md` — mitigation action notes by attack family

---

## License

Use and redistribute according to your course / organization rules. Add an explicit `LICENSE` file if you open-source publicly.

---

## Authors

Built as an academic / lab project for real-time DDoS detection and mitigation on edge-style infrastructure.

Repository: [github.com/Mohanapriyavelmurugan/DDoS-Detection-Mitigation](https://github.com/Mohanapriyavelmurugan/DDoS-Detection-Mitigation)
