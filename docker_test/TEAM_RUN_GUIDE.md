# EdgeGuard Lab — Team Run Guide

Simple step-by-step for running the local Docker lab, watching the dashboard, and generating known + unknown attacks.

---

## What you need

- Docker Desktop running (Windows / Mac / Linux)
- Project folder: `docker_test` inside this repo (e.g. `DDoS-Detection-Mitigation/docker_test`)
- Browser: open **http://localhost:5000**

---

## Terminal layout (use 2 terminals)

| Terminal | Purpose |
|----------|---------|
| **Terminal 1** | Start / watch the whole stack (`docker compose`) |
| **Terminal 2** | Run attacks inside the attacker container |

Optional: keep the browser on **Live Monitor** while attacking.

---

## Step 1 — Start the system (Terminal 1)

```powershell
cd docker_test
docker compose up --build
```

Wait until you see something like:

- `ddos_dashboard` — Uvicorn running on port 5000  
- `ddos_detector` — `Sniffing on eth0...`

Leave this terminal open. Do **not** close it while testing.

### Open the dashboard

1. Go to **http://localhost:5000**
2. Hard refresh once: **Ctrl + R**
3. Click **Live Monitor**
4. Confirm badge shows **Detector Online**

---

## Step 2 — Open the attacker shell (Terminal 2)

```powershell
cd docker_test
docker exec -it ddos_attacker bash
```

You are now inside the attacker container. All attack commands below are run **from this shell**.

Target IP (detector): `172.20.0.10`

---

## Step 3 — Generate known attacks (one by one)

Run **one attack at a time**. After each attack:

1. Watch **Terminal 1** for `ATTACK DETECTED & MITIGATION APPLIED`
2. Watch the dashboard for a **red toast** (top-right) and a new Live Monitor row
3. Wait ~10–15 seconds before the next attack (or wait for the block to expire ~60–90s)

### Known attack command list

Copy-paste from inside the attacker container (`docker exec -it ddos_attacker bash`):

| # | Attack name | Command |
|---|-------------|---------|
| 1 | **SYN Flood** | `python3 /attack_scripts/run_attack.py --target 172.20.0.10 --type syn --duration 20` |
| 2 | **UDP Flood** | `python3 /attack_scripts/run_attack.py --target 172.20.0.10 --type udp --duration 20` |
| 3 | **ICMP Flood (Ping Flood)** | `python3 /attack_scripts/run_attack.py --target 172.20.0.10 --type icmp --duration 20` |
| 4 | **HTTP Flood** | `python3 /attack_scripts/run_attack.py --target 172.20.0.10 --type http --duration 25` |
| 5 | **Slowloris** | `python3 /attack_scripts/run_attack.py --target 172.20.0.10 --type slowloris --duration 25` |
| 6 | **ACK Flood** | `python3 /attack_scripts/run_attack.py --target 172.20.0.10 --type ack --duration 20` |
| 7 | **PSH+ACK Flood** | `python3 /attack_scripts/run_attack.py --target 172.20.0.10 --type pshack --duration 20` |
| 8 | **RST+FIN Flood** | `python3 /attack_scripts/run_attack.py --target 172.20.0.10 --type rstfin --duration 20` |
| 9 | **XMAS Flood** | `python3 /attack_scripts/run_attack.py --target 172.20.0.10 --type xmas --duration 20` |
| 10 | **Port Scan (Recon)** | `python3 /attack_scripts/run_attack.py --target 172.20.0.10 --type scan --duration 20` |
| 11 | **Fragmentation Flood** | `python3 /attack_scripts/run_attack.py --target 172.20.0.10 --type frag --duration 20` |
| 12 | **Normal traffic (benign check)** | `python3 /attack_scripts/run_attack.py --target 172.20.0.10 --type normal --duration 15` |

### Run all built-in known attacks in sequence

```bash
python3 /attack_scripts/run_attack.py --target 172.20.0.10 --type all
```

This runs normal → SYN → UDP → ICMP → ACK → PSH+ACK → scan (with short pauses).

---

## Step 4 — Generate unknown / anomaly attacks

These are **not** standard trained flood labels. Expect something like **UNKNOWN-ANOMALY** (or IF-led layers) on the dashboard.

Still inside the attacker container:

| # | Unknown probe name | Command |
|---|--------------------|---------|
| 1 | **Zero-day mix (strongest IF probe)** | `python3 /attack_scripts/unknown_if_probe.py --target 172.20.0.10 --mode zeroday --duration 30` |
| 2 | **Jitter (same as zeroday)** | `python3 /attack_scripts/unknown_if_probe.py --target 172.20.0.10 --mode jitter --duration 30` |
| 3 | **NULL pulse** | `python3 /attack_scripts/unknown_if_probe.py --target 172.20.0.10 --mode nullpulse --duration 25` |
| 4 | **Size storm** | `python3 /attack_scripts/unknown_if_probe.py --target 172.20.0.10 --mode sizestorm --duration 25` |
| 5 | **Protocol mix** | `python3 /attack_scripts/unknown_if_probe.py --target 172.20.0.10 --mode protomix --duration 25` |

Recommended first unknown test:

```bash
python3 /attack_scripts/unknown_if_probe.py --target 172.20.0.10 --mode zeroday --duration 30
```

---

## Step 5 — What success looks like

For a real attack you should see **all three**:

1. **Detector log (Terminal 1)**  
   `ATTACK DETECTED & MITIGATION APPLIED` / `ATTACK MITIGATED`
2. **Dashboard toast** (top-right, red)  
   `ATTACK DETECTED & MITIGATED` (or `ATTACK ONGOING — ALREADY MITIGATED` if IP is still blocked)
3. **Live Monitor / Attack History**  
   New row with label, score, mitigation action

If the IP is still blocked from a previous run, you may see toast text like **already mitigated**. That is expected. Lab block/throttle expires in about **60–90 seconds**.

---

## Useful extra commands

### Restart only detector / dashboard (from host Terminal 1 / new PowerShell)

```powershell
cd docker_test
docker compose restart detector dashboard
```

### Stop everything

```powershell
cd docker_test
docker compose down
```

### Clean restart (wipes dashboard history volume too)

```powershell
cd docker_test
docker compose down -v
docker compose up --build
```

### Check containers are up

```powershell
docker ps
```

You should see: `ddos_detector`, `ddos_dashboard`, `ddos_attacker`.

---

## Quick checklist for a teammate

1. Docker Desktop is running  
2. Place Phase-6 `.pkl` models in `models_v6/`  
3. Terminal 1: `cd docker_test` → `docker compose up --build`  
4. Browser: http://localhost:5000 → **Ctrl+F5** → **Live Monitor**  
5. Terminal 2: `docker exec -it ddos_attacker bash`  
6. Run one known attack (start with SYN)  
7. Confirm toast + history row  
8. Try one unknown probe (`zeroday`)  
9. When done: `docker compose down` in Terminal 1 (Ctrl+C first, then down if needed)

---

## Notes

- Always attack **`172.20.0.10`** (detector), not the dashboard IP.
- Keep **Live Monitor** open while testing notifications.
- If toast does not appear: hard refresh (**Ctrl+F5**), confirm **Detector Online**, wait for block expiry, run the attack again.
- Path on your machine may differ — always `cd` into the folder that contains `docker-compose.yml`.
