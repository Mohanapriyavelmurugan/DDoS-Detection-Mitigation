"""
=============================================================================
MININET TOPOLOGY & EXPERIMENT CONTROLLER
Edge-Intelligent Hybrid DDoS Detection System for 5G Networks
=============================================================================

Run this INSIDE the Mininet Ubuntu VM:
  sudo python3 mininet_setup.py

What it does:
  1. Creates a virtual 5G MEC network (h1, h2, h3, s1)
  2. Starts the real-time defender on h2 (MEC Node)
  3. Launches 4 DDoS experiments automatically
  4. Shows live detection results on screen

Requirements (install inside the VM):
  sudo apt-get install -y hping3 slowhttptest
  pip3 install scapy joblib xgboost scikit-learn numpy pandas
=============================================================================
"""

from mininet.net import Mininet
from mininet.node import Controller, OVSKernelSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI
import time
import os

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
MODEL_PATH   = "/home/mininet/models_v5"   # Copy models_v5 folder here
DEFENDER     = "/home/mininet/realtime_defender_v2.py"
RESULTS_DIR  = "/home/mininet/results"

EXPERIMENT_DURATION = 60   # seconds per experiment
BENIGN_INTERVAL     = 2    # benign h3 sends request every 2 seconds


def run():
    setLogLevel('info')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    info("\n*** Creating Mininet 5G MEC Topology ***\n")
    net = Mininet(controller=Controller, switch=OVSKernelSwitch, link=TCLink)

    # Add controller
    c0 = net.addController('c0')

    # Add Hosts
    h1 = net.addHost('h1', ip='10.0.0.1/24')   # Attacker (malicious IoT)
    h2 = net.addHost('h2', ip='10.0.0.2/24')   # MEC Node (your defender)
    h3 = net.addHost('h3', ip='10.0.0.3/24')   # Benign IoT device

    # Add Switch (represents 5G base station / MEC switch)
    s1 = net.addSwitch('s1')

    # Connect hosts to switch
    net.addLink(h1, s1)
    net.addLink(h2, s1)
    net.addLink(h3, s1)

    # Start network
    net.start()
    info("*** Network started. Topology: h1 --+-- s1 --+-- h2\n")
    info("***                                  +-- h3\n")
    info(f"*** h1 (Attacker) : {h1.IP()}\n")
    info(f"*** h2 (MEC Node) : {h2.IP()}\n")
    info(f"*** h3 (Benign)   : {h3.IP()}\n\n")

    # Test connectivity
    info("*** Testing Connectivity (ping)...\n")
    net.ping([h1, h2, h3])

    # ------------------------------------------------------------------
    # STEP 1: Start Real-Time Defender on h2
    # ------------------------------------------------------------------
    info("\n*** [h2] Starting Real-Time Defender (AI Detection Engine)...\n")
    h2.cmd(f"python3 {DEFENDER} --interface h2-eth0 --model-dir {MODEL_PATH} > {RESULTS_DIR}/defender_log.txt 2>&1 &")
    time.sleep(3)  # Give the model time to load

    # ------------------------------------------------------------------
    # RUN EXPERIMENTS
    # ------------------------------------------------------------------
    experiments = [
        {
            "name": "1 — SYN Flood",
            "cmd":  f"hping3 -S --flood -p 80 {h2.IP()}",
            "logfile": "exp1_syn_flood.txt"
        },
        {
            "name": "2 — UDP Flood",
            "cmd":  f"hping3 --udp --flood -p 53 {h2.IP()}",
            "logfile": "exp2_udp_flood.txt"
        },
        {
            "name": "3 — ICMP Flood",
            "cmd":  f"hping3 --icmp --flood {h2.IP()}",
            "logfile": "exp3_icmp_flood.txt"
        },
        {
            "name": "4 — SlowLoris",
            "cmd":  f"slowhttptest -c 1000 -H -i 10 -r 200 -t GET -u http://{h2.IP()}:5000",
            "logfile": "exp4_slowloris.txt"
        },
    ]

    for exp in experiments:
        info(f"\n{'='*60}\n")
        info(f"*** EXPERIMENT {exp['name']}\n")
        info(f"*** h1 is attacking h2 for {EXPERIMENT_DURATION} seconds...\n")
        info(f"*** h3 is sending benign traffic simultaneously...\n")
        info(f"{'='*60}\n")

        # Start benign traffic from h3 in background
        h3.cmd(f"while true; do wget -q -O /dev/null http://{h2.IP()} 2>&1; sleep {BENIGN_INTERVAL}; done &")

        # Start attack from h1
        h1.cmd(exp["cmd"] + f" > {RESULTS_DIR}/{exp['logfile']} 2>&1 &")

        # Wait for experiment duration
        time.sleep(EXPERIMENT_DURATION)

        # Stop attack and benign traffic
        h1.cmd("kill %1 2>/dev/null; killall hping3 2>/dev/null; killall slowhttptest 2>/dev/null")
        h3.cmd("kill %1 2>/dev/null; killall wget 2>/dev/null")

        info(f"\n*** Experiment {exp['name']} complete. Pausing 5s before next...\n")
        time.sleep(5)

    # ------------------------------------------------------------------
    # STOP DEFENDER AND PRINT SUMMARY
    # ------------------------------------------------------------------
    h2.cmd("kill %1 2>/dev/null")
    time.sleep(1)

    info("\n" + "="*60)
    info("\n*** ALL EXPERIMENTS COMPLETE. READING RESULTS...\n")
    info("="*60 + "\n")

    log_path = f"{RESULTS_DIR}/defender_log.txt"
    if os.path.exists(log_path):
        with open(log_path) as f:
            print(f.read())
    else:
        info("WARNING: defender_log.txt not found. Check that defender started correctly.\n")

    info("\n*** Dropping into Mininet CLI for manual inspection. Type 'exit' to quit.\n")
    CLI(net)

    net.stop()


if __name__ == "__main__":
    run()
