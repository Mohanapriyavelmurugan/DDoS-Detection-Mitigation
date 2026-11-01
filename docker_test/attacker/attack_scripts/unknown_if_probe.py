#!/usr/bin/env python3
"""
=================================================================
UNKNOWN / ZERO-DAY STYLE PROBE  — Isolation Forest test
=================================================================
Traffic is crafted to be OUTSIDE XGBoost's 34 training classes and
BELOW DPI volumetric flood locks, so Isolation Forest must lead.

Usage:
  python3 /attack_scripts/unknown_if_probe.py --mode zeroday --duration 30
  python3 /attack_scripts/unknown_if_probe.py --mode nullpulse --duration 25
  python3 /attack_scripts/unknown_if_probe.py --mode sizestorm --duration 25
  python3 /attack_scripts/unknown_if_probe.py --mode protomix --duration 25
=================================================================
"""

from __future__ import annotations

import argparse
import random
import sys
import time

try:
    from scapy.all import IP, TCP, UDP, Raw, send  # type: ignore
except ImportError:
    print("[!] scapy required: pip3 install scapy")
    sys.exit(1)


def banner(mode: str, seconds: int, target: str):
    print("\n" + "=" * 60)
    print("  UNKNOWN ATTACK PROBE  (not in XGBoost label set)")
    print(f"  Mode   : {mode}")
    print(f"  Target : {target}")
    print(f"  Time   : {seconds}s")
    print("=" * 60)
    print("  Expect on dashboard: UNKNOWN-ANOMALY")
    print("  Layer: L1-IF-VETO or L2-DISAGREE")
    print("=" * 60 + "\n")


def _sleep_jitter(base: float, spread: float):
    time.sleep(max(0.0005, random.uniform(base - spread, base + spread)))


def mode_zeroday(target: str, seconds: int):
    """
    Strong IF-oriented probe:
      - rare IP protocols (99 / 132)
      - TCP NULL + URG-only
      - extreme TTL + size oscillation
      - chaotic inter-arrival (bursts + long gaps)
    Stays under SYN/UDP/ICMP flood DPI rate locks.
    """
    banner("zeroday (strong IF anomaly mix)", seconds, target)
    end = time.time() + seconds
    n = 0
    while time.time() < end:
        burst = random.randint(12, 28)
        for _ in range(burst):
            kind = random.choice(["proto99", "proto132", "null", "urg", "udpbig", "udptiny", "ack"])
            ttl = random.choice([1, 2, 3, 4, 250, 255])
            if kind == "proto99":
                pkt = IP(dst=target, proto=99, ttl=ttl) / Raw(
                    load=bytes([random.randint(0, 255) for _ in range(random.choice([1, 3, 700, 1200]))])
                )
            elif kind == "proto132":
                pkt = IP(dst=target, proto=132, ttl=ttl) / Raw(
                    load=b"\x00\x01" + bytes(random.randint(0, 200))
                )
            elif kind == "null":
                pkt = IP(dst=target, ttl=ttl) / TCP(
                    sport=random.randint(1024, 65535),
                    dport=random.randint(1, 1023),
                    flags="",
                    seq=random.randint(1, 2**20),
                    window=random.choice([0, 1, 65535]),
                )
            elif kind == "urg":
                pkt = IP(dst=target, ttl=ttl) / TCP(
                    sport=random.randint(40000, 60000),
                    dport=random.choice([21, 23, 445, 3389, 5900]),
                    flags="U",
                    urgptr=0xFFFF,
                ) / Raw(load=b"\xff" * random.randint(0, 5))
            elif kind == "udpbig":
                pkt = IP(dst=target, ttl=ttl) / UDP(
                    sport=random.randint(20000, 60000),
                    dport=random.choice([12345, 54321, 9999]),
                ) / Raw(load=bytes(random.randint(1100, 1300)))
            elif kind == "udptiny":
                pkt = IP(dst=target, ttl=ttl) / UDP(
                    sport=random.randint(20000, 60000),
                    dport=random.choice([12345, 54321, 9999]),
                ) / Raw(load=bytes(random.randint(0, 2)))
            else:
                pkt = IP(dst=target, ttl=ttl) / TCP(
                    sport=random.randint(50000, 60000),
                    dport=random.choice([80, 443, 8080]),
                    flags="A",
                    window=0,
                ) / Raw(load=bytes(random.randint(0, 8)))
            send(pkt, verbose=0)
            n += 1
            _sleep_jitter(0.008, 0.006)
        # Long quiet gap → extreme IAT variance without high average rate
        time.sleep(random.uniform(0.45, 1.1))
        if n % 60 == 0:
            print(f"[*] {n} zeroday packets sent…")
    print(f"[+] zeroday done ({n} pkts)")


def mode_nullpulse(target: str, seconds: int):
    banner("nullpulse (NULL-flag TCP pulses)", seconds, target)
    end = time.time() + seconds
    n = 0
    while time.time() < end:
        for _ in range(random.randint(15, 30)):
            pkt = IP(dst=target, ttl=random.randint(1, 8)) / TCP(
                sport=random.randint(1024, 65535),
                dport=random.randint(1, 1024),
                flags="",
                seq=random.randint(1, 2**20),
                urgptr=random.randint(0, 65535),
                window=random.choice([0, 1, 65535]),
            )
            send(pkt, verbose=0)
            n += 1
            time.sleep(0.008)
        time.sleep(random.uniform(0.4, 1.0))
        if n % 50 == 0:
            print(f"[*] {n} NULL packets sent…")
    print(f"[+] nullpulse done ({n} pkts)")


def mode_sizestorm(target: str, seconds: int):
    banner("sizestorm (extreme UDP size variance)", seconds, target)
    end = time.time() + seconds
    n = 0
    while time.time() < end:
        big = n % 2 == 0
        payload = bytes(random.randint(1100, 1400) if big else random.randint(0, 2))
        pkt = IP(dst=target, ttl=random.choice([1, 2, 255])) / UDP(
            sport=random.randint(20000, 60000),
            dport=random.choice([12345, 54321, 9999, 4242]),
        ) / Raw(load=payload)
        send(pkt, verbose=0)
        n += 1
        _sleep_jitter(0.02, 0.015)
        if n % 50 == 0:
            print(f"[*] {n} size-storm packets sent…")
    print(f"[+] sizestorm done ({n} pkts)")


def mode_protomix(target: str, seconds: int):
    banner("protomix (URG/UDP/ACK cocktail)", seconds, target)
    end = time.time() + seconds
    n = 0
    while time.time() < end:
        kind = random.choice(["urg", "udp", "ack", "proto99", "null"])
        ttl = random.choice([1, 2, 64, 255])
        if kind == "urg":
            pkt = IP(dst=target, ttl=ttl) / TCP(
                sport=random.randint(40000, 50000),
                dport=random.choice([21, 23, 445, 3389]),
                flags="U",
                urgptr=0xFFFF,
            ) / Raw(load=b"\x00" * random.randint(0, 3))
        elif kind == "udp":
            pkt = IP(dst=target, ttl=ttl) / UDP(
                sport=random.randint(1024, 2048),
                dport=random.choice([1900, 5353, 11211, 27015]),
            ) / Raw(load=bytes([0xDE, 0xAD, random.randint(0, 255)]))
        elif kind == "proto99":
            pkt = IP(dst=target, proto=99, ttl=ttl) / Raw(load=bytes(random.randint(20, 400)))
        elif kind == "null":
            pkt = IP(dst=target, ttl=ttl) / TCP(
                sport=random.randint(1024, 65535), dport=random.randint(1, 1024), flags=""
            )
        else:
            pkt = IP(dst=target, ttl=ttl) / TCP(
                sport=random.randint(50000, 60000), dport=443, flags="A", window=0
            )
        send(pkt, verbose=0)
        n += 1
        _sleep_jitter(0.015, 0.012)
        if n % 50 == 0:
            print(f"[*] {n} mix packets sent…")
    print(f"[+] protomix done ({n} pkts)")


# keep old name as alias
mode_jitter = mode_zeroday

MODES = {
    "zeroday": mode_zeroday,
    "jitter": mode_zeroday,
    "nullpulse": mode_nullpulse,
    "sizestorm": mode_sizestorm,
    "protomix": mode_protomix,
}


def main():
    ap = argparse.ArgumentParser(description="Unknown-attack probe for Isolation Forest")
    ap.add_argument("--target", default="172.20.0.10")
    ap.add_argument("--mode", required=True, choices=list(MODES))
    ap.add_argument("--duration", type=int, default=30)
    args = ap.parse_args()
    print("[*] scapy send() needs NET_RAW — docker attacker has it")
    MODES[args.mode](args.target, args.duration)
    print("\n[*] Check Live Monitor for UNKNOWN-ANOMALY / L1-IF-VETO / L2-DISAGREE")


if __name__ == "__main__":
    main()
