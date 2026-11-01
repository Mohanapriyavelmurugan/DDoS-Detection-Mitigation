"""
=============================================================================
MITIGATION ENGINE v1  —  Phase 7 Automated Response Module
Edge-Intelligent DDoS Mitigation for 5G/MEC Networks
=============================================================================

SCIENTIFIC BASIS
  SYN Cookie defence       : RFC 4987 - validates TCP handshake without
                             allocating kernel connection state.
  Token-bucket rate limit  : RFC 2697 - iptables hashlimit implements the
                             token-bucket algorithm per source IP.
  Stateful connection cap  : netfilter connlimit - limits concurrent sessions
                             to prevent connection-table exhaustion.
  DNS amplification guard  : RFC 5358 - drops outbound DNS responses > 512 B
                             that are the hallmark of amplification abuse.
  DSCP quarantine marking  : RFC 2474 - re-marks attack traffic to CS1
                             (background) so it is de-prioritised by QoS.
  SlowLoris timeout tuning : Reduces TCP established-state conntrack timeout
                             and caps new-connection rate from offending IP.

MITIGATION TIERS  (fusion-score driven)
  Tier 0  MONITOR    (0.40-0.54)  Log + alert only, zero network change.
  Tier 1  THROTTLE   (0.55-0.69)  Rate-limit to RATE_LIMIT_PPS pps via
                                   iptables hashlimit (token bucket).
  Tier 2  QUARANTINE (0.70-0.89)  Connection cap + DSCP CS1 re-mark.
  Tier 3  BLOCK      (0.90-1.00)  Hard DROP on INPUT/OUTPUT/FORWARD.

ATTACK-SPECIFIC DISPATCH  (when XGBoost provides a label)
  SYN_FLOOD      ->  SYN cookies (sysctl) + hashlimit SYN/sec per src IP
  UDP_FLOOD      ->  hashlimit UDP pps per src IP
  ICMP_FLOOD     ->  DROP icmp echo-request from src IP
  HTTP_FLOOD     ->  connlimit > 50 concurrent on port 80/8080
  SLOWLORIS      ->  conntrack timeout reduction + new-conn rate cap
  DNS_AMP        ->  DROP outbound UDP:53 > 512 B + inbound DNS rate cap
  RECON          ->  new-connection rate cap (slows scan below effective rate)
  TCP_FLOOD      ->  hashlimit all-TCP pps per src IP (generic TCP flood)
  FRAG_FLOOD     ->  DROP fragmented packets from src IP (ACK/ICMP frag)
  TCP_FLAG_FLOOD ->  DROP specific TCP flag combos (PSH+ACK, RST+FIN)
  GRE_FLOOD      ->  DROP GRE protocol (proto 47) from src IP (Mirai)
  BRUTEFORCE     ->  hashlimit new connections to 2/min (SSH/HTTP auth)
  MALWARE        ->  HARD_BLOCK immediately (device compromised)
  DNS_SPOOF      ->  DROP inbound DNS responses from src IP
  UPLOAD_FLOOD   ->  connlimit on 80/8080 + DROP oversized payloads
  SQLI/CMDI/XSS  ->  String matching for malicious payloads (HTTP only)
  UNKNOWN        ->  tier-based agnostic response (Tier 1/2/3 rules above)

AUTO-EXPIRY
  Every rule is stored with an expiry timestamp.
  A daemon thread (cleanup_loop) runs every CLEANUP_INTERVAL_SEC and
  inverts (-D) each iptables -A rule when it expires.
  This prevents permanent blocking of IPs that may be false positives.

DRY-RUN MODE
  Set DRY_RUN = True to log all commands without executing them.
  Use this when testing inside a Docker container that lacks root.
=============================================================================
"""

import os
import sys
import time
import logging
import subprocess
import threading
import csv
from datetime import datetime

# ---------------------------------------------------------------------------
# Auto-install iptables if the binary is missing (common in slim Docker images)
# ---------------------------------------------------------------------------
def _ensure_iptables() -> None:
    """Check if iptables is available; install it silently if not."""
    result = subprocess.run(["which", "iptables"], capture_output=True)
    if result.returncode != 0:
        print("[MITIGATION INIT] iptables not found — installing now...", flush=True)
        subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "iptables"], capture_output=True)
        verify = subprocess.run(["which", "iptables"], capture_output=True)
        if verify.returncode == 0:
            print("[MITIGATION INIT] iptables installed successfully.", flush=True)
        else:
            print("[MITIGATION INIT] WARNING: iptables install failed. "
                  "Rules will be logged only (DRY_RUN mode).", flush=True)
            global DRY_RUN
            DRY_RUN = True
    else:
        print("[MITIGATION INIT] iptables found. Firewall rules ACTIVE.", flush=True)

_ensure_iptables()

# =============================================================================
# CONFIGURATION  -- edit these to match your MEC environment
# =============================================================================
# LOCAL lab uses shorter expiry so re-tests can re-trigger toast/mitigation
# without waiting 5 minutes. MEC/production keeps the longer windows.
_DEPLOY = str(os.environ.get("DEPLOY_ENV", os.environ.get("MEC_MODE", "LOCAL"))).upper()
if _DEPLOY in ("1", "TRUE", "YES"):
    _DEPLOY = "MEC"
if _DEPLOY == "LOCAL":
    BLOCK_DURATION_SEC    = int(os.environ.get("BLOCK_DURATION_SEC", "90"))
    THROTTLE_DURATION_SEC = int(os.environ.get("THROTTLE_DURATION_SEC", "60"))
else:
    BLOCK_DURATION_SEC    = int(os.environ.get("BLOCK_DURATION_SEC", "300"))
    THROTTLE_DURATION_SEC = int(os.environ.get("THROTTLE_DURATION_SEC", "180"))
CLEANUP_INTERVAL_SEC  = 30    # Cleanup thread check period (s)

LOG_FILE = "/app/mitigation_log.csv"   # Inside the Docker container

DRY_RUN  = False   # True -> print commands only, do NOT run them

# Set this to False if traffic is mostly HTTPS, where string matching is useless
PAYLOAD_INSPECT_ENABLED = True

# Token-bucket parameters (Tier 1 throttle + attack-specific rate limits)
RATE_LIMIT_PPS   = 100   # packets / second allowed before DROP
RATE_LIMIT_BURST = 200   # token-bucket burst capacity

# Maximum concurrent connections per IP (Tier 2 quarantine + HTTP flood)
CONNLIMIT_ABOVE = 20

# DNS amplification: responses larger than this are dropped (RFC 5358)
DNS_RESPONSE_SIZE_BYTES = 512

# -- Trusted IPs -- NEVER blocked ----------------------------------------------
# Add your MEC gateway, management interface, and lab hosts here.
WHITELIST_IPS = {
    "127.0.0.1",
    "10.0.0.1",       # typical MEC gateway / UPF
    "192.168.1.1",    # lab router
    # "192.168.100.5",  # add your MEC management IP here
}

# Auto-detect and whitelist all local interface IPs so the detector
# never accidentally blocks its own container / host.
try:
    import socket
    _local_ips = socket.gethostbyname_ex(socket.gethostname())[2]
    WHITELIST_IPS.update(_local_ips)
except Exception:
    pass
try:
    import subprocess as _sp
    _ip_out = _sp.run(["hostname", "-I"], capture_output=True, text=True).stdout
    for _ip in _ip_out.split():
        WHITELIST_IPS.add(_ip.strip())
except Exception:
    pass

# -- XGBoost label -> normalised attack family ---------------------------------
# Covers all 34 classes from the CIC-IoT-2023 / NIDD dataset used in training.
ATTACK_FAMILY_MAP = {
    # SYN Flood variants
    "DDOS-SYN_FLOOD":            "SYN_FLOOD",
    "DOS-SYN_FLOOD":             "SYN_FLOOD",
    "DDOS-SYNONYMOUS_IP_FLOOD":  "SYN_FLOOD",   # exact model label
    "DDOS-SYNONYMOUSIP_FLOOD":   "SYN_FLOOD",   # alternate spelling in dataset
    "DDOS-SYNONYMOUS_IP_FLOOD":  "SYN_FLOOD",
    "SYN_FLOOD":                 "SYN_FLOOD",
    # UDP Flood variants
    "DDOS-UDP_FLOOD":            "UDP_FLOOD",
    "DOS-UDP_FLOOD":             "UDP_FLOOD",
    "DDOS-UDP_FRAGMENTATION":    "UDP_FLOOD",
    "UDP_FLOOD":                 "UDP_FLOOD",
    # Mirai UDP plain — same token-bucket defence as generic UDP flood
    "MIRAI-UDPPLAIN":            "UDP_FLOOD",
    # ICMP / Ping variants
    "DDOS-ICMP_FLOOD":           "ICMP_FLOOD",
    "DOS-ICMP_FLOOD":            "ICMP_FLOOD",
    "DDOS-PINGSWEEP":            "ICMP_FLOOD",
    "RECON-PINGSWEEP":           "ICMP_FLOOD",
    "ICMP_FLOOD":                "ICMP_FLOOD",
    # HTTP Flood
    "DDOS-HTTP_FLOOD":           "HTTP_FLOOD",
    "DOS-HTTP_FLOOD":            "HTTP_FLOOD",
    "HTTP_FLOOD":                "HTTP_FLOOD",
    # SlowLoris
    "DOS-SLOWLORIS":             "SLOWLORIS",
    "DDOS-SLOWLORIS":            "SLOWLORIS",
    "SLOWLORIS":                 "SLOWLORIS",
    # DNS Amplification
    "DDOS-DNS_AMPLIFICATION":    "DNS_AMP",
    "DNS_AMPLIFICATION":         "DNS_AMP",
    # Recon / Brute-force / MITM -- use soft rate cap
    "RECON-PORTSCAN":            "RECON",
    "BRUTE_FORCE":               "RECON",
    "MITM-ARPSPOOFING":          "RECON",
    "RECON-HOSTDISCOVERY":       "RECON",
    "RECON-OSSCAN":              "RECON",
    "RECON-VULNERABILITYSCAN":   "RECON",
    "VULNERABILITYSCAN":         "RECON",
    "NETWORK_DISCOVERY":         "RECON",
    # ---------------------------------------------------------------
    # NEW: Generic TCP Flood variants
    # ---------------------------------------------------------------
    "DDOS-TCP_FLOOD":            "TCP_FLOOD",
    "DOS-TCP_FLOOD":             "TCP_FLOOD",
    "TCP_FLOOD":                 "TCP_FLOOD",
    # ---------------------------------------------------------------
    # NEW: Fragmentation Floods (ACK-frag + ICMP-frag)
    # ---------------------------------------------------------------
    "DDOS-ACK_FRAGMENTATION":    "FRAG_FLOOD",
    "DDOS-ICMP_FRAGMENTATION":   "FRAG_FLOOD",
    "ACK_FRAGMENTATION":         "FRAG_FLOOD",
    "ICMP_FRAGMENTATION":        "FRAG_FLOOD",
    # ---------------------------------------------------------------
    # NEW: TCP Flag Floods (PSH+ACK, RST+FIN)
    # ---------------------------------------------------------------
    "DDOS-PSHACK_FLOOD":         "TCP_FLAG_FLOOD",
    "DDOS-RSTFINFLOOD":          "TCP_FLAG_FLOOD",
    "PSHACK_FLOOD":              "TCP_FLAG_FLOOD",
    "RSTFINFLOOD":               "TCP_FLAG_FLOOD",
    # ---------------------------------------------------------------
    # NEW: Mirai GRE Floods (GRE Ethernet + GRE IP)
    # ---------------------------------------------------------------
    "MIRAI-GREETH_FLOOD":        "GRE_FLOOD",
    "MIRAI-GREIP_FLOOD":         "GRE_FLOOD",
    "GREETH_FLOOD":              "GRE_FLOOD",
    "GREIP_FLOOD":               "GRE_FLOOD",
    # ---------------------------------------------------------------
    # NEW: Group 1 (App-Layer with Network mitigation)
    # ---------------------------------------------------------------
    "DICTIONARYBRUTEFORCE":      "BRUTEFORCE",
    "BACKDOOR_MALWARE":          "MALWARE",
    "DNS_SPOOFING":              "DNS_SPOOF",
    "UPLOADING_ATTACK":          "UPLOAD_FLOOD",
    # ---------------------------------------------------------------
    # NEW: Group 2 (App-Layer String Match - HTTP only)
    # ---------------------------------------------------------------
    "SQLINJECTION":              "SQLI",
    "COMMANDINJECTION":          "CMDI",
    "XSS":                       "XSS",
    # Isolation-Forest novel / zero-day label (not in XGB training set)
    "UNKNOWN-ANOMALY":           "UNKNOWN",
    "UNKNOWN_ANOMALY":           "UNKNOWN",
    "UNKNOWN":                   "UNKNOWN",
}

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[MITIGATE] %(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("MitigationEngine")

_csv_lock = threading.Lock()


def _write_csv(row: dict) -> None:
    """Thread-safe append of one mitigation event to the CSV audit log."""
    fields = [
        "timestamp", "src_ip", "attack_type", "family",
        "fusion_score", "tier", "action", "command", "status",
    ]
    try:
        file_exists = os.path.isfile(LOG_FILE)
        with _csv_lock:
            with open(LOG_FILE, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({k: row.get(k, "") for k in fields})
    except Exception as exc:
        log.error(f"CSV write error: {exc}")


# =============================================================================
# RULE STATE TRACKER
# =============================================================================
class _RuleState:
    """
    Tracks active mitigation rules per source IP.
    Each entry: tier, attack family, expiry epoch, list of applied commands.
    """

    def __init__(self):
        self._rules: dict = {}
        self._lock  = threading.Lock()

    def is_active(self, ip: str) -> bool:
        with self._lock:
            r = self._rules.get(ip)
            return r is not None and r["expire"] > time.time()

    def get_tier(self, ip: str) -> int:
        with self._lock:
            r = self._rules.get(ip)
            return r["tier"] if r else 0

    def add(self, ip: str, tier: int, family: str,
            duration: int, commands: list) -> None:
        with self._lock:
            self._rules[ip] = {
                "tier":     tier,
                "family":   family,
                "expire":   time.time() + duration,
                "commands": commands,
            }

    def get_expired(self) -> list:
        now = time.time()
        with self._lock:
            return [(ip, r) for ip, r in list(self._rules.items())
                    if r["expire"] <= now]

    def remove(self, ip: str) -> None:
        with self._lock:
            self._rules.pop(ip, None)


_state = _RuleState()

# Per-IP mutex: prevents concurrent threads from adding duplicate rules
# for the same source IP when multiple flows arrive in the same window.
_ip_locks: dict = {}
_ip_locks_lock = threading.Lock()

def _get_ip_lock(ip: str) -> threading.Lock:
    with _ip_locks_lock:
        if ip not in _ip_locks:
            _ip_locks[ip] = threading.Lock()
        return _ip_locks[ip]


# =============================================================================
# COMMAND EXECUTION
# =============================================================================
def _is_optional_cmd(cmd: list) -> bool:
    """sysctl is best-effort inside Docker (may lack binary or CAP_SYS_ADMIN)."""
    return bool(cmd) and cmd[0] == "sysctl"


def _run(cmd: list) -> bool:
    """Execute a shell command list. Returns True on success."""
    cmd_str = " ".join(cmd)
    optional = _is_optional_cmd(cmd)
    if DRY_RUN:
        log.info(f"[DRY-RUN]  {cmd_str}")
        return True
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            if optional:
                log.info(f"[OPTIONAL SKIP] {cmd_str} | {r.stderr.strip()}")
                return True  # do not fail overall mitigation
            log.warning(f"CMD failed [{r.returncode}]: {cmd_str} | {r.stderr.strip()}")
            return False
        return True
    except subprocess.TimeoutExpired:
        log.error(f"CMD timed out: {cmd_str}")
        return False
    except FileNotFoundError:
        if optional:
            log.info(f"[OPTIONAL SKIP] {cmd[0]} not installed — continuing with iptables rules.")
            return True
        log.error(f"Binary not found for: {cmd_str}")
        return False
    except Exception as exc:
        if optional:
            log.info(f"[OPTIONAL SKIP] {cmd_str} ({exc})")
            return True
        log.error(f"CMD exception ({exc}): {cmd_str}")
        return False


def _invert(cmd: list) -> list:
    """Convert iptables -A (append) to -D (delete) for rule cleanup."""
    return ["-D" if c == "-A" else c for c in cmd]


# =============================================================================
# ATTACK-SPECIFIC MITIGATION BUILDERS
# Each function returns a list-of-lists (commands to run in sequence).
# =============================================================================

def _cmds_syn_flood(ip: str) -> list:
    """
    RFC 4987 SYN Cookie + per-IP SYN rate limit.
    SYN cookies allow the kernel to respond with a cryptographic sequence
    number, deferring resource allocation until the ACK is confirmed.
    The hashlimit rule limits new SYN bursts to 10/sec (burst=20).
    """
    tag = ip.replace(".", "_")
    return [
        ["sysctl", "-w", "net.ipv4.tcp_syncookies=1"],
        ["iptables", "-A", "INPUT", "-p", "tcp", "--syn",
         "-s", ip,
         "-m", "hashlimit",
         "--hashlimit-name",  f"syn_{tag}",
         "--hashlimit-above", "10/sec",
         "--hashlimit-burst", "20",
         "--hashlimit-mode",  "srcip",
         "-j", "DROP"],
    ]


def _cmds_udp_flood(ip: str) -> list:
    """
    Token-bucket rate limit on all UDP from src IP (RFC 2697).
    Drops packets above RATE_LIMIT_PPS while allowing burst through.
    """
    tag = ip.replace(".", "_")
    return [
        ["iptables", "-A", "INPUT", "-p", "udp",
         "-s", ip,
         "-m", "hashlimit",
         "--hashlimit-name",  f"udp_{tag}",
         "--hashlimit-above", f"{RATE_LIMIT_PPS}/sec",
         "--hashlimit-burst", str(RATE_LIMIT_BURST),
         "--hashlimit-mode",  "srcip",
         "-j", "DROP"],
    ]


def _cmds_icmp_flood(ip: str) -> list:
    """
    Drop ICMP echo-request from source IP.
    High-rate ICMP echo has no legitimate use case; total drop is safe.
    Other ICMP types (TTL-exceeded, Unreachable) are preserved.
    """
    return [
        ["iptables", "-A", "INPUT", "-p", "icmp",
         "--icmp-type", "echo-request",
         "-s", ip,
         "-j", "DROP"],
    ]


def _cmds_http_flood(ip: str) -> list:
    """
    connlimit on ports 80 and 8080.
    Caps concurrent HTTP sessions per IP at 50.
    Real browsers rarely exceed 6-8 concurrent connections (RFC 7230 s6.4).
    """
    return [
        ["iptables", "-A", "INPUT", "-p", "tcp", "--dport", "80",
         "-s", ip,
         "-m", "connlimit", "--connlimit-above", "50",
         "--connlimit-mask", "32",
         "-j", "DROP"],
        ["iptables", "-A", "INPUT", "-p", "tcp", "--dport", "8080",
         "-s", ip,
         "-m", "connlimit", "--connlimit-above", "50",
         "--connlimit-mask", "32",
         "-j", "DROP"],
    ]


def _cmds_slowloris(ip: str) -> list:
    """
    Defend against Slowloris (incomplete HTTP header attack).
    1. Reduce conntrack TCP established timeout (default 432000s -> 120s)
       so stale sockets are reaped quickly.
    2. Rate-cap new TCP connections from attacker to 5/sec, preventing
       hundreds of slow sockets from being opened.
    The two rules together: existing slow connections time out fast,
    and new ones cannot be opened faster than 5/sec.
    """
    tag = ip.replace(".", "_")
    return [
        ["sysctl", "-w",
         "net.netfilter.nf_conntrack_tcp_timeout_established=120"],
        ["iptables", "-A", "INPUT", "-p", "tcp",
         "-s", ip,
         "-m", "conntrack", "--ctstate", "NEW",
         "-m", "hashlimit",
         "--hashlimit-name",  f"slow_{tag}",
         "--hashlimit-above", "5/sec",
         "--hashlimit-burst", "10",
         "--hashlimit-mode",  "srcip",
         "-j", "DROP"],
    ]


def _cmds_dns_amp(ip: str) -> list:
    """
    DNS Amplification two-pronged defence (RFC 5358).
    1. Drop outbound DNS responses > 512 bytes to attacker IP.
       Legitimate answers rarely exceed 512 B; amplified ANY/TXT
       responses can reach 4096 B.
    2. Rate-limit inbound DNS queries from attacker to 5/sec.
    """
    tag = ip.replace(".", "_")
    return [
        ["iptables", "-A", "OUTPUT", "-p", "udp",
         "--sport", "53",
         "-d", ip,
         "-m", "length", "--length", f"{DNS_RESPONSE_SIZE_BYTES}:65535",
         "-j", "DROP"],
        ["iptables", "-A", "INPUT", "-p", "udp",
         "--dport", "53",
         "-s", ip,
         "-m", "hashlimit",
         "--hashlimit-name",  f"dns_{tag}",
         "--hashlimit-above", "5/sec",
         "--hashlimit-burst", "10",
         "--hashlimit-mode",  "srcip",
         "-j", "DROP"],
    ]


def _cmds_recon(ip: str) -> list:
    """
    Port-scan / recon / brute-force: cap new-connection rate.
    Limits attacker to 5 new connections/sec, reducing scan throughput
    below the effective rate needed for timely recon, without fully blocking.
    """
    tag = ip.replace(".", "_")
    return [
        ["iptables", "-A", "INPUT",
         "-s", ip,
         "-m", "conntrack", "--ctstate", "NEW",
         "-m", "hashlimit",
         "--hashlimit-name",  f"recon_{tag}",
         "--hashlimit-above", "5/sec",
         "--hashlimit-burst", "10",
         "--hashlimit-mode",  "srcip",
         "-j", "DROP"],
    ]


def _cmds_tcp_flood(ip: str) -> list:
    """
    Generic TCP Flood (DDOS-TCP_FLOOD, DOS-TCP_FLOOD).
    Applies a token-bucket rate limit on ALL TCP traffic from the source IP.
    Uses RATE_LIMIT_PPS (same as UDP flood) because the attack characteristic
    is identical — overwhelming packet volume — just over TCP instead of UDP.
    This is deliberately broader than the SYN-only rule in _cmds_syn_flood:
    TCP floods send a mix of flags (ACK, PSH, SYN) to avoid simple SYN filters.
    """
    tag = ip.replace(".", "_")
    return [
        ["iptables", "-A", "INPUT", "-p", "tcp",
         "-s", ip,
         "-m", "hashlimit",
         "--hashlimit-name",  f"tcpflood_{tag}",
         "--hashlimit-above", f"{RATE_LIMIT_PPS}/sec",
         "--hashlimit-burst", str(RATE_LIMIT_BURST),
         "--hashlimit-mode",  "srcip",
         "-j", "DROP"],
    ]


def _cmds_fragmentation(ip: str) -> list:
    """
    Fragmentation Flood defence (DDOS-ACK_FRAGMENTATION, DDOS-ICMP_FRAGMENTATION).
    Drops ALL fragmented IP packets from the attacking source.

    Scientific basis:
      Fragmentation floods send oversized packets split into many IP fragments.
      The victim must reassemble all fragments before processing — exhausting
      CPU and memory in the IP reassembly queue.
      RFC 1858 and RFC 3128 document fragment-based attack patterns.

    Why dropping all fragments from this IP is safe:
      Legitimate traffic rarely requires fragmentation on modern networks
      (Path MTU Discovery, RFC 1191, prevents it). When fragmented packets
      arrive in a flood pattern from a single IP, they are overwhelmingly
      malicious. The two rules cover:
        1. First fragment (MORE-FRAGMENTS bit set, frag-offset = 0)
        2. Subsequent fragments (frag-offset > 0)
    """
    return [
        # Drop first fragment (MF=1, offset=0) — the trigger packet
        ["iptables", "-A", "INPUT",
         "-s", ip,
         "-m", "ipv4options",
         "-f",          # match fragmented packets
         "-j", "DROP"],
        # Fallback: drop any packet with a non-zero fragment offset
        ["iptables", "-A", "INPUT",
         "-s", ip,
         "--fragment",
         "-j", "DROP"],
    ]


def _cmds_tcp_flag_flood(ip: str) -> list:
    """
    TCP Flag Flood defence (DDOS-PSHACK_FLOOD, DDOS-RSTFINFLOOD).
    Drops specifically crafted flag-combination packets from the src IP.

    PSH+ACK flood: Sends TCP packets with PSH and ACK flags set simultaneously.
      Normal data transfer uses PSH+ACK legitimately, but at flood rates it
      exhausts the server's socket receive buffers. We rate-limit rather than
      drop entirely to preserve any real PSH+ACK during a false-positive scenario.

    RST+FIN flood: Sends TCP packets with both RST and FIN flags set — an
      invalid combination per RFC 793. Legitimate stacks never set both.
      It is completely safe to DROP RST+FIN packets unconditionally.

    Two separate rules: one rate-cap for PSH+ACK (preserves legitimate bursts),
    one hard drop for RST+FIN (invalid flag combo — always malicious).
    """
    tag = ip.replace(".", "_")
    return [
        # Rule 1: Rate-limit PSH+ACK flood (--tcp-flags PSH,ACK PSH,ACK)
        ["iptables", "-A", "INPUT", "-p", "tcp",
         "-s", ip,
         "--tcp-flags", "PSH,ACK", "PSH,ACK",
         "-m", "hashlimit",
         "--hashlimit-name",  f"pshack_{tag}",
         "--hashlimit-above", "50/sec",
         "--hashlimit-burst", "100",
         "--hashlimit-mode",  "srcip",
         "-j", "DROP"],
        # Rule 2: Hard DROP RST+FIN (RFC 793 invalid — never legitimate)
        ["iptables", "-A", "INPUT", "-p", "tcp",
         "-s", ip,
         "--tcp-flags", "RST,FIN", "RST,FIN",
         "-j", "DROP"],
    ]


def _cmds_gre_flood(ip: str) -> list:
    """
    Mirai GRE Flood defence (MIRAI-GREETH_FLOOD, MIRAI-GREIP_FLOOD).
    Drops all GRE (Generic Routing Encapsulation, IP protocol 47) packets
    from the attacking source IP.

    Scientific basis:
      Mirai botnet GRE floods encapsulate arbitrary payloads inside GRE
      tunnels to bypass simple port-based filters. GRE carries no port
      numbers, so connlimit and hashlimit (port-based) do not apply.
      RFC 2784 defines GRE (protocol number 47).

    Why dropping all GRE from this IP is safe:
      GRE is used by VPN tunnels (PPTP, GRE tunnels) but a typical 5G/MEC
      UE device has no legitimate reason to send GRE traffic to an application
      server. If the IP has been flagged as a Mirai bot, all GRE from it is
      malicious. A second rule also rate-limits all other IP traffic from the
      same source as a belt-and-suspenders precaution against mixed floods.
    """
    tag = ip.replace(".", "_")
    return [
        # Rule 1: DROP all GRE (protocol 47) from attacking IP
        ["iptables", "-A", "INPUT",
         "-s", ip,
         "-p", "47",   # GRE protocol number
         "-j", "DROP"],
        # Rule 2: Rate-limit all remaining traffic (belt-and-suspenders)
        ["iptables", "-A", "INPUT",
         "-s", ip,
         "-m", "hashlimit",
         "--hashlimit-name",  f"gre_{tag}",
         "--hashlimit-above", f"{RATE_LIMIT_PPS}/sec",
         "--hashlimit-burst", str(RATE_LIMIT_BURST),
         "--hashlimit-mode",  "srcip",
         "-j", "DROP"],
    ]


def _cmds_bruteforce(ip: str) -> list:
    """
    Dictionary Brute Force defence.
    Caps new connection attempts to SSH (22) and HTTP (80/8080) to 2 per minute.
    """
    tag = ip.replace(".", "_")
    return [
        ["iptables", "-A", "INPUT", "-p", "tcp",
         "-s", ip,
         "-m", "multiport", "--dports", "22,80,8080",
         "-m", "conntrack", "--ctstate", "NEW",
         "-m", "hashlimit",
         "--hashlimit-name",  f"brute_{tag}",
         "--hashlimit-above", "2/min",
         "--hashlimit-burst", "5",
         "--hashlimit-mode",  "srcip",
         "-j", "DROP"],
    ]


def _cmds_malware(ip: str) -> list:
    """
    Backdoor/Malware defence.
    Device is compromised. Apply HARD BLOCK.
    """
    return _cmds_tier3_block(ip)


def _cmds_dns_spoof(ip: str) -> list:
    """
    DNS Spoofing defence.
    Drop inbound DNS responses (UDP src port 53) from the attacker IP.
    """
    return [
        ["iptables", "-A", "INPUT", "-p", "udp",
         "-s", ip,
         "--sport", "53",
         "-j", "DROP"],
    ]


def _cmds_upload_flood(ip: str) -> list:
    """
    Uploading Attack defence.
    Drop oversized HTTP payloads from the attacker IP.
    """
    return [
        ["iptables", "-A", "INPUT", "-p", "tcp",
         "-s", ip,
         "-m", "multiport", "--dports", "80,8080",
         "-m", "length", "--length", "2000:65535",
         "-j", "DROP"],
    ]


def _cmds_payload_filter(ip: str, patterns: list) -> list:
    """
    App-Layer String Match defence.
    Only applicable if PAYLOAD_INSPECT_ENABLED is True.
    """
    if not PAYLOAD_INSPECT_ENABLED:
        return _cmds_tier1_throttle(ip)  # Fallback
    
    cmds = []
    for p in patterns:
        cmds.append([
            "iptables", "-A", "INPUT", "-p", "tcp",
            "-s", ip,
            "-m", "multiport", "--dports", "80,8080",
            "-m", "string", "--algo", "bm", "--string", p,
            "-j", "DROP"
        ])
    return cmds


# -- Tier-based agnostic rules (used when attack family is UNKNOWN) ------------

def _cmds_tier1_throttle(ip: str) -> list:
    """Tier 1: Token-bucket rate limit on all traffic from IP."""
    tag = ip.replace(".", "_")
    return [
        ["iptables", "-A", "INPUT",
         "-s", ip,
         "-m", "hashlimit",
         "--hashlimit-name",  f"t1_{tag}",
         "--hashlimit-above", f"{RATE_LIMIT_PPS}/sec",
         "--hashlimit-burst", str(RATE_LIMIT_BURST),
         "--hashlimit-mode",  "srcip",
         "-j", "DROP"],
    ]


def _cmds_tier2_quarantine(ip: str) -> list:
    """
    Tier 2: Connection cap + DSCP CS1 re-mark (RFC 2474).
    CS1 (DSCP 8) is the 'background' class; QoS-aware switches
    de-prioritise this traffic without dropping it, preserving
    partial connectivity while reducing damage.
    """
    return [
        ["iptables", "-A", "INPUT",
         "-s", ip,
         "-m", "connlimit", "--connlimit-above", str(CONNLIMIT_ABOVE),
         "--connlimit-mask", "32",
         "-j", "DROP"],
        ["iptables", "-t", "mangle", "-A", "PREROUTING",
         "-s", ip,
         "-j", "DSCP", "--set-dscp-class", "CS1"],
    ]


def _cmds_tier3_block(ip: str) -> list:
    """Tier 3: Hard DROP on INPUT, OUTPUT, and FORWARD chains."""
    return [
        ["iptables", "-A", "INPUT",   "-s", ip, "-j", "DROP"],
        ["iptables", "-A", "OUTPUT",  "-d", ip, "-j", "DROP"],
        ["iptables", "-A", "FORWARD", "-s", ip, "-j", "DROP"],
    ]


# =============================================================================
# PUBLIC API  --  called by realtime_defender_v2.py
# =============================================================================

def is_ip_mitigated(src_ip: str) -> bool:
    """True when an active mitigation rule already exists for this IP."""
    return _state.is_active(src_ip)


def apply_mitigation(src_ip: str,
                     attack_label: str,
                     fusion_score: float) -> dict:
    """
    Dispatch the correct mitigation action for a detected attack flow.

    Returns a dict describing what happened (for dashboard telemetry).
    """
    empty = {
        "mitigated": False,
        "status": "skipped",
        "family": "",
        "action": "",
        "tier": 0,
        "tier_name": "NONE",
        "expires_in": 0,
        "commands": [],
        "summary": "",
    }

    # 1. Whitelist check
    if src_ip in WHITELIST_IPS:
        log.info(f"[WHITELIST] {src_ip} is trusted -- skipping mitigation.")
        return {**empty, "status": "whitelisted", "summary": f"{src_ip} is whitelisted"}

    # Acquire per-IP lock BEFORE checking state.
    ip_lock = _get_ip_lock(src_ip)
    if not ip_lock.acquire(blocking=False):
        return {**empty, "status": "busy", "summary": "Mitigation already in progress for this IP"}

    try:
        # 2. Resolve attack family
        family = ATTACK_FAMILY_MAP.get(attack_label.upper().strip(), "UNKNOWN")

        # Fix floating point precision bug: float32 0.90 is 0.899999976 in Python
        fusion_score = round(fusion_score, 3)

        # 3. Determine tier and duration
        if fusion_score >= 0.90:
            tier, duration = 3, BLOCK_DURATION_SEC
        elif fusion_score >= 0.70:
            tier, duration = 2, THROTTLE_DURATION_SEC
        elif fusion_score >= 0.55:
            tier, duration = 1, THROTTLE_DURATION_SEC
        else:
            tier, duration = 0, 0   # Monitor only (score 0.40-0.54)

        tier_name = {0: "MONITOR", 1: "THROTTLE", 2: "QUARANTINE", 3: "HARD BLOCK"}.get(tier, "NONE")

        # Tier 0 -- log only, no iptables changes
        if tier == 0:
            log.warning(
                f"[TIER-0 MONITOR] {src_ip} | {family} | "
                f"score={fusion_score:.3f} -- logging only, no network change."
            )
            _write_csv({
                "timestamp": datetime.now().isoformat(),
                "src_ip": src_ip, "attack_type": attack_label,
                "family": family, "fusion_score": round(fusion_score, 3),
                "tier": 0, "action": "MONITOR", "command": "none", "status": "logged",
            })
            return {
                "mitigated": True,
                "status": "monitor",
                "family": family,
                "action": "MONITOR",
                "tier": 0,
                "tier_name": "MONITOR",
                "expires_in": 0,
                "commands": [],
                "summary": f"Attack {attack_label} logged — Tier 0 monitor only",
            }

        # 4. Cooldown / escalation check
        if _state.is_active(src_ip):
            existing = _state.get_tier(src_ip)
            if existing >= tier:
                return {
                    "mitigated": True,
                    "status": "already_active",
                    "family": family,
                    "action": "COOLDOWN",
                    "tier": existing,
                    "tier_name": {1: "THROTTLE", 2: "QUARANTINE", 3: "HARD BLOCK"}.get(existing, "ACTIVE"),
                    "expires_in": duration,
                    "commands": [],
                    "summary": f"Mitigation already active at tier {existing} for {src_ip}",
                }
            log.info(f"[ESCALATE] {src_ip}: Tier {existing} -> Tier {tier}")
            _cleanup_ip(src_ip)  # remove weaker rules before applying stronger

        # 5. Build command list
        if family == "SYN_FLOOD":
            cmds, action = _cmds_syn_flood(src_ip),       "SYN_COOKIES+RATE_LIMIT"
        elif family == "UDP_FLOOD":
            cmds, action = _cmds_udp_flood(src_ip),       "UDP_RATE_LIMIT"
        elif family == "ICMP_FLOOD":
            cmds, action = _cmds_icmp_flood(src_ip),      "ICMP_DROP"
        elif family == "HTTP_FLOOD":
            cmds, action = _cmds_http_flood(src_ip),      "CONNLIMIT_HTTP"
        elif family == "SLOWLORIS":
            cmds, action = _cmds_slowloris(src_ip),       "TIMEOUT+RATE_LIMIT"
        elif family == "DNS_AMP":
            cmds, action = _cmds_dns_amp(src_ip),         "DNS_SIZE_FILTER"
        elif family == "RECON":
            cmds, action = _cmds_recon(src_ip),           "RECON_RATE_CAP"
        elif family == "TCP_FLOOD":
            cmds, action = _cmds_tcp_flood(src_ip),       "TCP_RATE_LIMIT"
        elif family == "FRAG_FLOOD":
            cmds, action = _cmds_fragmentation(src_ip),   "FRAGMENT_DROP"
        elif family == "TCP_FLAG_FLOOD":
            cmds, action = _cmds_tcp_flag_flood(src_ip),  "FLAGFLOOD_DROP+RATE_LIMIT"
        elif family == "GRE_FLOOD":
            cmds, action = _cmds_gre_flood(src_ip),       "GRE_DROP+RATE_LIMIT"
        elif family == "BRUTEFORCE":
            cmds, action = _cmds_bruteforce(src_ip),      "BRUTEFORCE_RATE_CAP"
        elif family == "MALWARE":
            cmds, action = _cmds_malware(src_ip),         "HARD_BLOCK_COMPROMISED"
            tier = 3 # force hard block
        elif family == "DNS_SPOOF":
            cmds, action = _cmds_dns_spoof(src_ip),       "DNS_SPOOF_DROP"
        elif family == "UPLOAD_FLOOD":
            cmds, action = _cmds_upload_flood(src_ip),    "PAYLOAD_SIZE_DROP"
        elif family == "SQLI":
            cmds, action = _cmds_payload_filter(src_ip, ["SELECT ", "UNION ", "DROP TABLE", "' OR '1'='1"]), "SQLI_STRING_MATCH"
        elif family == "CMDI":
            cmds, action = _cmds_payload_filter(src_ip, [";/bin/sh", "&&/etc/", "|bash", "$(curl"]), "CMDI_STRING_MATCH"
        elif family == "XSS":
            cmds, action = _cmds_payload_filter(src_ip, ["<script>", "javascript:", "onerror=", "onload="]), "XSS_STRING_MATCH"
        else:
            # Unknown attack -- use tier-based agnostic response
            if tier == 1:
                cmds, action = _cmds_tier1_throttle(src_ip), "THROTTLE"
            elif tier == 2:
                cmds, action = _cmds_tier2_quarantine(src_ip), "QUARANTINE+DSCP"
            else:
                cmds, action = _cmds_tier3_block(src_ip), "HARD_BLOCK"

        # 6. For known attacks at max confidence, also hard-block
        if family != "UNKNOWN" and tier == 3:
            cmds  += _cmds_tier3_block(src_ip)
            action += "+HARD_BLOCK"

        # 7. Execute and audit log each command
        all_ok = True
        for cmd in cmds:
            ok = _run(cmd)
            if not ok:
                all_ok = False
            _write_csv({
                "timestamp":    datetime.now().isoformat(),
                "src_ip":       src_ip,
                "attack_type":  attack_label,
                "family":       family,
                "fusion_score": round(fusion_score, 3),
                "tier":         tier,
                "action":       action,
                "command":      " ".join(cmd),
                "status":       "OK" if ok else "FAILED",
            })

        # 8. Register state for auto-expiry
        _state.add(src_ip, tier, family, duration, cmds)

        # ------------------------------------------------------------------
        # READABLE MITIGATION ALERT — printed to stdout so you can see it
        # ------------------------------------------------------------------
        status_str = "SUCCESS" if all_ok else "PARTIAL (some commands failed)"
        cmd_strs = [" ".join(c) for c in cmds]

        print(flush=True)
        print("=" * 65, flush=True)
        print(f"  !! ATTACK DETECTED & MITIGATION APPLIED !!", flush=True)
        print("=" * 65, flush=True)
        print(f"  SPECIFIC ATTACK : {attack_label}", flush=True)
        print(f"  Attack Family   : {family}", flush=True)
        print("-" * 65, flush=True)
        print(f"  Time            : {datetime.now().strftime('%H:%M:%S')}", flush=True)
        print(f"  Attacker IP     : {src_ip}", flush=True)
        print(f"  Threat Score    : {fusion_score:.3f} / 1.000", flush=True)
        print(f"  Mitigation Tier : Tier {tier} — {tier_name}", flush=True)
        print(f"  Action Applied  : {action}", flush=True)
        print(f"  Auto-Expires In : {duration} seconds", flush=True)
        print(f"  Firewall Status : {status_str}", flush=True)
        print("=" * 65, flush=True)
        print(flush=True)

        log.warning(
            f"[MITIGATE] Tier-{tier} | {src_ip} | {family} | "
            f"score={fusion_score:.3f} | action={action} | "
            f"expires={duration}s | {status_str}"
        )

        return {
            "mitigated": True,
            "status": "applied" if all_ok else "partial",
            "family": family,
            "action": action,
            "tier": tier,
            "tier_name": tier_name,
            "expires_in": duration,
            "commands": cmd_strs,
            "summary": (
                f"{attack_label} from {src_ip} → Tier {tier} {tier_name}: {action} "
                f"({status_str}, expires {duration}s)"
            ),
        }

    finally:
        ip_lock.release()


# =============================================================================
# AUTO-EXPIRY / CLEANUP ENGINE
# =============================================================================

def _cleanup_ip(ip: str) -> None:
    """
    Remove all iptables rules for the given IP by inverting
    each stored -A command to a -D (delete) command.
    """
    with _state._lock:
        rule = _state._rules.get(ip)
    if rule is None:
        return
    for cmd in rule.get("commands", []):
        _run(_invert(cmd))
    _state.remove(ip)
    print(flush=True)
    print("-" * 65, flush=True)
    print(f"  [AUTO-UNBLOCK] {datetime.now().strftime('%H:%M:%S')}", flush=True)
    print(f"  IP {ip} mitigation expired (Tier {rule['tier']}, {rule['family']}).", flush=True)
    print(f"  Firewall rules for this IP have been REMOVED.", flush=True)
    print("-" * 65, flush=True)
    print(flush=True)
    log.info(
        f"[UNBLOCK] {ip} -- rules removed "
        f"(tier={rule['tier']}, family={rule['family']})"
    )


def _cleanup_loop() -> None:
    """
    Daemon thread: checks for expired rules every CLEANUP_INTERVAL_SEC.
    When a rule expires it inverts (deletes) the iptables entries and
    removes the IP from the state tracker, auto-unblocking the IP.
    This prevents false-positive IPs from being permanently blocked.
    """
    log.info(f"[CLEANUP] Auto-expiry thread started (interval={CLEANUP_INTERVAL_SEC}s).")
    while True:
        time.sleep(CLEANUP_INTERVAL_SEC)
        for ip, rule in _state.get_expired():
            log.info(
                f"[EXPIRE] {ip} rule expired "
                f"(tier={rule['tier']}, family={rule['family']}) -- removing."
            )
            _cleanup_ip(ip)


# =============================================================================
# MODULE INITIALISATION -- start cleanup thread on import
# =============================================================================
_cleanup_thread = threading.Thread(
    target=_cleanup_loop, daemon=True, name="MitigationCleanup"
)
_cleanup_thread.start()

log.info(
    "[INIT] Mitigation Engine v1 loaded | "
    f"DRY_RUN={DRY_RUN} | "
    f"block={BLOCK_DURATION_SEC}s | throttle={THROTTLE_DURATION_SEC}s | "
    f"whitelist={len(WHITELIST_IPS)} IPs"
)
