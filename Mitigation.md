Yes — the **Markdown table is causing the alignment problem when copied into Word/Google Docs**.

Use this **plain-text format** instead. It will copy cleanly and retain a proper structure:

---

# PHASE 7 – Automated Mitigation

## 7.1 Type-Specific Mitigation (When Attack Type is Known)

**1. SYN Flood**

Mitigation Action: Enable SYN Cookies  
Tool Used: `sysctl net.ipv4.tcp_syncookies=1`  
Why This Specifically: Validates connection requests without allocating resources until the handshake completes.

**2. UDP Flood**

Mitigation Action: Rate-limit UDP from source IP  
Tool Used: `iptables -m hashlimit`  
Why This Specifically: Drops UDP packets above the threshold while allowing DNS responses through.

**3. ICMP Flood**

Mitigation Action: Block all ICMP from source  
Tool Used: `iptables -p icmp -j DROP`  
Why This Specifically: ICMP flood has no legitimate high-rate use case, making it safe to block entirely.

**4. HTTP Flood**

Mitigation Action: Connection limiting per IP  
Tool Used: `iptables -m connlimit --connlimit-above 50`  
Why This Specifically: Limits concurrent HTTP connections per source to 50.

**5. SlowLoris**

Mitigation Action: Timeout short connections  
Tool Used: Set Apache/Nginx minimum request rate  
Why This Specifically: Forces incomplete HTTP requests to close after 15 seconds.

**6. DNS Amplification**

Mitigation Action: Filter outbound large DNS responses  
Tool Used: iptables UDP port 53 size limit  
Why This Specifically: Blocks responses over 512 bytes that indicate amplification.

---

## 7.2 Type-Agnostic Mitigation (When Attack Type is Unknown – Novel Attacks)

When Isolation Forest scores high (above 0.8), but XGBoost confidence is low (below 0.6), and DPI finds no match, the fusion score falls in the ALERT tier (0.4–0.69). You do not know what the attack is, but you must still protect the system.

**1. Rate Limiting**

What It Does: Limits the source IP to 20% of normal bandwidth using the token bucket algorithm.  
Why It Is Safe for Unknown Attacks: Works against any flood regardless of protocol or payload content.

**2. Connection Capping**

What It Does: Caps concurrent connections from the source IP at 20.  
Why It Is Safe for Unknown Attacks: Works against SlowLoris, HTTP floods, and other connection-exhaustion attacks.

**3. Traffic Logging**

What It Does: Logs the full 5-tuple and all 45 features to a file.  
Why It Is Safe for Unknown Attacks: Enables human review and future model retraining.

**4. Quarantine**

What It Does: Routes suspicious traffic through a separate low-priority queue.  
Why It Is Safe for Unknown Attacks: Does not block traffic entirely; it reduces damage while preserving some connectivity.

**5. Alert Dashboard**

What It Does: Triggers a visual alert on the monitoring dashboard.  
Why It Is Safe for Unknown Attacks: Allows a human operator to investigate and manually block the traffic if needed.