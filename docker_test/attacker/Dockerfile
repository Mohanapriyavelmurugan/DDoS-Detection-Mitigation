# ── Attacker Container ─────────────────────────────────────────────────
# Has all the attack tools pre-installed
# ───────────────────────────────────────────────────────────────────────
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    hping3 \
    nmap \
    python3 \
    python3-pip \
    iputils-ping \
    net-tools \
    curl \
    wget \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --default-timeout=1000 --no-cache-dir scapy

COPY attack_scripts/ ./attack_scripts/

# Keep container running so we can exec into it
CMD ["bash", "-c", "echo 'Attacker ready. Use: docker exec -it ddos_attacker bash' && tail -f /dev/null"]
