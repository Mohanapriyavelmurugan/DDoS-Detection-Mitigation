# ══════════════════════════════════════════════════════════════════════
# DETECTOR — Dual-Mode Image (Dev + MEC Production)
# Phase 6 Hybrid DDoS Detection Engine
# ══════════════════════════════════════════════════════════════════════
# DEV_MODE=true  (default) — Models NOT baked in; host volume mount used.
#                             Fast builds (~minutes), no C:\ disk pressure.
# DEV_MODE=false            — Models baked inside image; fully self-contained
#                             for MEC deployment (no host volume needed).
#
# Local dev   : docker-compose up --build                (uses DEV_MODE=true)
# MEC deploy  : docker-compose -f docker-compose.yml \
#               -f docker-compose.mec.yml up --build     (uses DEV_MODE=false)
# ══════════════════════════════════════════════════════════════════════

# ── Build argument (override only for MEC production builds) ───────────
ARG DEV_MODE=true

# ══════════════════════════════════════════════════════════════════════
# STAGE 1 — base: Install OS tools + Python deps + scripts (always runs)
# ══════════════════════════════════════════════════════════════════════
FROM python:3.10-slim AS base

LABEL maintainer="DDoS Detection Project"
LABEL version="6.0"
LABEL description="Phase 6 Confidence-Adaptive 3-Layer Fusion DDoS Detector"
LABEL architecture="x86_64"

WORKDIR /app

# ── Install OS-level packet capture tools ─────────────────────────────
# libpcap-dev : raw packet capture library (required by Scapy)
# gcc         : compiles C extensions for some Python packages
# tcpdump     : packet debugging tool
# net-tools   : ifconfig, netstat — for interface inspection
# iputils-ping: ping utility for connectivity testing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev \
    gcc \
    tcpdump \
    net-tools \
    iputils-ping \
    curl \
    iptables \
    procps \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python ML dependencies ────────────────────────────────────
# Versions are PINNED to exactly match training environment
COPY requirements.txt ./requirements.txt
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# ── Copy detection scripts ─────────────────────────────────────────────
COPY realtime_demo.py     ./realtime_demo.py
COPY verify_deployment.py ./verify_deployment.py
COPY telemetry.py         ./telemetry.py
COPY mitigation_engine.py ./mitigation_engine.py

# ══════════════════════════════════════════════════════════════════════
# STAGE 2a — prod-models: Bake models in for MEC (only used when DEV_MODE=false)
# ══════════════════════════════════════════════════════════════════════
FROM base AS prod-models
COPY models_v6/ ./models_v6/
RUN python verify_deployment.py

# ══════════════════════════════════════════════════════════════════════
# STAGE 2b — dev-only: Placeholder stage for local development
# Models are volume-mounted from host, not baked in.
# ══════════════════════════════════════════════════════════════════════
FROM base AS dev-only
RUN mkdir -p /app/models_v6

# ══════════════════════════════════════════════════════════════════════
# STAGE 3 — final: Selects dev-only or prod-models via DEV_MODE arg.
# Docker BuildKit resolves this at build time.
# ══════════════════════════════════════════════════════════════════════
ARG DEV_MODE=true
FROM dev-only AS final-true
FROM prod-models AS final-false
FROM final-${DEV_MODE} AS final

# ── Health check ───────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import joblib, json; \
        joblib.load('/app/models_v6/xgboost_model.pkl'); \
        json.load(open('/app/models_v6/optimal_threshold.json')); \
        print('OK')" || exit 1

# ── Default: simulation mode (safe — no raw packet access needed)
# For live capture on MEC, override with: --live --iface eth0
CMD ["python", "realtime_demo.py", "--samples", "100"]
