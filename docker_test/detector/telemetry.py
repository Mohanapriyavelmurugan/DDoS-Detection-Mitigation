"""
Lightweight telemetry bridge — posts detection + mitigation events to the dashboard
and appends NDJSON to /app/logs/detections.ndjson for history durability.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional

LOG_PATH = os.environ.get("DETECTION_LOG", "/app/logs/detections.ndjson")
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://172.20.0.30:5000").rstrip("/")
DEPLOY_ENV = os.environ.get("DEPLOY_ENV", os.environ.get("MEC_MODE", "LOCAL"))
if str(DEPLOY_ENV).lower() in ("1", "true", "yes"):
    DEPLOY_ENV = "MEC"
elif str(DEPLOY_ENV).upper() not in ("LOCAL", "MEC"):
    DEPLOY_ENV = "LOCAL"

_lock = threading.Lock()


def _tier(score: float) -> str:
    if score >= 0.90:
        return "T3-BLOCK"
    if score >= 0.70:
        return "T2-QUARANTINE"
    if score >= 0.55:
        return "T1-THROTTLE"
    return "T0-MONITOR"


def _layer_name(layer: int) -> str:
    return {1: "L1-IF-VETO", 2: "L2-DISAGREE", 3: "L3-FUSION"}.get(int(layer), "L3-FUSION")


def build_event(
    result: Dict[str, Any],
    *,
    flow_id: int = 0,
    src_ip: str = "",
    dst_ip: str = "172.20.0.10",
    proto: str = "TCP",
    packets: int = 0,
    rate: float = 0.0,
    latency_ms: Optional[Dict[str, float]] = None,
    mitigation_action: str = "",
    mitigation: Optional[Dict[str, Any]] = None,
    true_label: str = "",
    notify_phase: str = "",
) -> Dict[str, Any]:
    score = float(result.get("fusion_score", 0.0))
    is_attack = bool(result.get("is_attack", score >= 0.61))
    label = str(result.get("xgb_class", "UNKNOWN"))
    now = datetime.now(timezone.utc).isoformat()
    mit = mitigation or {}

    lat = latency_ms or {}
    action = mitigation_action or mit.get("action") or ""
    phase = (notify_phase or "").strip().lower()
    if not phase:
        if mit.get("mitigated") or mit.get("status") in ("applied", "partial", "already_active", "monitor"):
            phase = "mitigated"
        elif is_attack:
            phase = "detected"
        else:
            phase = "benign"
    if is_attack and not action and phase == "detected":
        action = "PENDING"

    return {
        "ts": now,
        "env": "MEC" if os.environ.get("MEC_MODE", "").lower() in ("1", "true") else DEPLOY_ENV,
        "flow_id": int(flow_id),
        "src_ip": src_ip or "unknown",
        "dst_ip": dst_ip,
        "proto": proto,
        "packets": int(packets),
        "rate": round(float(rate), 2),
        "xgb_score": float(result.get("xgb_prob", 0.0)),
        "xgb_conf_pct": float(result.get("xgb_conf_pct", 0.0)),
        "if_score": float(result.get("if_prob", 0.0)),
        "dpi_score": float(result.get("dpi_score", 0.0)),
        "fusion_score": score,
        "layer": int(result.get("layer_used", 3)),
        "layer_name": _layer_name(result.get("layer_used", 3)),
        "conf_mode": str(result.get("conf_mode", "")),
        "label": label,
        "decision": "ATTACK" if is_attack else "BENIGN",
        "tier": mit.get("tier_name") or (_tier(score) if is_attack else ""),
        "notify_phase": phase,
        "mitigation_action": action,
        "mitigated": bool(mit.get("mitigated", False)),
        "mitigation_status": str(mit.get("status", "")),
        "mitigation_family": str(mit.get("family", "")),
        "mitigation_tier": int(mit.get("tier", 0) or 0),
        "mitigation_tier_name": str(mit.get("tier_name", "")),
        "mitigation_expires_in": int(mit.get("expires_in", 0) or 0),
        "mitigation_commands": list(mit.get("commands") or []),
        "mitigation_summary": str(mit.get("summary", "")),
        "was_overridden": bool(result.get("was_overridden", False)),
        "override_reason": str(result.get("override_reason", "")),
        "true_label": true_label,
        "latency_ms": {
            "feature": float(lat.get("feature", 0.0)),
            "inference": float(lat.get("inference", 0.0)),
            "mitigation": float(lat.get("mitigation", 0.0)),
            "total": float(
                lat.get(
                    "total",
                    float(lat.get("feature", 0.0))
                    + float(lat.get("inference", 0.0))
                    + float(lat.get("mitigation", 0.0)),
                )
            ),
        },
    }


def _append_log(event: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
        with _lock:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[telemetry] log write failed: {exc}", flush=True)


def _post(event: Dict[str, Any]) -> None:
    url = f"{DASHBOARD_URL}/api/events"
    data = json.dumps(event).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            resp.read()
    except Exception:
        pass


def emit(event: Dict[str, Any]) -> None:
    _append_log(event)

    def _worker():
        _post(event)

    threading.Thread(target=_worker, daemon=True).start()


def emit_from_result(result: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    event = build_event(result, **kwargs)
    emit(event)
    return event


def _post_heartbeat() -> None:
    url = f"{DASHBOARD_URL}/api/heartbeat"
    req = urllib.request.Request(url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            resp.read()
    except Exception:
        pass


def start_heartbeat(interval_sec: float = 20.0) -> None:
    """Keep dashboard 'Detector Online' while the sniff loop is running."""

    def _loop():
        while True:
            _post_heartbeat()
            time.sleep(interval_sec)

    threading.Thread(target=_loop, daemon=True, name="dashboard-heartbeat").start()
