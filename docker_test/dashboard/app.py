"""
Interactive DDoS Monitoring Dashboard
Serves Home · Live Monitor · Attack History for LOCAL Docker and MEC.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
LOG_PATH = Path(os.environ.get("DETECTION_LOG", "/app/logs/detections.ndjson"))
DB_PATH = DATA_DIR / "history.db"
DEPLOY_ENV = os.environ.get("DEPLOY_ENV", "LOCAL").upper()
MEC_IP = os.environ.get("MEC_IP", "192.168.10.12")
STATIC_DIR = Path(__file__).resolve().parent / "static"
SEED_PATH = Path(__file__).resolve().parent / "measurements_seed.json"
MEASUREMENTS_PATH = DATA_DIR / "measurements.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Fresh start: ignore any pre-dashboard log file left on the shared volume
if LOG_PATH.exists():
    try:
        LOG_PATH.unlink()
    except Exception:
        pass

# ── In-memory live state ──────────────────────────────────────────────
_clients: Set[WebSocket] = set()
_lock = asyncio.Lock()
_stats = {
    "started_at": datetime.now(timezone.utc).isoformat(),
    "total_events": 0,
    "attacks": 0,
    "benign": 0,
    "last_event_ts": None,
    "last_attack_ts": None,
    "detector_seen_at": None,
    "avg_latency_ms": 0.0,
    "avg_feature_ms": 0.0,
    "avg_inference_ms": 0.0,
    "avg_mitigation_ms": 0.0,  # detect → mitigate time
    "latency_samples": 0,
    "mitigation_samples": 0,
    "throughput_fps": 0.0,
    "window_events": 0,
    "window_started": time.time(),
}
_recent: List[Dict[str, Any]] = []
_latency_series: List[Dict[str, Any]] = []
_throughput_series: List[Dict[str, Any]] = []
RECENT_MAX = 200
SERIES_MAX = 60


class DetectionEvent(BaseModel):
    ts: Optional[str] = None
    env: Optional[str] = None
    flow_id: int = 0
    src_ip: str = "unknown"
    dst_ip: str = ""
    proto: str = "TCP"
    packets: int = 0
    rate: float = 0.0
    xgb_score: float = 0.0
    xgb_conf_pct: float = 0.0
    if_score: float = 0.0
    dpi_score: float = 0.0
    fusion_score: float = 0.0
    layer: int = 3
    layer_name: str = "L3-FUSION"
    conf_mode: str = ""
    label: str = "UNKNOWN"
    decision: str = "BENIGN"
    tier: str = ""
    mitigation_action: str = ""
    mitigated: bool = False
    mitigation_status: str = ""
    mitigation_family: str = ""
    mitigation_tier: int = 0
    mitigation_tier_name: str = ""
    mitigation_expires_in: int = 0
    mitigation_commands: List[str] = Field(default_factory=list)
    mitigation_summary: str = ""
    notify_phase: str = ""
    was_overridden: bool = False
    override_reason: str = ""
    true_label: str = ""
    latency_ms: Dict[str, float] = Field(default_factory=dict)


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                env TEXT,
                flow_id INTEGER,
                src_ip TEXT,
                dst_ip TEXT,
                proto TEXT,
                packets INTEGER,
                rate REAL,
                xgb_score REAL,
                xgb_conf_pct REAL,
                if_score REAL,
                dpi_score REAL,
                fusion_score REAL,
                layer INTEGER,
                layer_name TEXT,
                conf_mode TEXT,
                label TEXT,
                decision TEXT,
                tier TEXT,
                mitigation_action TEXT,
                was_overridden INTEGER,
                override_reason TEXT,
                true_label TEXT,
                latency_total REAL,
                payload TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_src ON events(src_ip)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_decision ON events(decision)"
        )
        conn.commit()


def _event_row_values(event: Dict[str, Any]):
    lat = event.get("latency_ms") or {}
    return (
        event.get("ts"),
        event.get("env"),
        event.get("flow_id", 0),
        event.get("src_ip"),
        event.get("dst_ip"),
        event.get("proto"),
        event.get("packets", 0),
        event.get("rate", 0.0),
        event.get("xgb_score", 0.0),
        event.get("xgb_conf_pct", 0.0),
        event.get("if_score", 0.0),
        event.get("dpi_score", 0.0),
        event.get("fusion_score", 0.0),
        event.get("layer", 3),
        event.get("layer_name"),
        event.get("conf_mode"),
        event.get("label"),
        event.get("decision"),
        event.get("tier"),
        event.get("mitigation_action"),
        1 if event.get("was_overridden") else 0,
        event.get("override_reason"),
        event.get("true_label"),
        float(lat.get("total", 0.0) or 0.0),
        json.dumps(event),
    )


def _is_pending_action(action: Any) -> bool:
    return str(action or "").strip().upper() in ("", "PENDING", "MONITOR")


def _payload_phase(payload_text: Any) -> str:
    try:
        return str(json.loads(payload_text or "{}").get("notify_phase") or "").lower()
    except Exception:
        return ""


def _find_attack_row(conn: sqlite3.Connection, event: Dict[str, Any]):
    """Find a recent DB row for this attack (same flow, else same label within ~30s)."""
    src = event.get("src_ip")
    label = event.get("label")
    now = event.get("ts") or ""
    try:
        fid = int(event.get("flow_id") or 0)
    except Exception:
        fid = 0

    row = conn.execute(
        """
        SELECT id, ts, mitigation_action, payload FROM events
        WHERE src_ip = ? AND flow_id = ? AND decision = 'ATTACK'
        ORDER BY id DESC LIMIT 1
        """,
        (src, fid),
    ).fetchone()
    # Only reuse the flow row if it belongs to this burst (not an old session)
    if row and _within_seconds(row["ts"], now, 60):
        return row

    # Fallback: same source + attack family in the last ~30s (flow_id may differ
    # between detect and mitigate notify phases).
    if src and label:
        recent = conn.execute(
            """
            SELECT id, ts, mitigation_action, payload FROM events
            WHERE src_ip = ? AND label = ? AND decision = 'ATTACK'
            ORDER BY id DESC LIMIT 8
            """,
            (src, label),
        ).fetchall()
        for r in recent:
            if _within_seconds(r["ts"], now, 30):
                return r
    return None


def _within_seconds(ts_a: Any, ts_b: Any, window: float) -> bool:
    try:
        a = datetime.fromisoformat(str(ts_a).replace("Z", "+00:00"))
        b = datetime.fromisoformat(str(ts_b).replace("Z", "+00:00"))
        return abs((a - b).total_seconds()) <= window
    except Exception:
        return False


def _update_event_row(conn: sqlite3.Connection, row_id: int, event: Dict[str, Any], *, keep_ts: Optional[str] = None) -> int:
    merged = dict(event)
    if keep_ts:
        # Preserve first-seen (detect) timestamp; record mitigate time in payload
        merged["detected_ts"] = keep_ts
        merged["mitigated_ts"] = event.get("ts")
        merged["ts"] = keep_ts
    vals = _event_row_values(merged)
    conn.execute(
        """
        UPDATE events SET
            ts=?, env=?, flow_id=?, src_ip=?, dst_ip=?, proto=?, packets=?, rate=?,
            xgb_score=?, xgb_conf_pct=?, if_score=?, dpi_score=?, fusion_score=?,
            layer=?, layer_name=?, conf_mode=?, label=?, decision=?, tier=?,
            mitigation_action=?, was_overridden=?, override_reason=?, true_label=?,
            latency_total=?, payload=?
        WHERE id=?
        """,
        vals + (int(row_id),),
    )
    return int(row_id)


def persist_event(event: Dict[str, Any]) -> int:
    """Insert a new event, or merge detect→mitigate into one history row."""
    phase = str(event.get("notify_phase") or "").lower()
    decision = str(event.get("decision") or "").upper()
    status = str(event.get("mitigation_status") or "").lower()
    action = str(event.get("mitigation_action") or "").upper()
    # Residual "already mitigated" updates must not create duplicate history rows —
    # refresh the existing row payload so live/history stay current.
    if status == "already_active" or action == "ALREADY_MITIGATED":
        with _db() as conn:
            existing = _find_attack_row(conn, event)
            if existing:
                rid = _update_event_row(
                    conn, int(existing["id"]), event, keep_ts=existing["ts"]
                )
                conn.commit()
                return rid
            return 0
    with _db() as conn:
        if decision == "ATTACK":
            existing = _find_attack_row(conn, event)
            if existing:
                exist_phase = _payload_phase(existing["payload"])
                exist_pending = _is_pending_action(existing["mitigation_action"])

                # Late "detected" after mitigation already stored → keep mitigated row
                if phase == "detected" and (not exist_pending or exist_phase == "mitigated"):
                    conn.commit()
                    return int(existing["id"])

                # Mitigation completes an open detect row → UPDATE (keep detect time)
                if phase == "mitigated" and (exist_pending or exist_phase in ("", "detected")):
                    keep_ts = existing["ts"] if exist_pending or exist_phase == "detected" else None
                    rid = _update_event_row(conn, int(existing["id"]), event, keep_ts=keep_ts)
                    conn.commit()
                    return rid

                # Duplicate mitigated for same attack → refresh same row
                if phase == "mitigated" and not exist_pending:
                    rid = _update_event_row(conn, int(existing["id"]), event, keep_ts=existing["ts"])
                    conn.commit()
                    return rid

                # Duplicate detected while still pending → refresh same row
                if phase == "detected" and exist_pending:
                    rid = _update_event_row(conn, int(existing["id"]), event, keep_ts=existing["ts"])
                    conn.commit()
                    return rid

        cur = conn.execute(
            """
            INSERT INTO events (
                ts, env, flow_id, src_ip, dst_ip, proto, packets, rate,
                xgb_score, xgb_conf_pct, if_score, dpi_score, fusion_score,
                layer, layer_name, conf_mode, label, decision, tier,
                mitigation_action, was_overridden, override_reason, true_label,
                latency_total, payload
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            _event_row_values(event),
        )
        conn.commit()
        return int(cur.lastrowid)


def cleanup_duplicate_attack_rows() -> int:
    """Merge leftover PENDING + applied twins from earlier double-ingest bugs."""
    removed = 0
    with _db() as conn:
        # Remove residual "already mitigated" spam rows
        cur = conn.execute(
            """
            DELETE FROM events
            WHERE mitigation_action = 'ALREADY_MITIGATED'
               OR IFNULL(json_extract(payload, '$.mitigation_status'), '') = 'already_active'
            """
        )
        removed += int(cur.rowcount or 0)
        conn.commit()

    with _db() as conn:
        rows = conn.execute(
            """
            SELECT id, ts, src_ip, label, mitigation_action, payload
            FROM events WHERE decision = 'ATTACK'
            ORDER BY id ASC
            """
        ).fetchall()

        delete_ids: set = set()
        n = len(rows)
        for i in range(n):
            if rows[i]["id"] in delete_ids:
                continue
            a = rows[i]
            a_pending = _is_pending_action(a["mitigation_action"])
            a_phase = _payload_phase(a["payload"])
            for j in range(i + 1, min(i + 5, n)):
                if rows[j]["id"] in delete_ids:
                    continue
                b = rows[j]
                if str(a["src_ip"]) != str(b["src_ip"]) or str(a["label"]) != str(b["label"]):
                    continue
                if not _within_seconds(a["ts"], b["ts"], 15):
                    continue
                b_pending = _is_pending_action(b["mitigation_action"])
                b_phase = _payload_phase(b["payload"])
                is_twin = (a_pending != b_pending) or (
                    {"detected", "mitigated"} <= {a_phase, b_phase}
                )
                if not is_twin:
                    continue

                # Keep mitigated / richer row; preserve earliest timestamp
                if a_pending and not b_pending:
                    keep, drop = b, a
                elif b_pending and not a_pending:
                    keep, drop = a, b
                elif a_phase == "mitigated" and b_phase != "mitigated":
                    keep, drop = a, b
                elif b_phase == "mitigated" and a_phase != "mitigated":
                    keep, drop = b, a
                else:
                    keep, drop = (b, a) if int(b["id"]) > int(a["id"]) else (a, b)

                keep_ts = a["ts"] if str(a["ts"]) <= str(b["ts"]) else b["ts"]
                try:
                    payload = json.loads(keep["payload"] or "{}")
                except Exception:
                    payload = {}
                if not _is_pending_action(keep["mitigation_action"]):
                    payload["notify_phase"] = "mitigated"
                payload["detected_ts"] = keep_ts
                if str(keep["ts"]) != str(keep_ts):
                    payload["mitigated_ts"] = keep["ts"]
                payload["ts"] = keep_ts
                conn.execute(
                    "UPDATE events SET ts = ?, payload = ? WHERE id = ?",
                    (keep_ts, json.dumps(payload), int(keep["id"])),
                )
                delete_ids.add(int(drop["id"]))
                break

        for did in sorted(delete_ids):
            conn.execute("DELETE FROM events WHERE id = ?", (did,))
            removed += 1
        conn.commit()
    if removed:
        print(f"[dashboard] cleaned {removed} duplicate attack history row(s)", flush=True)
    return removed


def normalize_event(raw: Dict[str, Any]) -> Dict[str, Any]:
    event = dict(raw)
    if not event.get("ts"):
        event["ts"] = datetime.now(timezone.utc).isoformat()
    if not event.get("env"):
        event["env"] = DEPLOY_ENV
    decision = str(event.get("decision", "BENIGN")).upper()
    if decision not in ("ATTACK", "BENIGN"):
        decision = "ATTACK" if float(event.get("fusion_score", 0)) >= 0.61 else "BENIGN"
    event["decision"] = decision
    lat = event.get("latency_ms") or {}
    if not isinstance(lat, dict):
        lat = {}
    event["latency_ms"] = {
        "feature": float(lat.get("feature", 0) or 0),
        "inference": float(lat.get("inference", 0) or 0),
        "mitigation": float(lat.get("mitigation", 0) or 0),
        "total": float(lat.get("total", 0) or 0),
    }
    if event["latency_ms"]["total"] <= 0:
        event["latency_ms"]["total"] = (
            event["latency_ms"]["feature"]
            + event["latency_ms"]["inference"]
            + event["latency_ms"]["mitigation"]
        )
    return event


async def broadcast(message: Dict[str, Any]) -> None:
    dead = []
    payload = json.dumps(message)
    async with _lock:
        clients = list(_clients)
    for ws in clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    if dead:
        async with _lock:
            for ws in dead:
                _clients.discard(ws)


def update_stats(event: Dict[str, Any], *, is_update: bool = False) -> None:
    phase = str(event.get("notify_phase") or "").lower()
    # Mitigation updates enrich an existing attack — do not count a second attack
    if not is_update:
        _stats["total_events"] += 1
        if event["decision"] == "ATTACK" and phase != "mitigated":
            _stats["attacks"] += 1
            _stats["last_attack_ts"] = event["ts"]
        elif event["decision"] != "ATTACK":
            _stats["benign"] += 1

    _stats["last_event_ts"] = event["ts"]
    _stats["detector_seen_at"] = datetime.now(timezone.utc).isoformat()
    if event["decision"] == "ATTACK":
        _stats["last_attack_ts"] = event["ts"]

    lat = event.get("latency_ms") or {}
    feat = float(lat.get("feature", 0) or 0)
    infer = float(lat.get("inference", 0) or 0)
    mit = float(lat.get("mitigation", 0) or 0)
    total_lat = float(lat.get("total", 0) or 0)
    if total_lat <= 0:
        total_lat = feat + infer + mit

    if total_lat > 0 or feat > 0 or infer > 0:
        n = _stats["latency_samples"]
        def _avg(key, value):
            prev = _stats[key]
            _stats[key] = round((prev * n + value) / (n + 1), 3)

        _avg("avg_latency_ms", total_lat)
        _avg("avg_feature_ms", feat)
        _avg("avg_inference_ms", infer)
        _stats["latency_samples"] = n + 1
        _latency_series.append(
            {
                "ts": event["ts"],
                "total": total_lat,
                "feature": feat,
                "inference": infer,
                "mitigation": mit,
                "detect_to_mitigate": mit,
            }
        )
        if len(_latency_series) > SERIES_MAX:
            del _latency_series[0 : len(_latency_series) - SERIES_MAX]

    # Detect → mitigate is only meaningful after mitigation runs
    if mit > 0 or str(event.get("notify_phase") or "").lower() == "mitigated":
        m = _stats["mitigation_samples"]
        prev = _stats["avg_mitigation_ms"]
        _stats["avg_mitigation_ms"] = round((prev * m + mit) / (m + 1), 3) if (m + 1) else mit
        _stats["mitigation_samples"] = m + 1

    if not is_update:
        _stats["window_events"] += 1
        elapsed = max(time.time() - _stats["window_started"], 1.0)
        if elapsed >= 5.0:
            _stats["throughput_fps"] = round(_stats["window_events"] / elapsed, 2)
            _throughput_series.append(
                {"ts": event["ts"], "fps": _stats["throughput_fps"]}
            )
            if len(_throughput_series) > SERIES_MAX:
                del _throughput_series[0 : len(_throughput_series) - SERIES_MAX]
            _stats["window_events"] = 0
            _stats["window_started"] = time.time()

    # Always keep ONE live row per attack flow (src_ip + flow_id).
    # Prefer mitigated over detected when both arrive (HTTP + log-tail race).
    _upsert_recent(event)
    if len(_recent) > RECENT_MAX:
        del _recent[RECENT_MAX:]


def _same_attack_flow(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if str(a.get("src_ip") or "") != str(b.get("src_ip") or ""):
        return False
    if str(a.get("decision") or "").upper() != "ATTACK":
        return False
    if str(b.get("decision") or "").upper() != "ATTACK":
        return False
    fa = a.get("flow_id", 0)
    fb = b.get("flow_id", 0)
    try:
        return int(fa or 0) == int(fb or 0)
    except Exception:
        return str(fa) == str(fb)


def _phase_rank(ev: Dict[str, Any]) -> int:
    phase = str(ev.get("notify_phase") or "").lower()
    if phase == "mitigated" or ev.get("mitigation_action") not in ("", None, "PENDING"):
        return 2
    if phase == "detected":
        return 1
    return 0


def _upsert_recent(event: Dict[str, Any]) -> None:
    """Insert or merge into _recent so one attack never appears twice."""
    if str(event.get("decision") or "").upper() == "ATTACK":
        for i, existing in enumerate(_recent):
            if _same_attack_flow(existing, event):
                # Keep richer mitigation state; never leave a stale "detected" twin
                if _phase_rank(event) >= _phase_rank(existing):
                    merged = {**existing, **event}
                    if _phase_rank(event) >= 2:
                        merged["notify_phase"] = "mitigated"
                    _recent[i] = merged
                # Move merged row to front
                row = _recent.pop(i)
                _recent.insert(0, row)
                return
    _recent.insert(0, event)


def _dedupe_recent_attacks(events: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Collapse detected+mitigated twins into one row per attack flow."""
    src = events if events is not None else _recent
    out: List[Dict[str, Any]] = []
    seen = set()
    for ev in src:
        if str(ev.get("decision") or "").upper() != "ATTACK":
            out.append(ev)
            continue
        try:
            fid = int(ev.get("flow_id") or 0)
        except Exception:
            fid = ev.get("flow_id") or 0
        key = (str(ev.get("src_ip") or ""), fid)
        if key in seen:
            continue
        seen.add(key)
        twins = [x for x in src if _same_attack_flow(x, ev)]
        best = sorted(twins, key=_phase_rank, reverse=True)[0]
        out.append(best)
    return out


def compact_recent() -> None:
    collapsed = _dedupe_recent_attacks(list(_recent))
    _recent.clear()
    _recent.extend(collapsed[:RECENT_MAX])


async def ingest_event(raw: Dict[str, Any]) -> Dict[str, Any]:
    event = normalize_event(raw)
    status = str(event.get("mitigation_status") or "").lower()
    action = str(event.get("mitigation_action") or "").upper()
    phase = str(event.get("notify_phase") or "").lower()
    # Residual "already mitigated" pings: refresh the same history/live row
    # and broadcast so the UI can toast — never insert a duplicate row.
    if status == "already_active" or action == "ALREADY_MITIGATED":
        event_id = persist_event(event)
        if not event_id:
            # No prior row (rare) — still push live toast without DB insert
            event["id"] = event.get("id") or f"active-{event.get('src_ip')}-{event.get('flow_id')}"
            _upsert_recent(event)
            await broadcast({"type": "event", "data": event})
            await broadcast({"type": "stats", "data": public_stats()})
            return event
        event["id"] = event_id
        update_stats(event, is_update=True)
        await broadcast({"type": "event", "data": event})
        await broadcast({"type": "stats", "data": public_stats()})
        return event
    event_id = persist_event(event)
    if not event_id:
        return event
    event["id"] = event_id
    is_update = phase == "mitigated"
    update_stats(event, is_update=is_update)
    await broadcast({"type": "event", "data": event})
    await broadcast({"type": "stats", "data": public_stats()})
    return event


def public_stats() -> Dict[str, Any]:
    detector_ok = False
    if _stats["detector_seen_at"]:
        try:
            seen = datetime.fromisoformat(_stats["detector_seen_at"])
            detector_ok = (datetime.now(timezone.utc) - seen).total_seconds() < 120
        except Exception:
            detector_ok = False
    total = _stats["total_events"]
    return {
        "env": DEPLOY_ENV,
        "mec_ip": MEC_IP,
        "started_at": _stats["started_at"],
        "total_events": total,
        "attacks": _stats["attacks"],
        "benign": _stats["benign"],
        "attack_rate_pct": round(_stats["attacks"] / total * 100, 1) if total else None,
        "last_event_ts": _stats["last_event_ts"],
        "last_attack_ts": _stats["last_attack_ts"],
        "detector_online": detector_ok,
        "avg_latency_ms": _stats["avg_latency_ms"] if _stats["latency_samples"] else None,
        "avg_feature_ms": _stats["avg_feature_ms"] if _stats["latency_samples"] else None,
        "avg_inference_ms": _stats["avg_inference_ms"] if _stats["latency_samples"] else None,
        "avg_mitigation_ms": _stats["avg_mitigation_ms"] if _stats["mitigation_samples"] else None,
        "latency_samples": _stats["latency_samples"],
        "mitigation_samples": _stats["mitigation_samples"],
        "throughput_fps": _stats["throughput_fps"] if _stats["throughput_fps"] else None,
        "recent_count": len(_recent),
    }


def tail_log_file() -> None:
    """Background: ingest any NDJSON lines written by the detector."""
    position = 0
    if LOG_PATH.exists():
        position = LOG_PATH.stat().st_size
    while True:
        try:
            if LOG_PATH.exists():
                size = LOG_PATH.stat().st_size
                if size < position:
                    position = 0
                if size > position:
                    with open(LOG_PATH, "r", encoding="utf-8") as f:
                        f.seek(position)
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                raw = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            # HTTP is primary. Skip log lines already covered by
                            # the same attack flow (ts alone is wrong after mitigate
                            # replaces the detected row with a new timestamp).
                            src = raw.get("src_ip")
                            ts = raw.get("ts")
                            fid = raw.get("flow_id")
                            phase = str(raw.get("notify_phase") or "").lower()
                            decision = str(raw.get("decision") or "").upper()
                            already = False
                            raw_label = str(raw.get("label") or "")
                            for r in _recent[:40]:
                                if r.get("ts") == ts and r.get("src_ip") == src:
                                    already = True
                                    break
                                if (
                                    decision == "ATTACK"
                                    and str(r.get("decision") or "").upper() == "ATTACK"
                                    and r.get("src_ip") == src
                                ):
                                    try:
                                        same_flow = int(r.get("flow_id") or 0) == int(fid or 0)
                                    except Exception:
                                        same_flow = str(r.get("flow_id")) == str(fid)
                                    same_label = raw_label and str(r.get("label") or "") == raw_label
                                    r_phase = str(r.get("notify_phase") or "").lower()
                                    r_action = str(r.get("mitigation_action") or "")
                                    r_mitigated = r_phase == "mitigated" or (
                                        r_action and r_action.upper() not in ("", "PENDING", "MONITOR")
                                    )
                                    # Same flow: collapse duplicate phases
                                    if same_flow:
                                        if phase == "detected" and r_phase in ("detected", "mitigated"):
                                            already = True
                                            break
                                        if phase == r_phase:
                                            already = True
                                            break
                                    # Same attack family shortly after: skip late detected
                                    # when mitigation already landed (flow_id may differ)
                                    if same_label and phase == "detected" and r_mitigated:
                                        if _within_seconds(r.get("ts"), ts, 30):
                                            already = True
                                            break
                            if already:
                                continue
                            asyncio.run_coroutine_threadsafe(
                                ingest_event(raw), _loop
                            )
                        position = f.tell()
        except Exception:
            pass
        time.sleep(1.0)


_loop: Optional[asyncio.AbstractEventLoop] = None


def rehydrate_stats_from_db() -> None:
    """Rebuild live KPI averages + recent feed from SQLite after restart."""
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT ts, env, flow_id, src_ip, dst_ip, proto, packets, rate, label, decision,
                   fusion_score, xgb_score, if_score, dpi_score, layer_name, tier,
                   mitigation_action, latency_total, payload
            FROM events
            ORDER BY id DESC
            LIMIT 200
            """
        ).fetchall()
        counts = conn.execute(
            """
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN decision = 'ATTACK' THEN 1 ELSE 0 END) AS attacks,
              SUM(CASE WHEN decision != 'ATTACK' THEN 1 ELSE 0 END) AS benign
            FROM events
            """
        ).fetchone()

    if counts and counts["total"]:
        _stats["total_events"] = int(counts["total"] or 0)
        _stats["attacks"] = int(counts["attacks"] or 0)
        _stats["benign"] = int(counts["benign"] or 0)

    feat_vals, infer_vals, mit_vals, total_vals = [], [], [], []
    recent = []
    for r in reversed(rows):
        detail = {}
        if r["payload"]:
            try:
                detail = json.loads(r["payload"])
            except Exception:
                detail = {}
        lat = detail.get("latency_ms") or {}
        feat = float(lat.get("feature", 0) or 0)
        infer = float(lat.get("inference", 0) or 0)
        mit = float(lat.get("mitigation", 0) or 0)
        total = float(lat.get("total", r["latency_total"] or 0) or 0)
        if total <= 0:
            total = feat + infer + mit
        if feat or infer or mit or total:
            feat_vals.append(feat)
            infer_vals.append(infer)
            total_vals.append(total)
            _latency_series.append(
                {
                    "ts": r["ts"],
                    "feature": feat,
                    "inference": infer,
                    "mitigation": mit,
                    "detect_to_mitigate": mit,
                    "total": total,
                }
            )
        if mit > 0:
            mit_vals.append(mit)

        event = {
            "id": f"db-{r['ts']}-{r['src_ip']}-{r['flow_id']}",
            "ts": r["ts"],
            "env": r["env"],
            "flow_id": r["flow_id"] if r["flow_id"] is not None else detail.get("flow_id", 0),
            "src_ip": r["src_ip"],
            "dst_ip": r["dst_ip"],
            "proto": r["proto"],
            "packets": r["packets"],
            "rate": r["rate"],
            "label": r["label"],
            "decision": r["decision"],
            "fusion_score": r["fusion_score"] if r["fusion_score"] is not None else detail.get("fusion_score"),
            "xgb_score": r["xgb_score"],
            "if_score": r["if_score"],
            "dpi_score": r["dpi_score"],
            "layer_name": r["layer_name"],
            "tier": r["tier"],
            "mitigation_action": r["mitigation_action"],
            "mitigation_status": detail.get("mitigation_status", ""),
            "mitigation_summary": detail.get("mitigation_summary", ""),
            "latency_ms": {
                "feature": feat,
                "inference": infer,
                "mitigation": mit,
                "total": total,
            },
            "notify_phase": detail.get("notify_phase", ""),
        }
        recent.append(event)

    if len(_latency_series) > SERIES_MAX:
        del _latency_series[0 : len(_latency_series) - SERIES_MAX]

    def _mean(vals):
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    if total_vals:
        _stats["avg_latency_ms"] = _mean(total_vals)
        _stats["avg_feature_ms"] = _mean(feat_vals)
        _stats["avg_inference_ms"] = _mean(infer_vals)
        _stats["latency_samples"] = len(total_vals)
    if mit_vals:
        _stats["avg_mitigation_ms"] = _mean(mit_vals)
        _stats["mitigation_samples"] = len(mit_vals)

    _recent.clear()
    _recent.extend(list(reversed(recent))[:RECENT_MAX])
    compact_recent()
    if recent:
        _stats["last_event_ts"] = recent[-1]["ts"]
        attacks = [e for e in recent if e.get("decision") == "ATTACK"]
        if attacks:
            _stats["last_attack_ts"] = attacks[-1]["ts"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop
    init_db()
    try:
        cleanup_duplicate_attack_rows()
    except Exception as exc:
        print(f"[dashboard] history cleanup failed: {exc}", flush=True)
    try:
        rehydrate_stats_from_db()
    except Exception as exc:
        print(f"[dashboard] rehydrate failed: {exc}", flush=True)
    _loop = asyncio.get_running_loop()
    t = threading.Thread(target=tail_log_file, daemon=True)
    t.start()
    yield


app = FastAPI(title="DDoS Detection Dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {"ok": True, "env": DEPLOY_ENV, "service": "dashboard"}


@app.post("/api/heartbeat")
async def heartbeat():
    """Detector keep-alive so the UI stays Online between attacks."""
    _stats["detector_seen_at"] = datetime.now(timezone.utc).isoformat()
    await broadcast({"type": "stats", "data": public_stats()})
    return {"ok": True}


@app.get("/api/stats")
async def stats():
    return public_stats()


@app.get("/api/live")
async def live(limit: int = Query(50, ge=1, le=200)):
    compact_recent()
    return {
        "events": _recent[:limit],
        "latency": _latency_series[-30:],
        "throughput": _throughput_series[-30:],
    }


@app.get("/api/history")
async def history(
    q: str = "",
    decision: str = "",
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    clauses = []
    params: List[Any] = []
    if q:
        clauses.append("(src_ip LIKE ? OR label LIKE ? OR dst_ip LIKE ? OR mitigation_action LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like, like, like])
    if decision and decision.upper() in ("ATTACK", "BENIGN"):
        clauses.append("decision = ?")
        params.append(decision.upper())
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.extend([limit, offset])
    with _db() as conn:
        rows = conn.execute(
            f"""
            SELECT id, ts, env, flow_id, src_ip, dst_ip, proto, packets, rate,
                   xgb_score, if_score, dpi_score, fusion_score, layer_name,
                   label, decision, tier, mitigation_action, latency_total, payload
            FROM events
            {where}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        count_params = params[:-2]
        total = conn.execute(
            f"SELECT COUNT(*) AS c FROM events {where}", count_params
        ).fetchone()["c"]
    items = []
    for r in rows:
        item = dict(r)
        if item.get("payload"):
            try:
                item["detail"] = json.loads(item["payload"])
            except Exception:
                item["detail"] = {}
        items.append(item)
    return {"total": total, "items": items}


@app.get("/api/history/export")
async def export_history(decision: str = ""):
    clauses = []
    params: List[Any] = []
    if decision and decision.upper() in ("ATTACK", "BENIGN"):
        clauses.append("decision = ?")
        params.append(decision.upper())
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with _db() as conn:
        rows = conn.execute(
            f"""
            SELECT ts, env, src_ip, dst_ip, proto, packets, rate, label,
                   decision, fusion_score, xgb_score, if_score, dpi_score,
                   layer_name, tier, mitigation_action, latency_total, payload
            FROM events {where}
            ORDER BY id DESC
            """,
            params,
        ).fetchall()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "ts", "env", "src_ip", "dst_ip", "proto", "packets", "rate", "label",
            "decision", "fusion_score", "xgb_score", "if_score", "dpi_score",
            "layer", "tier", "mitigation_action", "mitigation_status",
            "mitigation_summary", "latency_feature_ms", "latency_inference_ms",
            "latency_detect_to_mitigate_ms", "latency_total_ms",
        ]
    )
    for r in rows:
        detail = {}
        if r["payload"]:
            try:
                detail = json.loads(r["payload"])
            except Exception:
                detail = {}
        lat = detail.get("latency_ms") or {}
        writer.writerow(
            [
                r["ts"], r["env"], r["src_ip"], r["dst_ip"], r["proto"], r["packets"],
                r["rate"], r["label"], r["decision"], r["fusion_score"], r["xgb_score"],
                r["if_score"], r["dpi_score"], r["layer_name"], r["tier"],
                r["mitigation_action"],
                detail.get("mitigation_status", ""),
                detail.get("mitigation_summary", ""),
                lat.get("feature", ""),
                lat.get("inference", ""),
                lat.get("mitigation", ""),
                r["latency_total"] if r["latency_total"] is not None else lat.get("total", ""),
            ]
        )
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=attack_history.csv"},
    )


def _load_measurement_seed() -> Dict[str, Any]:
    for path in (MEASUREMENTS_PATH, SEED_PATH):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
    return {}


def _percentile(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return round(s[0], 3)
    idx = (len(s) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return round(s[lo] * (1 - frac) + s[hi] * frac, 3)


def _live_m1() -> Dict[str, Any]:
    feats = [float(p.get("feature") or 0) for p in _latency_series]
    infers = [float(p.get("inference") or 0) for p in _latency_series]
    mits = [float(p.get("mitigation") or 0) for p in _latency_series if float(p.get("mitigation") or 0) > 0]
    totals = [float(p.get("total") or 0) for p in _latency_series]

    def block(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {"mean": None, "median": None, "p95": None, "min": None, "max": None, "n": 0}
        return {
            "mean": round(sum(vals) / len(vals), 3),
            "median": _percentile(vals, 0.5),
            "p95": _percentile(vals, 0.95),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
            "n": len(vals),
        }

    stages = [
        {"id": "feature", "label": "Feature extraction", "ms": _stats["avg_feature_ms"] if _stats["latency_samples"] else 0},
        {"id": "inference", "label": "Model inference", "ms": _stats["avg_inference_ms"] if _stats["latency_samples"] else 0},
        {"id": "mitigation", "label": "Detect → mitigate", "ms": _stats["avg_mitigation_ms"] if _stats["mitigation_samples"] else 0},
    ]
    return {
        "source": "live detector pipeline",
        "stages": stages,
        "feature": block(feats),
        "inference": block(infers),
        "mitigation": block(mits),
        "total": block(totals),
        "series": _latency_series[-40:],
    }


def _live_m2() -> Dict[str, Any]:
    return {
        "source": "live session throughput",
        "current_fps": _stats["throughput_fps"] or 0,
        "series": _throughput_series[-40:],
    }


def _live_m3_session() -> Dict[str, Any]:
    """Session offload proxy from history: mitigated attacks vs packets seen pre-mitigation."""
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT packets, rate, mitigation_action, payload
            FROM events WHERE decision = 'ATTACK'
            ORDER BY id DESC LIMIT 200
            """
        ).fetchall()
    if not rows:
        return {"available": False, "attacks": 0}

    attacks = len(rows)
    mitigated = 0
    packets_pre = 0
    for r in rows:
        detail = {}
        if r["payload"]:
            try:
                detail = json.loads(r["payload"])
            except Exception:
                detail = {}
        status = str(detail.get("mitigation_status") or "").lower()
        action = str(r["mitigation_action"] or "")
        if status in ("applied", "partial", "already_active", "success") or (
            action and action.upper() not in ("", "PENDING", "MONITOR")
        ):
            mitigated += 1
        packets_pre += int(r["packets"] or 0)

    # Conservative session estimate: mitigated share of attacks × packets observed before block
    mit_ratio = mitigated / max(attacks, 1)
    estimated_blocked = int(packets_pre * mit_ratio)
    return {
        "available": True,
        "attacks": attacks,
        "mitigated_attacks": mitigated,
        "mitigation_rate_pct": round(mit_ratio * 100, 1),
        "packets_seen_pre_mitigation": packets_pre,
        "estimated_blocked_packets": estimated_blocked,
        "note": "Session proxy from live detections (iptables drops are not re-counted after block).",
    }


def build_measurements() -> Dict[str, Any]:
    seed = _load_measurement_seed()
    m3 = seed.get("m3_offloading") or {}
    m4 = seed.get("m4_scalability") or {}
    m2b = seed.get("m2_throughput_bench") or {}
    return {
        "env": DEPLOY_ENV,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "m1_latency": _live_m1(),
        "m2_throughput": {
            **_live_m2(),
            "bench": m2b,
        },
        "m3_offloading": {
            "lab": m3,
            "session": _live_m3_session(),
        },
        "m4_scalability": {
            "lab": m4,
            "session": {
                "unique_attack_sources": len(
                    {e.get("src_ip") for e in _recent if e.get("decision") == "ATTACK"}
                ),
                "avg_latency_ms": _stats["avg_latency_ms"] if _stats["latency_samples"] else None,
                "avg_mitigation_ms": _stats["avg_mitigation_ms"] if _stats["mitigation_samples"] else None,
            },
        },
    }


@app.get("/api/analytics")
async def analytics():
    with _db() as conn:
        by_label = conn.execute(
            """
            SELECT label, COUNT(*) AS c FROM events
            WHERE decision = 'ATTACK'
            GROUP BY label ORDER BY c DESC LIMIT 12
            """
        ).fetchall()
        by_layer = conn.execute(
            """
            SELECT layer_name, COUNT(*) AS c FROM events
            WHERE decision = 'ATTACK'
            GROUP BY layer_name ORDER BY c DESC
            """
        ).fetchall()
        top_ips = conn.execute(
            """
            SELECT src_ip, COUNT(*) AS c, MAX(fusion_score) AS max_score
            FROM events WHERE decision = 'ATTACK'
            GROUP BY src_ip ORDER BY c DESC LIMIT 10
            """
        ).fetchall()
    return {
        "by_label": [dict(r) for r in by_label],
        "by_layer": [dict(r) for r in by_layer],
        "top_ips": [dict(r) for r in top_ips],
        "latency_series": _latency_series[-40:],
        "throughput_series": _throughput_series[-40:],
    }


@app.get("/api/measurements")
async def measurements():
    return build_measurements()


@app.post("/api/events")
async def post_event(event: DetectionEvent):
    saved = await ingest_event(event.model_dump())
    return {"ok": True, "id": saved.get("id")}


@app.post("/api/events/batch")
async def post_batch(events: List[DetectionEvent]):
    ids = []
    for e in events:
        saved = await ingest_event(e.model_dump())
        ids.append(saved.get("id"))
    return {"ok": True, "count": len(ids), "ids": ids}


@app.delete("/api/history")
async def clear_history():
    with _db() as conn:
        conn.execute("DELETE FROM events")
        conn.commit()
    _recent.clear()
    _latency_series.clear()
    _throughput_series.clear()
    _stats["total_events"] = 0
    _stats["attacks"] = 0
    _stats["benign"] = 0
    _stats["last_event_ts"] = None
    _stats["last_attack_ts"] = None
    _stats["avg_latency_ms"] = 0.0
    _stats["avg_feature_ms"] = 0.0
    _stats["avg_inference_ms"] = 0.0
    _stats["avg_mitigation_ms"] = 0.0
    _stats["latency_samples"] = 0
    _stats["mitigation_samples"] = 0
    await broadcast({"type": "stats", "data": public_stats()})
    await broadcast({"type": "cleared"})
    return {"ok": True}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    async with _lock:
        _clients.add(ws)
    try:
        await ws.send_text(json.dumps({"type": "stats", "data": public_stats()}))
        compact_recent()
        await ws.send_text(
            json.dumps(
                {
                    "type": "snapshot",
                    "data": {
                        "events": _recent[:40],
                        "latency": _latency_series[-30:],
                        "throughput": _throughput_series[-30:],
                    },
                }
            )
        )
        while True:
            # Keep-alive / ignore client pings
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with _lock:
            _clients.discard(ws)
