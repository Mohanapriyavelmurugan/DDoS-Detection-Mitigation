(() => {
  const state = {
    page: "home",
    paused: false,
    attacksOnly: true,
    events: [],
    selectedId: null,
    latency: [],
    throughput: [],
    ws: null,
    lastToastAt: {},
    lastSeenAttackTs: "",
  };

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));
  const text = (sel, value) => {
    const el = typeof sel === "string" ? $(sel) : sel;
    if (el) el.textContent = value == null ? "" : String(value);
  };

  function pad2(n) {
    return String(n).padStart(2, "0");
  }

  function fmtTime(ts) {
    // Display in the browser's local timezone (IST for this lab).
    if (!ts) return "";
    try {
      const d = new Date(ts);
      if (Number.isNaN(d.getTime())) return String(ts).slice(0, 19);
      return (
        `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())} ` +
        `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`
      );
    } catch {
      return String(ts).slice(0, 19);
    }
  }

  function fmtShort(ts) {
    if (!ts) return "";
    try {
      return new Date(ts).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
      });
    } catch {
      return String(ts).slice(11, 19);
    }
  }

  function blank(v) {
    return v === null || v === undefined || v === "" ? "" : v;
  }

  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("edgeguard-theme", theme);
  }

  function toggleTheme() {
    const cur = document.documentElement.getAttribute("data-theme") || "light";
    setTheme(cur === "light" ? "dark" : "light");
    // Re-draw measurement charts so theme colors refresh
    if (state.page === "measure") loadMeasurements().catch(() => {});
  }

  function navigate(page) {
    state.page = page;
    $$(".page").forEach((p) => p.classList.remove("active"));
    $$(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.nav === page));
    const el = document.getElementById(`page-${page}`);
    if (el) el.classList.add("active");
    if (page === "history") loadHistory().catch((e) => console.warn(e));
    if (page === "measure") loadMeasurements().catch((e) => console.warn(e));
    if (page === "monitor") renderFeed();
  }

  function fillStrip(id, items) {
    const el = $(id);
    if (!el) return;
    el.innerHTML = (items || [])
      .map(
        (it) =>
          `<div class="mini"><span>${it.label}</span><strong>${it.value == null || it.value === "" ? "—" : it.value}</strong></div>`
      )
      .join("");
  }

  function notifyKey(event) {
    return `${event.src_ip || ""}|${String(event.label || "")}`;
  }

  function shouldToast(event) {
    if (!event) return false;
    const key = notifyKey(event);
    const now = Date.now();
    const last = Number(state.lastToastAt[key] || 0);
    // Rate-limit per attacker+label so residual floods don't spam
    if (now - last < 10000) return false;
    state.lastToastAt[key] = now;
    return true;
  }

  function showToast(event) {
    const host = $("#toastHost");
    if (!host || !event) return;
    const phase = (event.notify_phase || "").toLowerCase();
    const status = String(event.mitigation_status || "").toLowerCase();
    const action = String(event.mitigation_action || "").toUpperCase();
    if (phase !== "mitigated" && phase !== "detected") return;

    const toast = document.createElement("div");
    toast.className = "toast toast-attack";
    const mit = event.mitigation_summary || event.mitigation_action || event.mitigation_status || "";
    const ongoing = status === "already_active" || action === "ALREADY_MITIGATED" || action === "COOLDOWN";
    if (ongoing) {
      toast.innerHTML =
        `<strong>ATTACK ONGOING — ALREADY MITIGATED</strong> · ${blank(event.label)} from <em>${blank(event.src_ip)}</em>` +
        `<br/>${blank(mit) || "mitigation still active"}`;
    } else if (phase === "mitigated") {
      toast.innerHTML =
        `<strong>ATTACK DETECTED & MITIGATED</strong> · ${blank(event.label)} from <em>${blank(event.src_ip)}</em>` +
        `<br/>${blank(mit)}`;
    } else {
      toast.innerHTML =
        `<strong>ATTACK DETECTED</strong> · ${blank(event.label)} from <em>${blank(event.src_ip)}</em>` +
        `<br/>score ${Number(event.fusion_score || 0).toFixed(3)}`;
    }
    host.prepend(toast);
    setTimeout(() => toast.remove(), 14000);
    try {
      if (window.Notification && Notification.permission === "granted") {
        new Notification("EdgeGuard", {
          body: `${blank(event.label)} from ${blank(event.src_ip)}`,
        });
      }
    } catch (_) {
      /* ignore */
    }
  }

  function flashBanner(event) {
    const banner = $("#alertBanner");
    const alertText = $("#alertText");
    if (!banner || !alertText || !event) return;
    banner.classList.remove("hidden");
    const phase = (event.notify_phase || "").toLowerCase();
    const status = String(event.mitigation_status || "").toLowerCase();
    const mit = event.mitigation_summary || event.mitigation_action || event.mitigation_status || "";
    if (status === "already_active") {
      alertText.textContent =
        `ONGOING · ${blank(event.label)} from ${blank(event.src_ip)} · already mitigated`;
    } else if (phase === "mitigated") {
      alertText.textContent =
        `DETECTED & MITIGATED · ${blank(event.label)} from ${blank(event.src_ip)} · ${blank(mit)}`;
    } else {
      alertText.textContent =
        `DETECTED · ${blank(event.label)} from ${blank(event.src_ip)} · score ${Number(event.fusion_score || 0).toFixed(3)}`;
    }
    clearTimeout(flashBanner._t);
    flashBanner._t = setTimeout(() => banner.classList.add("hidden"), 15000);
  }

  function mitLabel(e) {
    if (!e) return "";
    if (e.mitigation_action) return e.mitigation_action;
    if (e.mitigation_summary) return e.mitigation_summary;
    if (e.mitigated) return e.mitigation_status || "APPLIED";
    return "";
  }

  function updateStats(stats) {
    if (!stats) return;
    text("#envBadge", blank(stats.env));
    text("#homeEnv", blank(stats.env));

    const online = !!stats.detector_online;
    const health = $("#healthBadge");
    if (health) {
      health.textContent = online ? "Detector Online" : "Waiting for detector";
      health.classList.toggle("chip-online", online);
      health.classList.toggle("chip-offline", !online);
      health.classList.remove("chip-attack");
      if (online && stats.last_attack_ts) {
        const age = Date.now() - new Date(stats.last_attack_ts).getTime();
        if (age < 15000) health.classList.add("chip-attack");
      }
    }

    const hasEvents = Number(stats.total_events || 0) > 0;
    text("#homeEvents", hasEvents ? String(stats.total_events) : "");
    text("#homeAttacks", hasEvents ? String(stats.attacks ?? "") : "");
    text(
      "#homeLatency",
      Number(stats.avg_mitigation_ms) > 0
        ? `${Number(stats.avg_mitigation_ms).toFixed(2)} ms`
        : Number(stats.avg_latency_ms) > 0
          ? `${Number(stats.avg_latency_ms).toFixed(1)} ms`
          : ""
    );

    text("#kpiAttacks", hasEvents ? String(stats.attacks ?? "") : "");
    const mitigatedCount = state.events.filter(
      (e) =>
        e.decision === "ATTACK" &&
        (String(e.notify_phase || "").toLowerCase() === "mitigated" ||
          String(e.mitigation_status || "").toLowerCase() === "applied" ||
          (e.mitigation_action && !["PENDING", "MONITOR", ""].includes(String(e.mitigation_action).toUpperCase())))
    ).length;
    text("#kpiMitigated", hasEvents ? String(mitigatedCount) : "");
    text(
      "#kpiLastAttack",
      stats.last_attack_ts ? fmtShort(stats.last_attack_ts) : ""
    );
    text("#kpiDetector", online ? "Online" : "Waiting");
  }

  function renderFeed() {
    const body = $("#feedBody");
    if (!body) return;
    const rows = dedupeAttackFeed(state.events.filter((e) => e.decision === "ATTACK"));
    text("#feedMeta", `${rows.length} attack${rows.length === 1 ? "" : "s"} · ${state.paused ? "paused" : "live"}`);
    body.innerHTML = rows
      .slice(0, 80)
      .map((e) => {
        const sel = e.id === state.selectedId ? "selected" : "";
        const score =
          e.fusion_score == null || e.fusion_score === ""
            ? ""
            : Number(e.fusion_score).toFixed(3);
        const status =
          String(e.notify_phase || "").toLowerCase() === "mitigated" ||
          String(e.mitigation_status || "").toLowerCase() === "applied"
            ? "MITIGATED"
            : e.mitigation_action && String(e.mitigation_action).toUpperCase() !== "PENDING"
              ? blank(e.mitigation_status || "APPLIED")
              : "DETECTED";
        return `<tr class="attack ${sel}" data-id="${e.id}">
          <td>${fmtShort(e.ts)}</td>
          <td>${blank(e.src_ip)}</td>
          <td>${blank(e.proto)}</td>
          <td>${blank(e.label)}</td>
          <td class="score">${score}</td>
          <td>${blank(e.layer_name)}</td>
          <td>${blank(mitLabel(e))}</td>
          <td><span class="pill attack">${status}</span></td>
        </tr>`;
      })
      .join("");
  }

  function showDetail(event) {
    if (!event) return;
    state.selectedId = event.id;
    const empty = $("#detailEmpty");
    const box = $("#detailBody");
    if (empty) empty.classList.add("hidden");
    if (!box) return;
    box.classList.remove("hidden");
    const n = (v, d = 4) => (v == null || v === "" ? "" : Number(v).toFixed(d));
    const mitAction = String(blank(event.mitigation_action) || "").replace(/\+/g, " + ");
    box.innerHTML = `
      <div class="detail-grid">
        <div><span>Source IP</span><strong>${blank(event.src_ip)}</strong></div>
        <div><span>Destination</span><strong>${blank(event.dst_ip)}</strong></div>
        <div><span>Protocol</span><strong>${blank(event.proto)}</strong></div>
        <div><span>Packets / Rate</span><strong>${blank(event.packets)} · ${blank(event.rate)}${event.rate != null && event.rate !== "" ? "/s" : ""}</strong></div>
        <div><span>Attack type</span><strong>${blank(event.label)}</strong></div>
        <div><span>Phase</span><strong>${blank(event.notify_phase || event.decision)}</strong></div>
        <div><span>Fusion score</span><strong>${n(event.fusion_score)}</strong></div>
        <div><span>Tier</span><strong>${blank(event.tier || event.mitigation_tier_name)}</strong></div>
        <div><span>Layer</span><strong>${blank(event.layer_name || event.layer)}</strong></div>
        <div><span>XGBoost</span><strong>${n(event.xgb_score)}</strong></div>
        <div><span>Isolation Forest</span><strong>${n(event.if_score)}</strong></div>
        <div><span>DPI</span><strong>${n(event.dpi_score)}</strong></div>
        <div class="span-2"><span>Mitigation action</span><strong class="mono-break">${mitAction}</strong></div>
        <div><span>Mitigation status</span><strong>${blank(event.mitigation_status)}</strong></div>
        <div><span>Expires in</span><strong>${event.mitigation_expires_in ? event.mitigation_expires_in + "s" : ""}</strong></div>
        <div><span>Environment</span><strong>${blank(event.env)}</strong></div>
      </div>
      ${event.mitigation_summary ? `<p class="detail-summary"><strong>Mitigation:</strong> ${blank(event.mitigation_summary)}</p>` : ""}
      ${(event.mitigation_commands && event.mitigation_commands.length)
        ? `<p class="muted detail-cmds">${event.mitigation_commands.slice(0, 4).map(blank).join("<br/>")}</p>`
        : ""}
      ${event.was_overridden ? `<p class="muted" style="margin-top:10px">Type lock: ${blank(event.override_reason)}</p>` : ""}
    `;
    renderFeed();
  }

  function mapLatencyPoint(p) {
    return {
      total: Number((p && p.total) || 0),
      feature: Number((p && p.feature) || 0),
      inference: Number((p && p.inference) || 0),
      mitigation: Number((p && (p.mitigation || p.detect_to_mitigate)) || 0),
    };
  }

  function sameAttackFlow(a, b) {
    if (!a || !b) return false;
    if (String(a.src_ip || "") !== String(b.src_ip || "")) return false;
    if (String(a.decision || "").toUpperCase() !== "ATTACK") return false;
    if (String(b.decision || "").toUpperCase() !== "ATTACK") return false;
    return String(a.flow_id ?? 0) === String(b.flow_id ?? 0);
  }

  function phaseRank(e) {
    const phase = String((e && e.notify_phase) || "").toLowerCase();
    if (phase === "mitigated") return 2;
    const act = String((e && e.mitigation_action) || "");
    if (act && act !== "PENDING") return 2;
    if (phase === "detected") return 1;
    return 0;
  }

  function upsertEvent(event) {
    if (!event.id) {
      event.id = `${event.ts}-${event.src_ip}-${event.flow_id}-${event.notify_phase || ""}`;
    }
    if (String(event.decision || "").toUpperCase() === "ATTACK") {
      const idx = state.events.findIndex((e) => sameAttackFlow(e, event));
      if (idx >= 0) {
        const prev = state.events[idx];
        const merged =
          phaseRank(event) >= phaseRank(prev)
            ? { ...prev, ...event, notify_phase: phaseRank(event) >= 2 ? "mitigated" : event.notify_phase || prev.notify_phase }
            : prev;
        state.events.splice(idx, 1);
        state.events.unshift(merged);
        return merged;
      }
    }
    state.events.unshift(event);
    return event;
  }

  function dedupeAttackFeed(list) {
    const out = [];
    const seen = new Set();
    for (const e of list || []) {
      if (String(e.decision || "").toUpperCase() !== "ATTACK") {
        out.push(e);
        continue;
      }
      const key = `${e.src_ip}|${e.flow_id ?? 0}`;
      if (seen.has(key)) continue;
      seen.add(key);
      // Prefer mitigated twin if a weaker detected copy appears first
      const best = (list || [])
        .filter((x) => sameAttackFlow(x, e))
        .sort((a, b) => phaseRank(b) - phaseRank(a))[0];
      out.push(best || e);
    }
    return out;
  }

  function onEvent(event) {
    if (!event) return;
    const phase = (event.notify_phase || "").toLowerCase();
    upsertEvent(event);
    if (state.events.length > 200) state.events.length = 200;

    const lat = event.latency_ms || {};
    if (lat.total || lat.feature || lat.inference || lat.mitigation) {
      state.latency.push(mapLatencyPoint(lat));
      if (state.latency.length > 40) state.latency.shift();
    }

    if (!state.paused) renderFeed();
    // Toast on every fresh attack notify (rate-limited). Do not require a
    // phase-rank increase — that hid toasts when the IP was already blocked
    // or when live bootstrap already held the same flow.
    if (
      String(event.decision || "").toUpperCase() === "ATTACK" &&
      (phase === "detected" || phase === "mitigated")
    ) {
      if (event.ts) state.lastSeenAttackTs = event.ts;
      if (shouldToast(event)) {
        showToast(event);
        flashBanner(event);
      }
    }
  }

  function fillDistList(id, block) {
    const el = $(id);
    if (!el) return;
    if (!block || !block.n) {
      el.innerHTML = `<div class="dist-item"><span>Samples</span><strong>0</strong></div>`;
      return;
    }
    const items = [
      ["Min", block.min],
      ["Median", block.median],
      ["Mean", block.mean],
      ["P95", block.p95],
      ["Max", block.max],
    ];
    el.innerHTML = items
      .map(
        ([lab, v]) =>
          `<div class="dist-item"><span>${lab}</span><strong>${v == null ? "—" : Number(v).toFixed(1)}</strong></div>`
      )
      .join("");
  }

  function fillOverview(m1, m2, lab3, lab4, bench2) {
    const el = $("#measureOverview");
    if (!el) return;
    const last = (lab4.points || [])[(lab4.points || []).length - 1] || {};
    const cards = [
      {
        k: "M1 Latency",
        v: m1.total?.mean != null ? `${Number(m1.total.mean).toFixed(1)} ms` : "—",
        h: "mean pipeline",
      },
      {
        k: "M2 Throughput",
        v: bench2.peak_flows_per_sec
          ? `${Number(bench2.peak_flows_per_sec).toLocaleString()} /s`
          : m2.current_fps
            ? `${Number(m2.current_fps).toFixed(1)} /s`
            : "—",
        h: "lab peak flows/s",
      },
      {
        k: "M3 Offload",
        v: lab3.offload_pct != null ? `${lab3.offload_pct}%` : "—",
        h: "blocked at MEC edge",
      },
      {
        k: "M4 Det @100 UE",
        v: last.det_rate_pct != null ? `${last.det_rate_pct}%` : "—",
        h: "detection under load",
      },
    ];
    el.innerHTML = cards
      .map(
        (c) =>
          `<article class="ov-card"><div class="ov-k">${c.k}</div><div class="ov-v">${c.v}</div><div class="ov-h">${c.h}</div></article>`
      )
      .join("");
  }

  async function loadMeasurements() {
    const C = window.EdgeCharts;
    if (!C) return;
    const data = await (await fetch("/api/measurements")).json();
    const m1 = data.m1_latency || {};
    const m2 = data.m2_throughput || {};
    const m3 = data.m3_offloading || {};
    const m4 = data.m4_scalability || {};
    const lab3 = m3.lab || {};
    const lab4 = m4.lab || {};
    const bench2 = m2.bench || {};
    const sess3 = m3.session || {};
    const sess4 = m4.session || {};

    fillOverview(m1, m2, lab3, lab4, bench2);

    text("#m1Source", m1.source || "live detector pipeline");
    text(
      "#m1Finding",
      m1.total && m1.total.mean != null
        ? `Mean detect pipeline ${m1.total.mean} ms (p95 ${m1.total.p95} ms) across ${m1.total.n} samples.`
        : "Waiting for live latency samples from the detector."
    );
    fillStrip("#m1Strip", [
      { label: "Feature", value: m1.feature?.mean != null ? `${m1.feature.mean} ms` : "—" },
      { label: "Inference", value: m1.inference?.mean != null ? `${m1.inference.mean} ms` : "—" },
      { label: "Mitigate", value: m1.mitigation?.mean != null ? `${m1.mitigation.mean} ms` : "—" },
      { label: "Total", value: m1.total?.mean != null ? `${m1.total.mean} ms` : "—" },
    ]);
    C.stackedLatency($("#m1Stack"), m1.stages || [], m1.total?.mean || 0);
    C.multiLine(
      $("#m1Trend"),
      m1.series || [],
      ["feature", "inference", "mitigation", "total"],
      ["Feature", "Inference", "Mitigate", "Total"],
      {
        height: 230,
        empty: "No live latency series yet",
        xLabel: "Sample →",
        yLabel: "Latency (ms)",
      }
    );
    fillDistList("#m1Dist", m1.total);

    text("#m2Source", "live session + lab capacity bench");
    text(
      "#m2Finding",
      bench2.finding ||
        (m2.current_fps
          ? `Current session throughput ${Number(m2.current_fps).toFixed(1)} flows/s.`
          : "Live throughput fills during active detection; lab curve shows peak capacity.")
    );
    fillStrip("#m2Strip", [
      { label: "Live flows/s", value: m2.current_fps ? Number(m2.current_fps).toFixed(1) : "—" },
      { label: "Lab peak", value: bench2.peak_flows_per_sec ? `${bench2.peak_flows_per_sec.toLocaleString()} /s` : "—" },
      { label: "Bench UEs", value: (bench2.points || []).length ? `${bench2.points[0].n_ues}–${bench2.points[bench2.points.length - 1].n_ues}` : "—" },
      { label: "Det rate", value: "100%" },
    ]);
    C.areaLine($("#m2Live"), m2.series || [], "fps", {
      empty: "No live throughput yet this session",
      xLabel: "Sample →",
      yLabel: "Throughput (flows/s)",
    });
    C.comboScale(
      $("#m2Bench"),
      (bench2.points || []).map((p) => ({
        n_ues: p.n_ues,
        flows_per_sec: p.flows_per_sec,
        det_rate_pct: p.det_rate_pct,
        batch_ms: p.batch_ms,
      })),
      { xLabel: "Concurrent UEs →", yLabel: "Throughput (flows/s)" }
    );

    text("#m3Source", lab3.source || "lab offloading proof");
    text("#m3Finding", lab3.finding || "");
    fillStrip("#m3Strip", [
      { label: "Offload", value: lab3.offload_pct != null ? `${lab3.offload_pct}%` : "—" },
      { label: "Blocked", value: lab3.packets_blocked_edge != null ? lab3.packets_blocked_edge.toLocaleString() : "—" },
      { label: "Reached core", value: lab3.packets_reached_core != null ? lab3.packets_reached_core.toLocaleString() : "—" },
      { label: "Reduction", value: lab3.traffic_reduction_pct != null ? `${lab3.traffic_reduction_pct}%` : "—" },
    ]);
    if (lab3.packets_blocked_edge != null) {
      C.donut(
        $("#m3Donut"),
        [
          { label: "Blocked at MEC", value: lab3.packets_blocked_edge },
          { label: "Reached core", value: lab3.packets_reached_core },
        ],
        { centerSub: "offloaded", label: "Offload share" }
      );
      C.groupedBars(
        $("#m3Bars"),
        [
          { label: "Before", values: [lab3.rate_before_pps || 0] },
          { label: "After", values: [lab3.rate_after_pps || 0] },
        ],
        {
          legend: ["pkts/s to server"],
          colors: ["#b42318", "#0f7a45"],
          xLabel: "Mitigation phase",
          yLabel: "Packets/s",
        }
      );
      C.multiLine(
        $("#m3Timeline"),
        lab3.timeline || [],
        ["sent", "reached", "blocked"],
        ["Sent", "Reached", "Blocked"],
        {
          height: 230,
          label: "Offload timeline",
          xLabel: "Time sample →",
          yLabel: "Cumulative packets",
        }
      );
    }
    text(
      "#m3SessionNote",
      sess3.available
        ? `Live session: ${sess3.mitigated_attacks}/${sess3.attacks} attacks mitigated (${sess3.mitigation_rate_pct}%).`
        : "Live session note appears after attacks are mitigated."
    );

    text("#m4Source", lab4.source || "lab scalability run");
    text("#m4Finding", lab4.finding || "");
    const last = (lab4.points || [])[(lab4.points || []).length - 1] || {};
    fillStrip("#m4Strip", [
      { label: "UE range", value: lab4.ue_range ? `${lab4.ue_range[0]}–${lab4.ue_range[1]}` : "—" },
      { label: "Det @100 UE", value: last.det_rate_pct != null ? `${last.det_rate_pct}%` : "—" },
      { label: "Flows @100 UE", value: last.flows_per_sec != null ? last.flows_per_sec.toLocaleString() : "—" },
      {
        label: "Session sources",
        value: sess4.unique_attack_sources != null ? String(sess4.unique_attack_sources) : "—",
      },
    ]);
    C.comboScale($("#m4Combo"), lab4.points || [], {
      xLabel: "Concurrent UEs →",
      yLabel: "Throughput (flows/s)",
    });
    const body = $("#m4Table");
    if (body) {
      body.innerHTML = (lab4.points || [])
        .map(
          (r) => `<tr>
          <td>${r.n_ues}</td>
          <td>${r.batch_ms}</td>
          <td>${r.per_flow_ms}</td>
          <td>${Number(r.flows_per_sec).toLocaleString()}</td>
          <td>${r.det_rate_pct}%</td>
        </tr>`
        )
        .join("");
    }
  }

  async function loadHistory() {
    const qEl = $("#historyQuery");
    const dEl = $("#historyDecision");
    const q = qEl ? qEl.value.trim() : "";
    const decision = dEl ? dEl.value : "";
    const params = new URLSearchParams({ limit: "150" });
    if (q) params.set("q", q);
    if (decision) params.set("decision", decision);
    const res = await fetch(`/api/history?${params}`);
    const data = await res.json();
    text("#historyMeta", `${data.total} records`);
    const body = $("#historyBody");
    if (!body) return;
    body.innerHTML = (data.items || [])
      .map((e) => {
        const cls = e.decision === "ATTACK" ? "attack" : "benign";
        const score =
          e.fusion_score == null || e.fusion_score === ""
            ? ""
            : Number(e.fusion_score).toFixed(3);
        const detail = e.detail || {};
        const mitAct = blank(e.mitigation_action || detail.mitigation_action || mitLabel(detail));
        const mitStat = blank(e.mitigation_status || detail.mitigation_status);
        return `<tr class="${cls}">
          <td>${fmtTime(e.ts)}</td>
          <td>${blank(e.env)}</td>
          <td>${blank(e.src_ip)}</td>
          <td>${blank(e.dst_ip)}</td>
          <td>${blank(e.proto)}</td>
          <td>${blank(e.label)}</td>
          <td class="score">${score}</td>
          <td>${blank(e.layer_name)}</td>
          <td><span class="pill ${cls}">${blank(e.decision)}</span></td>
          <td>${mitAct}</td>
          <td>${mitStat}</td>
        </tr>`;
      })
      .join("");

    try {
      const analytics = await (await fetch("/api/analytics")).json();
      const fill = (id, rows, labelKey, countKey) => {
        const el = $(id);
        if (!el) return;
        if (!rows || !rows.length) {
          el.innerHTML = "";
          return;
        }
        el.innerHTML = rows
          .map(
            (r) =>
              `<li><span>${r[labelKey]}</span><span class="count">${r[countKey]}${
                r.max_score != null ? ` · max ${Number(r.max_score).toFixed(2)}` : ""
              }</span></li>`
          )
          .join("");
      };
      fill("#topIps", analytics.top_ips, "src_ip", "c");
      fill("#topLabels", analytics.by_label, "label", "c");
      fill("#topLayers", analytics.by_layer, "layer_name", "c");
    } catch (err) {
      console.warn(err);
    }
  }

  function connectWs() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    state.ws = ws;
    ws.onmessage = (msg) => {
      let payload;
      try {
        payload = JSON.parse(msg.data);
      } catch {
        return;
      }
      try {
        if (payload.type === "stats") updateStats(payload.data);
        if (payload.type === "event") onEvent(payload.data);
        if (payload.type === "snapshot") {
          state.events = dedupeAttackFeed(payload.data.events || []);
          state.latency = (payload.data.latency || []).map(mapLatencyPoint);
          state.throughput = payload.data.throughput || [];
          renderFeed();
        }
        if (payload.type === "cleared") {
          state.events = [];
          state.latency = [];
          state.throughput = [];
          renderFeed();
          if (state.page === "history") loadHistory().catch(() => {});
        }
      } catch (err) {
        console.warn("ws handler", err);
      }
    };
    ws.onclose = () => setTimeout(connectWs, 2000);
    ws.onopen = () => {
      const ping = () => {
        if (ws.readyState === WebSocket.OPEN) ws.send("ping");
      };
      clearInterval(connectWs._ping);
      connectWs._ping = setInterval(ping, 20000);
    };
  }

  async function pollLiveFallback() {
    try {
      const live = await (await fetch("/api/live")).json();
      const events = dedupeAttackFeed(live.events || []);
      const newest = events.find(
        (e) =>
          String(e.decision || "").toUpperCase() === "ATTACK" &&
          ["detected", "mitigated"].includes(String(e.notify_phase || "").toLowerCase())
      );
      if (newest && newest.ts && newest.ts !== state.lastSeenAttackTs) {
        const age = Date.now() - new Date(newest.ts).getTime();
        if (age >= 0 && age < 20000 && shouldToast(newest)) {
          showToast(newest);
          flashBanner(newest);
        }
        state.lastSeenAttackTs = newest.ts;
      }
      // Keep feed in sync even if a WS frame was dropped
      if (!state.paused && events.length) {
        state.events = events;
        renderFeed();
      }
    } catch (_) {
      /* ignore */
    }
  }

  async function bootstrap() {
    try {
      const stats = await (await fetch("/api/stats")).json();
      updateStats(stats);
      const live = await (await fetch("/api/live")).json();
      state.events = dedupeAttackFeed(live.events || []);
      state.latency = (live.latency || []).map(mapLatencyPoint);
      state.throughput = live.throughput || [];
      const newest = state.events.find(
        (e) => String(e.decision || "").toUpperCase() === "ATTACK"
      );
      if (newest && newest.ts) state.lastSeenAttackTs = newest.ts;
      renderFeed();
    } catch (err) {
      console.warn(err);
    }
    connectWs();
    try {
      if (window.Notification && Notification.permission === "default") {
        Notification.requestPermission().catch(() => {});
      }
    } catch (_) {
      /* ignore */
    }
    clearInterval(bootstrap._poll);
    bootstrap._poll = setInterval(pollLiveFallback, 3000);
  }

  function bind(sel, event, fn) {
    const el = $(sel);
    if (el) el.addEventListener(event, fn);
  }

  try {
    $$("[data-nav]").forEach((el) =>
      el.addEventListener("click", () => navigate(el.dataset.nav))
    );
    bind("#themeToggle", "click", toggleTheme);
    bind("#pauseStream", "click", () => {
      state.paused = !state.paused;
      text("#pauseStream", state.paused ? "Resume" : "Pause");
      if (!state.paused) renderFeed();
    });
    bind("#feedBody", "click", (e) => {
      const tr = e.target.closest("tr[data-id]");
      if (!tr) return;
      const id = tr.getAttribute("data-id");
      const event = state.events.find((x) => String(x.id) === String(id));
      showDetail(event);
    });
    bind("#refreshMeasure", "click", () => loadMeasurements().catch(console.warn));
    bind("#refreshHistory", "click", () => loadHistory().catch(console.warn));
    bind("#historyQuery", "keydown", (e) => {
      if (e.key === "Enter") loadHistory().catch(console.warn);
    });
    bind("#historyDecision", "change", () => loadHistory().catch(console.warn));
    bind("#exportCsv", "click", async () => {
      const decision = ($("#historyDecision") || {}).value || "";
      const url = decision
        ? `/api/history/export?decision=${encodeURIComponent(decision)}`
        : "/api/history/export";
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Export failed (${res.status})`);
        const blob = await res.blob();
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "attack_history.csv";
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(a.href);
      } catch (err) {
        alert("CSV export failed. Make sure the dashboard is running.");
        console.warn(err);
      }
    });
    bind("#clearHistory", "click", async () => {
      if (!confirm("Clear all history records?")) return;
      await fetch("/api/history", { method: "DELETE" });
      loadHistory().catch(console.warn);
    });

    setInterval(() => {
      text("#footerClock", new Date().toLocaleString());
    }, 1000);

    const saved = localStorage.getItem("edgeguard-theme") || "light";
    setTheme(saved);
    bootstrap();
  } catch (err) {
    console.error("dashboard init failed", err);
    const host = document.body;
    if (host) {
      const note = document.createElement("div");
      note.style.cssText =
        "position:fixed;bottom:12px;left:12px;right:12px;padding:12px;background:#fff;color:#b42318;border:1px solid #b42318;z-index:9999;font-family:serif";
      note.textContent = "Dashboard UI failed to start. Hard-refresh (Ctrl+F5) or rebuild the dashboard container.";
      host.appendChild(note);
    }
  }
})();
