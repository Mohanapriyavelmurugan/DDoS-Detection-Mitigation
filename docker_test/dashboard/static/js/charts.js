/* EdgeGuard SVG analytics — axis labels, spaced legends, no overlapping text */
window.EdgeCharts = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const FONT = "Times New Roman, Times, serif";

  function theme() {
    const s = getComputedStyle(document.documentElement);
    return {
      text: s.getPropertyValue("--text").trim() || "#0b3154",
      muted: s.getPropertyValue("--muted").trim() || "#4d6d8c",
      grid: s.getPropertyValue("--chart-grid").trim() || "rgba(15,68,115,.15)",
      blue: "#1f6fb5",
      blue2: "#3d86c6",
      blue3: "#0f4473",
      warn: "#b54708",
      danger: "#b42318",
      ok: "#0f7a45",
    };
  }

  function el(name, attrs = {}, kids = []) {
    const n = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, String(v)));
    kids.forEach((c) => n.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return n;
  }

  function clear(host) {
    if (!host) return null;
    host.innerHTML = "";
    return host;
  }

  function empty(host, msg) {
    const t = theme();
    const svg = el("svg", { viewBox: "0 0 440 110", width: "100%", height: "110", role: "img" });
    svg.appendChild(
      el("text", { x: 16, y: 58, fill: t.muted, "font-size": 13, "font-family": FONT }, [msg || "No data yet"])
    );
    clear(host).appendChild(svg);
  }

  function legendRow(svg, items, colors, y, x0 = 12) {
    const t = theme();
    let x = x0;
    items.forEach((lab, i) => {
      svg.appendChild(el("rect", { x, y: y - 8, width: 10, height: 10, rx: 2, fill: colors[i % colors.length] }));
      svg.appendChild(
        el("text", { x: x + 14, y, fill: t.muted, "font-size": 11, "font-family": FONT }, [lab])
      );
      x += Math.max(88, lab.length * 7 + 28);
    });
  }

  function axisLabels(svg, W, H, pad, xLabel, yLabel) {
    const t = theme();
    if (xLabel) {
      svg.appendChild(
        el("text", {
          x: pad.l + (W - pad.l - pad.r) / 2,
          y: H - 4,
          fill: t.muted,
          "font-size": 11,
          "text-anchor": "middle",
          "font-family": FONT,
        }, [xLabel])
      );
    }
    if (yLabel) {
      const cx = 14;
      const cy = pad.t + (H - pad.t - pad.b) / 2;
      svg.appendChild(
        el("text", {
          x: cx,
          y: cy,
          fill: t.muted,
          "font-size": 11,
          "text-anchor": "middle",
          "font-family": FONT,
          transform: `rotate(-90 ${cx} ${cy})`,
        }, [yLabel])
      );
    }
  }

  function fmtTick(val) {
    if (val >= 1000) return val.toFixed(0);
    if (val >= 100) return val.toFixed(0);
    if (val >= 10) return val.toFixed(1);
    return val.toFixed(2);
  }

  function stackedLatency(host, stages, totalMs) {
    if (!stages || !stages.length) return empty(host, "No latency stages yet");
    const t = theme();
    const W = 480;
    const H = 128;
    const svg = el("svg", {
      viewBox: `0 0 ${W} ${H}`,
      width: "100%",
      height: String(H),
      role: "img",
      "aria-label": "Latency pipeline breakdown",
    });
    const colors = [t.blue2, t.blue3, t.warn];
    const sum = stages.reduce((a, s) => a + Math.max(0, Number(s.ms) || 0), 0) || 1;
    const barY = 36;
    const barH = 26;
    const barX = 12;
    const barW = W - 24;
    let x = barX;
    stages.forEach((s, i) => {
      const w = (Math.max(0, Number(s.ms) || 0) / sum) * barW;
      if (w <= 0) return;
      svg.appendChild(el("rect", { x, y: barY, width: Math.max(w, 2), height: barH, rx: 4, fill: colors[i % colors.length] }));
      if (w > 56) {
        svg.appendChild(
          el("text", {
            x: x + w / 2,
            y: barY + 17,
            fill: "#fff",
            "font-size": 11,
            "text-anchor": "middle",
            "font-family": FONT,
          }, [`${Number(s.ms).toFixed(1)}`])
        );
      }
      x += w;
    });
    svg.appendChild(
      el("text", {
        x: 12,
        y: 20,
        fill: t.text,
        "font-size": 13,
        "font-weight": 700,
        "font-family": FONT,
      }, [`Total ${Number(totalMs || sum).toFixed(2)} ms`])
    );
    legendRow(
      svg,
      stages.map((s) => `${s.label} ${Number(s.ms || 0).toFixed(1)} ms`),
      colors,
      100
    );
    clear(host).appendChild(svg);
  }

  function multiLine(host, series, keys, labels, opts = {}) {
    if (!series || !series.length) return empty(host, opts.empty || "No samples yet");
    const t = theme();
    const W = opts.width || 540;
    const H = opts.height || 230;
    const pad = { l: 56, r: 16, t: 34, b: 42 };
    const colors = opts.colors || [t.blue2, t.blue3, t.warn, t.blue, t.ok];
    const svg = el("svg", {
      viewBox: `0 0 ${W} ${H}`,
      width: "100%",
      height: String(H),
      role: "img",
      "aria-label": opts.label || "Line chart",
    });

    const all = [];
    keys.forEach((k) => series.forEach((p) => all.push(Number(p[k] || 0))));
    const max = Math.max(...all, 0.001);
    const innerW = W - pad.l - pad.r;
    const innerH = H - pad.t - pad.b;

    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (innerH * i) / 4;
      svg.appendChild(el("line", { x1: pad.l, y1: y, x2: W - pad.r, y2: y, stroke: t.grid, "stroke-width": 1 }));
      const val = max * (1 - i / 4);
      svg.appendChild(
        el("text", {
          x: pad.l - 8,
          y: y + 3,
          fill: t.muted,
          "font-size": 10,
          "text-anchor": "end",
          "font-family": FONT,
        }, [fmtTick(val)])
      );
    }

    // X tick marks: sample index
    const step = Math.max(1, Math.ceil((series.length - 1) / 5));
    for (let i = 0; i < series.length; i += step) {
      const x = pad.l + (i / Math.max(series.length - 1, 1)) * innerW;
      svg.appendChild(
        el("text", {
          x,
          y: H - pad.b + 14,
          fill: t.muted,
          "font-size": 10,
          "text-anchor": "middle",
          "font-family": FONT,
        }, [String(i + 1)])
      );
    }

    keys.forEach((key, ki) => {
      const pts = series
        .map((p, i) => {
          const x = pad.l + (i / Math.max(series.length - 1, 1)) * innerW;
          const y = pad.t + innerH - (Number(p[key] || 0) / max) * innerH;
          return `${x},${y}`;
        })
        .join(" ");
      svg.appendChild(
        el("polyline", {
          points: pts,
          fill: "none",
          stroke: colors[ki % colors.length],
          "stroke-width": ki === keys.length - 1 ? 2.4 : 1.7,
          "stroke-linejoin": "round",
          "stroke-linecap": "round",
        })
      );
    });

    legendRow(svg, labels || keys, colors, 18, pad.l);
    axisLabels(svg, W, H, pad, opts.xLabel || "Sample →", opts.yLabel || "Latency (ms)");
    clear(host).appendChild(svg);
  }

  function areaLine(host, series, key, opts = {}) {
    if (!series || !series.length) return empty(host, opts.empty || "No throughput yet");
    const mapped = series.map((p) => ({ y: Number(p[key] || p.fps || 0) }));
    const t = theme();
    const W = opts.width || 540;
    const H = opts.height || 200;
    const pad = { l: 56, r: 16, t: 28, b: 42 };
    const max = Math.max(...mapped.map((p) => p.y), 0.001);
    const svg = el("svg", {
      viewBox: `0 0 ${W} ${H}`,
      width: "100%",
      height: String(H),
      role: "img",
      "aria-label": opts.label || "Throughput",
    });
    const innerW = W - pad.l - pad.r;
    const innerH = H - pad.t - pad.b;
    const coords = mapped.map((p, i) => {
      const x = pad.l + (i / Math.max(mapped.length - 1, 1)) * innerW;
      const y = pad.t + innerH - (p.y / max) * innerH;
      return [x, y];
    });
    const area =
      `M ${coords[0][0]} ${pad.t + innerH} ` +
      coords.map(([x, y]) => `L ${x} ${y}`).join(" ") +
      ` L ${coords[coords.length - 1][0]} ${pad.t + innerH} Z`;
    for (let i = 0; i <= 3; i++) {
      const y = pad.t + (innerH * i) / 3;
      svg.appendChild(el("line", { x1: pad.l, y1: y, x2: W - pad.r, y2: y, stroke: t.grid }));
      const val = max * (1 - i / 3);
      svg.appendChild(
        el("text", {
          x: pad.l - 8,
          y: y + 3,
          fill: t.muted,
          "font-size": 10,
          "text-anchor": "end",
          "font-family": FONT,
        }, [fmtTick(val)])
      );
    }
    const step = Math.max(1, Math.ceil((mapped.length - 1) / 5));
    for (let i = 0; i < mapped.length; i += step) {
      const x = pad.l + (i / Math.max(mapped.length - 1, 1)) * innerW;
      svg.appendChild(
        el("text", {
          x,
          y: H - pad.b + 14,
          fill: t.muted,
          "font-size": 10,
          "text-anchor": "middle",
          "font-family": FONT,
        }, [String(i + 1)])
      );
    }
    svg.appendChild(el("path", { d: area, fill: t.blue2, opacity: 0.2 }));
    svg.appendChild(
      el("polyline", {
        points: coords.map(([x, y]) => `${x},${y}`).join(" "),
        fill: "none",
        stroke: t.blue,
        "stroke-width": 2.4,
      })
    );
    const last = mapped[mapped.length - 1].y;
    svg.appendChild(
      el("text", {
        x: W - pad.r,
        y: 16,
        fill: t.text,
        "font-size": 12,
        "text-anchor": "end",
        "font-family": FONT,
      }, [`${last.toFixed(1)} flows/s`])
    );
    axisLabels(svg, W, H, pad, opts.xLabel || "Sample →", opts.yLabel || "Throughput (flows/s)");
    clear(host).appendChild(svg);
  }

  function donut(host, parts, opts = {}) {
    if (!parts || !parts.length) return empty(host, "No offload data");
    const t = theme();
    const W = 300;
    const H = 170;
    const cx = 78;
    const cy = 86;
    const r = 48;
    const stroke = 18;
    const svg = el("svg", {
      viewBox: `0 0 ${W} ${H}`,
      width: "100%",
      height: String(H),
      role: "img",
      "aria-label": opts.label || "Offload share",
    });
    const total = parts.reduce((a, p) => a + Math.max(0, p.value), 0) || 1;
    const colors = [t.ok, t.danger, t.blue2];
    const C = 2 * Math.PI * r;
    let offset = 0;
    parts.forEach((p, i) => {
      const frac = Math.max(0, p.value) / total;
      const dash = frac * C;
      svg.appendChild(
        el("circle", {
          cx,
          cy,
          r,
          fill: "none",
          stroke: colors[i % colors.length],
          "stroke-width": stroke,
          "stroke-dasharray": `${dash} ${C - dash}`,
          "stroke-dashoffset": String(-offset),
          transform: `rotate(-90 ${cx} ${cy})`,
        })
      );
      offset += dash;
    });
    const pct = ((parts[0].value / total) * 100).toFixed(1);
    svg.appendChild(
      el("text", {
        x: cx,
        y: cy + 1,
        fill: t.text,
        "font-size": 18,
        "font-weight": 700,
        "text-anchor": "middle",
        "font-family": FONT,
      }, [`${pct}%`])
    );
    svg.appendChild(
      el("text", {
        x: cx,
        y: cy + 18,
        fill: t.muted,
        "font-size": 10,
        "text-anchor": "middle",
        "font-family": FONT,
      }, [opts.centerSub || "offloaded"])
    );
    parts.forEach((p, i) => {
      const ly = 42 + i * 42;
      svg.appendChild(el("rect", { x: 150, y: ly, width: 12, height: 12, rx: 2, fill: colors[i % colors.length] }));
      svg.appendChild(
        el("text", { x: 170, y: ly + 10, fill: t.text, "font-size": 12, "font-family": FONT }, [p.label])
      );
      svg.appendChild(
        el("text", {
          x: 170,
          y: ly + 26,
          fill: t.muted,
          "font-size": 11,
          "font-family": FONT,
        }, [`${Number(p.value).toLocaleString()} pkts`])
      );
    });
    clear(host).appendChild(svg);
  }

  function groupedBars(host, groups, opts = {}) {
    if (!groups || !groups.length) return empty(host, "No comparison data");
    const t = theme();
    const W = 320;
    const H = 200;
    const pad = { l: 48, r: 14, t: 30, b: 44 };
    const svg = el("svg", {
      viewBox: `0 0 ${W} ${H}`,
      width: "100%",
      height: String(H),
      role: "img",
      "aria-label": opts.label || "Comparison",
    });
    const max = Math.max(...groups.flatMap((g) => g.values.map(Number)), 0.001);
    const innerW = W - pad.l - pad.r;
    const innerH = H - pad.t - pad.b;
    const slot = innerW / groups.length;
    const colors = opts.colors || [t.danger, t.ok];
    const barW = Math.min(36, slot * 0.45);

    for (let i = 0; i <= 3; i++) {
      const y = pad.t + (innerH * i) / 3;
      svg.appendChild(el("line", { x1: pad.l, y1: y, x2: W - pad.r, y2: y, stroke: t.grid }));
      const val = max * (1 - i / 3);
      svg.appendChild(
        el("text", {
          x: pad.l - 6,
          y: y + 3,
          fill: t.muted,
          "font-size": 10,
          "text-anchor": "end",
          "font-family": FONT,
        }, [fmtTick(val)])
      );
    }

    groups.forEach((g, gi) => {
      g.values.forEach((v, vi) => {
        const h = (Number(v) / max) * innerH;
        const x = pad.l + gi * slot + (slot - barW) / 2 + vi * (barW + 6);
        const y = pad.t + innerH - h;
        svg.appendChild(el("rect", { x, y, width: barW, height: Math.max(h, 1), rx: 4, fill: colors[vi % colors.length] }));
        svg.appendChild(
          el("text", {
            x: x + barW / 2,
            y: y - 6,
            fill: t.text,
            "font-size": 11,
            "text-anchor": "middle",
            "font-family": FONT,
          }, [String(Math.round(Number(v)))])
        );
      });
      svg.appendChild(
        el("text", {
          x: pad.l + gi * slot + slot / 2,
          y: H - pad.b + 16,
          fill: t.muted,
          "font-size": 11,
          "text-anchor": "middle",
          "font-family": FONT,
        }, [g.label])
      );
    });
    if (opts.legend && opts.legend.length) {
      legendRow(svg, opts.legend, colors, 16, pad.l);
    }
    axisLabels(svg, W, H, pad, opts.xLabel || "Phase", opts.yLabel || "Packets/s");
    clear(host).appendChild(svg);
  }

  function comboScale(host, points, opts = {}) {
    if (!points || !points.length) return empty(host, "No scalability data");
    const t = theme();
    const W = 560;
    const H = 250;
    const pad = { l: 56, r: 52, t: 36, b: 46 };
    const svg = el("svg", {
      viewBox: `0 0 ${W} ${H}`,
      width: "100%",
      height: String(H),
      role: "img",
      "aria-label": "Scalability under UE load",
    });
    const fps = points.map((p) => Number(p.flows_per_sec || 0));
    const batch = points.map((p) => Number(p.batch_ms || p.per_flow_ms || 0));
    const maxFps = Math.max(...fps, 1);
    const maxBatch = Math.max(...batch, 1);
    const innerW = W - pad.l - pad.r;
    const innerH = H - pad.t - pad.b;
    const n = points.length;
    const slot = innerW / n;

    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (innerH * i) / 4;
      svg.appendChild(el("line", { x1: pad.l, y1: y, x2: W - pad.r, y2: y, stroke: t.grid }));
      const val = maxFps * (1 - i / 4);
      svg.appendChild(
        el("text", {
          x: pad.l - 8,
          y: y + 3,
          fill: t.muted,
          "font-size": 10,
          "text-anchor": "end",
          "font-family": FONT,
        }, [fmtTick(val)])
      );
      // right axis: batch ms
      const bval = maxBatch * (1 - i / 4);
      svg.appendChild(
        el("text", {
          x: W - pad.r + 8,
          y: y + 3,
          fill: t.warn,
          "font-size": 10,
          "text-anchor": "start",
          "font-family": FONT,
        }, [fmtTick(bval)])
      );
    }

    points.forEach((p, i) => {
      const h = (Number(p.flows_per_sec || 0) / maxFps) * innerH;
      const x = pad.l + i * slot + slot * 0.18;
      const w = slot * 0.5;
      svg.appendChild(
        el("rect", {
          x,
          y: pad.t + innerH - h,
          width: w,
          height: Math.max(h, 1),
          rx: 3,
          fill: t.blue2,
          opacity: 0.88,
        })
      );
      svg.appendChild(
        el("text", {
          x: pad.l + i * slot + slot / 2,
          y: H - pad.b + 14,
          fill: t.muted,
          "font-size": 10,
          "text-anchor": "middle",
          "font-family": FONT,
        }, [`${p.n_ues}`])
      );
    });

    const detPts = points
      .map((p, i) => {
        const x = pad.l + i * slot + slot / 2;
        const y = pad.t + innerH - (Number(p.det_rate_pct || 0) / 100) * innerH;
        return `${x},${y}`;
      })
      .join(" ");
    svg.appendChild(el("polyline", { points: detPts, fill: "none", stroke: t.ok, "stroke-width": 2.3 }));
    points.forEach((p, i) => {
      const x = pad.l + i * slot + slot / 2;
      const y = pad.t + innerH - (Number(p.det_rate_pct || 0) / 100) * innerH;
      svg.appendChild(el("circle", { cx: x, cy: y, r: 3, fill: t.ok }));
    });

    const batchPts = points
      .map((p, i) => {
        const x = pad.l + i * slot + slot / 2;
        const y = pad.t + innerH - (Number(p.batch_ms || 0) / maxBatch) * innerH;
        return `${x},${y}`;
      })
      .join(" ");
    svg.appendChild(el("polyline", { points: batchPts, fill: "none", stroke: t.warn, "stroke-width": 1.8 }));

    legendRow(svg, ["Flows/s (left)", "Det %", "Batch ms (right)"], [t.blue2, t.ok, t.warn], 18, pad.l);
    axisLabels(svg, W, H, pad, opts.xLabel || "Concurrent UEs →", opts.yLabel || "Throughput (flows/s)");
    // right y-axis title
    const rcx = W - 12;
    const rcy = pad.t + innerH / 2;
    svg.appendChild(
      el("text", {
        x: rcx,
        y: rcy,
        fill: t.warn,
        "font-size": 10,
        "text-anchor": "middle",
        "font-family": FONT,
        transform: `rotate(90 ${rcx} ${rcy})`,
      }, ["Batch latency (ms)"])
    );
    clear(host).appendChild(svg);
  }

  /** Kept for API compatibility — prefer HTML dist list in Measurements page */
  function percentileStrip(host, block, title) {
    if (!block || !block.n) return empty(host, "No latency samples");
    const t = theme();
    const rows = [
      ["min", block.min],
      ["median", block.median],
      ["mean", block.mean],
      ["p95", block.p95],
      ["max", block.max],
    ];
    const W = 480;
    const H = 70 + rows.length * 22;
    const svg = el("svg", {
      viewBox: `0 0 ${W} ${H}`,
      width: "100%",
      height: String(H),
      role: "img",
      "aria-label": title || "Latency distribution",
    });
    svg.appendChild(
      el("text", {
        x: 12,
        y: 18,
        fill: t.text,
        "font-size": 12,
        "font-weight": 700,
        "font-family": FONT,
      }, [title || "Distribution (ms)"])
    );
    const max = Math.max(...rows.map((r) => Number(r[1]) || 0), 0.001);
    rows.forEach((r, i) => {
      const y = 40 + i * 22;
      const w = (Number(r[1]) / max) * (W - 140);
      svg.appendChild(
        el("text", { x: 12, y: y + 4, fill: t.muted, "font-size": 11, "font-family": FONT }, [r[0]])
      );
      svg.appendChild(el("rect", { x: 70, y: y - 8, width: Math.max(w, 2), height: 12, rx: 3, fill: t.blue2 }));
      svg.appendChild(
        el("text", {
          x: 78 + Math.max(w, 2),
          y: y + 3,
          fill: t.text,
          "font-size": 11,
          "font-family": FONT,
        }, [`${Number(r[1]).toFixed(1)}`])
      );
    });
    clear(host).appendChild(svg);
  }

  return {
    stackedLatency,
    multiLine,
    areaLine,
    donut,
    groupedBars,
    comboScale,
    percentileStrip,
    empty,
  };
})();
