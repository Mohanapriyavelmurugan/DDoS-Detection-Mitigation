# Journal Results Package — EdgeGuard MEC DDoS Detection

Style used for generated figures: **white background**, **Times New Roman 14 pt**, labeled axes, 300 dpi PNG + PDF.

Regenerate anytime:
```bash
python paper_results/generate_paper_figures.py
```

---

## Recommended Results Section Structure

### 1. Experimental setup (short paragraph + table)
- Models: XGBoost + Isolation Forest + DPI fusion (threshold 0.61)
- Environments: **Laptop/central** vs **MEC edge node**
- Metrics: M1 latency, M2 throughput, M3 offloading, M4 multi-UE scalability
- Attack types: SYN / UDP / ICMP / HTTP flood, Slowloris

**Table A — Testbed configuration** (write manually in paper):
| Item | Laptop | MEC |
|------|--------|-----|
| Role | Central / baseline | Edge detection node |
| Models | `models_v6` | `models_v6` |
| Scripts | `measure_*.py` | same inside `/app` |

---

### 2. Figures to include (already generated)

| Fig | File | What it shows | Paper message |
|-----|------|---------------|---------------|
| **Fig. 1** | `fig1_m1_latency_laptop_vs_mec` | Bar: stage latency Laptop vs MEC | MEC is faster end-to-end |
| **Fig. 2** | `fig2_m1_latency_stack` | Stacked stage composition | Inference dominates latency |
| **Fig. 3** | `fig3_m2_throughput_vs_ues` | Line: flows/s vs UEs | Throughput scales with load |
| **Fig. 4** | `fig4_m2_throughput_batch_dual` | Dual-axis flows/s + batch ms | Capacity vs compute cost |
| **Fig. 5** | `fig5_m3_offload_share` | Donut: blocked vs reached | ~89% stopped at edge |
| **Fig. 6** | `fig6_m3_offload_timeline` | Line: cumulative packets | After mitigation, core stays flat |
| **Fig. 7** | `fig7_m4_scalability` | Multi-line M4 metrics | Accuracy stays 100% to 100 UEs |
| **Fig. 8** | `fig8_m4_detection_rate` | Bars of detection rate | Robust under concurrency |
| **Fig. 9** | `fig9_summary_scorecard` | Summary strength bars | Optional overview graphic |

---

### 3. Tables to include (CSV ready)

| Table | File | Content |
|-------|------|---------|
| **Table 1** | `tables/table1_m1_latency.csv` | Feature / Inference / Mitigation / Total (Laptop vs MEC) |
| **Table 2** | `tables/table2_m2_throughput.csv` | UE sweep throughput |
| **Table 3** | `tables/table3_m3_offloading.csv` | Offload summary counts & % |
| **Table 4** | `tables/table4_m4_scalability.csv` | Batch / per-flow / flows/s / det rate |

Suggested **Table 5 (comparison summary)** for the paper text:

| Metric | Laptop / Local | MEC | Prefer |
|--------|----------------|-----|--------|
| Mean total latency | ~38.3 ms (bench) / ~500–950 ms live Docker | **~23–30 ms** live bench table | Lower |
| Peak throughput | up to ~3900 flows/s (lab PC) | **~2267 flows/s** | Context-dependent |
| Edge offload | ~87.8% (lab) | **88.9%** | Higher |
| Det. rate @ 100 UEs | 100% | **100%** | Higher / hold |

---

### 4. Extra results journals often expect (optional next)

If reviewers ask for more:

1. **Confusion-style attack-type table** — per attack family (SYN/UDP/ICMP/HTTP/Slowloris): count detected, mitigated, mean fusion score  
2. **Latency CDF / boxplot** — distribution of total ms over 100 flows (from `measure_latency.py` raw logs)  
3. **Ablation** — XGB-only vs IF-only vs full 3-layer fusion (accuracy / latency)  
4. **Before–after mitigation rate plot** — server pps before vs after (from M3 timeline)  
5. **Resource plot** — CPU% / memory during M2/M4 (if you logged `top`/`psutil`)  
6. **False-positive note** — benign traffic session with attack rate ≈ 0 (if available)

---

### 5. Caption examples (copy-ready)

**Fig. 1.** Mean detection-to-mitigation latency by pipeline stage on laptop and MEC. Lower values indicate faster protection.

**Fig. 3.** Detection throughput (flows/s) as concurrent UE load increases on the MEC node. Detection rate remained 100% across the sweep.

**Fig. 5.** Share of SYN-flood packets blocked at the MEC edge versus packets that reached the simulated 5G core server (15 s, 200 pps).

**Fig. 7.** Multi-UE scalability on MEC: throughput, detection rate, and per-flow latency for 1–100 concurrent UEs.

---

### 6. Key claims supported by YOUR MEC numbers

1. **Low edge latency:** total pipeline ≈ **23–30 ms** on MEC.  
2. **High offloading:** **88.9%** attack packets stopped at MEC; only **11.1%** reached core.  
3. **Stable accuracy under load:** **100%** detection from 1 to 100 UEs.  
4. **Sustained throughput:** peak ≈ **2,267 flows/s** with 100% detection.

---

### 7. Honesty notes (important for journal)

- M1 “Laptop” numbers in Fig.1 use `measure_latency.py` bench means; live Docker sniffing latency is higher — state which setting each figure uses.  
- M3 summary “before/after rate = 0%” line in the script text is inconsistent with the timeline; **use packet counts + timeline figure**, not that buggy summary line.  
- M2 peak on MEC (~2267) can differ from a powerful lab PC; discuss hardware context.
