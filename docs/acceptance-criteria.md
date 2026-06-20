# Apollo Autonomous-Loop Acceptance Framework

**Status:** spec / driving document for the brutal acceptance gates.
**Scope:** the two-tier accept framework that decides KEEP vs REVERT for every change the autonomous loop makes.
**Hardware basis:** Apple Silicon M1, 8 GiB RAM, pagesize 16384, swap total 2 GiB. All calibrated numbers below are one-box numbers — recalibrate on different hardware.

> Supervision doctrine applies. This document never declares anything "closed" on small-N evidence. Floors are calibrated from n=31 steady-state rows over ~15h on ONE box — preliminary, not closed (project bar: N≥500). Where a number is a bootstrap default that re-derives live, it says so.

---

## 1. What the user actually asked for, and the honest translation

The user wants Apollo to behave like supercomputer-grade resource management:

- stable while ~50 apps are open,
- app launch that is fast and precise,
- no crashes,
- no scarcity-thrashing,
- no stalls.

These are stated as absolutes ("zero thrashing", "zero latency", "does not crash", "like a supercomputer"). Literal zero is mostly **impossible**, and pretending otherwise would paper over the one thing this framework exists to catch. So each demand is restated as a measurable SLO together with the reason literal-zero cannot hold.

### The scarcity-vs-benign-churn distinction (the load-bearing idea)

The single most important honesty point: **a high `thrashing_score` is not proof of scarcity.** On this box, Brave's page-cache churn routinely spikes `thrashing_score` past 50,000 while swap is completely flat and memory pressure is well under the survival floor. That is *benign churn* by a cooperative app, not memory scarcity — and the daemon already correctly declines to act on it (the `maintenance_purge_skipped_pressure_low_total` counter sits in the thousands, proving the defense works). This is production scar `533bad6`.

Therefore "zero thrashing" is translated as **zero scarcity-thrashing**, where scarcity is a *cross-gated* condition that requires swap to actually be moving (or OOM to be genuinely imminent), not just a raw thrash number. Benign churn is a deliberate PASS.

```
scarcity ≡ (swap_delta_bps > 0  AND  memory_pressure ≥ 0.65) sustained ≥ 2 reads
           OR  si_p_oom_30s ≥ 0.80
```

### Honest SLO translation table

| User said | Why literal-zero is impossible | HONEST SLO |
|---|---|---|
| "zero thrashing" | `thrashing_score` spikes 50k+ on benign Brave page-cache churn with flat swap (scar `533bad6`). Raw thrash ≠ scarcity. | **scarcity-thrashing == 0** under the cross-gated definition above. High thrash + flat swap + pressure < 0.65 is PASS by design. |
| "zero latency" | A cycle is real work; the healthy steady-state p95 floor is 59–92 ms. | **p95_cycle_ms ≤ 92 ms** (steady p90). Single transient spikes (observed 153/193 ms rode through fine at AIS 95.43/93.22) are only a fail on **2 consecutive** reads, never one. |
| "does not crash" | This is the one achievable hard-zero. | **failures == 0 AND last_error == null AND cycles strictly advancing AND no protected process killed.** |
| "fast precise app launch" | Real GUI cold-launch is high-noise (LaunchServices/dyld). | **synthetic CLI target cold-launch ≤ 2.5× quiesced baseline** (STRESS tier). Real-GUI launch is an advisory `--real-apps` demo, never a gate. |
| "stable with ~50 apps / like a supercomputer" | Launching 50 GUI apps is non-deterministic. | **synthetic N-holder stress** (RSS ≈ 1.4× physical, forces real swap) judged on memory-bounded + pressure-recovery + no-wrongful-kill (STRESS tier). |

### The two tiers and the one mandatory predicate

```
KEEP(change) ⟺  FAST_PASS(change)
                AND ( is_risky_or_structural(change) ⟹ STRESS_PASS(change) )
```

- **FAST gate** runs on every keep-decision and every deploy. Cheap, reads existing telemetry, hard-reverts.
- **STRESS gate** runs only before promoting a change classified RISKY/structural. Heavy (~4–5 min), induces real scarcity, judged on kernel-truth the daemon cannot author.

A change that fails FAST is reverted. A risky/structural change that passes FAST but fails STRESS is **not promoted**.

---

## 2. FAST gate — hard-SLO table (every number cites its percentile)

Runs in `apollo-deploy-gate.sh`'s post-deploy window and is re-runnable by the regression probe each tick. Reads only `runtime_metrics.json` + one `ps` set-diff + `launchctl print`. Reuses the existing warmup poll (`cycles ≥ 800 AND ais_score > 0`, 90s baseline with up to 720s budget — v3's `cycles ≥ 400` false-FAILed, so v4 = 800).

All percentiles are from steady-state n=31 (the warmup row is dropped — it alone inflates AIS σ from 1.34 to 16.28 and p95 σ from ~26 to ~307, so warmup exclusion is mandatory, not optional).

### 2a. Absolute hard gates

| # | Gate | Threshold | Derived from | Revert tier |
|---|---|---|---|---|
| H1 | AIS floor | `ais_score ≥ 92.0` | steady-state **p25 = 92.46**, rounded down for one-box headroom. Replaces the flat 87 which lets a 95→88 collapse PASS. | Tier-B human |
| H2 | crash/error | `failures == 0` AND `last_error == null` AND `cycles_end > cycles_start` (strict advance) | `failures` is a **full-sample invariant** (all 32 rows == 0). The strict cycle-advance catches a tombstoned daemon that started exactly 1 cycle, which a bare `cycles > 0` check misses. | **Tier-A auto** |
| H3 | latency ceiling | `p95_cycle_ms ≤ 92`; fail only on **2 consecutive** reads > 92 | steady-state **p90 = 92.0**. A single transient (92–193 ms) is not a hard fail — 153/193 ms rode through at AIS 95.43/93.22. | Tier-B human |
| H4 | scarcity-thrashing | `== 0` (cross-gated definition, §1) AND `memory_pressure ≤ 0.71` | pressure ceiling = steady **observed max 0.712** (survivable). Raw thrash never hard-fails. | Tier-B human |
| H5 | no wrongful kill | protected-set `ps` set-diff: every hard-listed process present at T+0 is still present at T+window | `safety.rs` hard-list (confirmed present: `kernel_task, launchd, WindowServer, loginwindow, configd, …` + `Finder, Antigravity, Claude, Brave Browser, language_server`). Deterministic, lowest-noise gate. Synthetic holders MAY die — that is correct. | **Tier-A auto** |
| H6 | test evidence | existing 3-tier presence check; `--skip-test-check` bypass preserved | pre-deploy abort (exit 1), not a rollback. Presence-only is acceptable as a hard gate; TDD-rigor belongs in CI. | pre-deploy abort |

### 2b. Regression-delta hard gates (close the never-compared `PRE_SNAP`)

The current gate captures `PRE_SNAP` and never compares it. These gates use that already-captured snapshot — **zero new reads** — to catch a regression that is still inside the absolute floor.

| # | Gate | Threshold | Derived from | Revert tier |
|---|---|---|---|---|
| R1 | AIS no-regress | `POST_AIS ≥ PRE_AIS − 3.0` | 3 pp ≈ **2.2σ** of steady AIS sd (1.34). Ships **SOFT (log-only) for the first 20 deploys**, then promotes to HARD — a `coding`→`build` workload shift can move AIS several points benignly. | Tier-B human |
| R2 | failures no-rise | `POST_FAILS ≤ PRE_FAILS` | belt-and-suspenders with H2. | **Tier-A auto** |
| R3 | latency no-regress | `POST_p95 ≤ PRE_p95 × 1.25` | no >25% latency regression even when under the 92 ms absolute ceiling. | Tier-B human |

### 2c. Composite no-regression (death-by-a-thousand-cuts)

A weighted **geometric mean** so one collapsed dimension cannot be averaged away (the AIS-vs-D4-drag lesson — a strong AIS must not mask a resource-score collapse). Each metric is normalized to a `[0,1]` goodness `gᵢ` using named production constants / observed steady extremes (no invented anchors), then:

```
S = 100 · Π gᵢ^(wᵢ / Σwⱼ)
```

Normalizers:

```
g_ais       = clamp01((ais_score - 80) / (96 - 80))             # 80 usefulness floor, 96 best observed
g_d4        = clamp01(ais_resource)                              # already [0,1]; live 0.692 (the drag)
g_p95       = clamp01((110 - p95_ms) / (110 - 59))              # 59 best steady, 110 bad
g_pressure  = clamp01((0.70 - memory_pressure) / (0.70 - 0.45)) # 0.70 EMERGENCY_PURGE_FLOOR, ~0.45 min obs
g_swapdelta = clamp01(1 - swap_delta_bps / (4*1024*1024))       # 0 B/s = 1.0; 4 MB/s = 0
g_compress  = clamp01((0.30 - compressed_memory_ratio) / 0.30)  # 0.30 saturation onset
g_stall     = clamp01(1 - stall_fraction / 0.20)                # 0.20 heavy contention
g_fluidity  = clamp01((fluidity_score - 0.65) / (1.0 - 0.65))   # 0.65 fluidity_degraded floor
```

Weights (true-scarcity + UX outweigh self-reported scores):

```
swapdelta 2.5, pressure 2.0, fluidity 2.0, d4 1.5, p95 1.5, compress 1.5, ais 1.0, stall 1.0
```

(Stress-only metrics join `S` in the STRESS tier; in FAST these 8 self-reported terms are used.)

**Composite acceptance:** `S_post ≥ S_baseline − ε_S`, where `ε_S = 1.5 · σ_S`.

---

## 3. No-regression vs rolling baseline (the brutal core)

This is the mechanism that closes the seam where the current gate captures `PRE_SNAP` but never compares it.

### Baseline

For each guarded metric:

```
baseline_m = median(last K accepted-AND-stable runs)
K = 7
```

K = 7 spans ≈ 3.4h at the observed ~29 min/row cadence — long enough to cross a workload-mix change, short enough to track a real improvement. The median resists a single-run yank.

### Noise band

```
band_m = max(1.5 · σ_m, floor_m)        # k = 1.5 → one-tailed false-revert ≈ 6.7% (the knee)
```

`σ` is taken from the **steady-state** trend table (warmup excluded — mandatory).

| metric | σ (steady) | 1.5σ | floor | effective band |
|---|---|---|---|---|
| ais_score | 1.34 | 2.01 | 0.5 pp | **2.0 pp** |
| p95_cycle_ms | robust ≈ 9 (raw 26 is contaminated by 2 transient spikes → use MAD·1.4826 / trimmed body) | 13.5 | 5 ms | **14 ms** |
| memory_pressure | 0.051 | 0.077 | 0.02 | **0.06** |
| swap_delta_bps | ~0 (flat all window) | — | absolute | **2 MB/s** (growth matters, not magnitude) |
| ais_resource, compressed_ratio, fluidity, stall | not in the 7-field trend log → **measured live** during the gate's own baseline window | — | per-metric | computed live, NOT hardcoded |

### Per-metric regression test

For each guarded metric, with worse-direction `dir`:

```
regressed(m) = (dir == higher ∧ post > base + band)
             ∨ (dir == lower  ∧ post < base − band)
```

**ANY single `regressed(m)` true → revert.** The composite `S` does NOT vote it back in. A change that lifts AIS 4 pp but pushes `swap_delta_bps` 3 MB/s over band is REVERTED — survival beats every other consideration.

### Baseline-update rule (anti-ratchet / anti-boiling-frog)

- The baseline advances **only** when a run is accepted AND internally stable: intra-window p95 σ < 2× baseline σ, no Tier-H flap, monotonic cycles.
- Worse-direction baseline movement per update is clamped to `≤ 0.5 · band` (prevents a slow boil where each accepted run is a hair worse than the last).
- Rejected, reverted, and warmup rows never enter the K-buffer.

---

## 4. STRESS gate scorecard (Gate 3.5)

Runs after FAST passes, before promoting a RISKY/structural change. It deliberately induces **true scarcity** — the load must actually move swap, or it proves nothing. Heavy (~4–5 min).

### Load (synthetic holders, deterministic)

- 8 holders × 1400 MiB = **11,200 MiB ≈ 1.37× physical** → forces swapouts. Pagesize 16384 confirmed (load-bearing for the swapout-bytes math; it is NOT 4096).
- Holders re-touch their pages each second → working set stays hot → forces real eviction → crosses from benign churn into true scarcity.
- Ramp one holder every 3 s (pressure gradient), hold 120 s, release, recover 30 s.
- `trap 'kill $HOLDER_PIDS' EXIT INT TERM` guarantees cleanup. Anon touch-only → zero filesystem/data risk.

### Observer (250 ms cadence, kernel-truth, not daemon-authored)

`vm_stat` deltas (swapouts/swapins/compressions per sec), `sysctl vm.swapusage`, `ps -axo comm` ∩ protected-baseline, `launchctl print system/<label>` liveness, a `nanosleep(2ms)` overrun probe (median-of-5/sample), and the daemon's own `cycles/failures/last_error/pressure` read ONLY to verify it is alive and advancing. Output → JSONL at `/tmp/apollo_stress_$$.jsonl`.

### Scorecard

| # | Line | Threshold | Type | Derived from |
|---|---|---|---|---|
| S1 | No wrongful kill | protected-baseline set-diff `== ∅` at every sample + post | **HARD** | `safety.rs` hard-list (scars `03472d7`/`a98b33a`). THE invariant. |
| S2 | Daemon survived | `alive == 1` every sample AND `cycles` strictly increasing AND post `failures == 0` AND `last_error == null` | **HARD** | Gate-3 today, but now *under load*. |
| S3 | Pressure recovered | peak may exceed 0.65 (we *want* scarcity) but crosses back **< 0.65 within 20 s** of last holder exit | **HARD** | `CRITICAL_PRESSURE_BYPASS = 0.65`; recovery is the supercomputer claim. |
| S4 | Swap bounded | median `swapouts_per_sec` over hold **≤ 2000 pg/s** (≈ 31 MB/s at 16 KB) AND post-recovery `swap_used` ≤ baseline + 256 MB | **HARD** | good manager leans on the compressor, not the SSD; no permanent leak. |
| S5 | Compressor absorbed | peak `compressed_memory_ratio` **< 0.55** | ADVISORY | leading scarcity indicator (saturation onset). |
| S6 | p95 under load | `p95_cycle_ms` during hold **≤ 120 ms** | ADVISORY | steady p90 = 92 + 30% stress headroom (high p95 under deliberate load is NOT a regression). |
| S7 | FG responsive | median `overrun_us` **≤ 8000 µs** AND deadline-miss (>16 ms) **≤ 10%** of hold samples | ADVISORY | 16 ms frame budget; median-of-N (jitter is noisy → median, not mean). |
| S8 | Cold-launch under load | synthetic CLI target spawn→ready **≤ 2.5× quiesced baseline** | ADVISORY | relative (absolute is box-dependent); fixed target (real-GUI too noisy). |
| S9 | No benign false-alarm | if `thrashing_score` spikes > 50k but `swap_delta_bps ≈ 0` AND pressure < 0.65, NO protected action is taken (S1 holds through the spike) | ADVISORY | scar `533bad6` — proves the churn-vs-scarcity discrimination actually fires. |

### Scoring

```
HARD (S1–S4):  ALL must pass, else STRESS_PASS = false (score = 0).
ADVISORY (S5–S9): 20 pts each → 100 max.
stress_score = all_hard_pass ? sum(advisory_pass × 20) : 0
STRESS_PASS ⟺ all_hard_pass AND stress_score ≥ 80      # 60–79 = PASS-MARGINAL (surface to human)
```

---

## 5. Wiring into the autonomous loop

### FAST accept predicate (mandatory, every keep-decision)

```
FAST_PASS ⟺  (H1 ∧ H2 ∧ H3 ∧ H4 ∧ H5 ∧ H6)
            ∧ (R1 ∧ R2 ∧ R3)
            ∧ (∀ m ∈ guarded:  ¬regressed(m))
            ∧ (S_post ≥ S_baseline − 1.5·σ_S)
            ∧ (no NOISY proxy cross-confirmed by a TRUE-scarcity breach)
```

The FAST gate runs in the deploy-gate post-deploy window AND can be re-run by the regression probe on every tick.

### STRESS gate placement

The STRESS gate runs **after FAST passes** and **before promoting** any change the regression-fix-loop's SAFE/RISKY classifier flags as RISKY/structural — gating/suppression/calibration thresholds, starving-gate consumers, anything that changes daemon behavior under load. SAFE changes (telemetry plumbing, additive protection guards, cap-at-mutation, doc fixes) do **not** require STRESS.

### Doctrine: it REVERTS on fail and never auto-applies gating/calibration

- A FAST fail reverts. A STRESS fail blocks promotion (revert suggested).
- Revert is **tiered**:
  - **Tier-A auto-revert (no human):** H2, H5, R2 — unambiguous, deterministic, externally observed, dangerous to leave running. The gate executes `sudo cp <prior-binary> $BINARY_DST && bootout && bootstrap`, captures `incident_pre/post.json`, and re-verifies the prior binary re-establishes baseline.
  - **Tier-B surface-for-human:** H1, H3, H4, R1, R3, composite-`S`, and all STRESS fails — calibration-sensitive or benign-churn-adjacent. The gate prints the full revert command plus evidence, exits nonzero, and the human pulls the trigger.
- **NEVER auto-apply gating / suppression / calibration changes** (project doctrine). A STRESS-gated RISKY change additionally needs ≥500-obs validation before being called "closed" — a STRESS pass is a promotion gate, not a closure claim.
- Tier-A auto-execute is a behavioral change vs today's suggest-only gate, so it ships behind `--auto-revert` defaulting **OFF**; it requires explicit user opt-in (the one decision that needs the user's call).

### Scripts

- **`scripts/apollo-accept-fast.py`** — the FAST tier. Reads `PRE_SNAP`/`POST_SNAP` JSON (passed as args, reusing the existing gate's snapshots), does the `ps` set-diff and `launchctl print`, implements H1–H5, R1–R3, the composite `S`, the rolling-baseline buffer (`/var/lib/apollo/accept_baseline.jsonl`, K=7), and the per-metric regression test. Exits 0 on PASS, nonzero with a tier label (auto vs human). Pure stdlib python3.
- **`scripts/apollo-stress-gate.sh`** — the STRESS orchestrator + observer + scorecard. Baseline 15 s → ramp 24 s → hold 120 s → recovery 30 s. Exit 0 PASS / 5 STRESS FAIL. Supports `--real-apps` (advisory demo, off) and `--skip-stress` (hotfix, logged).
- Supporting (compiled to `/tmp` at runtime): **`scripts/apollo-stress-holder.c`** (synthetic memory holder), **`scripts/apollo-sleep-overrun.c`** (`nanosleep(2ms)` overrun probe).
- **`scripts/apollo-accept-gate.sh`** — the thin orchestrator that wraps the unmodified `apollo-deploy-gate.sh`: runs it (Gates 1/2/deploy/3), passes its `PRE_SNAP`/`POST_SNAP` to `apollo-accept-fast.py`, and — only if the change is tagged RISKY/structural (flag `--risky` or auto-detected from the diff touching gating/calibration files) — invokes `apollo-stress-gate.sh`. Honors `--auto-revert` (default off).
- The live `apollo-deploy-gate.sh` is wrapped/orchestrated, **never edited in place**.

---

## 6. Limits & honesty — what is NOT guaranteed, and why

1. **Literal zero is impossible** for thrashing and latency. The framework guarantees *zero scarcity-thrashing* (cross-gated) and a *bounded* p95, not the absolutes the user phrased. The only genuine hard-zero is "does not crash" (H2/H5/S1/S2).
2. **n = 31 steady rows, ~15h, ONE M1 8GB box** — preliminary, NOT closed (project bar: N≥500). The AIS/pressure/p95 floors are well-supported (low variance, warmup excluded). The noise bands re-derive live as the K-buffer fills; the hardcoded bands are bootstrap defaults.
3. **Benchmark noise is real.** The `k=1.5` band gives a ≈6.7% one-tailed false-revert rate by design — accepted as the knee. Transient single-cycle latency spikes are explicitly *not* failures; only 2-consecutive breaches fail H3.
4. **`92 ms` and `92.0 AIS` are one-box numbers** — not portable. Recalibrate on different hardware or a sustained different `current_workload`.
5. **R1 (3 pp AIS no-regress) ships SOFT for 20 deploys**, then HARD — a benign `coding`→`build` workload shift can move AIS several points, and that variance must be characterized before the bar bites.
6. **One stress run = one observation.** STRESS is a binary promotion gate, not a distribution claim. Closure (N≥500) requires tracking the score across many deploys.
7. **H1/H3 stay self-reported.** AIS is kept as the best single composite but is NOT the survival proof. Survival proof = H5 + S1–S4 (kernel-truth from the STRESS observer); that external observer is what breaks the current gate's self-reported tautology.
8. **Synthetic holders ≠ real 50-app page-access patterns** — an inherent ceiling. The `--real-apps` demo partially covers it, advisory only.

### Metrics deliberately excluded from gates

- **`max_boost_single_name`** — its trend (89→53) is decaying pre-fix residue, so its p25=65.5 is a regression's corpse, not a distribution. Use the *detector's* fresh-30-min-window threshold (>20 HIGH) when needed, never the trend p-values.
- **`refault_peak_per_sec`** — a sticky max-hold watermark (984935 ×6). If gated at all, gate on `refault_delta_per_sec` (p90=7977; storm line 100k), advisory only.

### Still uncertain (flagged for log-then-promote)

Whether the 3 pp R1 delta and the `0.5·band` anti-boiling-frog clamp are too tight under legitimate workload shifts. There is no production data on this yet; both ship in log-only/soft mode first.
