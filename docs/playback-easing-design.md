# Playback-Aware Memory Easing — Design + Acceptance Criteria

Origin: 2026-06-21 multi-agent design workflow (10 agents) for the recurring
"occasional microstutter on high-quality 4K content" symptom on M1 8GB.

## Diagnosis (corrected)

Earlier verdict "Apollo is clean" was **incomplete**. It held for `maintenance_purge`
(8× ever) but the workflow found two **latent** paths that can themselves cause the
stutter:

- **page_reclaim.rs** at pressure ≥0.80 purges *through* the foreground gate → can
  blanket-refault the live 4K working set (separate path from maintenance_purge).
- **zombie-hunter** nominates Brave renderer/GPU helpers for jetsam-demote (was
  firing 974×, always failing) — a live renderer demote would frame-drop.

Root of the symptom itself is still organic: on 8GB the 4K working set is LIVE and
irreducible; under pressure the kernel reclaims it and faulting it back = stall
(refault_peak hit 593k/s). A GC frees garbage; you cannot GC live memory. The only
lever is to free OTHER idle memory to lower total pressure — and that only helps
when idle RAM exists to free.

## Verdict: Wave 1 (PROTECT) ships; Wave 2 (EASE) is data-gated

Adversarial red-team conclusion: **Wave 1 is safe and worth shipping; Wave 2 easing
is likely not worth building on 8GB as specified** (the actuator it would reuse,
`run_warn_limits`, already pokes Brave renderers — the "idle-only additive" story is
false; and the lever cannot move the LIVE working set). Decide Wave 2 from data.

### WAVE 1 — PROTECT (additive, closes latent regressions)

| ID | Change | File:line | Risk | Status |
|----|--------|-----------|------|--------|
| P1 | zombie-hunter early-out also `is_chromium_family` (kills 974× renderer demote) | `main.rs:4092` | LOW-add | DONE |
| P2 | SetMemorystatus execute chokepoint also `is_chromium_family` (complete mediation) | `execute_actions.rs:1074` | LOW-add, safety-crit | DONE |
| P3 | `mediaplaybackd` → name-keyed hard-list | `safety.rs:60` | LOW-add | DONE |
| P4 | `page_reclaim` skips purge during refault-storm/high-bw, fed RAW pressure (C1) | `page_reclaim.rs:149` + `main.rs:3487` | **MED suppression** | TODO (separate gate) |
| pmset | browser `Video Wake Lock`/`Playing audio` assertion → media-app detection | `user_context.rs:465` | LOW-add | TODO (Wave 2 prep) |

C1 fix for P4 (adversarial): page_reclaim must be fed the **raw-preferring**
`high_bw_physical_pressure` (main.rs:4067), not the effective
`snapshot.pressure.memory_pressure`, or the survival escape fires early.

### WAVE 2 — EASE (gated experiment, NEVER blind-applied, ≥500 obs)

Build ONLY if instrumentation shows idle/reclaimable RAM exists during the user's
stutter windows. If 8GB is already tight with no slack, easing does nothing —
protect-only is the honest verdict.

- Trigger `is_hq_playback_likely()` = `is_audio_running_somewhere() &&
  foreground_is_media_app && !is_realtime_call_active()`, survival escape at
  pressure ≥0.70. Honest blind spot: muted 4K is undetectable (no audio signal),
  and no resolution/bitrate signal exists on this M1 — "playback" ≠ "4K".
- Action: arm the existing `run_warn_limits` with a playback pressure floor (E1 —
  the ONE new calibration threshold, floor 0.45 first deploy). NOTE adversarial
  C2: `run_warn_limits` already pokes Brave renderers via `is_bg_renderer` — the
  exclusion story must be fixed before E1 ships.
- **Blocking pre-condition (C3):** `memory_pressure_raw` reads missing/0 in the
  serialized `runtime_metrics.json`; the in-memory snapshot field exists. Verify it
  is populated before gating E1 on it.

## Acceptance criteria (measurable)

Primary metric: **`refault_delta_per_sec` / `refault_peak_per_sec` during detected
playback** (the prime "the video working set is being reclaimed" proxy). For sub-cycle
transients, sample `vm_stat` Decompressions-Δ out-of-band ≥2Hz, A/B feature-off vs -on
over a fixed 4K clip.

Hard gates (reuse `scripts/apollo-accept-gate.sh`):
- H5 zero protected-kill — **zero** `vm_pressure_send`/demote/throttle landing on a
  Brave/mediaplaybackd pid in `journal.jsonl` (the absolute safety gate).
- H4 scarcity-thrashing==0; watch the over-strip failure mode (easing trades
  video-refault for app-switch-refault).
- H3 p95≤92ms; R1/R3 no-regression vs rolling baseline.
- Wave-1 success signal: zero zombie action emitted for Chromium names (was 974×);
  no new `memorystatus-send-failed:` naming Brave helpers.

Benchmark (TODO): `scripts/apollo-playback-bench.py` — a controlled high-page-touch
"playback-like" working set under background pressure, sampling the primary metric +
vm_stat + pressure before/during/after, scorecard 0-100, bounded + cleanup-on-signal.

## Honest limits (8GB physics — do not over-promise)

1. The 4K working set is LIVE — this reduces stutter **frequency/magnitude, NOT
   occurrence**. "Less often, less severe" is a PASS.
2. When no idle RAM exists to free, easing does nothing.
3. Muted 4K is undetectable; "playback detected" ≠ "4K".
4. Report **preliminary**, never closed, until ≥500 playback-cycle obs in production.
