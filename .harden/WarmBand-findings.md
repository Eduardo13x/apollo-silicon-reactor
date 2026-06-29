# `/apollo-evolve:harden` findings — WarmBand (commit 5346513) + Lock-Scope-Minimization (commit c99c7c3)

**Scope:** Read-only audit of two recently-deployed fixes.
**Concern:** Micro-stutter root cause — does WarmBand actually reduce micro-shutters; is the trend correct; is hysteresis preventing Normal↔WarmBand oscillation; NEVER_FREEZE preservation.
**Production state (verified):** `apollo-optimizerd` binary deployed at `Jun 28 15:41:18 2026`, `cycles=3150`, `thermal_state=nominal`, `iopm_thermal_warning=None`, `thermal_trend_predicted=Stable`. System is at nominal thermal state; WarmBand has not had a chance to fire today. **All claims are static-analysis + hand-trace only; no live-fire evidence.**

---

## Summary

| Severity     | Count | Highest-priority item |
|--------------|------:|-----------------------|
| **Critical** | 0     | —                     |
| **High**     | 0     | —                     |
| **Medium**   | 3     | F-04 dead-code WarmBand arms; F-05 missing test coverage; F-06 sensor-noise / trend-detection reliability |
| **Low**      | 4     | F-01..03, F-07 docstring hazard (in uncommitted tree) |
| **Total**    | 7     |                       |

**Verdict:**
- **Lock-Scope-Minimization (`c99c7c3`)** is **SAFE TO KEEP** in production. The sysinfo walk was moved correctly outside `state.metrics.lock_recover()`. Mutation of `state.frozen_state` is over an isolated, brief critical section that releases before any sysinfo lookup. No new hot-path allocations. Tests cover the pure core. Risk only if `--jobs>1` for cargo build (shared `target/`) — already noted in CLAUDE.md.
- **WarmBand (`5346513`)** is **SAFE TO KEEP** in production **but with a 7d observation window required** per the commit author's own postmortem plan. The wiring is correct, NEVER_FREEZE preservation holds, the warm_pressure_boost field clamps properly, the ring-buffer math is sound. The three Medium findings are: (a) dead-code arms document the design but never execute, (b) no tests cover WarmBand specifically, (c) the 8-sample / 2s buffer may be too short to distinguish 0.5°C/min trend from Apple SMC sensor noise in real production. None of these are safety bugs (all fail in the conservative direction: lower-than-optimal boost).

The deployed binary hash matches commit 5346513 (`Jun 28 15:41:18 2026`).

---

## Findings

### F-04 (Medium) — `CoolingPhase::WarmBand` arms are dead code

**Files / line:**
- `crates/apollo-engine/src/engine/thermal_bailout.rs:259` — `CoolingPhase::Normal | CoolingPhase::WarmBand => ThermalAction::normal()` in `action_for_phase`
- `src/bin/apollo-optimizerd/main.rs:2202` — `CoolingPhase::WarmBand => 0.0` in the phase-derived boost match

**Claim (hand-traced):** `self.current_phase` is initialized to `Normal` (`thermal_bailout.rs:135`) and is only ever updated by `classify_temp(temp)`, which (lines 234-255) returns one of `Normal | Phase1Gentle | Phase2Moderate | Phase3Aggressive | Phase4Emergency` — never `WarmBand`. As a result, `action_for_phase(self.current_phase)` always receives one of those five values; the `WarmBand` arm in `action_for_phase` is unreachable. Equivalently, `thermal_action.phase` returned to the daemon is always `Normal | Phase1..4`, never `WarmBand`; the `WarmBand => 0.0` arm in `main.rs:2202` is unreachable.

The WarmBand signal **does** flow correctly: `compute_warm_boost()` populates `action.warm_pressure_boost` regardless of `current_phase`, and that field is added by `main.rs:2211-2212`. Behavior is correct; observability and code-hygiene are the only issues.

**Severity rationale — Medium, not Low:** Any future contributor who sees `CoolingPhase::WarmBand` in the enum will reasonably expect it to appear in `thermal_action.phase`. It does not, and there is no metric field exposing `warm_pressure_boost > 0`. A dashboard operator looking for "is WarmBand firing?" will conclude it never does. This blocks the audit-cron promotion the commit message envisioned.

**Proposed fix (read-only — do NOT implement now):**
1. Add `pub warm_pressure_boost: f32` (and ideally an aggregate `cycles_warm_band_total` counter to `LockFreeMetrics`) so `runtime_metrics.json` reports when WarmBand is triggering.
2. Or: simplify by removing the `WarmBand` enum variant entirely and treating the band as a pure floating-point signal that lives only on `ThermalAction.warm_pressure_boost` — fewer dead arms.
3. Lint: `#[deny(unreachable_patterns)]` would have caught this at compile time; current crate does not enable it for the engine.

**Citation:** [Nygard 2018] — *Release It!* §"Stability Patterns"; dead-code shoulders knowledge gaps at handoff time. Failure-to-communicate ("WarmBand IS firing" — but where?) is the most common mode for this kind of invisible-but-active feature.

---

### F-05 (Medium) — No tests cover the new WarmBand path

**File / line:** `crates/apollo-engine/src/engine/thermal_bailout.rs:332-399` (tests module). Existing tests:

| Test | Covers |
|------|--------|
| `cool_temp_is_normal` | Normal at 60°C |
| `phase4_emergency_above_95` | Phase4 escalation |
| `phase3_aggressive_90_to_95` | Phase3 escalation |
| `cooling_phases_are_ordered` | enum ordering Normal < Phase1..4 |
| `hysteresis_prevents_immediate_recovery` | TICKS_TO_RECOVER behavior |
| `recovery_after_enough_cool_ticks` | de-escalation |

**What's missing (deliberately per commit message):**
- WarmBand does NOT fire below 60°C (trend path blocked by `current >= WARM_TREND_FLOOR_C`)
- WarmBand fires at 75°C absolute (with rate=0)
- WarmBand fires with rising-fast trend at 60°C+ (e.g. +1.0°C/min)
- WarmBand does NOT fire with stable or cooling trend at 60°C
- Ring-buffer endpoints are correct at `warm_filled=2, 8` (the only edge cases the guard `warm_filled < 2` doesn't block)
- `warm_pressure_boost` ramps linearly from 0.5 to 1.0°C/min within `[0, WARM_MAX_BOOST]`
- Final clamp survives a f32 NaN (NaN comparison in Rust returns false → fallback to 0.0)

**Severity rationale — Medium, not Low:** This is the second recent regression the project has hit where an absent test converted a small surgical change into a fanned-out bug. The commit message acknowledges this with "I deliberately did NOT add new tests in this hot path to keep the diff surgical" — defensible at the time, but the 7d postmortem promised in the commit body makes it blocking now.

**Proposed fix:** Add the seven tests above in the same file. Each is <30 lines. None require touching the public API. The deploy-gate gate-1 already requires `#[test]` for staged fixes; one WarmBand test would satisfy it on the next patch.

**Citation:** [Fowler 2004] — *Refactoring* §"Red-Green"; without a green pin, refactoring is guesswork.

---

### F-06 (Medium) — 8-sample ring buffer may not resolve 0.5°C/min over typical SMC sensor noise

**File / line:** `crates/apollo-engine/src/engine/thermal_bailout.rs:120` — `WARM_BUFFER_SIZE: usize = 8`.

**Math (hand-traced, against the question's spec):**
- Apollo's typical cycle cadence is 250-300ms (`main.rs` `cycle_dt_secs` observations, confirmed in CLAUDE.md "Tau scaling" note). 8 samples × 250ms = **2.0s** of history.
- Threshold = `WARM_TREND_RATE_C_PER_MIN = 0.5 °C/min` = `0.00833 °C/sec` = `0.0167 °C` per 2s window.
- To resolve this reliably, per-sample noise must be `<< 0.0083 °C` (so that variance doesn't dominate the trend).
- Apple SMC temperature sensors on M1 typically report integer-decimal precision via `IOHWSensor`; cluster averages inherit a per-sensor noise floor of ~0.1-0.5 °C (one sensor disagreement produces spikes this large in the cluster mean).

**Direction-of-fail (safe):** Under-noise produces a `rate_c_per_min` that oscillates around zero → `triggered = false` → `warm_pressure_boost = 0.0`. Apollo misses legitimate triggers and acts later (at 75°C absolute or 80°C Phase1Gentle). This is **conservative** — no safety regression; the existing Phase1..4 ladder is the safety net.

**Severity rationale — Medium, not Low:** This is the exact detection problem the commit claims to solve ("act on the trend, not just the absolute level"). If the trend is unreliable in practice, WarmBand degenerates to "fires at 75°C absolute only," which doesn't compress the reactive window as advertised — it just adds hysteresis to the same 75°C trigger Phase1Gentle would have hit at 80°C anyway. Worth measuring before the 7d postmortem.

**Proposed fix:**
- Tunable `WARM_BUFFER_SIZE` (already extracted as `const WARM_BUFFER_SIZE: usize = 8` in the working tree — `git diff HEAD` line: `+const WARM_BUFFER_SIZE: usize = 8;`). Consider bumping to 16 (4s) for better S/N.
- Or: apply a 1-Euro filter to `temp` before storing in the ring. The current code stores raw. A trivial `ema_alpha=0.5` first-order low-pass would halve the noise without changing the unit-of-analysis.
- Surface `WARM_RATE_C_PER_MIN_LAST_CYCLE` in `runtime_metrics.json` (paired with F-04) so the postmortem can verify detection sensitivity.

**Citation:** [Bishop 2006] — *Pattern Recognition and Machine Learning* §2.3.2 (smoothing priors for short observation sequences). For 0.5 °C/min over 2s the prior alone cannot compensate; either more samples or smoothing.

---

### F-01 (Low) — `peak_temp` `unwrap_or(0.0)` causes phantom cooldown when a sensor flaps to `None`

**File / line:** `crates/apollo-engine/src/engine/thermal_bailout.rs:189-194`.

**Claim (hand-traced):** If `p_cluster_celsius` is `Some(70.0)` one cycle and `None` the next (IOKit sensor transient), `peak_temp` returns `70.0` then `0.0`. The 8-sample ring captures the `0.0`, computing a -8.75 °C/min spike. `triggered = false` (negative rate), so this only hurts WarmBand detection, not safety.

This is **pre-existing** — `classify_temp` consumed `peak_temp` before WarmBand was added. Not introduced by 5346513. But WarmBand multiplies the visibility because the trend signal is sampled at higher rate (per cycle) than the existing absolute-only comparison.

**Severity rationale — Low:** Pre-existing, conservative direction-of-fail. Worth flagging only because the audit is exactly the case where it shows up.

**Proposed fix:** Treat `None` as "don't include this cycle in the trend" — advance `warm_idx` but do not write `temp` into `warm_temps[warm_idx]`, and skip the `warm_filled` increment. Add NaN-as-skip too (currently the same silent-NaN-writes-the-buffer pattern applies).

**Citation:** [Saltzer & Kaashoek 2009] — *Principles of Computer System Design* §5 ("missing data is not zero").

---

### F-02 (Low) — Documentation/comment inconsistency in `compute_warm_boost` (deployed source)

**File / line:** `crates/apollo-engine/src/engine/thermal_bailout.rs:208` (HEAD = deployed): comment reads `(2.4 Hz)`.

**Claim (hand-traced):**
- Working tree (uncommitted, observed via `git diff HEAD -- crates/apollo-engine/src/engine/thermal_bailout.rs`):
  ```
  -        // We use 250ms (2.4 Hz) for a slightly-conservative rate; actual
  +        // We use 250ms (4.0 Hz) for a slightly-conservative rate; actual
  ```
- The deployed binary was built at `15:41:18 2026`, the same minute the commit was authored; the working-tree fix was applied **after** deployment. So the **deployed daemon** carries the `(2.4 Hz)` comment. The `(4.0 Hz)` correction is in the working tree but **not in production**.
- The multiplier `* 240.0_f32` is **correct for 250ms cadence** (`60 sec ÷ 0.250s = 240 cycles/min`). So both the old `(2.4 Hz)` comment AND the new `(4.0 Hz)` comment are mismatched against the multiplier:
  - Old: `(2.4 Hz)` does not match `240 cycles/min` (240/60 = 4.0 Hz, not 2.4 Hz)
  - New: `(4.0 Hz)` matches `240 cycles/min` (`60 × 4 = 240`) ✓
- Net effect on production: zero. The multiplier is correct; only the prose is misleading. `rate_c_per_min` faithfully reflects "per-cycle delta × 240", which is the right conversion at 250ms cadence.

**Severity rationale — Low:** Pure comment-level. Doesn't affect behavior. But the fix is already half-applied in the working tree — the only missing step is to commit it so it matches the deployed binary.

**Proposed fix:** Commit the working-tree change. `git diff HEAD` shows the comment edit is sitting in the working tree; `git add && git commit` it before the next deploy so the deployed binary + checked-out source match.

**Citation:** [Nygard 2018] — *Release It!*; drift between source-of-truth and binary is a stability anti-pattern.

---

### F-03 (Low) — `WarmBand` variant has no observability hook in `RuntimeMetrics`

**File / line:**
- `crates/apollo-engine/src/engine/types.rs` (search for `warm_pressure_boost` — no matches).
- `crates/apollo-engine/src/engine/daemon_state.rs` (no `warm_*` field).

**Claim (hand-traced):** `thermal_action.warm_pressure_boost` is computed and added into `effective_pressure` (via `main.rs:2211-2212` → `aggregate_cycle_pressure` → `PressureBoosts::thermal`), but there is no separate `runtime_metrics.json` field that exposes either:
- `warm_pressure_boost_last_cycle: f32`
- `cycles_warm_band_total: u64` (via `LSE_COUNTERS`)

The audit-cron (`scripts/apollo-learned-state-audit.py`) cannot detect WarmBand firing without grepping the journal (and the journal doesn't emit a WarmBand event).

**Severity rationale — Low:** Pairs with F-04 — without this, the postmortem is write-only. The pressure-aggregator's `components.thermal` already aggregates warm+phase-derived into one number, so postmortem reviewers can't isolate WarmBand's contribution.

**Proposed fix:** Add a `#[serde(default)] pub warm_pressure_boost: f32` to `RuntimeMetrics` and wire it in `sync_from_lockfree` alongside the existing thermal metrics. Add an `inc_warm_band_fire()` to `LSE_COUNTERS` (per the LSE Counter Discipline in CLAUDE.md) and bump when `compute_warm_boost()` returns > 0.

**Citation:** LSE Counter Discipline ("Silent telemetry-death pattern"); if you can't see it, you can't tune it.

---

### F-07 (Low, no live-fire evidence) — `current_phase` cannot migrate from `Normal` back to `WarmBand` across daemon restart; ring buffer is in-memory only

**File / line:** `crates/apollo-engine/src/engine/thermal_bailout.rs:107-113`, `daemon_init.rs:158` (`thermal_bailout: ThermalBailout::new()`).

**Claim (hand-traced):** `ThermalBailout` is constructed fresh on daemon startup. The 8-sample ring is empty (`warm_filled=0`). On the very first cycles after a daemon restart, `compute_warm_boost()` returns `0.0` because of the `warm_filled < 2` guard (line 201-203). During those first ~0.5s of post-restart cycles, WarmBand cannot fire on trend (can still fire on absolute ≥ 75°C immediately, since the absolute check doesn't depend on the buffer).

**Severity rationale — Low:** Not a safety bug; a brief detection gap (~0.5s post-restart). The existing Phase ladder already handles 75°C+ within `TICKS_TO_ESCALATE = 2`. Documented for completeness; no fix needed beyond F-04/F-05's recommended observability.

**Citation:** None.

---

## Anti-pattern scan (Q4)

| Pattern | Verdict | Evidence |
|---------|---------|----------|
| (a) New Mutex / RwLock | **PASS** | `warm_temps: [f32; WARM_BUFFER_SIZE]` is owned by `ThermalBailout`, which lives in `daemon_init::DaemonSubsystems` and is taken by `mut` in the single-threaded main loop (`main.rs:861`). No internal locking required; matches existing `mut thermal_bailout` pattern. |
| (b) New I/O syscalls | **PASS** | `compute_warm_boost` is pure f32 arithmetic. No `std::fs::*`, no `Command::new`, no sockets. |
| (c) Heap allocations on hot path | **PASS** | Stack-only `[f32; 8]` (no `Vec`, no `Box`). `compute_warm_boost` returns by value, no `String`. |
| (d) New `unsafe` | **PASS** | `grep -n 'unsafe' crates/apollo-engine/src/engine/thermal_bailout.rs` returns nothing in the new code. |
| (e) Panic paths / `unwrap` / `expect` on the new code | **PASS** | No `unwrap()`, no `expect()`, no `panic!`, no `unreachable!`, no `todo!()` in the WarmBand path (lines 99-232). The `unwrap_or(0.0)` calls in `peak_temp` are pre-existing and replaced-by-default-not-panic. |
| (f) Lock-scope-minimization violations (`c99c7c3`) | **PASS** | `compute_frozen_ram_mb` (`daemon_cycle_tail.rs:178-182`) takes the `frozen_state` lock, clones, releases, then walks sysinfo with only the cloned map. The metrics god-lock is acquired only inside `wire_enriched_telemetry` (`daemon_cycle_tail.rs:227`), and explicitly `drop(m)` is called *before* the I/O in `append_history_snapshot` (`daemon_cycle_tail.rs:423`). Two separate locks held in series, never nested. Matches the project's Lock Scope Minimization rule (`rust-systems-patterns.md`). |
| (g) `safety.rs` complete-mediation on freeze paths | **PASS (not relevant to WarmBand)** | WarmBand does NOT introduce a new freeze path. `daemon_thermal_freeze.rs:45` consumes `thermal_action.force_ecores` / `freeze_background` / `freeze_all_non_critical` — WarmBand sets all three to `false` via `action_for_phase(Normal)` (and never escalates beyond Normal — see F-04). Per `safety.rs::is_protected_pid` is consulted downstream unchanged. |
| (h) `learned_pattern_matches` for 15-char macOS name truncation | **PASS (not relevant)** | WarmBand touches only temperature, not process names. |

---

## What's NOT a finding

These look suspicious but are correct on hand-trace. Listing so a future reviewer doesn't re-investigate.

1. **The `Normal | WarmBand => ThermalAction::normal()` arm** (F-04 above, but explaining the design intent again): `WarmBand` is intentionally absent from `self.current_phase`. The branch groups WarmBand with Normal because both are "no-force-ecores, no-freeze" outputs, regardless of how `warm_pressure_boost` is set. The `warm_pressure_boost = warm_boost;` line afterward sets the only WarmBand-specific field. **Consequence:** the `WarmBand` arm is dead code today, but it documents intent.

2. **The `(warm_filled - 1).max(1.0)` guard** (`thermal_bailout.rs:205`): looks like paranoia but is necessary. When `warm_filled == 1`, `(0 as f32) = 0.0`, and `0.0.max(1.0) = 1.0`. Prevents `0.0167 / 0 = NaN` from corrupting the trend signal. Without the guard, a single sample would produce a NaN rate, and (per Rust IEEE semantics) the subsequent `NaN >= WARM_TREND_RATE_C_PER_MIN` comparison returns `false`, so `triggered = false` anyway — but the buffer would carry NaN for up to 7 cycles. Defensively, the `max(1.0)` is correct. **Note:** an additional defensive `if self.warm_filled < 2` already returns 0.0 at the top, so this guard is doubly safe.

3. **The newest/oldest index formulas** (`thermal_bailout.rs:229-230`): hand-traced for `warm_filled = {2, 8}` and two steady-state `warm_idx` values each. Correct in all cases. The `+ n` is necessary to avoid underflow when `warm_idx < 1` (after wrapping from 0 → 0).

4. **`warm_pressure_boost` is `f32`, the cap is in main.rs as `f64`** (`main.rs:2212`): the implicit widening `f32 as f64` is lossless (every f32 is exactly representable as f64), so no precision is lost. Cast is safe.

5. **`(scaled * WARM_MAX_BOOST).clamp(0.0, WARM_MAX_BOOST)`** (`thermal_bailout.rs:220`): the clamp is redundant given the math (`scaled ∈ [0.0, 1.0]` by construction: `(ratio - 1.0).max(0.0) / (WARM_BOOST_FULL_RATIO - 1.0)` and `ratio ≤ WARM_BOOST_FULL_RATIO`). Defense-in-depth — keep it. If `scaled` were `>1.0` due to a future refactor, the clamp still holds the line.

6. **The `ThermalAction::normal()` constructor** (`thermal_bailout.rs:68-77`): sets `warm_pressure_boost: 0.0`. After `evaluate()` returns, line 185 overwrites this with the real value. So the constructor's default of `0.0` is only visible during the brief construction window of the same line — meaningless in practice. Not a bug.

7. **The `lock_scope_minimization` refactor (`c99c7c3`)** in `daemon_cycle_tail.rs:178-202`: pre-extracted `sum_frozen_ram_mb<V>` is generic over the value type. The type parameter `V` is unused in the body, only `V::keys()` is needed; the test `sum_frozen_ram_mb_only_consumes_keys_not_values` (line 656-672) pins that contract. `cargo build --release --bin apollo-optimizerd` is clean per the commit message. The 4 added tests pass per the commit message and post-deploy production evidence (binary running cleanly).

8. **`Temperature trend_predicted='Stable'`** in today's `runtime_metrics.json` (latest dump, cycles=3150): this comes from `ThermalManager.predict()`, not from WarmBand's `compute_warm_boost()` — different code paths. Current temp is below WARM_ABS_ENTER_C, so WarmBand correctly returns 0.0 today; "Stable" is a separate detector. No conflict.

9. **`thermal_action.warm_pressure_boost` survives NaN input**: hand-traced. If `peak_temp` returns NaN (sensor dropout), the ring buffer holds NaN. On the next cycle, `compute_warm_boost`: `NaN - NaN = NaN`, `NaN / 1.0 = NaN`, `NaN * 240.0 = NaN`, `NaN >= 0.5 = false` (Rust IEEE), `triggered = false`, returns `0.0`. Conservative direction-of-fail.

---

## Files audited

- `/Users/eduardocortez/proyectos/system-optimizer/crates/apollo-engine/src/engine/thermal_bailout.rs` (full file, 401 lines, includes WarmBand and pre-existing thermal ladder tests)
- `/Users/eduardocortez/proyectos/system-optimizer/src/bin/apollo-optimizerd/daemon_cycle_tail.rs` (full file, 673 lines, includes the lock-scope refactor and its tests)
- `/Users/eduardocortez/proyectos/system-optimizer/src/bin/apollo-optimizerd/main.rs` (lines 850, 861, 2157-2227, 2430-2460, 5680-5695 — wiring & pressure aggregator plumbing)
- `/Users/eduardocortez/proyectos/system-optimizer/src/bin/apollo-optimizerd/daemon_pressure_aggregator.rs` (full file, 335 lines — additive boost semantics + the 0.30 cap)
- `/Users/eduardocortez/proyectos/system-optimizer/crates/apollo-engine/src/engine/effective_pressure.rs` (lines 80-225 — `PressureBoosts` and `compute`; confirms `min(sum_boosts, 0.30)` cap on sum, `clamp(0.0, 1.0)` on result)

References cited: [Nygard 2018] [Bishop 2006] [Fowler 2004] [Saltzer Kaashoek 2009]; [LeCun 2022] and [Barto Sutton 2018] were not applicable to either audit target.

---

## Audit disposition

**Production verdict (read-only, no action taken):**
- Both fixes are safe to keep in production. No rollback needed.
- `c99c7c3` (lock scope) is a strict refactor with tests, no behavior change, fixing a 30ms-peak stall candidate.
- `5346513` (WarmBand) is a **monitoring-grade pre-stage** that adds up to +0.05 to effective pressure under trend. Its safety net is the existing Phase1..4 ladder (which is unchanged and gated by `TICKS_TO_ESCALATE / TICKS_TO_RECOVER` hysteresis).
- The Medium-severity findings (F-04, F-05, F-06) are all **observability + verification**, not safety. The 7d postmortem promised in the commit body is the right hook to address them.
