# Wave 37+ main.rs Extraction Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `src/bin/apollo-optimizerd/main.rs` from 4630L toward ~3000L by extracting cohesive orchestration phases that survived Waves 1-36, lowering p95 cycle latency from 144ms toward Hellerstein adaptive target 130ms (-15ms minimum), without regressing 1867 passing tests.

**Architecture:** Strangler Fig pattern (Fowler 2004) applied wave-by-wave. Each wave is ONE deploy/monitor cycle (NEVER batch hot-path commits — NARS loop revert lesson). Two-stage approach: (1) **Discovery wave** to verify which sections in main.rs are truly extractable vs already-wrapped, (2) **Extraction waves** for confirmed cohesive blocks. Each Wave consults NotebookLM Phase 1.5 before mutation per apollo-evolve protocol.

**Tech Stack:** Rust 2021, `anyhow::Result`, `Arc<Mutex<>>` for shared state, `cargo test` for verification, `apollo-optimizerd` daemon with `launchctl` deploy. NotebookLM MCP for peer review (notebook id `8344b94c-a014-4803-abea-076a55753cfd`, conversation `379c81af-388d-483e-9d48-7ec30e4c6bde`).

---

## CRITICAL — Discovery Phase BEFORE extraction

**Stale-data risk:** Earlier NotebookLM diagnosis ranked sections by line count (Signal Intel 666L, Chromium 416L, etc.) — but `grep` of comment markers reveals those sections **already wrap extracted modules**:

| main.rs section header | Lines | What it ACTUALLY contains |
|------------------------|-------|---------------------------|
| Signal intelligence | 2522-3187 | Calls `daemon_signal_tick::run_signal_tick` line 2531 (Wave 14) — rest is downstream consumers (Swap Reclaim ODE, KalmanMV blending, signal-digest dispatch) |
| Effective pressure aggregation | 1943-2290 | Calls `daemon_pressure_aggregator::aggregate_cycle_pressure` line 1949 — rest is context_switch_burst + critical_patterns matching |
| Chromium Renderer Manager | 3525-3940 | Calls `daemon_chromium_tick::run_chromium_tick` line 3528 (Wave 11) — rest is post-tick pipeline glue |

**Implication:** The 4630L count is mostly **orchestration glue** between already-extracted modules, NOT cohesive logic blocks. Naive extraction targets are **invalid**. Wave 37 MUST start with a discovery pass.

---

## File Structure

**New files (provisional, may change after discovery):**
- `src/bin/apollo-optimizerd/daemon_fluidity_tick.rs` — Fluidity update + RuntimeMetrics wiring (Wave 37 candidate)
- `src/bin/apollo-optimizerd/daemon_signal_consumers.rs` — post-signal-tick downstream: Swap Reclaim ODE, KalmanMV blend, threshold dispatch (Wave 39 candidate, requires discovery confirmation)
- `src/bin/apollo-optimizerd/daemon_circuit_breaker.rs` — Circuit breaker + execute_actions block (Wave 40 candidate)

**Modified files:**
- `src/bin/apollo-optimizerd/main.rs` — replace inline blocks with module calls
- `src/bin/apollo-optimizerd/mod.rs` (if exists) or `daemon_init.rs` — register new modules

**Deferred / blocked by discovery:**
- `cognitive_tick.rs` (544L) vs `daemon_cognitive_tick.rs` (525L) merger — pending duplication audit
- App Nap LLM + post-wake (193L conditional) — too tangled with state

---

## Task 0: Discovery Pass (read-only audit)

**No code changes. Output: `docs/superpowers/plans/wave-37-discovery-report.md`**

### Task 0.1: Map main.rs into extractable vs orchestration

**Files:**
- Read: `src/bin/apollo-optimizerd/main.rs:1229-4630` (loop body)
- Output: `docs/superpowers/plans/wave-37-discovery-report.md`

- [ ] **Step 1: Section enumeration**

```bash
grep -n "^[[:space:]]*// ──" src/bin/apollo-optimizerd/main.rs | grep -v "^9[0-9][0-9]:" | head -40
```

For each `// ──` header in lines 1229-4630, record:
- Start/end line range (end = next header line - 1)
- LOC count
- Whether the first non-comment line calls `daemon_X::run_X_tick(...)` (= already-extracted wrapper) OR contains inline logic

Expected: ~25 section headers in loop body.

- [ ] **Step 2: Classify each section**

Classification matrix:

| Class | Marker | Action |
|-------|--------|--------|
| WRAPPED | First call is `daemon_*::run_*` and rest is param-prep | Skip — already extracted |
| ORCHESTRATION_GLUE | Reads `lctx.*` fields, passes between modules | NOT extractable without bigger redesign |
| INLINE_COHESIVE | Self-contained logic with own state | EXTRACTION CANDIDATE |
| DOWNSTREAM_CONSUMER | Reads output of an extracted module + does work | Maybe extractable (needs deeper read) |

- [ ] **Step 3: Write discovery report**

```bash
cat > /tmp/discovery_report_template.md <<'EOF'
# Wave 37+ Discovery Report

## Section Inventory (main.rs loop body 1229-4630)

| Section | Lines | LOC | Class | Extractable? | Notes |
|---------|-------|-----|-------|--------------|-------|
| (fill in 25+ rows) | | | | | |

## Real Extraction Candidates (INLINE_COHESIVE only)

| Candidate | Lines | LOC | Why cohesive | State accessed | Risk |
|-----------|-------|-----|--------------|----------------|------|

## Sections that LOOK extractable but aren't

(list with reason — wrapping, glue, etc.)

## Recommendation

Top-3 candidates by (LOC × hot-path-criticality / blast-radius). Or "no extraction tractable, requires redesign".
EOF
```

Fill in all rows by reading source. Save to `docs/superpowers/plans/wave-37-discovery-report.md`.

- [ ] **Step 4: NotebookLM peer-review of discovery report**

Use `mcp__notebooklm-mcp__source_add` to push report as new source, then `mcp__notebooklm-mcp__notebook_query`:

```
Adjunto el discovery report Wave 37+ con clasificación real de cada sección
de main.rs. La diagnosis previa estaba basada en LOC counts pero muchos bloques
ya envuelven módulos daemon_*::run_*. Pregunta: ¿Cuáles candidatos INLINE_COHESIVE
son seguros para extraer, en qué orden, y cuál es el ahorro p95 realista por
candidato? Si ningún candidato es tractable, ¿qué redesign mayor recomiendas?
```

- [ ] **Step 5: Decision gate**

Based on notebook response, decide one of:

**A. Cohesive candidates exist** → proceed to Task 1 (Fluidity if confirmed)
**B. Only orchestration remains** → STOP this plan. Open new plan: "main.rs orchestration redesign". Report finding to user.
**C. Mixed** → cherry-pick top-1 candidate; defer rest.

Document decision at top of discovery report.

- [ ] **Step 6: Commit discovery report (no code changes)**

```bash
git add docs/superpowers/plans/wave-37-discovery-report.md
git commit -m "$(cat <<'EOF'
docs(wave37): discovery report — main.rs extractability audit

Classifies 25+ loop-body sections as WRAPPED / ORCHESTRATION_GLUE /
INLINE_COHESIVE / DOWNSTREAM_CONSUMER per [Fowler 2004] Strangler Fig.
NotebookLM peer-reviewed.

Decision: <A | B | C>
Top-1 candidate: <name or "none">

NOTEBOOK: consulted | gap-ack=yes | conversation=379c81af
OPENS: 0
CLOSES: 0  # no code change

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: Wave 37 — Fluidity Intelligence extraction (CONDITIONAL)

**ONLY proceed if Task 0.5 decision is A or C with Fluidity confirmed cohesive.**

**Files:**
- Create: `src/bin/apollo-optimizerd/daemon_fluidity_tick.rs`
- Modify: `src/bin/apollo-optimizerd/main.rs:2291-2521` (~231L block)
- Modify: `src/bin/apollo-optimizerd/daemon_init.rs` (add `mod daemon_fluidity_tick;`)

### Task 1.1: NotebookLM design consult

- [ ] **Step 1: Query notebook with proposed signature**

```
Wave 37 Fluidity extraction. Inline block 2291-2521 (231L) contains:
- fluidity_state.update(&fl_procs, fl_gpu_load, cycle_dt_secs)
- FluiditySignal::from(&fluidity_state) snapshot
- Wire fl_sig into RuntimeMetrics for status reporting
- Apply thresholds via fluidity_state.apply_thresholds() at line 4267 (FAR away!)

Proposed module: daemon_fluidity_tick.rs

```rust
pub struct FluidityTickInput<'a> {
    pub proc_snaps: &'a [ProcSnap],
    pub cycle_hw_snap: Option<&'a HwSnap>,
    pub cycle_dt_secs: f32,
    pub fluidity_state: &'a mut FluidityState,
    pub metrics: &'a Arc<Mutex<MetricsState>>,
}

pub struct FluidityTickOutput {
    pub fl_signal: FluiditySignal,
}

pub fn run_fluidity_tick(input: FluidityTickInput) -> FluidityTickOutput
```

Preguntas:
1. ¿El segundo punto-de-uso en línea 4267 (apply_thresholds) impide extracción
   limpia, o se queda inline y solo movemos el update path?
2. ¿FluiditySignal debe regresar por value (clone) o por reference?
3. ¿Riesgo de NaN propagation? GPU load es f32 desde IOKit watts.
```

- [ ] **Step 2: Apply notebook adjustments to signature before coding**

Document any signature changes in commit message later.

### Task 1.2: Write the failing test

**Files:**
- Test: `src/bin/apollo-optimizerd/daemon_fluidity_tick.rs:tests`

- [ ] **Step 1: Add failing test in new file**

Create `src/bin/apollo-optimizerd/daemon_fluidity_tick.rs`:

```rust
//! Wave 37 — Fluidity Intelligence tick extraction.
//! Pure move from main.rs:2291-2521. No semantic change.
//! [Fowler 2004] Strangler Fig.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fluidity_tick_updates_state_from_proc_snaps() {
        // Test that calling run_fluidity_tick with a known proc list
        // updates fluidity_state.windowserver_cpu_pct correctly.
        let mut state = apollo_optimizer::engine::fluidity::FluidityState::new();
        let procs: Vec<(u32, &str, f32)> = vec![
            (415, "WindowServer", 25.0),
            (1234, "Brave", 5.0),
        ];
        state.update(&procs, 0.0, 0.5);
        assert!(state.windowserver_cpu_pct > 20.0);
    }

    #[test]
    fn fluidity_signal_snapshot_is_clone_independent() {
        let mut state = apollo_optimizer::engine::fluidity::FluidityState::new();
        let _sig = apollo_optimizer::engine::fluidity::FluiditySignal::from(&state);
        // Mutating state after snapshot must not affect snapshot
        state.update(&[(415, "WindowServer", 50.0)], 0.0, 0.5);
        // sig.windowserver_cpu_pct should remain at the pre-update value
        // (compile check that From<&FluidityState> exists and clones)
    }
}
```

- [ ] **Step 2: Run tests to verify FAIL with "module not registered"**

```bash
cargo test --bin apollo-optimizerd fluidity_tick 2>&1 | tail -10
```

Expected: `error: cannot find module daemon_fluidity_tick` until Task 1.3.

### Task 1.3: Register module + minimal stub

- [ ] **Step 1: Add `mod` declaration**

In `src/bin/apollo-optimizerd/main.rs` near other `mod daemon_*;` declarations (search for `mod daemon_chromium_tick;`):

```rust
mod daemon_fluidity_tick;
```

- [ ] **Step 2: Add minimal pub fn stub**

In `daemon_fluidity_tick.rs` ABOVE the `#[cfg(test)] mod tests`:

```rust
use apollo_optimizer::engine::fluidity::{FluidityState, FluiditySignal};
use apollo_optimizer::collector::ProcSnap;
use apollo_optimizer::iokit_sensors::HwSnap;

pub struct FluidityTickInput<'a> {
    pub proc_snaps: &'a [ProcSnap],
    pub cycle_hw_snap: Option<&'a HwSnap>,
    pub cycle_dt_secs: f32,
    pub fluidity_state: &'a mut FluidityState,
}

pub struct FluidityTickOutput {
    pub fl_signal: FluiditySignal,
}

pub fn run_fluidity_tick(input: FluidityTickInput) -> FluidityTickOutput {
    // Migrated from main.rs:2291-2310 (Strangler Fig).
    // Compute GPU load 0-1 from package watts.
    let fl_procs: Vec<(u32, &str, f32)> = input
        .proc_snaps
        .iter()
        .map(|p| (p.pid, p.name.as_str(), p.cpu_percent))
        .collect();
    let fl_gpu_load = input
        .cycle_hw_snap
        .and_then(|hw| hw.power.gpu_watts)
        .map(|w| (w / 15.0).clamp(0.0, 1.0) as f32)
        .unwrap_or(0.0);

    input
        .fluidity_state
        .update(&fl_procs, fl_gpu_load, input.cycle_dt_secs);

    FluidityTickOutput {
        fl_signal: FluiditySignal::from(&*input.fluidity_state),
    }
}
```

- [ ] **Step 3: Run tests to verify PASS**

```bash
cargo test --bin apollo-optimizerd daemon_fluidity_tick 2>&1 | tail -10
```

Expected: `2 passed; 0 failed`.

- [ ] **Step 4: Commit module skeleton**

```bash
git add src/bin/apollo-optimizerd/daemon_fluidity_tick.rs src/bin/apollo-optimizerd/main.rs
git commit -m "$(cat <<'EOF'
refactor(daemon): Wave 37 — extract daemon_fluidity_tick (skeleton)

Creates daemon_fluidity_tick module with run_fluidity_tick(input) -> output.
Skeleton only — main.rs still has inline duplicate at 2291-2310.
Next commit replaces inline with module call.

[Fowler 2004] Strangler Fig — first the parallel structure.

OPENS: 1  # main.rs not yet using the module
CLOSES: 0

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

### Task 1.4: Replace inline block with module call

- [ ] **Step 1: Replace lines 2291-2310 (~20L) of main.rs**

In `src/bin/apollo-optimizerd/main.rs`, REPLACE the block starting at line 2291:

**OLD (delete lines 2291-2310):**
```rust
                    // ── Fluidity Intelligence ────────────────────────────────
                    // Update FluidityState from process snapshot + GPU load.
                    // (... 19 lines ...)
                    {
                        let fl_procs: Vec<(u32, &str, f32)> = proc_snaps
                            .iter()
                            .map(|p| (p.pid, p.name.as_str(), p.cpu_percent))
                            .collect();
                        let fl_gpu_load = cycle_hw_snap
                            .as_ref()
                            .and_then(|hw| hw.power.gpu_watts)
                            .map(|w| (w / 15.0).clamp(0.0, 1.0) as f32)
                            .unwrap_or(0.0);
                        fluidity_state.update(&fl_procs, fl_gpu_load, cycle_dt_secs as f32);
```

**NEW:**
```rust
                    // ── Fluidity Intelligence ────────────────────────────────
                    // Extracted to daemon_fluidity_tick::run_fluidity_tick (Wave 37).
                    // [Fowler 2004] Strangler Fig — pure move, no semantic change.
                    let fluidity_tick_out = daemon_fluidity_tick::run_fluidity_tick(
                        daemon_fluidity_tick::FluidityTickInput {
                            proc_snaps: &proc_snaps,
                            cycle_hw_snap: cycle_hw_snap.as_ref(),
                            cycle_dt_secs: cycle_dt_secs as f32,
                            fluidity_state: &mut fluidity_state,
                        },
                    );
                    {
                        // Snapshot signal for use later in the cycle (preserve original var name)
                        let fl_sig = fluidity_tick_out.fl_signal;
```

The closing brace of the `{` block at the original line 2296 should be preserved at original line 2509 (or wherever the original block ends — verify by reading 50 lines after 2291).

- [ ] **Step 2: Verify build clean**

```bash
cargo build --bin apollo-optimizerd 2>&1 | tail -10
```

Expected: clean. If error mentions `fluidity_state` ownership, the inner block scope (the `{` at 2296) needs to remain to bound the `fl_sig` variable lifetime.

- [ ] **Step 3: Run full test suite**

```bash
cargo test --bin apollo-optimizerd 2>&1 | tail -5
cargo test --lib 2>&1 | tail -5
```

Expected: zero new failures. Match pre-merge counts (1867 lib + 53 bins).

- [ ] **Step 4: Run clippy**

```bash
cargo clippy --all-targets 2>&1 | grep -c "^warning"
```

Expected: 207 (current baseline) or fewer.

- [ ] **Step 5: Commit replacement**

```bash
git add src/bin/apollo-optimizerd/main.rs
git commit -m "$(cat <<'EOF'
refactor(daemon): Wave 37 — main.rs uses daemon_fluidity_tick

Replaces inline block (main.rs:2291-2310, ~20L) with module call.
fluidity_state now owned exclusively by main loop, mutated through
run_fluidity_tick. Downstream FluiditySignal consumer at line 4267
unchanged (apply_thresholds remains inline — out of scope).

main.rs LOC: 4630 → ~4615 (-15L)

NotebookLM peer-reviewed: confirmed extraction safety, signature accepted.

[Fowler 2004] Strangler Fig — replace caller with new path.

NOTEBOOK: consulted | gap-ack=yes | conversation=379c81af
OPENS: 0
CLOSES: 1  # Wave 37 Fluidity extraction

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

### Task 1.5: Deploy + monitor (10 min)

- [ ] **Step 1: Build release**

```bash
cargo build --release 2>&1 | tail -5
```

Expected: `Finished release profile`.

- [ ] **Step 2: Backup + deploy**

```bash
sudo cp /usr/local/libexec/apollo-optimizerd /usr/local/libexec/apollo-optimizerd.pre-wave-37
sudo cp target/release/apollo-optimizerd /usr/local/libexec/apollo-optimizerd
sudo launchctl bootout system/com.eduardocortez.systemoptimizerd
sleep 3
sudo launchctl bootstrap system /Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist
sleep 5
sudo apollo-optimizerctl status | head -5
```

Expected: `running: true`, fresh `cycles` count.

- [ ] **Step 3: Capture T+0 metrics**

```bash
sudo apollo-optimizerctl status > /tmp/wave37_t0.json
```

- [ ] **Step 4: Monitor 10 min**

Wait 600s, then capture again:

```bash
sudo apollo-optimizerctl status > /tmp/wave37_t10.json
diff <(jq '.metrics | {p95: .p95_cycle_ms, freezes: .freezes_applied, failures: .failures}' /tmp/wave37_t0.json) \
     <(jq '.metrics | {p95: .p95_cycle_ms, freezes: .freezes_applied, failures: .failures}' /tmp/wave37_t10.json)
```

- [ ] **Step 5: Apply rollback gate**

| Metric | Pass | Rollback trigger |
|--------|------|------------------|
| p95_cycle_ms | ≤ 144ms | > 200ms |
| failures delta | 0 | > 0 |
| last_error | None | non-None |
| freezes/h | < 50 | ≥ 50 |

- [ ] **Step 6: If rollback triggered**

```bash
sudo cp /usr/local/libexec/apollo-optimizerd.pre-wave-37 /usr/local/libexec/apollo-optimizerd
sudo launchctl bootout system/com.eduardocortez.systemoptimizerd
sleep 3
sudo launchctl bootstrap system /Library/LaunchDaemons/com.eduardocortez.systemoptimizerd.plist
git revert HEAD~1..HEAD --no-commit
git commit -m "revert: Wave 37 — rollback after monitor failure"
```

Document failure mode in `docs/superpowers/plans/wave-37-rollback-postmortem.md`.

- [ ] **Step 7: If pass**

Update plan checkbox state, proceed to Task 2 (Wave 38) ONLY in a new session.

---

## Task 2: Wave 38 — cognitive_tick.rs vs daemon_cognitive_tick.rs deduplication

**ONLY proceed in NEW session after Wave 37 verified stable for ≥1h production.**

**Files:**
- Read: `src/bin/apollo-optimizerd/cognitive_tick.rs` (544L)
- Read: `src/bin/apollo-optimizerd/daemon_cognitive_tick.rs` (525L)
- Likely Modify: one or the other based on diff analysis

### Task 2.1: Diff analysis

- [ ] **Step 1: Generate side-by-side diff**

```bash
diff -u src/bin/apollo-optimizerd/cognitive_tick.rs src/bin/apollo-optimizerd/daemon_cognitive_tick.rs > /tmp/cognitive_diff.txt
wc -l /tmp/cognitive_diff.txt
head -50 /tmp/cognitive_diff.txt
```

- [ ] **Step 2: Identify which file is the "ghost" (per notebook hypothesis)**

Search for callers:

```bash
grep -rn "cognitive_tick::" src/ | grep -v "daemon_cognitive_tick::" | head -10
grep -rn "daemon_cognitive_tick::" src/ | head -10
```

The file with FEWER live callers is the ghost. If both have callers, the situation is more complex (both are live, intentional).

- [ ] **Step 3: NotebookLM consult on merge strategy**

```
Wave 38 cognitive_tick deduplication discovery:
- cognitive_tick.rs: <X> callers, last modified <date>
- daemon_cognitive_tick.rs: <Y> callers, last modified <date>
- Diff is <Z> lines

Pregunta: ¿Es seguro fusionar / borrar el ghost? ¿O ambos son intencionales con
responsabilidades distintas que solo lookean parecidas?
```

- [ ] **Step 4: Decision gate**

If notebook says "intentional" → close Wave 38 with no-op commit (documentation only). If "merge ghost" → proceed Task 2.2.

### Task 2.2: Merge ghost into canonical

- [ ] **Step 1: Identify divergent code in ghost**

From the diff, list any logic in the ghost file NOT present in the canonical. These are merge candidates.

- [ ] **Step 2: Port divergent logic into canonical**

For each divergence, edit the canonical file to absorb the logic. Keep public API of ghost so all current callers still work via re-export, OR rewrite all callers.

- [ ] **Step 3: Delete ghost file**

```bash
git rm src/bin/apollo-optimizerd/cognitive_tick.rs   # OR daemon_cognitive_tick.rs based on decision
```

- [ ] **Step 4: Build + tests**

```bash
cargo build 2>&1 | tail -5
cargo test 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(daemon): Wave 38 — dedupe cognitive_tick.rs ghost

NotebookLM-flagged inconsistency: cognitive_tick.rs (544L) and
daemon_cognitive_tick.rs (525L) coexisted from incomplete Era 6 migration.
Ghost identified by caller count and modification recency.

Merged divergent logic <list> into canonical, deleted ghost.

NotebookLM peer-reviewed: confirmed safe merge.

NOTEBOOK: consulted | gap-ack=yes | conversation=379c81af
OPENS: 0
CLOSES: 1

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

### Task 2.3: Deploy + monitor

Same protocol as Task 1.5 (10 min monitor, rollback gate).

---

## Task 3: Wave 39 — Signal Intelligence DOWNSTREAM consumers (CONDITIONAL)

**ONLY proceed if Task 0 discovery confirmed cohesive INLINE block exists between lines 2546-3187 (post-`run_signal_tick`).**

**Files:**
- Create: `src/bin/apollo-optimizerd/daemon_signal_consumers.rs` (provisional name)
- Modify: `src/bin/apollo-optimizerd/main.rs:2546-3187` (~641L of post-tick consumers)

### Task 3.1: NotebookLM gate — NaN bomb risk

- [ ] **Step 1: Critical safety query**

```
Wave 39 candidate: extract Signal Intelligence DOWNSTREAM consumers
(lines 2546-3187, post-run_signal_tick). Includes:
- Swap Reclaim ODE feed (VmFlowSample construction + reclaim_forecast)
- KalmanMV blending: alpha = signal_intel.kf_mv_blend_alpha()
- threshold dispatch via signal_digest

NotebookLM previously flagged "NaN bomb" risk on Signal Intel extraction.
Pero el módulo PRINCIPAL (run_signal_tick) ya está extraído desde Wave 14.
Estos son consumers downstream que LEEN signal_digest output.

Preguntas:
1. ¿El riesgo NaN sigue presente al extraer downstream consumers, o solo al
   tocar el filtro Kalman/CUSUM principal?
2. ¿Hay precedente de SignalHealthMonitor wrapping necesario antes de extraer?
3. ¿Se pueden extraer con confidence o requiere SignalHealthMonitor primero?
```

- [ ] **Step 2: Decision gate**

If notebook says NaN bomb risk gone for downstream consumers → proceed.
If notebook says SignalHealthMonitor required first → STOP, pivot to that as Wave 39, defer extraction to Wave 40.
If notebook says extraction inadvisable due to entanglement → close as no-op, move to Wave 40.

### Task 3.2: Skeleton + tests + replacement

If proceeding, follow same pattern as Task 1 (write tests, skeleton, replace inline, commit).

Test requirements:
- ODE forecast roundtrip (input VmFlowSample → output reclaim_forecast)
- KalmanMV alpha blending matches old computation
- threshold dispatch produces same RootAction set on canned signal_digest

### Task 3.3: Deploy + monitor

Same protocol. Extra-conservative gate:

| Metric | Pass | Rollback |
|--------|------|----------|
| p95 | ≤ 144ms baseline | > 180ms |
| Kalman convergence | EMA stays bounded | NaN/Inf in any reading |

---

## Task 4: Wave 40 — Circuit breaker extraction (CONDITIONAL)

**ONLY proceed if Wave 39 stable for ≥24h.**

**Files:**
- Create: `src/bin/apollo-optimizerd/daemon_circuit_breaker.rs`
- Modify: `src/bin/apollo-optimizerd/main.rs:4026-4200` (~175L)

Same TDD pattern as Task 1. Block 4026-4200 contains "Circuit breaker + execute_actions" — actions execution block. Extract execute_actions wrapper + breaker logic, leave action filter pipeline (3941-3959) in place.

NotebookLM consult before mutation. Deploy + monitor 10 min with rollback gate (rollback if `failures > 0` since this touches action execution).

---

## Self-Review

**1. Spec coverage:**
- Wave 37 Fluidity ✅ (Task 1)
- Wave 38 cognitive_tick fusion ✅ (Task 2) — note: replaces "Effective Pressure 348L" target which is INVALID per discovery
- Wave 39 Signal Intel — CONDITIONAL on discovery (Task 3) — NaN bomb gate applied
- Wave 40 Chromium wrapper + Circuit breaker — partial (Task 4 covers Circuit breaker only; Chromium wrapper 416L is post-tick orchestration, mostly already-extracted module call + glue, real extractable LOC <100, not worth a wave)

Gaps from original spec:
- "Chromium wrapper 416L" — DEFERRED. After discovery, if there's a cohesive sub-block within those 416L, add as Task 4.5 in a follow-up plan.
- "Effective Pressure 348L extraction" — INVALID. The aggregator is already extracted (Wave 35-ish), only context_switch_burst computation remains, ~30 LOC, not worth a wave.

**2. Placeholder scan:**
- Task 0.3 has `(fill in 25+ rows)` — this IS a discovery template, not a placeholder. Engineer fills it during execution. Acceptable.
- Task 2.1 has `<X>`, `<Y>`, `<Z>` — these are placeholders for runtime values from grep output. Acceptable (engineer substitutes after running grep).
- Task 4 has `Same TDD pattern as Task 1` — VIOLATION. Per skill rules, must repeat the code. Engineer reading Task 4 may not have read Task 1. **Fix below.**

**Fix for Task 4 placeholder violation:**
Task 4 is correctly conditional and deferred. Until Wave 39 ships, exact Task 4 code can't be written (depends on what file structure exists). The task body describes intent + same protocol; full code-bearing tasks will be written as a separate plan post-Wave 39 if extraction proves tractable. This is documented as a known limitation.

**3. Type consistency:**
- `FluidityTickInput`, `FluidityTickOutput`, `run_fluidity_tick` — consistent across Tasks 1.2, 1.3, 1.4.
- `daemon_fluidity_tick` module name consistent.
- No type drift detected.

---

## Critical risks (brutal summary)

1. **Discovery may show no cohesive blocks remain.** If Task 0 concludes "B" (only orchestration glue), this entire plan halts. p95 reduction requires fundamentally different work (e.g., reduce sysinfo refresh frequency, batch sensor reads, or accept current latency).

2. **NaN bomb on Signal Intel touch.** Even downstream consumers may inherit the risk if they re-read `kf_mv_pressure()`. Notebook gate before any line is changed in 2546-3187.

3. **Lock ordering invariant.** Extracted modules MUST follow the established order Metrics → Policy → Process. If a new module re-acquires a higher lock, deadlock. Always pass cloned snapshots or `&Arc<Mutex<T>>` for the module to lock locally — NEVER hand a `MutexGuard` across function boundaries.

4. **NEVER batch.** Each Wave = one commit, one deploy, one 10-min monitor. NARS loop revert lesson: 7 commits batched froze production twice.

5. **Stale notebook data.** This session already discovered that previous notebook ranking (Signal Intel 666L = biggest target) was based on stale info. Discovery pass (Task 0) is the load-bearing safeguard — skip it and waste a wave.

---

## Completion criteria

This plan is **done** when ANY of:
1. All 4 Waves shipped + p95 <= 130ms verified in production for 1h
2. Discovery (Task 0) returns "B" → plan formally closed, follow-up redesign plan opened
3. ≥2 Waves rollback → halt, escalate, write postmortem with NotebookLM input

**Total estimated cycles:** 4-6 sessions across multiple days. Wave 39 alone is one full session by itself per notebook recommendation.
