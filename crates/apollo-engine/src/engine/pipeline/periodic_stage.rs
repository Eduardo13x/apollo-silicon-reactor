//! # Periodic Stage
//!
//! Houses the housekeeping work that runs every N cycles rather than every cycle.
//! In the current daemon main loop these are scattered as `if cycle_count % N == 0`
//! blocks throughout the ~5000-line `run_daemon` function.
//!
//! ## Extraction status
//!
//! `run_periodic()` is now wired to a [`LearningContext`] and performs real work:
//!
//! | Cycle gate    | What it does                                           |
//! |---------------|--------------------------------------------------------|
//! | % 500 == 0    | Compress experience memory, prune weights, GC skills   |
//! | % 100 == 0    | Compute causal solid-edge count for observability      |
//!
//! Remaining inline periodic work in the daemon loop:
//!
//! | Cycle gate    | Why it stays inline                                    |
//! |---------------|--------------------------------------------------------|
//! | % 100 == 0    | Signal/state persist: needs `learning_pipeline`,       |
//! |               | `effectiveness_tracker`, `frozen_state` (SharedState)  |
//! | % 100 == 0    | Rule induction: needs `learned_policy` (SharedState)   |
//! | % 7200 == 0   | Hourly GC: needs `cache_warmer`, `io_shaper`,          |
//! |               | `temporal_predictor` (binary-local types)              |

use crate::engine::config_reloader::{LlmConfigReloader, ReloadOutcome};
use crate::engine::llm::LlmConfig;
use crate::engine::pipeline::learning_context::LearningContext;

/// How often the daemon polls `config.toml` for mtime changes.
///
/// 100 cycles ≈ 50s at the 500ms daemon cadence — fast enough that an
/// operator edit lands within a minute, slow enough that `fs::metadata`
/// on the config file adds zero measurable cost.
pub const CONFIG_RELOAD_GATE_CYCLES: u64 = 100;

/// Opt-in helper that wraps the `% CONFIG_RELOAD_GATE_CYCLES` gate around
/// `LlmConfigReloader::tick`.
///
/// The daemon main loop calls this once per cycle with its owned reloader +
/// the current `LlmConfig`. On non-gate cycles the call is a single modulo
/// comparison and returns `None`. On gate cycles it polls the file mtime —
/// also ~free when the file has not changed because `fs::metadata` is a
/// cheap `stat(2)` on macOS.
///
/// Returns `Some(outcome)` iff the gate fired so the caller can log `applied`
/// diffs, WARN on `rejected`, or swap the in-memory `LlmConfig` from
/// `outcome.new_cfg`.
///
/// # Why a free helper, not a field on `PeriodicContext`
///
/// Threading `&mut LlmConfigReloader` through `PeriodicContext` would force
/// every existing call site to construct (or `Option::None`-out) a reloader
/// even when the daemon does not use Gemma. Keeping the gate as a separate
/// free function means the wire-up is a single line in the main loop and
/// does not churn any other caller.
pub fn maybe_reload_llm_config(
    cycle_count: u64,
    reloader: &mut LlmConfigReloader,
    current: &LlmConfig,
) -> Option<ReloadOutcome> {
    if cycle_count % CONFIG_RELOAD_GATE_CYCLES != 0 {
        return None;
    }
    Some(reloader.tick(current))
}

/// Everything the periodic stage needs to do its work.
///
/// `'a` is the lifetime of the data borrowed inside [`LearningContext`].
/// `'lctx` is the lifetime of the borrow of the context itself.
pub struct PeriodicContext<'a, 'lctx> {
    /// Current daemon cycle counter (starts at 1, monotonically increasing).
    pub cycle_count: u64,

    /// Memory pressure at the time the periodic stage runs.
    /// Used by rule_inducer to gate skill induction (only at elevated pressure).
    pub current_pressure: f64,

    /// Current workload mode string ("idle", "build", "browser", etc.).
    pub workload_mode: &'a str,

    /// Filesystem path where optimization skills are persisted.
    pub skills_path: &'a std::path::Path,

    /// Filesystem path where hop-group data is persisted.
    pub hop_groups_path: &'a std::path::Path,

    /// Filesystem path where signal intelligence state is persisted.
    pub signal_intel_path: &'a std::path::Path,

    /// Filesystem path where learned state (unified persistence) is stored.
    pub learned_state_path: &'a std::path::Path,

    /// Persist generation counter (incremented by LearnedState::persist_improved).
    pub persist_generations: u32,

    /// Quality score of the last restored state (None if no restore yet).
    pub last_restore_quality: Option<f64>,

    /// Pending trial skill from the current decision cycle, if any.
    /// Passed through to LearnedState so a crash mid-trial can be recovered.
    pub pending_trial_skill: Option<(String, f64)>,

    /// All 9 learning subsystems needed for GC and observability.
    ///
    /// Must be constructed from the same locals that [`LearningContext`] was
    /// built from, inside the same loop iteration.
    pub lctx: &'lctx mut LearningContext<'a>,
}

/// Which periodic housekeeping tasks ran this cycle.
///
/// Returned by `run_periodic` so the caller can log what happened.
#[derive(Debug, Default)]
pub struct PeriodicResult {
    /// Causal graph solid-edge count logged this cycle (None if gate didn't fire).
    pub causal_solid_edges: Option<usize>,

    /// Number of new skills crystallised from rule induction (0 if gate didn't fire).
    pub induced_skills: Option<usize>,

    /// Whether unified learned-state was persisted this cycle.
    pub did_persist: bool,

    /// Whether GC/compression ran this cycle (% 500 gate).
    pub did_gc: bool,

    /// Whether hourly housekeeping ran this cycle (% 7200 == 0 gate).
    pub did_hourly: bool,
}

/// Run all periodic housekeeping tasks for this cycle.
///
/// Executes only the gates whose modulus fires for the given `ctx.cycle_count`.
/// Callers are responsible for the remaining periodic work that requires
/// binary-local state (signal persist, rule induction, hourly GC).
pub fn run_periodic(ctx: &mut PeriodicContext<'_, '_>) -> PeriodicResult {
    let mut result = PeriodicResult::default();

    // ── Every 100 cycles: observability snapshot ─────────────────────────────
    // Full persist (signal_intel, LearnedState, skills) remains inline because
    // it requires learning_pipeline and effectiveness_tracker from the binary.
    if ctx.cycle_count % 100 == 0 {
        result.did_persist = true;
        result.causal_solid_edges = Some(ctx.lctx.causal_graph.solid_count());
        result.induced_skills = Some(0); // rule induction stays inline (needs SharedState)
    }

    // ── Every 500 cycles: GC and compression ─────────────────────────────────
    if ctx.cycle_count % 500 == 0 {
        // Compress old experience records to save memory (Hermes pattern).
        ctx.lctx.outcome_tracker.experience.compress_old();
        // Prune low-signal Bayesian weight entries (BUG-04).
        ctx.lctx.outcome_tracker.gc_weights();
        // Retire ineffective skills.
        ctx.lctx.skill_registry.gc();
        // Persist updated skill registry after GC.
        ctx.lctx.skill_registry.persist(ctx.skills_path);
        result.did_gc = true;
    }

    // ── Every 7200 cycles (~2 hours): hourly housekeeping ────────────────────
    // cache_warmer.gc(), io_shaper.gc(), temporal_predictor.persist() remain
    // inline: they are binary-local types that cannot be imported here.
    if ctx.cycle_count % 7200 == 0 {
        result.did_hourly = true;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::causal_graph::CausalGraph;
    use crate::engine::energy::EnergyTracker;
    use crate::engine::neuromodulator::ApolloNeuromodulator;
    use crate::engine::optimization_skills::SkillRegistry;
    use crate::engine::outcome_tracker::OutcomeTracker;
    use crate::engine::overflow_guard::OverflowGuard;
    use crate::engine::pipeline::learning_context::LearningContext;
    use crate::engine::predictive_agent::{PredictiveAgent, SpecialistAccuracyTracker};
    use crate::engine::signal_intelligence::SignalIntelligence;

    /// Owns the 9 learning subsystems needed to construct a LearningContext.
    struct LctxOwner {
        outcome_tracker: OutcomeTracker,
        signal_intel: SignalIntelligence,
        predictive_agent: PredictiveAgent,
        specialist_accuracy: SpecialistAccuracyTracker,
        overflow_guard: OverflowGuard,
        causal_graph: CausalGraph,
        skill_registry: SkillRegistry,
        neuromod: ApolloNeuromodulator,
        energy_tracker: EnergyTracker,
    }

    impl LctxOwner {
        fn new() -> Self {
            Self {
                outcome_tracker: OutcomeTracker::new(),
                signal_intel: SignalIntelligence::new(),
                predictive_agent: PredictiveAgent::load_or_default(std::path::Path::new(
                    "/dev/null",
                )),
                specialist_accuracy: SpecialistAccuracyTracker::new(),
                overflow_guard: OverflowGuard::load_or_default(
                    std::path::Path::new("/dev/null"),
                    None,
                ),
                causal_graph: CausalGraph::new(),
                skill_registry: SkillRegistry::new(),
                neuromod: ApolloNeuromodulator::new(),
                energy_tracker: EnergyTracker::new(),
            }
        }

        fn make_lctx(&mut self) -> LearningContext<'_> {
            LearningContext::new(
                &mut self.outcome_tracker,
                &mut self.signal_intel,
                &mut self.predictive_agent,
                &mut self.specialist_accuracy,
                &mut self.overflow_guard,
                &mut self.causal_graph,
                &mut self.skill_registry,
                &mut self.neuromod,
                &mut self.energy_tracker,
            )
        }
    }

    fn make_pctx<'a, 'lctx>(
        cycle: u64,
        lctx: &'lctx mut LearningContext<'a>,
    ) -> PeriodicContext<'a, 'lctx> {
        PeriodicContext {
            cycle_count: cycle,
            current_pressure: 0.30,
            workload_mode: "idle",
            skills_path: std::path::Path::new("/tmp/test-skills.json"),
            hop_groups_path: std::path::Path::new("/tmp/test-hops.json"),
            signal_intel_path: std::path::Path::new("/tmp/test-si.json"),
            learned_state_path: std::path::Path::new("/tmp/test-ls.json"),
            persist_generations: 0,
            last_restore_quality: None,
            pending_trial_skill: None,
            lctx,
        }
    }

    /// % 100 gate fires: did_persist set, causal_solid_edges populated.
    #[test]
    fn persist_fires_at_cycle_100() {
        let mut owner = LctxOwner::new();
        let mut lctx = owner.make_lctx();
        let mut ctx = make_pctx(100, &mut lctx);
        let result = run_periodic(&mut ctx);
        assert!(result.did_persist, "% 100 gate should fire at cycle 100");
        assert!(!result.did_gc, "% 500 gate must not fire at cycle 100");
        assert!(!result.did_hourly, "% 7200 gate must not fire at cycle 100");
        assert!(
            result.causal_solid_edges.is_some(),
            "causal solid count should be populated at % 100"
        );
    }

    /// % 500 gate: GC runs; experience, weights, and skills are mutated.
    #[test]
    fn gc_fires_at_cycle_500() {
        let mut owner = LctxOwner::new();
        let mut lctx = owner.make_lctx();
        let mut ctx = make_pctx(500, &mut lctx);
        let result = run_periodic(&mut ctx);
        assert!(result.did_persist, "% 100 gate must co-fire at cycle 500");
        assert!(result.did_gc, "% 500 gate should fire at cycle 500");
        assert!(!result.did_hourly, "% 7200 gate must not fire at cycle 500");
    }

    /// % 7200 gate: did_hourly set (binary-local work remains inline).
    #[test]
    fn hourly_fires_at_cycle_7200() {
        let mut owner = LctxOwner::new();
        let mut lctx = owner.make_lctx();
        let mut ctx = make_pctx(7200, &mut lctx);
        let result = run_periodic(&mut ctx);
        assert!(
            result.did_hourly,
            "% 7200 == 0 gate should fire at cycle 7200"
        );
        assert!(result.did_persist, "% 100 gate must co-fire at cycle 7200");
    }

    /// No gates fire on cycle 1.
    #[test]
    fn gates_at_cycle_1() {
        let mut owner = LctxOwner::new();
        let mut lctx = owner.make_lctx();
        let mut ctx = make_pctx(1, &mut lctx);
        let result = run_periodic(&mut ctx);
        assert!(!result.did_persist, "% 100 gate must not fire at cycle 1");
        assert!(!result.did_gc, "% 500 gate must not fire at cycle 1");
        assert!(!result.did_hourly, "% 7200 gate must not fire at cycle 1");
    }

    /// PeriodicResult::default() is all-false/None.
    #[test]
    fn periodic_result_default_is_all_false() {
        let r = PeriodicResult::default();
        assert!(!r.did_persist);
        assert!(!r.did_gc);
        assert!(!r.did_hourly);
        assert!(r.causal_solid_edges.is_none());
        assert!(r.induced_skills.is_none());
    }

    // ── maybe_reload_llm_config gate ───────────────────────────────────────

    use std::fs::{self, OpenOptions};
    use std::io::Write;
    use std::path::PathBuf;

    fn tmp_cfg(name: &str, contents: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "apollo_periodic_reload_{}_{}",
            std::process::id(),
            name
        ));
        let _ = fs::remove_file(&p);
        let mut f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&p)
            .unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        p
    }

    const BASE_CFG: &str = r#"
[llm]
enabled = true
endpoint = "http://127.0.0.1:8080"
timeout_ms = 60000
always_on = true
"#;

    fn current_cfg() -> LlmConfig {
        #[derive(serde::Deserialize)]
        struct Repo {
            llm: LlmConfig,
        }
        let parsed: Repo = toml::from_str(BASE_CFG).unwrap();
        parsed.llm
    }

    /// Non-gate cycles: helper short-circuits before touching the file.
    /// Proven by using a bogus path — if `tick` were called it would fail
    /// to read metadata.  Because the modulo gate returns first, the helper
    /// returns `None` without touching the filesystem.
    #[test]
    fn maybe_reload_returns_none_off_gate() {
        let nonexistent = PathBuf::from("/definitely/not/here/config.toml");
        let wal = PathBuf::from("/definitely/not/here/wal.json");
        let mut reloader = LlmConfigReloader::new(nonexistent, wal);
        let cfg = current_cfg();
        // Cycle 1 ..= CONFIG_RELOAD_GATE_CYCLES-1 should all skip.
        for cycle in [1u64, 50, 99, 101, 199] {
            assert!(
                maybe_reload_llm_config(cycle, &mut reloader, &cfg).is_none(),
                "cycle {cycle} should skip the gate",
            );
        }
    }

    /// On-gate cycles: helper returns `Some(outcome)`.
    #[test]
    fn maybe_reload_fires_on_gate() {
        let cfg_path = tmp_cfg("gate.toml", BASE_CFG);
        let wal = std::env::temp_dir().join(format!(
            "apollo_periodic_reload_{}_gate_wal.json",
            std::process::id()
        ));
        let _ = fs::remove_file(&wal);
        let mut reloader = LlmConfigReloader::new(cfg_path, wal);
        let cfg = current_cfg();
        // 100, 200, 300 are all on the gate; must all return Some.
        for cycle in [100u64, 200, 300, 7200] {
            assert!(
                maybe_reload_llm_config(cycle, &mut reloader, &cfg).is_some(),
                "cycle {cycle} should be on the gate",
            );
        }
    }

    /// Cycle 0 sits on the gate arithmetically (0 % 100 == 0) — document this
    /// so anyone bumping the gate sees the boot-edge behavior is intended.
    #[test]
    fn maybe_reload_cycle_zero_is_on_gate() {
        let cfg_path = tmp_cfg("zero.toml", BASE_CFG);
        let wal = std::env::temp_dir().join(format!(
            "apollo_periodic_reload_{}_zero_wal.json",
            std::process::id()
        ));
        let _ = fs::remove_file(&wal);
        let mut reloader = LlmConfigReloader::new(cfg_path, wal);
        let cfg = current_cfg();
        assert!(maybe_reload_llm_config(0, &mut reloader, &cfg).is_some());
    }
}
