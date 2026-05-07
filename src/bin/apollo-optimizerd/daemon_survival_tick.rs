//! # Daemon Survival Tick
//!
//! Survival-mode overflow recording + activation handling extracted from
//! main.rs (Wave 27). [Fowler 2004] Strangler Fig — pure move, no semantic change.
//!
//! ## Responsibilities
//! - Detect survival mode (pressure >0.85 / swap thrashing / p_oom escalation)
//! - Record overflow into OverflowGuard + hazard model when real overflow detected
//! - Track swap growth streak for RL meta-gate
//! - Increment survival_mode_entry_count, demote Chromium renderers, last-resort purge
//! - overflow_guard.tick_decay each cycle (calm relaxation)
//!
//! ## Ordering invariant
//! Must run AFTER signal_digest is available and BEFORE neuromodulator / decide_actions.

use std::time::{Duration, Instant};

use apollo_optimizer::collector::SystemSnapshot;
use apollo_optimizer::engine::chromium_manager::ChromiumManager;
use apollo_optimizer::engine::daemon_state::SharedState;
use apollo_optimizer::engine::learned_state::LearnableParams;
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::overflow_guard::OverflowGuard;
use apollo_optimizer::engine::safety::{survival_mode_active_total, swap_exhaustion_threshold_bytes};
use apollo_optimizer::engine::signal_intelligence::SignalIntelligence;
use apollo_optimizer::engine::signal_intelligence::SignalDigest;

/// Run survival-mode detection, overflow recording, and threshold decay.
///
/// # Parameters
/// - `snapshot` — system snapshot for this cycle
/// - `signal_digest` — signal intelligence output (p_oom_30s, pressure_smooth)
/// - `cycle_count` — cycle counter (warmup gate for p_oom escalation)
/// - `overflow_guard` — lctx.overflow_guard (records events + decay)
/// - `signal_intel` — lctx.signal_intel (hazard model training)
/// - `learnable_params` — RL pressure/compressor bands
/// - `swap_growth_streak` — mutable swap-growth counter for RL meta-gate
/// - `state` — SharedState (survival_mode_entry_count metric)
/// - `chromium_mgr` — demote renderers in survival mode
/// - `last_purge_at` — rate-limit guard for `purge` command (once per 10 min)
#[allow(clippy::too_many_arguments)]
pub fn run_survival_tick(
    snapshot: &SystemSnapshot,
    signal_digest: &SignalDigest,
    cycle_count: u64,
    overflow_guard: &mut OverflowGuard,
    signal_intel: &mut SignalIntelligence,
    learnable_params: &LearnableParams,
    swap_growth_streak: &mut u32,
    state: &SharedState,
    chromium_mgr: &mut ChromiumManager,
    last_purge_at: &mut Option<Instant>,
) {
    let p_oom_escalation = cycle_count > 5
        && signal_digest.p_oom_30s > 0.80
        && snapshot.pressure.memory_pressure >= 0.70;
    let survival_mode = snapshot.pressure.memory_pressure > 0.85
        || snapshot.pressure.swap_delta_bytes_per_sec > 1_000_000.0
        || p_oom_escalation;

    // Overflow guard: only record when real pressure (≥ 0.60). Swap storms at
    // 36-42% were poisoning the guard with false positives.
    let real_overflow = survival_mode && snapshot.pressure.memory_pressure >= 0.60;
    if real_overflow {
        let heavy: Vec<String> = snapshot
            .top_processes
            .iter()
            .filter(|p| p.name != "apollo-optimizerd")
            .take(8)
            .map(|p| p.name.clone())
            .collect();
        overflow_guard.record_event(
            snapshot.pressure.memory_pressure,
            snapshot.pressure.swap_delta_bytes_per_sec,
            &heavy,
            "survival-mode",
            snapshot.pressure.compressor_pressure,
            &learnable_params.rl_pressure_bands,
            &learnable_params.rl_compressor_bands,
        );
        let sr = if snapshot.pressure.swap_total_bytes > 0 {
            snapshot.pressure.swap_used_bytes as f64
                / snapshot.pressure.swap_total_bytes as f64
        } else {
            0.0
        };
        // Only train hazard model when swap is actively growing (real OOM risk).
        let swap_growing = snapshot.pressure.swap_delta_bytes_per_sec > 524_288.0;
        if sr > 0.10 && swap_growing {
            signal_intel.record_overflow(
                snapshot.pressure.memory_pressure,
                sr,
                snapshot.pressure.memory_pressure,
            );
        }
    }

    // Track swap growth streak → RL meta-gate.
    if snapshot.pressure.swap_delta_bytes_per_sec > 1_048_576.0 {
        *swap_growth_streak = swap_growth_streak.saturating_add(1);
    } else {
        *swap_growth_streak = 0;
    }
    if let Some(rl) = overflow_guard.rl_agent.as_mut() {
        rl.set_swap_growth_streak(*swap_growth_streak);
    }

    // Observability: count one activation per cycle survival is active.
    let survival_active = survival_mode_active_total(
        snapshot.pressure.memory_pressure,
        snapshot.pressure.swap_used_bytes,
        snapshot.pressure.swap_total_bytes,
    );
    if survival_active {
        state.metrics.lock_recover().metrics.survival_mode_entry_count += 1;

        // Jetsam demotion: mark non-foreground Chromium renderers as BACKGROUND
        // so the kernel kills them first under OOM — softer than SIGSTOP.
        let _ = chromium_mgr.demote_background_renderers();

        // Last-resort page reclaim: spawn `purge` when swap crosses 80% of
        // exhaustion threshold. Rate-limited to once per 10 min.
        let threshold = swap_exhaustion_threshold_bytes(snapshot.pressure.swap_total_bytes);
        let swap_used = snapshot.pressure.swap_used_bytes;
        if swap_used as f64 >= threshold as f64 * 0.80 {
            let can_purge = last_purge_at
                .map(|t| t.elapsed() >= Duration::from_secs(600))
                .unwrap_or(true);
            if can_purge {
                if std::process::Command::new("purge").spawn().is_ok() {
                    *last_purge_at = Some(Instant::now());
                }
            }
        }
    }

    // Gradual decay: relax thresholds when system is calm.
    overflow_guard.tick_decay(
        snapshot.pressure.memory_pressure,
        snapshot.pressure.compressor_pressure,
        &learnable_params.rl_pressure_bands,
        &learnable_params.rl_compressor_bands,
    );
}
