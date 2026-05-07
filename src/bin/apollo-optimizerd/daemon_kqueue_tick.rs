//! # Daemon kqueue Tick
//!
//! kqueue VM pressure event consumer + reactor_weight adjuster extracted from
//! main.rs (Wave 26). [Fowler 2004] Strangler Fig — pure move, no semantic change.
//!
//! ## Responsibilities
//! - Drain kqueue events: VmPressure → boost reactor_weight + fast-tick
//! - Record overflow into OverflowGuard + hazard model (SignalIntelligence)
//! - Clean up frozen_state + display_turbo on ProcessExited
//!
//! ## Ordering invariant
//! Must run AFTER reactor_weight is initialized (decay already applied in caller)
//! and BEFORE pressure aggregation / decide_actions — kqueue events provide the
//! lowest-latency pressure signal (zero polling).

use std::path::Path;

use apollo_optimizer::engine::daemon_helpers::write_frozen_state;
use apollo_optimizer::engine::daemon_state::SharedState;
use apollo_optimizer::engine::display_turbo::DisplayTurbo;
use apollo_optimizer::engine::kqueue_pressure;
use apollo_optimizer::engine::learned_state::LearnableParams;
use apollo_optimizer::engine::lock_ext::LockRecover;
use apollo_optimizer::engine::overflow_guard::OverflowGuard;
use apollo_optimizer::engine::signal_intelligence::SignalIntelligence;
use apollo_optimizer::engine::identity_cache_manager::IdentityCacheManager;
use apollo_optimizer::collector::SystemSnapshot;
use std::time::{Duration, Instant};

/// Drain kqueue events for this cycle and update reactor_weight.
///
/// # Parameters
/// - `kq_frozen` — optional kqueue pressure handle (None if init failed)
/// - `reactor_weight` — mutable reactor weight (modified in-place)
/// - `state` — SharedState (metrics fast_tick_until + frozen_state + unfreezes_applied)
/// - `snapshot` — system snapshot for overflow context (top_processes, pressure)
/// - `overflow_guard` — lctx.overflow_guard (records kqueue overflow events)
/// - `signal_intel` — lctx.signal_intel (trains hazard model on real OOM indicators)
/// - `display_turbo` — display-off turbo state (clean up dead PIDs)
/// - `frozen_state_path` — for write_frozen_state WAL after ProcessExited
/// - `learnable_params` — RL pressure/compressor bands for overflow recording
#[allow(clippy::too_many_arguments)]
pub fn run_kqueue_tick(
    kq_frozen: &mut Option<kqueue_pressure::KqueuePressure>,
    reactor_weight: &mut f64,
    state: &SharedState,
    snapshot: &SystemSnapshot,
    overflow_guard: &mut OverflowGuard,
    signal_intel: &mut SignalIntelligence,
    display_turbo: &mut DisplayTurbo,
    frozen_state_path: &Path,
    learnable_params: &LearnableParams,
    identity_cache: &IdentityCacheManager,
) {
    let kq = match kq_frozen.as_mut() {
        Some(kq) => kq,
        None => return,
    };

    for event in kq.poll_events() {
        match event {
            kqueue_pressure::PressureEvent::VmPressure(level) => {
                use kqueue_pressure::VmPressureLevel;
                match level {
                    VmPressureLevel::Critical | VmPressureLevel::SuddenTerminate => {
                        *reactor_weight = 1.0;
                        state.metrics.lock_recover().fast_tick_until =
                            Some(Instant::now() + Duration::from_secs(30));
                        println!("kqueue: VM pressure {:?} — fast-tick engaged", level);

                        // Registrar overflow: ajustar thresholds para prevenir próxima vez.
                        // Excluir el propio daemon — aparece en top_processes durante
                        // survival-mode por el trabajo intensivo que hace, contaminando
                        // el diagnóstico de causa del overflow.
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
                            &format!("kqueue-{:?}", level),
                            snapshot.pressure.compressor_pressure,
                            &learnable_params.rl_pressure_bands,
                            &learnable_params.rl_compressor_bands,
                        );

                        // Teach hazard model only on real OOM indicators:
                        // swap must be GROWING (delta > 512KB/s) and present
                        // (> 10% used). kqueue Critical fires at ~80% pressure
                        // which is normal; training on it saturates base_rate.
                        let sr = if snapshot.pressure.swap_total_bytes > 0 {
                            snapshot.pressure.swap_used_bytes as f64
                                / snapshot.pressure.swap_total_bytes as f64
                        } else {
                            0.0
                        };
                        let swap_growing =
                            snapshot.pressure.swap_delta_bytes_per_sec > 524_288.0;
                        if sr > 0.10 && swap_growing {
                            signal_intel.record_overflow(
                                snapshot.pressure.memory_pressure,
                                sr,
                                snapshot.pressure.memory_pressure,
                            );
                        }
                    }
                    VmPressureLevel::Warning => {
                        *reactor_weight = (*reactor_weight + 0.5).min(1.0);
                    }
                    VmPressureLevel::Normal => {}
                }
            }
            kqueue_pressure::PressureEvent::ProcessExited(pid) => {
                // Frozen process died (jetsam/OOM) — clean up immediately.
                let mut frozen_state = state.frozen_state.lock_recover();
                if frozen_state.remove(&pid).is_some() {
                    write_frozen_state(frozen_state_path, &frozen_state);
                    state.metrics.lock_recover().metrics.unfreezes_applied += 1;
                }
                // Also clean up display turbo's set — prevents unbounded
                // growth if many processes die while frozen during turbo.
                display_turbo.remove_pid(pid);
                // Notify the identity cache manager of the exit. Without
                // this, kernel may recycle the PID within the TTL window
                // and the manager's PID-only lookup would return
                // CachedValid for a different process (ABA bug —
                // fix 2026-05-07; consolidated under
                // IdentityCacheManager 2026-05-07 Fase 2).
                identity_cache.notify_exited(pid);
            }
            kqueue_pressure::PressureEvent::TimerTick => {}
        }
    }
}
