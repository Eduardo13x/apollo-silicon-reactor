//! # Daemon Markov Tick
//!
//! FocusMarkov prediction + temporal predictor per-cycle tick extracted from
//! main.rs (Wave 29). [Fowler 2004] Strangler Fig — pure move, no semantic change.
//!
//! ## Responsibilities
//! - FocusMarkov miss check: score last high-confidence prediction
//! - Markov observe + predicted-app pre-warm (unfreeze + QoS + cache)
//! - Universal pre-thaw: categories matching predicted next app
//! - Temporal predictor: observe fg transitions, blend Markov + temporal, cache-warm
//!
//! ## Ordering invariant
//! Must run AFTER foreground detection (fg_state → foreground_app/pid) and BEFORE
//! the context-switch burst detector (which updates last_fg_name).

use std::path::Path;

use apollo_engine::collector::SystemCollector;
use apollo_engine::engine::cache_warmer::CacheWarmer;
use apollo_engine::engine::daemon_helpers::{unfreeze_pids, write_frozen_state};
use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::focus_markov::FocusMarkov;
use apollo_engine::engine::freeze_intelligence::FreezeIntelligence;
use apollo_engine::engine::jetsam_control;
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::mach_qos::SchedulingTier;
use apollo_engine::engine::temporal_predictor::TemporalPredictor;
use chrono::{Timelike, Utc};

pub struct MarkovTickOutput {
    pub temporal_hour: u8,
    pub temporal_weekday: u8,
}

/// Run FocusMarkov + temporal predictor for this cycle.
///
/// # Parameters
/// - `foreground_app` — current foreground app name (from ForegroundDetector)
/// - `foreground_pid` — current foreground PID
/// - `last_fg_name` — fg name from previous cycle (for transition detection)
/// - `cycle_count` — current cycle index
/// - `focus_markov` — mutable FocusMarkov (observe + predict)
/// - `temporal_predictor` — mutable TemporalPredictor (observe + predict)
/// - `last_markov_prethaw` — pending prediction being tracked for miss detection
/// - `markov_hit_count` — cumulative hit counter for accuracy audit
/// - `markov_miss_count` — cumulative miss counter for accuracy audit
/// - `state` — SharedState (frozen_state, mach_qos, metrics)
/// - `collector` — SystemCollector (process table for PID lookup)
/// - `cache_warmer` — for cache pre-warming predicted apps
/// - `frozen_state_path` — for write_frozen_state after pre-thaw
#[allow(clippy::too_many_arguments)]
pub fn run_markov_tick(
    foreground_app: Option<&str>,
    foreground_pid: Option<u32>,
    last_fg_name: Option<&str>,
    cycle_count: u64,
    focus_markov: &mut FocusMarkov,
    temporal_predictor: &mut TemporalPredictor,
    last_markov_prethaw: &mut Option<(String, u64, u32, i32)>,
    markov_hit_count: &mut u32,
    markov_miss_count: &mut u32,
    state: &SharedState,
    collector: &SystemCollector,
    cache_warmer: &mut CacheWarmer,
    frozen_state_path: &Path,
) -> MarkovTickOutput {
    // Fight-hunt fix (2026-06-10): prefetch is a luxury. Under pressure the
    // maintenance/survival paths are EVICTING file cache while these
    // warm_pid calls fault pages back in — Apollo fighting itself,
    // amplifying thrashing. Gate all speculative cache warming on pressure.
    let cache_warm_allowed = {
        let m = state.metrics.lock_recover();
        cache_warm_allowed_at(m.metrics.memory_pressure)
    };

    // ── FocusMarkov miss check ───────────────────────────────────────────────
    // [Sutton & Barto 1998 §6 — temporal difference credit assignment]
    if let Some((ref predicted, pred_cycle, warmed_pid, prior_jetsam)) = *last_markov_prethaw {
        let cycles_elapsed = cycle_count.saturating_sub(pred_cycle);
        if cycles_elapsed >= 1 {
            let hit = foreground_app
                .map(|fa| {
                    fa.to_ascii_lowercase()
                        .contains(&predicted.to_ascii_lowercase())
                })
                .unwrap_or(false);
            if hit {
                *markov_hit_count += 1;
                // Evolve iter-4: prediction landed — runningboard owns the
                // now-foreground app's priorities; drop the ledger backstop.
                apollo_engine::engine::effect_ledger::forget_global(
                    &apollo_engine::engine::effect_ledger::AppliedEffect::JetsamPriority {
                        pid: warmed_pid,
                        prior: 0,
                    },
                );
                apollo_engine::engine::effect_ledger::forget_global(
                    &apollo_engine::engine::effect_ledger::AppliedEffect::MachTier {
                        pid: warmed_pid,
                    },
                );
            } else {
                *markov_miss_count += 1;
                // Anti-ratchet (2026-06-10 fight-hunt): a missed prediction
                // left the pre-warmed process at jetsam FOREGROUND (immune
                // to kernel kills) + Mach Foreground tier FOREVER. Revert
                // both on miss — restore the jetsam priority we captured
                // before the pre-warm and drop the tier back to Normal.
                // (On HIT runningboard owns the app's lifecycle priorities;
                // no Apollo cleanup needed.)
                if let Some(restore) = prewarm_jetsam_restore(prior_jetsam) {
                    let _ = jetsam_control::set_priority(warmed_pid, restore);
                }
                let mut qos = state.mach_qos.lock_recover();
                qos.set_tier(warmed_pid, SchedulingTier::Normal);
                drop(qos);
                tracing::debug!(
                    pid = warmed_pid,
                    predicted = %predicted,
                    "markov-miss: reverted pre-warm jetsam/tier"
                );
                // Evolve iter-4: already reverted — drop the ledger backstop.
                apollo_engine::engine::effect_ledger::forget_global(
                    &apollo_engine::engine::effect_ledger::AppliedEffect::JetsamPriority {
                        pid: warmed_pid,
                        prior: 0,
                    },
                );
                apollo_engine::engine::effect_ledger::forget_global(
                    &apollo_engine::engine::effect_ledger::AppliedEffect::MachTier {
                        pid: warmed_pid,
                    },
                );
            }
            *last_markov_prethaw = None;
            let total = *markov_hit_count + *markov_miss_count;
            if total > 0 && total.is_multiple_of(50) {
                let accuracy = *markov_hit_count as f64 / total as f64;
                apollo_engine::engine::daemon_helpers::audit_log(&serde_json::json!({
                    "event": "markov_prediction_accuracy",
                    "hits": markov_hit_count,
                    "misses": markov_miss_count,
                    "accuracy": (accuracy * 1000.0).round() / 1000.0,
                }));
            }
        }
    }

    // ── Markov observe + predicted-app pre-warm ──────────────────────────────
    let markov_prediction = focus_markov.observe(foreground_app);
    if let Some(ref pred) = markov_prediction {
        let pred_name_lc = pred.app_name.to_ascii_lowercase();
        let predicted_pid: Option<u32> = collector
            .system()
            .processes()
            .iter()
            .find(|(_, p)| p.name().to_ascii_lowercase() == pred_name_lc)
            .map(|(pid, _)| pid.as_u32());

        if let Some(pid) = predicted_pid {
            let mut frozen_guard = state.frozen_state.lock_recover();
            if frozen_guard.remove(&pid).is_some() {
                unfreeze_pids(std::iter::once(pid));
                write_frozen_state(frozen_state_path, &frozen_guard);
                state.metrics.lock_recover().metrics.unfreezes_applied += 1;
            }
            drop(frozen_guard);

            if pred.probability >= 0.50 {
                // Anti-ratchet: capture the prior jetsam priority so a missed
                // prediction can restore it (-1 sentinel = unreadable, skip
                // the jetsam revert but still drop the tier).
                let prior_jetsam = jetsam_control::get_priority(pid).unwrap_or(-1);
                let _ = jetsam_control::set_priority(pid, jetsam_control::priority::FOREGROUND);
                // Cable C: Proactive QoS — route predicted app to P-cores BEFORE switch.
                {
                    let mut qos = state.mach_qos.lock_recover();
                    qos.set_tier(pid, SchedulingTier::Foreground);
                }
                // Evolve iter-4: ledger backstop. The miss-check 1 cycle later
                // is the fast revert; the ledger catches the slow paths (miss
                // check never ran — daemon restart, prediction window skipped).
                let (warm_sec, _) = apollo_engine::engine::daemon_helpers::pid_start_time(pid);
                apollo_engine::engine::effect_ledger::record_global(
                    apollo_engine::engine::effect_ledger::AppliedEffect::JetsamPriority {
                        pid,
                        prior: prior_jetsam,
                    },
                    apollo_engine::engine::effect_ledger::DEFAULT_TTL,
                    warm_sec,
                    "markov pre-warm: jetsam FOREGROUND",
                );
                apollo_engine::engine::effect_ledger::record_global(
                    apollo_engine::engine::effect_ledger::AppliedEffect::MachTier { pid },
                    apollo_engine::engine::effect_ledger::DEFAULT_TTL,
                    warm_sec,
                    "markov pre-warm: Foreground tier",
                );
                if cache_warm_allowed {
                    cache_warmer.warm_pid(pid);
                }
                *last_markov_prethaw =
                    Some((pred.app_name.clone(), cycle_count, pid, prior_jetsam));
            }
        }
    }

    // ── Universal pre-thaw: FocusMarkov → pre-thaw ALL frozen processes ──────
    // whose category matches the hint for the predicted next app.
    // [Altmann & Trafton 2002] Pre-activate resources before predicted task switch.
    if let Some(ref pred) = markov_prediction {
        if pred.probability >= 0.35 {
            let elapsed = focus_markov.elapsed_dwell_secs();
            let time_to_switch = pred.avg_dwell_secs - elapsed;
            if time_to_switch > -5.0 && time_to_switch < 10.0 {
                let hint_categories = FreezeIntelligence::pre_thaw_hint(&pred.app_name);
                let mut frozen_guard = state.frozen_state.lock_recover();
                let candidates: Vec<(u32, String)> = frozen_guard
                    .iter()
                    .filter_map(|(&pid, entry)| {
                        let pname = entry.process_name.as_deref().unwrap_or("");
                        if !pname.is_empty() {
                            let cat = FreezeIntelligence::classify(pname);
                            if hint_categories.contains(&cat) {
                                return Some((pid, pname.to_string()));
                            }
                        }
                        None
                    })
                    .collect();
                if !candidates.is_empty() {
                    for (pid, pname) in &candidates {
                        if frozen_guard.remove(pid).is_some() {
                            unfreeze_pids(std::iter::once(*pid));
                            tracing::info!(
                                pid = pid,
                                process = pname.as_str(),
                                predicted_app = pred.app_name.as_str(),
                                prob = pred.probability,
                                time_to_switch = time_to_switch,
                                "freeze_intelligence: universal pre-thaw — switch imminent"
                            );
                        }
                    }
                    write_frozen_state(frozen_state_path, &frozen_guard);
                }
            }
        }
    }

    // ── Temporal predictor ───────────────────────────────────────────────────
    // Shin et al. 2012 — temporal patterns predict app launches with ~80% accuracy.
    // Update hour/weekday unconditionally every cycle for pressure_headroom_for_incoming().
    let now_chrono = Utc::now();
    let mut temporal_hour = now_chrono.hour() as u8;
    let mut temporal_weekday = chrono::Datelike::weekday(&now_chrono).num_days_from_monday() as u8;

    if let Some(fg_name) = foreground_app {
        let now_chrono = Utc::now();
        let hour = now_chrono.hour() as u8;
        let weekday = chrono::Datelike::weekday(&now_chrono).num_days_from_monday() as u8;
        temporal_hour = hour;
        temporal_weekday = weekday;

        let fg_changed = last_fg_name != Some(fg_name);
        if fg_changed {
            temporal_predictor.observe(fg_name, hour, weekday);
        }

        let markov_probs: std::collections::HashMap<String, f64> = focus_markov
            .predict_top_n(fg_name, 5)
            .into_iter()
            .map(|p| (p.app_name, p.probability))
            .collect();
        let temporal_preds = temporal_predictor.predict(hour, weekday, &markov_probs);

        for tpred in &temporal_preds {
            if tpred.temporal_score > 0.3 && tpred.probability > 0.15 && tpred.markov_score < 0.30 {
                let pred_lc = tpred.app_name.to_ascii_lowercase();
                if let Some(pid) = collector
                    .system()
                    .processes()
                    .iter()
                    .find(|(_, p)| p.name().to_ascii_lowercase() == pred_lc)
                    .map(|(pid, _)| pid.as_u32())
                {
                    if cache_warm_allowed {
                        cache_warmer.warm_pid(pid);
                    }
                }
            }
        }

        // Suppress unused foreground_pid warning when not passed to jetsam paths above.
        let _ = foreground_pid;
    }

    MarkovTickOutput {
        temporal_hour,
        temporal_weekday,
    }
}

/// Anti-ratchet (2026-06-10): what jetsam priority to restore after a
/// missed pre-warm prediction. `-1` is the "prior unreadable" sentinel —
/// in that case we skip the jetsam revert (writing a guessed band could
/// fight runningboard) and rely on the tier drop alone.
fn prewarm_jetsam_restore(prior_jetsam: i32) -> Option<i32> {
    (prior_jetsam >= 0).then_some(prior_jetsam)
}

/// Fight-hunt fix (2026-06-10): speculative cache warming is allowed only
/// below this pressure — above it, the purge paths are evicting the same
/// cache the warmer would fill (self-fight, thrash amplification).
fn cache_warm_allowed_at(pressure: f64) -> bool {
    pressure < 0.60
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_warm_gated_above_060() {
        assert!(cache_warm_allowed_at(0.45));
        assert!(cache_warm_allowed_at(0.59));
        assert!(!cache_warm_allowed_at(0.60));
        assert!(!cache_warm_allowed_at(0.75));
    }

    #[test]
    fn prewarm_restore_sentinel_semantics() {
        // Unreadable prior → no jetsam write on miss.
        assert_eq!(prewarm_jetsam_restore(-1), None);
        // Captured priors restore verbatim, including IDLE (0).
        assert_eq!(prewarm_jetsam_restore(0), Some(0));
        assert_eq!(prewarm_jetsam_restore(2), Some(2));
        assert_eq!(prewarm_jetsam_restore(9), Some(9));
    }
}
