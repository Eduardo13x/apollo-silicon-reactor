//! # Daemon Swap Reclaim Tick
//!
//! Swap reclaim ODE reactor-weight boost extracted from main.rs (Wave 35).
//! [Fowler 2004] Strangler Fig — pure move, no semantic change.
//!
//! ## Responsibilities
//! - Match SaturationForecast risk level to pre-emptive reactor_weight boost
//! - Critical (T_sat ≤ 60s): +0.25 + tracing info
//! - Overflow (past threshold): =1.0 + tracing warn
//! - Building (net positive): +0.05 (early nudge)
//! - Safe: no-op
//!
//! ## Ordering invariant
//! Must run AFTER reclaim_forecast is computed (swap_reclaim.update()) and
//! BEFORE decision_stage.run() so the boosted reactor_weight flows into the
//! freeze gate.

use apollo_optimizer::engine::swap_reclaim::{SaturationForecast, SwapRisk, CRITICAL_ETA_SEC};

/// Apply pre-emptive reactor_weight boost based on ODE saturation forecast.
///
/// # Parameters
/// - `reclaim_forecast` — current cycle saturation forecast
/// - `reactor_weight` — mutable reactor weight (clamped to [0.0, 1.0])
pub fn apply_swap_reclaim_boost(
    reclaim_forecast: &SaturationForecast,
    reactor_weight: &mut f64,
) {
    match reclaim_forecast.risk {
        SwapRisk::Critical => {
            *reactor_weight = (*reactor_weight + 0.25).min(1.0);
            if let Some(eta) = reclaim_forecast.t_sat_sec {
                tracing::info!(
                    target: "apollo.swap_reclaim",
                    eta_sec = format!("{:.1}", eta),
                    net_mbps = format!("{:.2}", reclaim_forecast.net_rate_bps / (1024.0 * 1024.0)),
                    "swap reclaim ODE: Critical — reactor boosted +0.25"
                );
            }
        }
        SwapRisk::Overflow => {
            *reactor_weight = 1.0;
            tracing::warn!(
                target: "apollo.swap_reclaim",
                swap_ratio = format!("{:.2}", reclaim_forecast.swap_ratio),
                "swap reclaim ODE: Overflow — reactor_weight=1.0"
            );
        }
        SwapRisk::Building => {
            let _ = CRITICAL_ETA_SEC;
            *reactor_weight = (*reactor_weight + 0.05).min(1.0);
        }
        SwapRisk::Safe => {}
    }
}
