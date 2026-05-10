//! # Swap Reclaim ODE
//!
//! Models compressor/swap dynamics as a first-order linear ODE:
//!
//! ```text
//!     dS/dt = dirty_rate − reclaim_rate
//! ```
//!
//! where
//!
//! - `S`            — compressor occupancy (bytes held by the WKdm compressor).
//! - `dirty_rate`   — `compressions_per_sec × PAGE_SIZE`: pages actively flowing
//!                    from RAM into the compressor (working-set overflow).
//! - `reclaim_rate` — `(decompressions_per_sec + purges_per_sec) × PAGE_SIZE`:
//!                    pages freed back to the system (kernel reclaim or purge).
//!
//! The **net accumulation rate** `ṡ = dirty_rate − reclaim_rate` gives a
//! signed signal that predicts compressor saturation seconds before the level
//! crosses any threshold:
//!
//! ```text
//!     T_sat = (S_capacity × 0.85 − S_now) / ṡ   (when ṡ > 0)
//! ```
//!
//! This is strictly more informative than velocity-based prediction because
//! it exposes *which component* is driving growth — a throttled dirty_rate
//! recovers quickly; a collapsed reclaim_rate requires a different intervention
//! (freeze or purge).
//!
//! ## macOS mapping
//!
//! macOS does not expose "dirty pages" in the Linux sense.  The closest
//! equivalent is the **compressor** (WKdm anonymous memory compression):
//!
//! - `compressions_per_sec`   — pages newly compressed: memory that can no
//!   longer fit in the active working set.  Analogous to Linux dirty_rate.
//! - `decompressions_per_sec` — pages decompressed: kernel reusing compressor
//!   slots (soft reclaim — no I/O required).
//! - `purges_per_sec`         — file-cache pages purged (hard eviction).
//! - `swapouts_per_sec`       — overflow from compressor to SSD swap file.
//!   High swapouts = compressor full, true I/O pressure begins.
//!
//! `dirty_rate` and `reclaim_rate` are each EMA-smoothed (α = 0.2) to
//! suppress per-cycle noise from the 50–200 ms background collector window.
//!
//! ## Papers
//!
//! - [Aho & Ullman 1972] "The Theory of Parsing, Translation, and Compiling" —
//!   steady-state flow analysis (rate balance determines saturation).
//! - [Denning 1968] "The Working Set Model" — memory demand exceeding supply
//!   produces compressor/swap pressure.
//! - [Zhao et al. 2009] "Dynamic Memory Compression: Reduce Data Movement in
//!   Hierarchical Memory" — compression-first architecture matches Apple's
//!   WKdm design; rates model the compressor pipeline.

use serde::{Deserialize, Serialize};

/// macOS page size in bytes (Apple Silicon default 16 KiB, but compressor
/// tracks in 4 KiB logical pages for compatibility; we use 16 KiB to match
/// `vm_stat` page size on M-series).
pub const PAGE_SIZE_BYTES: u64 = 16_384;

/// EMA smoothing factor for rate estimates.  α = 0.2 → τ_ema ≈ 4 cycles
/// (one cycle ≈ 2 s → ~8 s half-life), which smooths burst noise while
/// still tracking genuine load shifts within 30 s.
pub const EMA_ALPHA: f64 = 0.2;

/// Warn early when swap saturation is predicted within this many seconds.
pub const CRITICAL_ETA_SEC: f64 = 60.0;

/// Fraction of swap capacity considered "full" (mirrors SwapPredictor).
pub const SWAP_CRITICAL_RATIO: f64 = 0.85;

/// Minimum net_rate (bytes/s) to trust — below this threshold the ODE is
/// effectively at rest and T_sat would be astronomically large or noisy.
pub const NET_RATE_FLOOR_BYTES_SEC: f64 = 4_096.0; // 4 KB/s

/// Minimum swap capacity before the model activates (avoids divide-by-zero
/// on systems with swap disabled or tiny swap files).
pub const MIN_SWAP_CAPACITY_BYTES: u64 = 64 * 1024 * 1024; // 64 MB

/// Minimum swapout rate (pages/s) to escalate to Critical.
/// Set to 0.5 pps (= 8 KB/s to SSD) rather than 1.0 to catch "slow-boil"
/// scenarios: a process generating 0.5 pps sustained will exhaust swap in
/// minutes without crossing a 1.0 floor.  The EMA (α=0.2, τ≈4 cycles) already
/// smooths burst noise, so 0.5 represents genuine sustained I/O, not a spike.
/// [Zhao et al. 2009] — swapout = compressor overflow event (observable I/O).
pub const SWAPOUT_FLOOR_PPS: f64 = 0.5;

/// Saturation risk classification derived from `T_sat` and `net_rate`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwapRisk {
    /// Reclaim is keeping up; no saturation risk.
    Safe,
    /// Net accumulation positive but T_sat > `CRITICAL_ETA_SEC`.
    Building,
    /// T_sat ≤ `CRITICAL_ETA_SEC`; early intervention warranted.
    Critical,
    /// Compressor already above `SWAP_CRITICAL_RATIO`; immediate action.
    Overflow,
}

impl SwapRisk {
    /// Score in [0.0, 1.0] — used for blending with existing pressure signals.
    pub fn score(self) -> f64 {
        match self {
            SwapRisk::Safe => 0.0,
            SwapRisk::Building => 0.3,
            SwapRisk::Critical => 0.7,
            SwapRisk::Overflow => 1.0,
        }
    }
}

/// Normalization ceiling for `net_rate_bps` → [0, 1].
/// 200 MB/s ≈ peak WKdm compression rate on M-series under extreme load.
pub const NET_RATE_CEILING_BPS: f64 = 200_000_000.0;

// ── CyberPhysical trait ────────────────────────────────────────────────────────

/// Trait for ODE-derived signals that normalize raw physical quantities to [0,1].
///
/// Unifies the ad-hoc `(CRITICAL_ETA_SEC / t).clamp(0,1)` and
/// `(net_rate / 200 MB/s).clamp(0,1)` patterns scattered across callers.
/// Each implementor owns its normalization constant — callers just call `.normalized()`.
pub trait CyberPhysicalSignal {
    /// Map this signal to [0.0, 1.0] for use in RL/LinUCB/neuromodulator contexts.
    fn normalized(&self) -> f64;
    /// Signal name for logging/observability.
    fn name(&self) -> &'static str;
}

/// T_sat urgency: (CRITICAL_ETA_SEC / t_sat_sec).clamp(0,1).
/// 0 = safe, 1 = saturation within `CRITICAL_ETA_SEC` seconds.
pub struct TsatUrgency(pub Option<f64>);

/// Net rate normalized: net_rate_bps / NET_RATE_CEILING_BPS, clamped [0,1].
pub struct NetRateNorm(pub f64);

impl CyberPhysicalSignal for TsatUrgency {
    fn normalized(&self) -> f64 {
        self.0
            .map(|t| (CRITICAL_ETA_SEC / t.max(1.0)).clamp(0.0, 1.0))
            .unwrap_or(0.0)
    }
    fn name(&self) -> &'static str {
        "t_sat_urgency"
    }
}

impl CyberPhysicalSignal for NetRateNorm {
    fn normalized(&self) -> f64 {
        (self.0 / NET_RATE_CEILING_BPS).clamp(0.0, 1.0)
    }
    fn name(&self) -> &'static str {
        "net_rate_norm"
    }
}

/// Snapshot produced each cycle — consumed by `SignalDigest` and decision logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaturationForecast {
    /// EMA-smoothed dirty rate (bytes/s).
    pub dirty_rate_bps: f64,
    /// EMA-smoothed reclaim rate (bytes/s).
    pub reclaim_rate_bps: f64,
    /// Net accumulation rate = dirty − reclaim (bytes/s).  Negative = draining.
    pub net_rate_bps: f64,
    /// EMA-smoothed swapout rate (pages/s).  > 0 = compressor spilling to SSD.
    pub swapouts_ema_pps: f64,
    /// Predicted seconds until compressor occupancy hits 85 % of swap capacity.
    /// `None` when net_rate ≤ 0 (system is draining) or capacity unknown.
    pub t_sat_sec: Option<f64>,
    /// Risk level derived from `t_sat_sec`, current occupancy, and swapout rate.
    pub risk: SwapRisk,
    /// True only on the first cycle transitioning into Overflow (swap_ratio ≥ 0.85).
    /// False on subsequent consecutive Overflow cycles. Callers should gate WARN logs
    /// on this flag to prevent log spam during sustained swap saturation.
    pub overflow_entered: bool,
    /// Current swap occupancy ratio in [0.0, 1.0] for reference.
    pub swap_ratio: f64,
    /// Empirical volatility of net_rate (bytes/s) — std-dev estimated via EMA
    /// of squared deviations [Welford 1962]. High σ at low mean = "sticky swap"
    /// harbinger: pressure will spike faster than T_sat predicts.
    /// [Øksendal 2003 §3] — σ is the diffusion term in the SDE dS = μ dt + σ dW.
    pub net_rate_volatility: f64,
}

/// Per-cycle input: caller provides the macOS VM flow rates and current swap
/// level.  All fields are optional so the model degrades gracefully when the
/// background collector has not produced a sample yet.
#[derive(Debug, Clone, Default)]
pub struct VmFlowSample {
    pub compressions_per_sec: f64,
    pub decompressions_per_sec: f64,
    pub purges_per_sec: f64,
    pub swapouts_per_sec: f64,
    /// Current swap occupancy in bytes.
    pub swap_used_bytes: u64,
    /// Total swap capacity in bytes.
    pub swap_total_bytes: u64,
}

/// The swap reclaim model — owns EMA state for dirty, reclaim, and swapout rates.
#[derive(Debug, Default)]
pub struct SwapReclaimModel {
    /// EMA of dirty_rate (bytes/s).
    dirty_ema_bps: f64,
    /// EMA of reclaim_rate (bytes/s).
    reclaim_ema_bps: f64,
    /// EMA of swapout rate (pages/s) — gates Critical escalation.
    swapout_ema_pps: f64,
    /// EMA of squared net_rate first-differences — Welford variance estimate.
    /// σ = sqrt(net_rate_var_ema) exposed in SaturationForecast.
    net_rate_var_ema: f64,
    /// Previous cycle's net_rate for first-difference volatility computation.
    net_rate_prev: f64,
    /// Number of samples ingested (warm-up guard).
    samples: u32,
    /// Consecutive cycles where risk == Overflow. Used to gate WARN log spam:
    /// only the first cycle in an overflow run emits WARN; subsequent cycles
    /// are suppressed until the system exits and re-enters Overflow.
    overflow_cycles: u32,
}

impl SwapReclaimModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest one cycle's VM flow rates and return a `SaturationForecast`.
    pub fn update(&mut self, sample: &VmFlowSample) -> SaturationForecast {
        // Convert page/sec → bytes/sec.
        let dirty_bps = sample.compressions_per_sec * PAGE_SIZE_BYTES as f64;
        // reclaim = voluntary decompressions + forced purges.
        // swapouts are NOT counted as reclaim — they represent compressor
        // overflow spilling to disk, which is the emergency state we predict.
        let reclaim_bps =
            (sample.decompressions_per_sec + sample.purges_per_sec) * PAGE_SIZE_BYTES as f64;

        // EMA update — on first sample seed the EMAs directly (no false lag).
        if self.samples == 0 {
            self.dirty_ema_bps = dirty_bps;
            self.reclaim_ema_bps = reclaim_bps;
            self.swapout_ema_pps = sample.swapouts_per_sec;
        } else {
            self.dirty_ema_bps = EMA_ALPHA * dirty_bps + (1.0 - EMA_ALPHA) * self.dirty_ema_bps;
            self.reclaim_ema_bps =
                EMA_ALPHA * reclaim_bps + (1.0 - EMA_ALPHA) * self.reclaim_ema_bps;
            self.swapout_ema_pps =
                EMA_ALPHA * sample.swapouts_per_sec + (1.0 - EMA_ALPHA) * self.swapout_ema_pps;
        }
        self.samples = self.samples.saturating_add(1);

        let net_rate = self.dirty_ema_bps - self.reclaim_ema_bps;

        // Volatility: EMA of squared first-differences of net_rate.
        // [Welford 1962] online variance; [Øksendal 2003 §3] σ = diffusion term.
        // δ = net_rate - prev_net_rate captures cycle-to-cycle change rate.
        // After warm-up (≥2 samples), σ² = EMA(δ²) is valid.
        let delta = net_rate - self.net_rate_prev;
        self.net_rate_var_ema =
            EMA_ALPHA * (delta * delta) + (1.0 - EMA_ALPHA) * self.net_rate_var_ema;
        let net_rate_volatility = if self.samples >= 2 {
            self.net_rate_var_ema.sqrt()
        } else {
            0.0
        };

        let swap_total = sample.swap_total_bytes;
        let swap_used = sample.swap_used_bytes;
        let swap_ratio = if swap_total > 0 {
            swap_used as f64 / swap_total as f64
        } else {
            0.0
        };

        // Risk classification.
        // Critical requires active swapouts (confirmed SSD I/O) to distinguish
        // from the M1 "sticky swap" baseline where reclaim_rate ≈ 0 and T_sat
        // is always short despite no real disk pressure [Zhao 2009].
        let has_io = self.swapout_ema_pps >= SWAPOUT_FLOOR_PPS;
        let risk = if swap_total < MIN_SWAP_CAPACITY_BYTES {
            SwapRisk::Safe // no swap configured
        } else if swap_ratio >= SWAP_CRITICAL_RATIO {
            SwapRisk::Overflow
        } else if net_rate < NET_RATE_FLOOR_BYTES_SEC {
            SwapRisk::Safe // draining or at rest
        } else {
            // T_sat = headroom / net_rate
            let headroom = (swap_total as f64 * SWAP_CRITICAL_RATIO).max(0.0) - swap_used as f64;
            let t_sat = headroom.max(0.0) / net_rate;
            if t_sat <= CRITICAL_ETA_SEC && has_io {
                SwapRisk::Critical
            } else {
                SwapRisk::Building
            }
        };

        // Compute T_sat for the forecast struct.
        let t_sat_sec = if swap_total >= MIN_SWAP_CAPACITY_BYTES
            && net_rate >= NET_RATE_FLOOR_BYTES_SEC
            && swap_ratio < SWAP_CRITICAL_RATIO
        {
            let headroom = (swap_total as f64 * SWAP_CRITICAL_RATIO) - swap_used as f64;
            Some((headroom.max(0.0) / net_rate).min(3600.0)) // cap at 1 hour
        } else {
            None
        };

        self.net_rate_prev = net_rate;

        // Track overflow run length to gate WARN log spam.
        let overflow_entered = if matches!(risk, SwapRisk::Overflow) {
            self.overflow_cycles = self.overflow_cycles.saturating_add(1);
            self.overflow_cycles == 1 // true only on entry cycle
        } else {
            self.overflow_cycles = 0;
            false
        };

        SaturationForecast {
            dirty_rate_bps: self.dirty_ema_bps,
            reclaim_rate_bps: self.reclaim_ema_bps,
            net_rate_bps: net_rate,
            swapouts_ema_pps: self.swapout_ema_pps,
            t_sat_sec,
            risk,
            overflow_entered,
            swap_ratio,
            net_rate_volatility,
        }
    }

    /// Reset EMA state (e.g., after a sleep/wake cycle).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn samples(&self) -> u32 {
        self.samples
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn gb(n: u64) -> u64 {
        n * 1024 * 1024 * 1024
    }

    fn sample(comp: f64, decomp: f64, purge: f64, used: u64, total: u64) -> VmFlowSample {
        VmFlowSample {
            compressions_per_sec: comp,
            decompressions_per_sec: decomp,
            purges_per_sec: purge,
            swapouts_per_sec: 0.0,
            swap_used_bytes: used,
            swap_total_bytes: total,
        }
    }

    fn sample_io(
        comp: f64,
        decomp: f64,
        purge: f64,
        swapouts: f64,
        used: u64,
        total: u64,
    ) -> VmFlowSample {
        VmFlowSample {
            compressions_per_sec: comp,
            decompressions_per_sec: decomp,
            purges_per_sec: purge,
            swapouts_per_sec: swapouts,
            swap_used_bytes: used,
            swap_total_bytes: total,
        }
    }

    #[test]
    fn safe_when_reclaim_exceeds_dirty() {
        let mut m = SwapReclaimModel::new();
        // comp < decomp → reclaim wins → net < 0 → Safe
        let f = m.update(&sample(100.0, 200.0, 0.0, gb(1), gb(8)));
        assert!(f.net_rate_bps < 0.0);
        assert_eq!(f.risk, SwapRisk::Safe);
        assert!(f.t_sat_sec.is_none());
    }

    #[test]
    fn building_when_net_positive_but_far() {
        let mut m = SwapReclaimModel::new();
        // comp > decomp, swap 10 % full → plenty of headroom
        let f = m.update(&sample(200.0, 100.0, 0.0, gb(1), gb(8)));
        assert!(f.net_rate_bps > 0.0);
        assert_eq!(f.risk, SwapRisk::Building);
        // T_sat should exist and be > CRITICAL_ETA_SEC
        let eta = f.t_sat_sec.unwrap();
        assert!(eta > CRITICAL_ETA_SEC, "eta={}", eta);
    }

    #[test]
    fn critical_when_t_sat_within_threshold_and_swapouts_active() {
        let mut m = SwapReclaimModel::new();
        // comp=10_000 pages/s, swapouts=10 pps → real SSD I/O confirmed
        // swap 80 % full, capacity 8 GB → headroom to 85% = 409 MB
        // T_sat = 409M / 163M ≈ 2.5 s → Critical
        let used = (gb(8) as f64 * 0.80) as u64;
        let f = m.update(&sample_io(10_000.0, 0.0, 0.0, 10.0, used, gb(8)));
        assert_eq!(f.risk, SwapRisk::Critical);
        let eta = f.t_sat_sec.unwrap();
        assert!(eta <= CRITICAL_ETA_SEC, "eta={}", eta);
    }

    #[test]
    fn building_when_t_sat_short_but_no_swapouts() {
        // M1 sticky-swap regression test: short T_sat without active swapouts
        // should stay at Building, not escalate to Critical.
        // Reclaim ≈ 0 (XNU does not defrag swap eagerly) but compressor not spilling.
        let mut m = SwapReclaimModel::new();
        let used = (gb(8) as f64 * 0.80) as u64;
        let f = m.update(&sample(10_000.0, 0.0, 0.0, used, gb(8)));
        assert_eq!(
            f.risk,
            SwapRisk::Building,
            "short T_sat without swapouts should be Building (sticky-swap false-alarm gate)"
        );
    }

    #[test]
    fn overflow_when_already_past_threshold() {
        let mut m = SwapReclaimModel::new();
        let used = (gb(8) as f64 * 0.90) as u64;
        let f = m.update(&sample(100.0, 0.0, 0.0, used, gb(8)));
        assert_eq!(f.risk, SwapRisk::Overflow);
        assert!(f.t_sat_sec.is_none()); // already over, eta undefined
    }

    #[test]
    fn critical_at_slow_boil_swapout_rate() {
        // Slow-boil regression: 0.6 pps sustained swapout (above SWAPOUT_FLOOR_PPS=0.5)
        // with short T_sat must escalate to Critical, not stay at Building.
        let mut m = SwapReclaimModel::new();
        let used = (gb(8) as f64 * 0.80) as u64;
        let f = m.update(&sample_io(10_000.0, 0.0, 0.0, 0.6, used, gb(8)));
        assert_eq!(
            f.risk,
            SwapRisk::Critical,
            "slow-boil 0.6 pps (> SWAPOUT_FLOOR_PPS=0.5) must escalate to Critical"
        );
    }

    #[test]
    fn safe_when_no_swap_configured() {
        let mut m = SwapReclaimModel::new();
        // swap_total below MIN_SWAP_CAPACITY_BYTES
        let f = m.update(&sample(1000.0, 0.0, 0.0, 0, 1024));
        assert_eq!(f.risk, SwapRisk::Safe);
    }

    #[test]
    fn ema_smooths_transient_spike() {
        let mut m = SwapReclaimModel::new();
        // Seed with quiet state: comp=10, decomp=50 → net < 0 → Safe
        for _ in 0..5 {
            m.update(&sample(10.0, 50.0, 0.0, gb(1), gb(8)));
        }
        // Moderate spike (50× quiet dirty rate) — EMA at α=0.2 absorbs 80%.
        // A 5000× spike CAN trigger Critical in 1 sample (expected — EMA is
        // not an outlier filter for that magnitude).  Use a realistic 50× spike:
        // dirty_ema ≈ 1.8 MB/s, T_sat ≈ 5000 s → Building, not Critical.
        let f = m.update(&sample(500.0, 0.0, 0.0, gb(1), gb(8)));
        assert_eq!(
            f.risk,
            SwapRisk::Building,
            "moderate spike (50×) should be Building, not Critical — got {:?}",
            f.risk
        );
    }

    #[test]
    fn sustained_dirty_with_swapouts_eventually_escalates() {
        let mut m = SwapReclaimModel::new();
        let used = (gb(8) as f64 * 0.80) as u64;
        // 20 cycles of high dirty_rate + active swapouts near threshold
        let mut last = SaturationForecast {
            dirty_rate_bps: 0.0,
            reclaim_rate_bps: 0.0,
            net_rate_bps: 0.0,
            swapouts_ema_pps: 0.0,
            t_sat_sec: None,
            risk: SwapRisk::Safe,
            overflow_entered: false,
            swap_ratio: 0.0,
            net_rate_volatility: 0.0,
        };
        for _ in 0..20 {
            last = m.update(&sample_io(10_000.0, 0.0, 0.0, 5.0, used, gb(8)));
        }
        // After sustained pressure + swapouts, EMA converges → Critical
        assert_eq!(last.risk, SwapRisk::Critical);
    }

    #[test]
    fn reset_clears_ema_state() {
        let mut m = SwapReclaimModel::new();
        for _ in 0..10 {
            m.update(&sample(5_000.0, 0.0, 0.0, gb(2), gb(8)));
        }
        m.reset();
        assert_eq!(m.samples(), 0);
        // After reset + 1 quiet sample, should be Safe
        let f = m.update(&sample(0.0, 100.0, 0.0, gb(1), gb(8)));
        assert_eq!(f.risk, SwapRisk::Safe);
    }

    #[test]
    fn overflow_entered_true_only_on_first_overflow_cycle() {
        let mut m = SwapReclaimModel::new();
        let over_used = (gb(8) as f64 * SWAP_CRITICAL_RATIO + 1.0) as u64;
        // Cycle 1: enters Overflow → overflow_entered = true
        let f1 = m.update(&sample(0.0, 0.0, 0.0, over_used, gb(8)));
        assert_eq!(f1.risk, SwapRisk::Overflow);
        assert!(
            f1.overflow_entered,
            "first overflow cycle must set overflow_entered"
        );
        // Cycle 2: still Overflow → overflow_entered = false (already in overflow)
        let f2 = m.update(&sample(0.0, 0.0, 0.0, over_used, gb(8)));
        assert_eq!(f2.risk, SwapRisk::Overflow);
        assert!(
            !f2.overflow_entered,
            "subsequent overflow cycles must NOT set overflow_entered"
        );
        // Drop back to Safe, then re-enter Overflow → overflow_entered = true again
        let safe_used = gb(1);
        let _fs = m.update(&sample(0.0, 0.0, 0.0, safe_used, gb(8)));
        let f3 = m.update(&sample(0.0, 0.0, 0.0, over_used, gb(8)));
        assert_eq!(f3.risk, SwapRisk::Overflow);
        assert!(
            f3.overflow_entered,
            "re-entry into overflow must set overflow_entered again"
        );
    }

    #[test]
    fn swap_ratio_reported_correctly() {
        let mut m = SwapReclaimModel::new();
        let f = m.update(&sample(0.0, 0.0, 0.0, gb(2), gb(8)));
        assert!((f.swap_ratio - 0.25).abs() < 0.01);
    }

    // ── CyberPhysicalSignal trait tests ─────────────────────────────────────

    #[test]
    fn tsat_urgency_none_is_zero() {
        use super::CyberPhysicalSignal;
        assert_eq!(super::TsatUrgency(None).normalized(), 0.0);
    }

    #[test]
    fn tsat_urgency_at_critical_eta_is_one() {
        use super::CyberPhysicalSignal;
        let u = super::TsatUrgency(Some(CRITICAL_ETA_SEC)).normalized();
        assert!(
            (u - 1.0).abs() < 1e-9,
            "t_sat == CRITICAL_ETA → urgency = 1.0, got {u}"
        );
    }

    #[test]
    fn tsat_urgency_far_future_is_small() {
        use super::CyberPhysicalSignal;
        let u = super::TsatUrgency(Some(3600.0)).normalized();
        assert!(u < 0.05, "t_sat=3600s → urgency should be tiny, got {u}");
    }

    #[test]
    fn net_rate_norm_ceiling_is_one() {
        use super::CyberPhysicalSignal;
        let n = super::NetRateNorm(NET_RATE_CEILING_BPS).normalized();
        assert!((n - 1.0).abs() < 1e-9, "ceiling rate → norm = 1.0, got {n}");
    }

    #[test]
    fn net_rate_norm_half_ceiling_is_half() {
        use super::CyberPhysicalSignal;
        let n = super::NetRateNorm(NET_RATE_CEILING_BPS * 0.5).normalized();
        assert!((n - 0.5).abs() < 1e-9, "half ceiling → 0.5, got {n}");
    }

    #[test]
    fn net_rate_norm_beyond_ceiling_clamped() {
        use super::CyberPhysicalSignal;
        let n = super::NetRateNorm(NET_RATE_CEILING_BPS * 10.0).normalized();
        assert!(
            (n - 1.0).abs() < 1e-9,
            "beyond ceiling → clamped 1.0, got {n}"
        );
    }

    // ── Volatility (σ) tests ────────────────────────────────────────────────

    #[test]
    fn volatility_zero_on_first_sample() {
        // [Welford 1962] warm-up: σ undefined until 2 samples; return 0.
        let mut m = SwapReclaimModel::new();
        let f = m.update(&sample(100.0, 50.0, 0.0, gb(1), gb(8)));
        assert_eq!(f.net_rate_volatility, 0.0, "first sample must have σ=0");
    }

    #[test]
    fn volatility_nonzero_after_rate_change() {
        // After a change in net_rate, σ must become positive.
        let mut m = SwapReclaimModel::new();
        m.update(&sample(100.0, 50.0, 0.0, gb(1), gb(8))); // seed
        let f = m.update(&sample(200.0, 50.0, 0.0, gb(1), gb(8))); // step up
        assert!(
            f.net_rate_volatility > 0.0,
            "σ must be positive after net_rate change"
        );
    }

    #[test]
    fn volatility_higher_for_volatile_signal() {
        // Alternating high/low dirty_rate → high σ vs steady → low σ.
        let mut m_stable = SwapReclaimModel::new();
        let mut m_volatile = SwapReclaimModel::new();
        for _ in 0..20 {
            m_stable.update(&sample(200.0, 100.0, 0.0, gb(1), gb(8)));
        }
        for i in 0..20 {
            let comp = if i % 2 == 0 { 10.0 } else { 5_000.0 };
            m_volatile.update(&sample(comp, 0.0, 0.0, gb(1), gb(8)));
        }
        let f_stable = m_stable.update(&sample(200.0, 100.0, 0.0, gb(1), gb(8)));
        let f_volatile = m_volatile.update(&sample(2_500.0, 0.0, 0.0, gb(1), gb(8)));
        assert!(
            f_volatile.net_rate_volatility > f_stable.net_rate_volatility,
            "volatile signal (σ={:.0}) must exceed stable (σ={:.0})",
            f_volatile.net_rate_volatility,
            f_stable.net_rate_volatility
        );
    }

    #[test]
    fn volatility_sticky_swap_harbinger() {
        // Sticky-swap: low mean net_rate but high volatility (oscillates around 0).
        // This is the pattern that T_sat alone misses.
        let mut m = SwapReclaimModel::new();
        for i in 0..30 {
            // Oscillate between slight compression and slight decompression.
            let (comp, decomp) = if i % 2 == 0 {
                (55.0, 50.0) // net slightly positive
            } else {
                (50.0, 55.0) // net slightly negative
            };
            m.update(&sample(comp, decomp, 0.0, gb(3), gb(8)));
        }
        let f = m.update(&sample(55.0, 50.0, 0.0, gb(3), gb(8)));
        // net_rate is small (near 0) but volatility is non-trivial.
        assert!(
            f.net_rate_volatility > 0.0,
            "oscillating signal must have σ > 0"
        );
    }
}
