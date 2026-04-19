//! # Unfreeze Decay Model
//!
//! First-order linear ODE describing how a process re-accumulates RSS after
//! `SIGCONT`:
//!
//! ```text
//!     dM/dt = (M∞ − M) / τ
//! ```
//!
//! with the closed-form solution
//!
//! ```text
//!     M(t) = M₀ + (M∞ − M₀) · (1 − e^(−t/τ))
//! ```
//!
//! where
//!
//! - `M₀`  — RSS at the moment of thaw (frozen processes usually sit near the
//!   compressor baseline).
//! - `M∞`  — asymptotic working-set size (learned per app as the running max of
//!   observed samples post-thaw).
//! - `τ`   — per-app time constant (seconds).  Small τ = reloads quickly.
//!
//! ## Why an ODE
//!
//! The existing unfreeze heuristic ("thaw after 30 cycles") ignores two facts:
//!
//! 1. Different apps reload at wildly different rates.  A Chromium renderer
//!    hits ~500 MB in a couple of seconds; a quiescent menubar app barely
//!    grows at all.
//! 2. Thaw timing affects pressure seconds later.  Without a model we thaw
//!    first and *then* discover we're back at 85 % pressure.
//!
//! Learning τ online lets `decide_actions` refuse to thaw candidates whose
//! predicted post-thaw RSS would overflow the current headroom.
//!
//! ## Design choices (all risk-mitigated)
//!
//! - **Fallback τ**: until `MIN_SAMPLES_FOR_LEARNING` observations accumulate,
//!   the model returns `DEFAULT_TAU_SEC` so callers are never stuck with a
//!   wildly wrong estimate.
//! - **Bounded τ**: learned τ is clamped to `[MIN_TAU_SEC, MAX_TAU_SEC]` so a
//!   noisy outlier cannot produce predictions with effectively infinite time
//!   constants.
//! - **Active-thaw GC**: entries are dropped once `5·τ` seconds have passed
//!   (decay is effectively complete) or `STALE_HARD_LIMIT_SEC` as a hard cap.
//! - **Map cap**: per-app τ estimates are LRU-capped at `MAX_APPS` entries.
//!
//! ## Papers
//!
//! - [Strogatz 2015] "Nonlinear Dynamics and Chaos" §2.3 — linear ODE with
//!   exponential relaxation.
//! - [Denning 1968] "The Working Set Model for Program Behavior" — processes
//!   have stable resident-set sizes to which they return.
//! - [Bansal & Modha 2004] "CAR: Clock with Adaptive Replacement" — adaptive
//!   time constants in caching.

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// Fallback τ when insufficient evidence exists to learn one.
pub const DEFAULT_TAU_SEC: f64 = 30.0;

/// Lower clamp on learned τ.  Prevents pathological "reloads instantly"
/// predictions from a single noisy sample.
pub const MIN_TAU_SEC: f64 = 5.0;

/// Upper clamp on learned τ.  A working set that takes longer than 5 minutes
/// to reload is behaving like a dormant process, not a live one.
pub const MAX_TAU_SEC: f64 = 300.0;

/// Minimum post-thaw samples required before the learned τ is used instead
/// of `DEFAULT_TAU_SEC`.
pub const MIN_SAMPLES_FOR_LEARNING: usize = 3;

/// Drop an active-thaw record once this many τ have elapsed — decay is
/// effectively complete after ~5τ (> 99 % of M∞).
pub const GC_TAU_MULTIPLIER: f64 = 5.0;

/// Hard floor GC: regardless of τ, drop active-thaw records older than this.
pub const STALE_HARD_LIMIT_SEC: f64 = 900.0; // 15 min

/// LRU cap on `tau_estimates` to bound persisted state.
pub const MAX_APPS: usize = 50;

/// Running statistics for one app's learned τ.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TauEstimate {
    /// EMA of τ (seconds).
    pub tau_sec: f64,
    /// Back-computed asymptotic RSS (bytes): m_0 + (M_last − m_0)/(1 − e^{−t_last/τ}).
    /// Stored so `predict_rss` has an M∞ prior even on a cold call.
    pub m_infinity: u64,
    /// Count of samples that contributed to the τ estimate (pairs fit).
    pub samples: u32,
    /// Seconds since `UNIX_EPOCH` of the last update — used for LRU eviction.
    pub last_updated_epoch_sec: u64,
}

impl TauEstimate {
    fn fresh(m_infinity: u64, now_epoch_sec: u64) -> Self {
        Self {
            tau_sec: DEFAULT_TAU_SEC,
            m_infinity,
            samples: 0,
            last_updated_epoch_sec: now_epoch_sec,
        }
    }
}

/// One active (still-relaxing) thaw event.  Stores the previous sample so τ
/// can be fit from a 2-sample ratio (eliminates M∞ from the fit — see
/// [Strogatz 2015 §2.3]: dividing two exponentials removes the asymptote).
#[derive(Debug, Clone)]
pub struct ThawEvent {
    pub app: String,
    pub m_0: u64,
    pub thaw_at: Instant,
    /// Most-recent `(observed_at, rss)` — `None` before the first sample.
    pub last_sample: Option<(Instant, u64)>,
}

/// The decay model itself — owns per-app τ estimates and active thaws.
#[derive(Debug, Default)]
pub struct UnfreezeDecayModel {
    tau_estimates: HashMap<String, TauEstimate>,
    active_thaws: HashMap<u32, ThawEvent>,
}

impl UnfreezeDecayModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a fresh thaw.  Call this from the unfreeze execution path
    /// immediately after `SIGCONT` returns success.
    pub fn record_thaw(&mut self, pid: u32, app: String, m_0: u64, now: Instant) {
        self.active_thaws.insert(
            pid,
            ThawEvent {
                app,
                m_0,
                thaw_at: now,
                last_sample: None,
            },
        );
    }

    /// Ingest a fresh RSS sample for a `pid` known to be relaxing.
    ///
    /// `wss_hint` — optional working-set size from `TASK_VM_INFO` (measured by
    /// `MemoryAnalyzer`).  When available, it anchors M∞ to a kernel-measured
    /// value instead of relying on the back-computed running maximum.  For a
    /// process that was frozen mid-session, pre-freeze WSS is the ground truth
    /// for its steady-state demand [Denning 1968].
    ///
    /// No-op if the pid isn't tracked (silently ignored — caller can feed
    /// every live process without bothering to filter).
    pub fn observe_sample(&mut self, pid: u32, current_rss: u64, now: Instant, now_epoch_sec: u64) {
        self.observe_sample_with_wss(pid, current_rss, now, now_epoch_sec, None);
    }

    /// Same as `observe_sample` but with an optional WSS hint from the kernel.
    pub fn observe_sample_with_wss(
        &mut self,
        pid: u32,
        current_rss: u64,
        now: Instant,
        now_epoch_sec: u64,
        wss_hint: Option<u64>,
    ) {
        // Clone just the fields we need (short read) — we need a mutable borrow
        // below to write back `last_sample`.
        let (app, m_0, thaw_at, prev_sample) = match self.active_thaws.get(&pid) {
            Some(e) => (e.app.clone(), e.m_0, e.thaw_at, e.last_sample),
            None => return,
        };
        let dt = now.saturating_duration_since(thaw_at).as_secs_f64();
        if dt < 1.0 {
            return; // too soon — signal dominated by noise.
        }

        // Always remember the freshest sample for the next call.
        if let Some(event) = self.active_thaws.get_mut(&pid) {
            event.last_sample = Some((now, current_rss));
        }

        // Need a prior sample to do the 2-sample ratio fit.
        let (prev_t, prev_rss) = match prev_sample {
            Some(s) => s,
            None => return,
        };
        // Require strict monotonic growth above baseline — anything else
        // violates the exponential approach model.
        if prev_rss <= m_0 || current_rss <= prev_rss {
            return;
        }
        let t1 = prev_t.saturating_duration_since(thaw_at).as_secs_f64();
        let t2 = dt;
        if t2 - t1 < 0.5 {
            return; // samples too close in time — ill-conditioned.
        }

        let r = (prev_rss - m_0) as f64 / (current_rss - m_0) as f64;
        // Model implies r ∈ (t1/t2, 1).  Outside that window → noise/outlier.
        if !(t1 / t2 < r && r < 1.0) {
            return;
        }
        let tau_sample = match solve_tau_ratio(t1, t2, r) {
            Some(t) => t.clamp(MIN_TAU_SEC, MAX_TAU_SEC),
            None => return,
        };

        // Back-compute M∞ self-consistently from the latest sample and τ:
        //   M(t) = m_0 + (M∞ − m_0)(1 − e^{−t/τ})  ⇒  M∞ = m_0 + (M(t) − m_0)/(1 − e^{−t/τ})
        let denom = 1.0 - (-t2 / tau_sample).exp();
        let m_inf_est = if denom > 0.01 {
            m_0.saturating_add(((current_rss - m_0) as f64 / denom).round() as u64)
        } else {
            current_rss
        };

        let entry = self
            .tau_estimates
            .entry(app)
            .or_insert_with(|| TauEstimate::fresh(m_inf_est, now_epoch_sec));
        entry.samples = entry.samples.saturating_add(1);
        // EMA once we're past the warm-up, straight average while warming.
        entry.tau_sec = if entry.samples >= MIN_SAMPLES_FOR_LEARNING as u32 {
            0.8 * entry.tau_sec + 0.2 * tau_sample
        } else {
            (entry.tau_sec * (entry.samples as f64 - 1.0) + tau_sample) / entry.samples as f64
        };
        // Track M∞ as a running max of self-consistent estimates — still a
        // running max, but of a well-conditioned quantity rather than raw RSS.
        // If the caller provides a WSS from TASK_VM_INFO, prefer it as the
        // ground-truth lower bound [Denning 1968 — WSS is the only reliable
        // predictor of steady-state RAM demand].
        let wss_lower = wss_hint.unwrap_or(0);
        entry.m_infinity = entry.m_infinity.max(m_inf_est).max(current_rss).max(wss_lower);
        entry.last_updated_epoch_sec = now_epoch_sec;

        if dt > GC_TAU_MULTIPLIER * entry.tau_sec || dt > STALE_HARD_LIMIT_SEC {
            self.active_thaws.remove(&pid);
        }
    }

    /// Per-cycle maintenance.  Drops active thaws older than the hard limit
    /// and LRU-caps the tau map.
    pub fn gc(&mut self, now: Instant, now_epoch_sec: u64) {
        self.active_thaws
            .retain(|_pid, ev| now.duration_since(ev.thaw_at).as_secs_f64() < STALE_HARD_LIMIT_SEC);

        if self.tau_estimates.len() > MAX_APPS {
            // Drop the oldest (smallest `last_updated_epoch_sec`) until we're at cap.
            let mut entries: Vec<(String, u64)> = self
                .tau_estimates
                .iter()
                .map(|(k, v)| (k.clone(), v.last_updated_epoch_sec))
                .collect();
            entries.sort_by_key(|(_, ts)| *ts);
            let drop_count = self.tau_estimates.len() - MAX_APPS;
            for (k, _) in entries.into_iter().take(drop_count) {
                self.tau_estimates.remove(&k);
            }
            let _ = now_epoch_sec;
        }
    }

    /// Predicted RSS (bytes) `t_sec` after thaw, given the baseline `m_0`.
    ///
    /// Uses the app's learned τ if available and we have enough samples;
    /// otherwise falls back to `DEFAULT_TAU_SEC` and `m_0` as its own M∞
    /// (i.e., assumes no growth — conservative when we truly have no prior).
    pub fn predict_rss(&self, app: &str, m_0: u64, t_sec: f64) -> u64 {
        let (tau, m_inf) = match self.tau_estimates.get(app) {
            Some(est) if est.samples >= MIN_SAMPLES_FOR_LEARNING as u32 => {
                (est.tau_sec, est.m_infinity.max(m_0))
            }
            _ => (DEFAULT_TAU_SEC, m_0),
        };
        if t_sec <= 0.0 {
            return m_0;
        }
        let delta = m_inf.saturating_sub(m_0) as f64;
        let growth = delta * (1.0 - (-t_sec / tau).exp());
        m_0.saturating_add(growth.round() as u64)
    }

    /// Seconds until the predicted RSS reaches `threshold`.  `None` if the
    /// app's `M∞` is below the threshold (it will never reach it).
    pub fn time_to_threshold(&self, app: &str, m_0: u64, threshold: u64) -> Option<f64> {
        if threshold <= m_0 {
            return Some(0.0);
        }
        let (tau, m_inf) = match self.tau_estimates.get(app) {
            Some(est) if est.samples >= MIN_SAMPLES_FOR_LEARNING as u32 => {
                (est.tau_sec, est.m_infinity)
            }
            _ => (DEFAULT_TAU_SEC, m_0),
        };
        if m_inf <= threshold {
            return None; // asymptote is under the threshold.
        }
        // M(t) = M₀ + (M∞−M₀)(1−e^(−t/τ)) = threshold
        //   ⇒ t = −τ·ln(1 − (threshold−M₀)/(M∞−M₀))
        let ratio = (threshold - m_0) as f64 / (m_inf - m_0) as f64;
        let ratio = ratio.clamp(0.0, 0.999);
        Some(-tau * (1.0 - ratio).ln())
    }

    /// Learned τ for `app`, or DEFAULT_TAU_SEC if unknown/insufficient samples.
    /// [Denning 1968] high τ = slow working-set re-growth → freeze is expensive.
    pub fn tau_for_app(&self, app: &str) -> f64 {
        self.tau_estimates
            .get(app)
            .filter(|e| e.samples >= MIN_SAMPLES_FOR_LEARNING as u32)
            .map(|e| e.tau_sec)
            .unwrap_or(DEFAULT_TAU_SEC)
    }

    /// Snapshot of the learned τ map for persistence.
    pub fn tau_snapshot(&self) -> HashMap<String, TauEstimate> {
        self.tau_estimates.clone()
    }

    /// Restore τ estimates from persistence.  Active thaws are never
    /// persisted — on restart we do not know which processes are still
    /// relaxing.
    pub fn restore(&mut self, snapshot: HashMap<String, TauEstimate>) {
        self.tau_estimates = snapshot;
    }

    pub fn active_thaw_count(&self) -> usize {
        self.active_thaws.len()
    }

    pub fn learned_app_count(&self) -> usize {
        self.tau_estimates.len()
    }

    /// Snapshot of the pids currently being tracked — the caller can then
    /// query its process collector for a fresh RSS reading and feed it back
    /// via `observe_sample`.
    pub fn active_thaw_pids(&self) -> Vec<u32> {
        self.active_thaws.keys().copied().collect()
    }
}

/// Solve `(1 − e^{−t1/τ}) / (1 − e^{−t2/τ}) = r` for τ by bisection.
///
/// The function is strictly monotonic in τ on `[MIN_TAU_SEC, MAX_TAU_SEC]`:
///   lim_{τ→0⁺} g(τ) = 1,   lim_{τ→∞} g(τ) = t1/t2.
/// So a zero of `g(τ) − r` exists iff `t1/t2 < r < 1`.  Caller guarantees this.
fn solve_tau_ratio(t1: f64, t2: f64, r: f64) -> Option<f64> {
    let f = |tau: f64| -> f64 {
        let a = 1.0 - (-t1 / tau).exp();
        let b = 1.0 - (-t2 / tau).exp();
        if b.abs() < 1e-12 {
            return 1.0 - r;
        }
        a / b - r
    };
    let mut lo = MIN_TAU_SEC;
    let mut hi = MAX_TAU_SEC;
    let flo = f(lo);
    let fhi = f(hi);
    // If the sign doesn't change across the bracket, the answer lies at the
    // nearest clamp boundary.  Return the clamp so the caller's subsequent
    // clamp() is a no-op.
    if flo.signum() == fhi.signum() {
        return Some(if flo.abs() < fhi.abs() { lo } else { hi });
    }
    // 40 iterations of bisection → 10⁻¹² precision on [5, 300].  Way overkill,
    // but cheap and deterministic.
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let fm = f(mid);
        if fm.signum() == flo.signum() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn now_epoch() -> u64 {
        1_700_000_000
    }

    #[test]
    fn predict_with_no_samples_returns_m0() {
        // No learned τ, no M∞ — with no prior we assume no growth (conservative).
        let model = UnfreezeDecayModel::new();
        let rss = model.predict_rss("chrome", 100_000_000, 60.0);
        assert_eq!(rss, 100_000_000);
    }

    #[test]
    fn predict_matches_ode_closed_form() {
        let mut model = UnfreezeDecayModel::new();
        // Synthesise: τ=10s, M₀=100MB, M∞=500MB, sample 3 points on the curve.
        let t0 = Instant::now();
        let m_0 = 100_000_000u64;
        let m_inf = 500_000_000u64;
        let tau_true = 10.0_f64;
        let curve = |t: f64| {
            (m_0 as f64 + (m_inf - m_0) as f64 * (1.0 - (-t / tau_true).exp())).round() as u64
        };

        model.record_thaw(42, "bench".into(), m_0, t0);
        // 4 samples → 3 pair-wise fits → passes the MIN_SAMPLES gate.
        for secs in [5.0_f64, 10.0, 20.0, 30.0] {
            let now = t0 + Duration::from_secs_f64(secs);
            model.observe_sample(42, curve(secs), now, now_epoch());
        }

        // With clean samples, learned τ should be close to truth
        // (± 25 % given EMA smoothing and log-noise near the asymptote).
        let est = model.tau_snapshot().get("bench").cloned().unwrap();
        assert!(
            (est.tau_sec - tau_true).abs() / tau_true < 0.25,
            "tau={} truth={}",
            est.tau_sec,
            tau_true
        );

        // Prediction at t=15s should be within 10 % of the closed form.
        let predicted = model.predict_rss("bench", m_0, 15.0);
        let truth = curve(15.0);
        let err = (predicted as f64 - truth as f64).abs() / truth as f64;
        assert!(err < 0.10, "predicted={} truth={} err={}", predicted, truth, err);
    }

    #[test]
    fn predict_clamps_tau_within_bounds() {
        let mut model = UnfreezeDecayModel::new();
        let t0 = Instant::now();
        let m_0 = 10u64;
        model.record_thaw(1, "edge".into(), m_0, t0);
        // Two samples both at ~99.9 % of some asymptote → implied τ ≈ 0.
        // Must be clamped to MIN_TAU_SEC.
        model.observe_sample(1, 9_990, t0 + Duration::from_secs_f64(2.0), now_epoch());
        model.observe_sample(1, 9_999, t0 + Duration::from_secs_f64(3.0), now_epoch());
        let est = model.tau_snapshot().get("edge").cloned().unwrap();
        assert!(
            est.tau_sec >= MIN_TAU_SEC && est.tau_sec <= MAX_TAU_SEC,
            "tau={} outside bounds",
            est.tau_sec
        );
    }

    #[test]
    fn time_to_threshold_none_when_asymptote_below() {
        let mut model = UnfreezeDecayModel::new();
        let t0 = Instant::now();
        let m_0 = 100u64;
        let m_inf = 500u64;
        let tau_true = 10.0_f64;
        let curve = |t: f64| {
            (m_0 as f64 + (m_inf - m_0) as f64 * (1.0 - (-t / tau_true).exp())).round() as u64
        };
        model.record_thaw(9, "capped".into(), m_0, t0);
        for secs in [5.0, 10.0, 20.0] {
            let now = t0 + Duration::from_secs_f64(secs);
            model.observe_sample(9, curve(secs), now, now_epoch());
        }
        // Asymptote ≈ 500 — threshold 1000 is unreachable.
        assert_eq!(model.time_to_threshold("capped", m_0, 1000), None);
    }

    #[test]
    fn time_to_threshold_zero_when_already_exceeded() {
        let model = UnfreezeDecayModel::new();
        assert_eq!(model.time_to_threshold("x", 500, 400), Some(0.0));
    }

    #[test]
    fn gc_drops_stale_active_thaws() {
        let mut model = UnfreezeDecayModel::new();
        let t0 = Instant::now();
        model.record_thaw(1, "a".into(), 100, t0);
        assert_eq!(model.active_thaw_count(), 1);
        let far_future = t0 + Duration::from_secs_f64(STALE_HARD_LIMIT_SEC + 10.0);
        model.gc(far_future, now_epoch());
        assert_eq!(model.active_thaw_count(), 0);
    }

    #[test]
    fn lru_caps_tau_map_size() {
        let mut model = UnfreezeDecayModel::new();
        for i in 0..(MAX_APPS + 5) {
            let app = format!("app{}", i);
            model
                .tau_estimates
                .insert(app, TauEstimate::fresh(1_000, now_epoch() + i as u64));
        }
        assert!(model.tau_estimates.len() > MAX_APPS);
        model.gc(Instant::now(), now_epoch() + 1_000_000);
        assert_eq!(model.tau_estimates.len(), MAX_APPS);
    }

    #[test]
    fn observe_noop_for_unknown_pid() {
        let mut model = UnfreezeDecayModel::new();
        let now = Instant::now();
        model.observe_sample(9999, 100_000, now, now_epoch());
        assert_eq!(model.learned_app_count(), 0);
    }

    #[test]
    fn restore_round_trips() {
        let mut model = UnfreezeDecayModel::new();
        let mut snap = HashMap::new();
        snap.insert(
            "persisted".to_string(),
            TauEstimate {
                tau_sec: 42.0,
                m_infinity: 1_234_567,
                samples: 7,
                last_updated_epoch_sec: now_epoch(),
            },
        );
        model.restore(snap.clone());
        assert_eq!(model.tau_snapshot(), snap);
        // Learned τ should be used now (samples >= MIN_SAMPLES_FOR_LEARNING).
        let rss = model.predict_rss("persisted", 0, 42.0);
        assert!(rss > 0, "should predict growth with restored τ");
    }
}
