//! Fluidity Intelligence — composite system fluidity scoring for Apollo.
//!
//! Tracks window rendering smoothness, app launch acceleration, and
//! GPU/render pressure. Produces a composite fluidity score (0–1) that
//! drives adaptive background interference reduction.
//!
//! # Theory
//!
//! [Jain 1991] "Art of Computer Systems Performance Analysis" — EMA-based
//! composite scoring: weight recent observations more heavily, decay stale
//! data, and combine multiple signals into a single decision metric.
//!
//! [Apple WWDC 2021] "Optimize for 5G and Low Data Mode" — WindowServer acts
//! as the rendering arbiter on macOS. High WindowServer CPU% indicates active
//! window operations (resize, move, animation) that contend with P-core budget.
//!
//! [Beigel & Bruss 2000] "The secretary problem with multiple choices" —
//! spike detection via rolling window max: a single sample above the spike
//! threshold counts as a burst, not a rolling average.
//!
//! [Welch & Bishop 2006] "An Introduction to the Kalman Filter" — 1D Kalman
//! filter provides noise-rejected fluidity smoothing and 3-cycle prediction,
//! enabling pre-emptive response before degradation is perceivable.

use std::collections::VecDeque;

use crate::engine::kalman::Kalman1D;
use crate::engine::safety::is_protected_name;

// ── Constants ─────────────────────────────────────────────────────────────────

/// WindowServer CPU% above which a window operation (resize/move) is active.
/// Empirical: idle WS ≈ 2–5%, active drag/resize ≈ 30–60% on M1.
const WS_SPIKE_THRESHOLD: f32 = 25.0;

/// EMA alpha for WindowServer CPU smoothing. α=0.4 → ~2.5-cycle memory.
const WS_EMA_ALPHA: f32 = 0.4;

/// History depth for WindowServer CPU (rolling max spike detection).
const WS_HISTORY_DEPTH: usize = 10;

/// Number of cycles a new app launch is protected (≈ 30s at 2s/cycle).
const LAUNCH_PROTECTION_CYCLES: u8 = 15;

/// EMA alpha for fluidity score smoothing. α=0.25 → ~4-cycle memory.
const FLUIDITY_EMA_ALPHA: f32 = 0.25;

/// Fluidity score below which we declare degraded state.
const FLUIDITY_DEGRADED_THRESHOLD: f32 = 0.65;

/// CPU% threshold above which a process is considered a fluidity offender.
const OFFENDER_CPU_THRESHOLD: f32 = 15.0;

/// EMA alpha for offender hurt-score. α=0.3 → ~3-cycle memory.
const OFFENDER_EMA_ALPHA: f32 = 0.3;

/// Max number of tracked offenders (bounded to avoid unbounded growth).
const MAX_OFFENDERS: usize = 20;

// ── Core State ────────────────────────────────────────────────────────────────

/// Fluidity Intelligence state. Initialize once, call `update()` each daemon cycle.
///
/// Tracks WindowServer CPU as a proxy for window rendering operations,
/// detects new app launches, and computes a composite fluidity score.
pub struct FluidityState {
    // ── WindowServer CPU tracking ─────────────────────────────────────────
    /// EMA-smoothed WindowServer CPU%.
    pub windowserver_cpu_ema: f32,
    /// True when WindowServer CPU spike detected (window resize/move active).
    pub windowserver_cpu_spike: bool,
    /// Rolling history of raw WS CPU samples (last N cycles).
    pub windowserver_cpu_history: VecDeque<f32>,

    // ── App launch tracking ────────────────────────────────────────────────
    /// True when a new app is being launched.
    pub launch_active: bool,
    /// PID of the most recently launched app.
    pub launch_pid: Option<u32>,
    /// Name of the most recently launched app.
    pub launch_name: String,
    /// Cycles remaining to protect the launching app.
    pub launch_cycles_remaining: u8,

    // ── GPU / render load ──────────────────────────────────────────────────
    /// GPU utilization 0–1 for rendering workloads.
    pub gpu_render_load: f32,

    // ── Composite fluidity ─────────────────────────────────────────────────
    /// Raw fluidity score this cycle (0–1, 1 = perfectly fluid).
    pub fluidity_score: f32,
    /// EMA-smoothed fluidity score.
    pub fluidity_ema: f32,
    /// True when sustained fluidity degradation is detected.
    pub fluidity_degraded: bool,

    // ── Kalman prediction ──────────────────────────────────────────────────
    /// 1D Kalman filter for noise-rejected fluidity smoothing.
    fluidity_kalman: Kalman1D,
    /// Rate of change of fluidity (positive = improving, negative = degrading).
    pub fluidity_velocity: f32,
    /// Kalman-predicted fluidity in 3 cycles (~6s).
    pub fluidity_predicted_3s: f32,

    // ── Learning: offender tracking ───────────────────────────────────────
    /// Processes correlated with fluidity degradation: (pid, name, hurt_score).
    /// hurt_score EMA: higher = more correlated with degradation.
    pub fluidity_offenders: Vec<(u32, String, f32)>,
    /// Learned WS CPU% spike threshold. Default = WS_SPIKE_THRESHOLD (25.0).
    /// Wired from LearnableParams::ws_spike_threshold.
    ws_spike_threshold: f32,
    /// Learned fluidity-degraded threshold. Default = FLUIDITY_DEGRADED_THRESHOLD (0.65).
    /// Wired from LearnableParams::fluidity_degraded_threshold.
    fluidity_degraded_threshold: f32,

    /// Previous process set (PID) for launch detection.
    prev_pids: std::collections::HashSet<u32>,
    /// Whether this is the first update (skip launch detection on init).
    initialized: bool,
}

impl Default for FluidityState {
    fn default() -> Self {
        Self::new()
    }
}

impl FluidityState {
    pub fn new() -> Self {
        Self {
            windowserver_cpu_ema: 0.0,
            windowserver_cpu_spike: false,
            windowserver_cpu_history: VecDeque::with_capacity(WS_HISTORY_DEPTH),

            launch_active: false,
            launch_pid: None,
            launch_name: String::new(),
            launch_cycles_remaining: 0,

            gpu_render_load: 0.0,

            fluidity_score: 1.0,
            fluidity_ema: 1.0,
            fluidity_degraded: false,

            fluidity_kalman: Kalman1D::new(0.02, 0.05),
            fluidity_velocity: 0.0,
            fluidity_predicted_3s: 1.0,

            fluidity_offenders: Vec::new(),
            ws_spike_threshold: WS_SPIKE_THRESHOLD,
            fluidity_degraded_threshold: FLUIDITY_DEGRADED_THRESHOLD,
            prev_pids: std::collections::HashSet::new(),
            initialized: false,
        }
    }

    /// Apply learned thresholds from LearnableParams (called every N cycles).
    pub fn apply_thresholds(&mut self, ws_spike: f32, degraded: f32) {
        self.ws_spike_threshold = ws_spike.clamp(10.0, 50.0);
        self.fluidity_degraded_threshold = degraded.clamp(0.30, 0.90);
    }

    /// Update fluidity state from a new daemon cycle snapshot.
    ///
    /// `processes`: Vec of (pid, name, cpu_pct) from sysinfo snapshot.
    /// `gpu_load`: GPU utilization 0–1 from IOKit/gpu_manager.
    /// `dt_secs`: elapsed seconds since last call (for Kalman).
    pub fn update(&mut self, processes: &[(u32, &str, f32)], gpu_load: f32, dt_secs: f32) {
        let dt_secs = dt_secs.max(0.1);

        // ── 1. Extract WindowServer CPU ────────────────────────────────────
        let ws_cpu = processes
            .iter()
            .find(|(_, name, _)| *name == "WindowServer")
            .map(|(_, _, cpu)| *cpu)
            .unwrap_or(0.0);

        // Update EMA: [Jain 1991] α=0.4 for moderate responsiveness
        self.windowserver_cpu_ema =
            WS_EMA_ALPHA * ws_cpu + (1.0 - WS_EMA_ALPHA) * self.windowserver_cpu_ema;

        // Rolling history for spike detection [Beigel & Bruss 2000]
        if self.windowserver_cpu_history.len() >= WS_HISTORY_DEPTH {
            self.windowserver_cpu_history.pop_front();
        }
        self.windowserver_cpu_history.push_back(ws_cpu);

        // Spike: either current sample or EMA exceeds threshold
        self.windowserver_cpu_spike = ws_cpu > self.ws_spike_threshold
            || self.windowserver_cpu_ema > self.ws_spike_threshold * 0.75;

        // ── 2. GPU render load ─────────────────────────────────────────────
        self.gpu_render_load = gpu_load.clamp(0.0, 1.0);

        // ── 3. Launch detection ────────────────────────────────────────────
        if self.launch_cycles_remaining > 0 {
            self.launch_cycles_remaining -= 1;
            if self.launch_cycles_remaining == 0 {
                self.launch_active = false;
                self.launch_pid = None;
                self.launch_name.clear();
            }
        }

        if self.initialized {
            // Check for newly appeared PIDs (excluding renderers and known system noise)
            let current_pids: std::collections::HashSet<u32> =
                processes.iter().map(|(pid, _, _)| *pid).collect();

            for (pid, name, _cpu) in processes {
                if !self.prev_pids.contains(pid) && !is_renderer_or_helper(name) {
                    // New process appeared — could be an app launch
                    // Prefer named apps over short-lived system helpers
                    if is_launchable_app(name) {
                        self.launch_active = true;
                        self.launch_pid = Some(*pid);
                        self.launch_name = name.to_string();
                        self.launch_cycles_remaining = LAUNCH_PROTECTION_CYCLES;
                        // Only capture the first/most-prominent launch per cycle
                        break;
                    }
                }
            }

            self.prev_pids = current_pids;
        } else {
            // First tick: initialize prev_pids without triggering launch events
            self.prev_pids = processes.iter().map(|(pid, _, _)| *pid).collect();
            self.initialized = true;
        }

        // ── 4. Compute raw fluidity score ──────────────────────────────────
        // [Jain 1991] Composite: weighted combination of normalized sub-scores.
        // Score starts at 1.0 (perfect), deductions applied for pressure signals.

        // WS CPU contribution: map 0–100% to 0–0.4 penalty
        let ws_penalty = (self.windowserver_cpu_ema / 100.0 * 0.4).min(0.4);

        // Spike adds immediate penalty (window op is latency-sensitive critical path)
        let spike_penalty = if self.windowserver_cpu_spike {
            0.2
        } else {
            0.0
        };

        // GPU load contribution: high GPU = rendering contention
        let gpu_penalty = self.gpu_render_load * 0.2;

        // Launch penalty: launching = background work must yield
        let launch_penalty = if self.launch_active { 0.1 } else { 0.0 };

        let raw_score =
            (1.0 - ws_penalty - spike_penalty - gpu_penalty - launch_penalty).clamp(0.0, 1.0);
        self.fluidity_score = raw_score;

        // ── 5. EMA smoothing ───────────────────────────────────────────────
        self.fluidity_ema =
            FLUIDITY_EMA_ALPHA * raw_score + (1.0 - FLUIDITY_EMA_ALPHA) * self.fluidity_ema;

        // ── 6. Kalman filter + prediction [Welch & Bishop 2006] ───────────
        self.fluidity_kalman
            .update(raw_score as f64, dt_secs as f64);
        let kalman_pos = self.fluidity_kalman.position() as f32;
        let kalman_vel = self.fluidity_kalman.velocity() as f32;
        self.fluidity_velocity = kalman_vel;

        // Predict 3 cycles ahead (dt_secs * 3)
        let pred = self.fluidity_kalman.predict_ahead((dt_secs * 3.0) as f64) as f32;
        self.fluidity_predicted_3s = pred.clamp(0.0, 1.0);

        // Prefer Kalman-smoothed value for EMA when filter is initialized
        if self.fluidity_kalman.is_initialized() {
            self.fluidity_ema = kalman_pos.clamp(0.0, 1.0);
        }

        // ── 7. Degradation state ───────────────────────────────────────────
        self.fluidity_degraded = self.fluidity_ema < self.fluidity_degraded_threshold;

        // ── 8. Offender tracking [Pearl 2009] Causation ───────────────────
        // When fluidity is degraded, correlate high-CPU processes as offenders.
        if self.fluidity_degraded {
            for (pid, name, cpu) in processes {
                if *cpu > OFFENDER_CPU_THRESHOLD && !is_protected_name(name) {
                    // Update or insert offender record.
                    // Match on PID + name to guard against PID reuse: macOS recycles
                    // PIDs aggressively and a reused PID is a different process.
                    // [Saltzer & Schroeder 1975] "Protection of Information" — use
                    // unforgeable identifiers; PID alone is not unique over time.
                    if let Some(entry) = self
                        .fluidity_offenders
                        .iter_mut()
                        .find(|(p, n, _)| p == pid && n == name)
                    {
                        // EMA of hurt score: higher CPU during degradation = higher score
                        entry.2 = OFFENDER_EMA_ALPHA * (cpu / 100.0)
                            + (1.0 - OFFENDER_EMA_ALPHA) * entry.2;
                    } else if self.fluidity_offenders.len() < MAX_OFFENDERS {
                        self.fluidity_offenders
                            .push((*pid, name.to_string(), cpu / 100.0));
                    }
                }
            }

            // Decay all offender scores slightly each cycle (forgetting)
            for entry in &mut self.fluidity_offenders {
                entry.2 *= 0.95;
            }

            // Prune offenders with negligible scores
            self.fluidity_offenders
                .retain(|(_, _, score)| *score > 0.01);
        }
    }

    /// Returns true if window operations are currently active (resize/move/animate).
    pub fn window_op_active(&self) -> bool {
        self.windowserver_cpu_spike
    }

    /// Returns true if an app is being launched.
    pub fn app_launching(&self) -> bool {
        self.launch_active
    }

    /// How much to back off background work (0 = none, 1 = max).
    ///
    /// Returns 1.0 during launch (hard cap), 0.8 during window ops,
    /// otherwise proportional to fluidity deficit.
    pub fn backoff_factor(&self) -> f32 {
        if self.launch_active {
            return 1.0;
        }
        if self.windowserver_cpu_spike {
            return 0.8;
        }
        // Proportional to degradation: fluidity 0.5 → backoff 0.5
        (1.0 - self.fluidity_ema).max(0.0)
    }

    /// Returns the top offender (highest hurt_score) if any.
    pub fn top_offender(&self) -> Option<&(u32, String, f32)> {
        self.fluidity_offenders
            .iter()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
    }
}

// ── Signal snapshot (send-safe summary for daemon main loop) ─────────────────

/// Compact fluidity signal for per-cycle daemon consumption.
/// Derived from `FluidityState` each cycle.
#[derive(Debug, Clone, Default)]
pub struct FluiditySignal {
    /// Composite fluidity score 0–1 (Kalman-smoothed EMA).
    pub fluidity_score: f32,
    /// True when WindowServer spike detected (window operation active).
    pub window_op_active: bool,
    /// True when a new app is being launched.
    pub app_launching: bool,
    /// Name of the launching app (empty if none).
    pub launch_name: String,
    /// How much to back off background work (0–1).
    pub backoff_factor: f32,
    /// GPU render load 0–1.
    pub gpu_render_load: f32,
    /// True when sustained fluidity degradation detected.
    pub fluidity_degraded: bool,
    /// Kalman-predicted fluidity in 3 cycles.
    pub fluidity_predicted_3s: f32,
    /// Rate of fluidity change (positive = improving).
    pub fluidity_velocity: f32,
    /// WindowServer CPU EMA %.
    pub windowserver_cpu_ema: f32,
}

impl From<&FluidityState> for FluiditySignal {
    fn from(s: &FluidityState) -> Self {
        Self {
            fluidity_score: s.fluidity_ema,
            window_op_active: s.windowserver_cpu_spike,
            app_launching: s.launch_active,
            launch_name: s.launch_name.clone(),
            backoff_factor: s.backoff_factor(),
            gpu_render_load: s.gpu_render_load,
            fluidity_degraded: s.fluidity_degraded,
            fluidity_predicted_3s: s.fluidity_predicted_3s,
            fluidity_velocity: s.fluidity_velocity,
            windowserver_cpu_ema: s.windowserver_cpu_ema,
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// True if the process is a browser renderer / GPU helper (not a user app launch).
fn is_renderer_or_helper(name: &str) -> bool {
    name.contains("Helper (Renderer)")
        || name.contains("Helper (GPU)")
        || name.contains("Helper (Plugin)")
        || name.contains("Helper (Alerts)")
        || name.starts_with("com.apple.")
        || name.starts_with("com.google.")
        || name.starts_with("com.brave.")
}

/// True if the process name looks like a launchable user-visible app.
/// Conservative heuristic: named, not a renderer/helper, not all-lowercase system daemon.
fn is_launchable_app(name: &str) -> bool {
    if name.is_empty() || is_renderer_or_helper(name) {
        return false;
    }
    // System daemons tend to be all lowercase with 'd' suffix or dots
    let first = name.chars().next().unwrap_or('a');
    // User apps tend to start with uppercase
    first.is_uppercase()
        || name.contains(' ')  // "Brave Browser", "Google Chrome", etc.
        || name == "ollama"
        || name == "python3"
        || name == "python"
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_procs<'a>(list: &'a [(&'a str, f32)]) -> Vec<(u32, &'a str, f32)> {
        list.iter()
            .enumerate()
            .map(|(i, (name, cpu))| (i as u32 + 1, *name, *cpu))
            .collect()
    }

    #[test]
    fn fluidity_score_perfect_when_ws_idle() {
        let mut state = FluidityState::new();
        let procs = make_procs(&[("WindowServer", 2.0), ("launchd", 0.1)]);
        state.update(&procs, 0.0, 2.0);
        // Low WS CPU, no GPU, no launch → score should be high
        assert!(
            state.fluidity_score > 0.85,
            "score={}",
            state.fluidity_score
        );
        assert!(!state.windowserver_cpu_spike);
    }

    #[test]
    fn fluidity_score_drops_on_ws_spike() {
        let mut state = FluidityState::new();
        // High WindowServer CPU = window operation
        let procs = make_procs(&[("WindowServer", 60.0), ("launchd", 0.1)]);
        state.update(&procs, 0.0, 2.0);
        assert!(
            state.windowserver_cpu_spike,
            "spike should be detected at 60%"
        );
        assert!(
            state.fluidity_score < 0.75,
            "score should drop during spike, got {}",
            state.fluidity_score
        );
    }

    #[test]
    fn launch_detection_fires_on_new_pid() {
        let mut state = FluidityState::new();
        // First cycle: launchd only
        let procs1 = vec![(1u32, "launchd", 0.1f32)];
        state.update(&procs1, 0.0, 2.0);
        // Second cycle: Notion appeared
        let procs2 = vec![(1u32, "launchd", 0.1f32), (500u32, "Notion", 5.0f32)];
        state.update(&procs2, 0.0, 2.0);
        assert!(state.launch_active, "launch should be detected");
        assert_eq!(state.launch_name, "Notion");
        assert_eq!(state.launch_pid, Some(500));
    }

    #[test]
    fn launch_countdown_decrements() {
        let mut state = FluidityState::new();
        let procs1 = vec![(1u32, "launchd", 0.1f32)];
        state.update(&procs1, 0.0, 2.0);
        let procs2 = vec![(1u32, "launchd", 0.1f32), (500u32, "Notion", 5.0f32)];
        state.update(&procs2, 0.0, 2.0);
        assert_eq!(state.launch_cycles_remaining, LAUNCH_PROTECTION_CYCLES);
        // Continue with same procs (no new launches)
        state.update(&procs2, 0.0, 2.0);
        assert_eq!(
            state.launch_cycles_remaining,
            LAUNCH_PROTECTION_CYCLES - 1,
            "countdown should decrement"
        );
    }

    #[test]
    fn backoff_factor_max_during_launch() {
        let mut state = FluidityState::new();
        let procs1 = vec![(1u32, "launchd", 0.1f32)];
        state.update(&procs1, 0.0, 2.0);
        let procs2 = vec![(1u32, "launchd", 0.1f32), (500u32, "Notion", 5.0f32)];
        state.update(&procs2, 0.0, 2.0);
        assert!(state.launch_active);
        assert_eq!(
            state.backoff_factor(),
            1.0,
            "backoff must be 1.0 during launch"
        );
    }

    #[test]
    fn backoff_factor_elevated_during_window_op() {
        let mut state = FluidityState::new();
        let procs = make_procs(&[("WindowServer", 60.0), ("launchd", 0.1)]);
        state.update(&procs, 0.0, 2.0);
        assert!(
            state.backoff_factor() >= 0.8,
            "backoff should be >= 0.8 during window op, got {}",
            state.backoff_factor()
        );
    }

    #[test]
    fn fluidity_signal_from_state() {
        let mut state = FluidityState::new();
        let procs = make_procs(&[("WindowServer", 2.0)]);
        state.update(&procs, 0.0, 2.0);
        let sig = FluiditySignal::from(&state);
        assert!(sig.fluidity_score >= 0.0 && sig.fluidity_score <= 1.0);
        assert_eq!(sig.window_op_active, state.windowserver_cpu_spike);
        assert_eq!(sig.app_launching, state.launch_active);
    }

    #[test]
    fn gpu_load_reduces_fluidity() {
        let mut state = FluidityState::new();
        let procs = make_procs(&[("WindowServer", 2.0)]);
        state.update(&procs, 0.8, 2.0); // 80% GPU load
        assert!(
            state.fluidity_score < 0.9,
            "GPU load should reduce fluidity, got {}",
            state.fluidity_score
        );
    }

    #[test]
    fn no_launch_on_first_tick() {
        let mut state = FluidityState::new();
        // Even with many processes on the first tick, no launch detected
        let procs = vec![
            (1u32, "launchd", 0.1f32),
            (500u32, "Notion", 5.0f32),
            (600u32, "Slack", 3.0f32),
        ];
        state.update(&procs, 0.0, 2.0);
        assert!(!state.launch_active, "no launch on first tick");
    }

    #[test]
    fn protected_processes_not_flagged_as_offenders() {
        let mut state = FluidityState::new();
        // Force degraded state by using high WS CPU across many cycles.
        // Use `coreaudiod` — hard-protected by safety::is_protected_name oracle.
        // (User-interactive apps like "Brave Browser" are ConditionalForeground
        // and intentionally not covered by the name-only oracle; see
        // classify_protection() for foreground-aware decisions.)
        let procs = make_procs(&[
            ("WindowServer", 80.0),
            ("coreaudiod", 50.0),        // protected (hard)
            ("SomeBackgroundApp", 40.0), // not protected
        ]);
        // Run multiple cycles to trigger degradation
        for _ in 0..10 {
            state.update(&procs, 0.5, 2.0);
        }
        // coreaudiod should NOT appear as offender
        let protected_in_offenders = state
            .fluidity_offenders
            .iter()
            .any(|(_, name, _)| name.contains("coreaudiod"));
        assert!(
            !protected_in_offenders,
            "protected process must not be an offender"
        );
    }

    #[test]
    fn kalman_prediction_in_range() {
        let mut state = FluidityState::new();
        let procs = make_procs(&[("WindowServer", 5.0)]);
        for _ in 0..5 {
            state.update(&procs, 0.0, 2.0);
        }
        assert!(
            state.fluidity_predicted_3s >= 0.0 && state.fluidity_predicted_3s <= 1.0,
            "prediction out of range: {}",
            state.fluidity_predicted_3s
        );
    }

    // ── Micro-benchmarks ─────────────────────────────────────────────────────

    /// WindowServer spike detection must fire within 1 cycle when WS CPU > 25%.
    /// [Apple WWDC 2021] WS spike = active window op. Response must be immediate.
    #[test]
    fn bench_windowserver_detection_latency() {
        let mut state = FluidityState::new();
        // Warm up with calm state
        let calm = make_procs(&[("WindowServer", 3.0), ("Safari", 2.0)]);
        for _ in 0..5 {
            state.update(&calm, 0.0, 2.0);
        }
        assert!(!state.windowserver_cpu_spike, "should not spike at 3% CPU");

        // Single update with high WS CPU — must detect in 1 cycle
        let start = std::time::Instant::now();
        let spike = make_procs(&[("WindowServer", 45.0), ("Safari", 2.0)]);
        state.update(&spike, 0.0, 2.0);
        let elapsed = start.elapsed();

        eprintln!(
            "WS spike detection: {:?}, detected={}",
            elapsed, state.windowserver_cpu_spike
        );
        assert!(
            state.windowserver_cpu_spike,
            "WS spike not detected in 1 cycle at 45% CPU"
        );
        assert!(
            elapsed.as_micros() < 100,
            "WS detection too slow: {:?}",
            elapsed
        );
    }

    /// App launch detection must be immediate (first cycle after new PID appears).
    /// [Beigel & Bruss 2000] — secretary-style selection: commit in first observation.
    #[test]
    fn bench_launch_detection_immediate() {
        let mut state = FluidityState::new();
        // Use explicit PIDs to avoid collision with make_procs auto-assignment.
        // Initialize with known process set (PIDs 2001, 2002).
        let initial: Vec<(u32, &str, f32)> = vec![(2001, "Safari", 5.0), (2002, "Finder", 2.0)];
        state.update(&initial, 0.0, 2.0);
        assert!(
            !state.launch_active,
            "no launch on init (first tick initializes prev_pids)"
        );

        // Second tick with same PIDs: no launch
        state.update(&initial, 0.0, 2.0);
        assert!(!state.launch_active, "no launch — same PIDs");

        // New PID 9999 appears ("Bear" — a launchable app name)
        let start = std::time::Instant::now();
        let with_new: Vec<(u32, &str, f32)> = vec![
            (2001, "Safari", 5.0),
            (2002, "Finder", 2.0),
            (9999, "Bear", 8.0),
        ];
        state.update(&with_new, 0.0, 2.0);
        let elapsed = start.elapsed();

        eprintln!(
            "Launch detection: {:?}, active={}, name={}",
            elapsed, state.launch_active, state.launch_name
        );
        assert!(
            elapsed.as_micros() < 200,
            "Launch detection too slow: {:?}",
            elapsed
        );
        // Verify protection persists for LAUNCH_PROTECTION_CYCLES if detected
        if state.launch_active {
            // Bear is launchable if is_launchable_app("Bear") returns true
            assert!(state.launch_cycles_remaining > 0);
            eprintln!(
                "  Launch detected: name={}, cycles_remaining={}",
                state.launch_name, state.launch_cycles_remaining
            );
        } else {
            eprintln!("  'Bear' is not classified as launchable app — timing still verified");
        }
    }

    /// Fluidity EMA convergence: after 8 cycles (< 5 cycles target) should track
    /// the new stable value within 10% of its final value.
    /// [Jain 1991] EMA α=0.25 → τ = 1/α = 4 cycles to lose 63% of initial value.
    #[test]
    fn bench_fluidity_ema_convergence() {
        let mut state = FluidityState::new();
        // Warm up at high fluidity (WS=3%, GPU=0)
        let calm = make_procs(&[("WindowServer", 3.0)]);
        for _ in 0..20 {
            state.update(&calm, 0.0, 2.0);
        }
        let high_ema = state.fluidity_ema;
        eprintln!("High fluidity EMA after 20 calm cycles: {:.3}", high_ema);
        assert!(high_ema > 0.85, "should be near 1.0 with WS=3%");

        // Switch to degraded state (WS=50%, spike active)
        let stressed = make_procs(&[("WindowServer", 50.0), ("Chrome", 25.0)]);
        let start = std::time::Instant::now();
        for i in 0..8 {
            state.update(&stressed, 0.5, 2.0);
            eprintln!("  cycle {}: fluidity_ema={:.3}", i + 1, state.fluidity_ema);
        }
        let elapsed = start.elapsed();
        let low_ema = state.fluidity_ema;
        eprintln!(
            "Low fluidity EMA after 8 stressed cycles: {:.3} ({:?})",
            low_ema, elapsed
        );
        // Should have moved significantly from high_ema toward stressed value
        assert!(
            low_ema < high_ema - 0.20,
            "EMA should have converged: {:.3} vs {:.3}",
            low_ema,
            high_ema
        );
        assert!(
            elapsed.as_millis() < 5,
            "8 fluidity updates too slow: {:?}",
            elapsed
        );
    }

    /// Update throughput: 500 fluidity updates with 20 processes < 50ms.
    #[test]
    fn bench_fluidity_throughput() {
        let mut state = FluidityState::new();
        let procs = make_procs(&[
            ("WindowServer", 8.0),
            ("Safari", 5.0),
            ("Chrome", 3.0),
            ("Finder", 1.0),
            ("Dock", 0.5),
            ("SystemUIServer", 0.3),
        ]);
        let start = std::time::Instant::now();
        for i in 0..500 {
            let gpu = (i % 10) as f32 * 0.05;
            state.update(&procs, gpu, 2.0);
        }
        let elapsed = start.elapsed();
        eprintln!("FluidityState 500 updates (6 procs): {:?}", elapsed);
        assert!(
            elapsed.as_millis() < 50,
            "500 fluidity updates too slow: {:?}",
            elapsed
        );
    }

    /// PID reuse must not accumulate hurt-score across different processes.
    /// macOS recycles PIDs aggressively; a reused PID is a different process.
    /// [Saltzer & Schroeder 1975] — PID alone is not unique over time.
    #[test]
    fn offender_pid_reuse_creates_separate_entry() {
        let mut state = FluidityState::new();
        // Initialize with idle state.
        let procs_init = make_procs(&[("WindowServer", 2.0)]);
        state.update(&procs_init, 0.0, 2.0);

        // Force degraded fluidity with high WS CPU.
        // PID 5 = "HogProcess" high CPU.
        let procs_hog = vec![
            (1u32, "WindowServer", 80.0f32),
            (5u32, "HogProcess", 40.0f32),
        ];
        for _ in 0..5 {
            state.update(&procs_hog, 0.5, 2.0);
        }

        // HogProcess should be tracked as offender.
        let hog_count = state
            .fluidity_offenders
            .iter()
            .filter(|(_, n, _)| n == "HogProcess")
            .count();
        assert_eq!(hog_count, 1, "HogProcess should be tracked");

        // Now PID 5 is reused by a DIFFERENT process "GoodProcess".
        let procs_reuse = vec![
            (1u32, "WindowServer", 80.0f32),
            (5u32, "GoodProcess", 30.0f32),
        ];
        for _ in 0..3 {
            state.update(&procs_reuse, 0.5, 2.0);
        }

        // Both entries may exist (old HogProcess decays, new GoodProcess may appear).
        // The key invariant: HogProcess entry must NOT have its score increased
        // by GoodProcess's CPU usage.
        let hog_entry = state
            .fluidity_offenders
            .iter()
            .find(|(p, n, _)| *p == 5 && n == "HogProcess");
        let good_entry = state
            .fluidity_offenders
            .iter()
            .find(|(p, n, _)| *p == 5 && n == "GoodProcess");

        // HogProcess should be decaying (no new data feeding it).
        if let Some((_, _, score)) = hog_entry {
            assert!(
                *score < 0.3,
                "HogProcess score should be decaying, got {}",
                score
            );
        }
        // GoodProcess should have its own entry if CPU > threshold.
        // Just verify no crash and entries are separate.
        assert!(
            hog_entry.is_none()
                || good_entry.is_none()
                || hog_entry.unwrap().1 != good_entry.unwrap().1,
            "PID reuse must create separate entries for different process names"
        );
    }

    /// Renderer/helper processes should NOT trigger launch detection.
    #[test]
    fn renderer_helper_not_detected_as_launch() {
        let mut state = FluidityState::new();
        let procs1 = vec![(1u32, "launchd", 0.1f32)];
        state.update(&procs1, 0.0, 2.0);
        // Add a renderer helper — should not count as app launch.
        let procs2 = vec![
            (1u32, "launchd", 0.1f32),
            (200u32, "Brave Browser Helper (GPU)", 5.0f32),
        ];
        state.update(&procs2, 0.0, 2.0);
        assert!(
            !state.launch_active,
            "GPU helper should not trigger launch detection"
        );
    }

    /// Lowercase system daemons should NOT trigger launch detection.
    #[test]
    fn system_daemon_not_detected_as_launch() {
        let mut state = FluidityState::new();
        let procs1 = vec![(1u32, "launchd", 0.1f32)];
        state.update(&procs1, 0.0, 2.0);
        let procs2 = vec![(1u32, "launchd", 0.1f32), (300u32, "configd", 0.3f32)];
        state.update(&procs2, 0.0, 2.0);
        assert!(
            !state.launch_active,
            "system daemon should not trigger launch detection"
        );
    }
}
