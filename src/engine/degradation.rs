//! Graceful degradation tiers for the optimization daemon.
//!
//! The daemon progresses through four modes depending on observed failure rates
//! and system health.  Transitions are driven by the main loop calling
//! `DegradationController::update()` each cycle.
//!
//! ```text
//!  Full ──────(3 failures / 60 s)──────► Conservative
//!  Conservative ──(kernel_task > 95%)──► Observe
//!  Any ──────(CB open > 5 min)──────────► Emergency
//!  Emergency ──(60 s clean)─────────────► Full
//! ```

use std::time::{Duration, Instant};

// ── Public types ─────────────────────────────────────────────────────────────

/// The four degradation tiers — ordered from least to most restrictive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationMode {
    /// All optimizations active: freeze, throttle, QoS, sysctl.
    Full,
    /// Only safe actions: unfreeze + QoS hints.  No SIGSTOP.
    Conservative,
    /// No actions, only metrics collection.
    Observe,
    /// Unfreeze everything, restore sysctls, hold and wait.
    Emergency,
}

impl OperationMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Conservative => "conservative",
            Self::Observe => "observe",
            Self::Emergency => "emergency",
        }
    }

    /// Returns true if freeze/SIGSTOP actions are permitted.
    pub fn allows_freeze(&self) -> bool {
        matches!(self, Self::Full)
    }

    /// Returns true if throttle actions are permitted.
    pub fn allows_throttle(&self) -> bool {
        matches!(self, Self::Full)
    }

    /// Returns true if sysctl writes are permitted.
    pub fn allows_sysctl(&self) -> bool {
        matches!(self, Self::Full | Self::Conservative)
    }

    /// Returns true if QoS hint actions are permitted.
    pub fn allows_qos(&self) -> bool {
        matches!(self, Self::Full | Self::Conservative)
    }

    /// Returns true if unfreeze actions should always proceed regardless of tier.
    pub fn allows_unfreeze(&self) -> bool {
        true // unfreeze is always safe
    }
}

/// Driving inputs evaluated each cycle.
#[derive(Debug, Default)]
pub struct DegradationInputs {
    /// New failures from `execute_actions` this cycle.
    pub new_failures: u64,
    /// kernel_task CPU % (0.0–100.0).
    pub kernel_task_cpu_pct: f64,
    /// Whether the circuit breaker is currently Open.
    pub circuit_open: bool,
    /// How long the circuit breaker has been open (None if not open).
    pub circuit_open_duration: Option<Duration>,
}

/// Tracks degradation state and drives transitions.
#[derive(Debug)]
pub struct DegradationController {
    pub mode: OperationMode,
    /// Timestamps of recent failures within the sliding window.
    failure_timestamps: Vec<Instant>,
    /// Failures-per-60s threshold before Conservative.
    pub failure_threshold_conservative: u32,
    /// kernel_task CPU threshold before Observe.
    pub kernel_cpu_threshold_observe: f64,
    /// Circuit open duration before Emergency.
    pub circuit_open_emergency: Duration,
    /// Clean time (zero failures) before recovering from Emergency → Full.
    pub emergency_recovery_secs: u64,
    /// When this controller entered Emergency mode.
    emergency_since: Option<Instant>,
    /// Timestamp of last failure (for clean-period tracking).
    last_failure_at: Option<Instant>,
    /// Total mode transitions recorded.
    pub transitions_total: u64,
    /// Total cycles spent in each mode.
    pub cycles_full: u64,
    pub cycles_conservative: u64,
    pub cycles_observe: u64,
    pub cycles_emergency: u64,
}

impl Default for DegradationController {
    fn default() -> Self {
        Self::new(3, 95.0, Duration::from_secs(300), 60)
    }
}

impl DegradationController {
    /// Create a controller with explicit thresholds.
    ///
    /// - `failure_threshold_conservative`: failures in 60 s → Conservative
    /// - `kernel_cpu_threshold_observe`: kernel_task CPU % → Observe
    /// - `circuit_open_emergency`: circuit open duration → Emergency
    /// - `emergency_recovery_secs`: clean seconds to exit Emergency
    pub fn new(
        failure_threshold_conservative: u32,
        kernel_cpu_threshold_observe: f64,
        circuit_open_emergency: Duration,
        emergency_recovery_secs: u64,
    ) -> Self {
        Self {
            mode: OperationMode::Full,
            failure_timestamps: Vec::new(),
            failure_threshold_conservative,
            kernel_cpu_threshold_observe,
            circuit_open_emergency,
            emergency_recovery_secs,
            emergency_since: None,
            last_failure_at: None,
            transitions_total: 0,
            cycles_full: 0,
            cycles_conservative: 0,
            cycles_observe: 0,
            cycles_emergency: 0,
        }
    }

    /// Evaluate inputs and potentially transition mode.  Call once per cycle.
    ///
    /// Returns the new (or unchanged) `OperationMode`.
    pub fn update(&mut self, inputs: &DegradationInputs) -> &OperationMode {
        let now = Instant::now();
        let window_60s = Duration::from_secs(60);

        // ── Record new failures ───────────────────────────────────────────────
        if inputs.new_failures > 0 {
            self.last_failure_at = Some(now);
            for _ in 0..inputs.new_failures {
                self.failure_timestamps.push(now);
            }
        }
        // Prune old failure timestamps outside 60 s window.
        self.failure_timestamps
            .retain(|t| now.duration_since(*t) <= window_60s);

        let failures_60s = self.failure_timestamps.len() as u32;

        // ── Emergency: circuit open for too long (any → Emergency) ────────────
        if let Some(open_dur) = inputs.circuit_open_duration {
            if open_dur >= self.circuit_open_emergency && self.mode != OperationMode::Emergency {
                self.transition(OperationMode::Emergency, "circuit-open-too-long");
                self.emergency_since = Some(now);
            }
        }

        // ── Emergency recovery → Full ─────────────────────────────────────────
        if self.mode == OperationMode::Emergency {
            let clean = self
                .last_failure_at
                .map(|t| now.duration_since(t).as_secs() >= self.emergency_recovery_secs)
                .unwrap_or(true); // No failures ever recorded → clean.
            if clean && !inputs.circuit_open {
                self.transition(OperationMode::Full, "emergency-recovery");
                self.emergency_since = None;
            }
        }

        // ── Observe: kernel_task CPU spike ────────────────────────────────────
        if self.mode == OperationMode::Conservative
            && inputs.kernel_task_cpu_pct > self.kernel_cpu_threshold_observe
        {
            self.transition(OperationMode::Observe, "kernel-task-cpu-high");
        }
        // Recover from Observe when kernel_task calms down.
        if self.mode == OperationMode::Observe
            && inputs.kernel_task_cpu_pct <= self.kernel_cpu_threshold_observe
            && failures_60s < self.failure_threshold_conservative
        {
            self.transition(OperationMode::Full, "kernel-task-cpu-recovered");
        }

        // ── Conservative: failure rate ────────────────────────────────────────
        if self.mode == OperationMode::Full
            && failures_60s >= self.failure_threshold_conservative
        {
            self.transition(OperationMode::Conservative, "failure-rate-high");
        }
        // Recover from Conservative when failures clear.
        if self.mode == OperationMode::Conservative
            && failures_60s == 0
            && !inputs.circuit_open
        {
            self.transition(OperationMode::Full, "failure-rate-clear");
        }

        // ── Cycle counters ────────────────────────────────────────────────────
        match self.mode {
            OperationMode::Full => self.cycles_full += 1,
            OperationMode::Conservative => self.cycles_conservative += 1,
            OperationMode::Observe => self.cycles_observe += 1,
            OperationMode::Emergency => self.cycles_emergency += 1,
        }

        &self.mode
    }

    /// Returns the failure rate per 60-second window (0.0–1.0 fraction of threshold).
    pub fn failure_rate_60s(&self) -> f32 {
        if self.failure_threshold_conservative == 0 {
            return 0.0;
        }
        let now = Instant::now();
        let window = Duration::from_secs(60);
        let count = self
            .failure_timestamps
            .iter()
            .filter(|t| now.duration_since(**t) <= window)
            .count() as f32;
        count / self.failure_threshold_conservative as f32
    }

    /// Returns the count of failures in the last 60 seconds.
    pub fn failures_in_last_60s(&self) -> u32 {
        let now = Instant::now();
        let window = Duration::from_secs(60);
        self.failure_timestamps
            .iter()
            .filter(|t| now.duration_since(**t) <= window)
            .count() as u32
    }

    fn transition(&mut self, to: OperationMode, reason: &str) {
        if self.mode == to {
            return;
        }
        tracing::warn!(
            from = self.mode.as_str(),
            to = to.as_str(),
            reason,
            "degradation: mode transition"
        );
        self.mode = to;
        self.transitions_total += 1;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(failures: u64) -> DegradationInputs {
        DegradationInputs {
            new_failures: failures,
            kernel_task_cpu_pct: 0.0,
            circuit_open: false,
            circuit_open_duration: None,
        }
    }

    #[test]
    fn starts_full() {
        let ctrl = DegradationController::default();
        assert_eq!(ctrl.mode, OperationMode::Full);
    }

    #[test]
    fn transitions_to_conservative_on_failure_rate() {
        let mut ctrl = DegradationController::new(3, 95.0, Duration::from_secs(300), 60);
        ctrl.update(&inputs(1));
        ctrl.update(&inputs(1));
        assert_eq!(ctrl.mode, OperationMode::Full); // 2 < 3
        ctrl.update(&inputs(1)); // 3 → Conservative
        assert_eq!(ctrl.mode, OperationMode::Conservative);
        assert_eq!(ctrl.transitions_total, 1);
    }

    #[test]
    fn transitions_to_emergency_when_circuit_open_too_long() {
        let mut ctrl = DegradationController::new(3, 95.0, Duration::from_millis(10), 60);
        let inp = DegradationInputs {
            new_failures: 0,
            kernel_task_cpu_pct: 0.0,
            circuit_open: true,
            circuit_open_duration: Some(Duration::from_millis(20)),
        };
        ctrl.update(&inp);
        assert_eq!(ctrl.mode, OperationMode::Emergency);
    }

    #[test]
    fn emergency_recovers_to_full() {
        let mut ctrl = DegradationController::new(3, 95.0, Duration::from_millis(10), 0);
        ctrl.update(&DegradationInputs {
            new_failures: 0,
            kernel_task_cpu_pct: 0.0,
            circuit_open: true,
            circuit_open_duration: Some(Duration::from_millis(20)),
        });
        assert_eq!(ctrl.mode, OperationMode::Emergency);
        // Now circuit closes and no failures
        ctrl.update(&inputs(0));
        assert_eq!(ctrl.mode, OperationMode::Full);
    }

    #[test]
    fn observe_triggers_on_kernel_cpu() {
        let mut ctrl = DegradationController::new(3, 90.0, Duration::from_secs(300), 60);
        // First push to Conservative
        ctrl.update(&inputs(1));
        ctrl.update(&inputs(1));
        ctrl.update(&inputs(1));
        assert_eq!(ctrl.mode, OperationMode::Conservative);
        // Now kernel_task spikes
        let inp = DegradationInputs {
            new_failures: 0,
            kernel_task_cpu_pct: 95.0,
            circuit_open: false,
            circuit_open_duration: None,
        };
        ctrl.update(&inp);
        assert_eq!(ctrl.mode, OperationMode::Observe);
    }

    #[test]
    fn operation_mode_flags() {
        assert!(OperationMode::Full.allows_freeze());
        assert!(!OperationMode::Conservative.allows_freeze());
        assert!(!OperationMode::Observe.allows_freeze());
        assert!(!OperationMode::Emergency.allows_freeze());

        assert!(OperationMode::Full.allows_sysctl());
        assert!(OperationMode::Conservative.allows_sysctl());
        assert!(!OperationMode::Observe.allows_sysctl());

        assert!(OperationMode::Emergency.allows_unfreeze());
        assert!(OperationMode::Observe.allows_unfreeze());
    }

    #[test]
    fn as_str_values() {
        assert_eq!(OperationMode::Full.as_str(), "full");
        assert_eq!(OperationMode::Conservative.as_str(), "conservative");
        assert_eq!(OperationMode::Observe.as_str(), "observe");
        assert_eq!(OperationMode::Emergency.as_str(), "emergency");
    }
}
