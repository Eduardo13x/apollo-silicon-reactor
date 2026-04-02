//! Circuit breaker for `execute_actions` — prevents cascading failures when
//! the action executor encounters repeated errors.
//!
//! State machine:
//!   Closed → Open  : `failure_threshold` failures within `window` seconds
//!   Open   → HalfOpen : after `timeout` elapses
//!   HalfOpen → Closed : `success_threshold` consecutive successes
//!   HalfOpen → Open   : any failure resets back to Open

use std::time::{Duration, Instant};

// ── Public types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation — all calls pass through.
    Closed,
    /// Failing — reject calls immediately and log a skip.
    Open,
    /// Testing recovery — allow the next call through; revert to Open on failure.
    HalfOpen,
}

impl CircuitState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Closed => "closed",
            Self::Open => "open",
            Self::HalfOpen => "half_open",
        }
    }
}

/// Wraps a fallible function and enforces the circuit breaker policy.
#[derive(Debug)]
pub struct CircuitBreaker {
    state: CircuitState,
    /// Timestamps of recent failures within the sliding window.
    failure_timestamps: Vec<Instant>,
    /// Number of failures within `window` that trips the breaker.
    pub failure_threshold: u32,
    /// Sliding window for counting failures (seconds).
    pub window_secs: u64,
    /// Consecutive successes needed to close from HalfOpen.
    pub success_threshold: u32,
    /// How long to stay Open before moving to HalfOpen.
    pub timeout: Duration,
    /// Consecutive successes accumulated while HalfOpen.
    half_open_successes: u32,
    /// When the circuit opened (for timeout tracking).
    opened_at: Option<Instant>,
    /// Total calls rejected because the circuit was Open.
    pub rejected_total: u64,
    /// Total transitions to Open.
    pub trips_total: u64,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(5, 60, 2, Duration::from_secs(30))
    }
}

impl CircuitBreaker {
    /// Create a circuit breaker.
    ///
    /// - `failure_threshold`: failures within `window_secs` → trip to Open
    /// - `window_secs`: sliding window size for failure counting
    /// - `success_threshold`: consecutive successes to close from HalfOpen
    /// - `timeout`: time spent Open before entering HalfOpen
    pub fn new(
        failure_threshold: u32,
        window_secs: u64,
        success_threshold: u32,
        timeout: Duration,
    ) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_timestamps: Vec::new(),
            failure_threshold,
            window_secs,
            success_threshold,
            timeout,
            half_open_successes: 0,
            opened_at: None,
            rejected_total: 0,
            trips_total: 0,
        }
    }

    /// Current state of the circuit.
    pub fn state(&self) -> &CircuitState {
        &self.state
    }

    /// Returns true if calls are currently allowed through.
    pub fn is_closed(&self) -> bool {
        matches!(self.state, CircuitState::Closed | CircuitState::HalfOpen)
    }

    /// Execute `f` through the circuit breaker.
    ///
    /// - If `Closed` or `HalfOpen`: runs `f`, records success/failure.
    /// - If `Open`: returns `Err(CircuitError::Open)` without calling `f`.
    ///
    /// After `timeout`, an Open circuit automatically transitions to HalfOpen
    /// on the next `call()` attempt.
    pub fn call<F, T, E>(&mut self, f: F) -> Result<T, CircuitError<E>>
    where
        F: FnOnce() -> Result<T, E>,
    {
        self.maybe_recover();

        match self.state {
            CircuitState::Open => {
                self.rejected_total += 1;
                return Err(CircuitError::Open);
            }
            CircuitState::Closed | CircuitState::HalfOpen => {
                match f() {
                    Ok(v) => {
                        self.record_success();
                        Ok(v)
                    }
                    Err(e) => {
                        self.record_failure();
                        Err(CircuitError::Inner(e))
                    }
                }
            }
        }
    }

    /// Record a successful operation (also usable when calling `execute_actions`
    /// indirectly — the caller can report outcomes externally).
    pub fn record_success(&mut self) {
        match self.state {
            CircuitState::HalfOpen => {
                self.half_open_successes += 1;
                if self.half_open_successes >= self.success_threshold {
                    self.close();
                }
            }
            CircuitState::Closed => {
                // Nothing to do — stay closed.
            }
            CircuitState::Open => {
                // Shouldn't happen via normal flow, but treat as recovery.
                self.close();
            }
        }
    }

    /// Record a failed operation.
    pub fn record_failure(&mut self) {
        let now = Instant::now();
        let window = Duration::from_secs(self.window_secs);

        // Prune old timestamps outside the sliding window.
        self.failure_timestamps
            .retain(|t| now.duration_since(*t) <= window);
        self.failure_timestamps.push(now);

        match self.state {
            CircuitState::HalfOpen => {
                // Any failure in HalfOpen resets to Open immediately.
                self.trip(now);
            }
            CircuitState::Closed => {
                if self.failure_timestamps.len() as u32 >= self.failure_threshold {
                    self.trip(now);
                }
            }
            CircuitState::Open => {
                // Already open — just update timestamp.
                self.opened_at = Some(now);
            }
        }
    }

    /// Elapsed time since the circuit tripped (None if not Open).
    pub fn open_duration(&self) -> Option<Duration> {
        if self.state == CircuitState::Open {
            self.opened_at.map(|t| t.elapsed())
        } else {
            None
        }
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    fn trip(&mut self, at: Instant) {
        self.state = CircuitState::Open;
        self.opened_at = Some(at);
        self.half_open_successes = 0;
        self.trips_total += 1;
        tracing::warn!(
            trips = self.trips_total,
            "circuit-breaker: tripped → Open"
        );
    }

    fn close(&mut self) {
        self.state = CircuitState::Closed;
        self.half_open_successes = 0;
        self.failure_timestamps.clear();
        self.opened_at = None;
        tracing::info!("circuit-breaker: recovered → Closed");
    }

    /// Transition Open → HalfOpen after timeout.
    fn maybe_recover(&mut self) {
        if self.state == CircuitState::Open {
            if let Some(opened_at) = self.opened_at {
                if opened_at.elapsed() >= self.timeout {
                    self.state = CircuitState::HalfOpen;
                    self.half_open_successes = 0;
                    tracing::info!("circuit-breaker: timeout elapsed → HalfOpen (testing recovery)");
                }
            }
        }
    }
}

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum CircuitError<E> {
    /// Circuit is Open — the inner function was not called.
    Open,
    /// Circuit allowed the call but the inner function returned an error.
    Inner(E),
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helpers ─────────────────────────────────────────────────────────────────

    fn ok_call(cb: &mut CircuitBreaker) -> bool {
        cb.call(|| -> Result<(), &str> { Ok(()) }).is_ok()
    }

    fn err_call(cb: &mut CircuitBreaker) -> bool {
        matches!(
            cb.call(|| -> Result<(), &str> { Err("boom") }),
            Err(CircuitError::Inner(_))
        )
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn starts_closed() {
        let cb = CircuitBreaker::default();
        assert_eq!(*cb.state(), CircuitState::Closed);
    }

    #[test]
    fn failure_counting_trips_breaker() {
        // threshold = 3 failures within 60s
        let mut cb = CircuitBreaker::new(3, 60, 2, Duration::from_secs(30));
        assert!(err_call(&mut cb));
        assert!(err_call(&mut cb));
        assert_eq!(*cb.state(), CircuitState::Closed); // 2 < 3
        assert!(err_call(&mut cb)); // 3rd → trips
        assert_eq!(*cb.state(), CircuitState::Open);
        assert_eq!(cb.trips_total, 1);
    }

    #[test]
    fn open_rejects_calls() {
        let mut cb = CircuitBreaker::new(1, 60, 1, Duration::from_secs(60));
        err_call(&mut cb);
        assert_eq!(*cb.state(), CircuitState::Open);
        let result = cb.call(|| -> Result<(), &str> { Ok(()) });
        assert!(matches!(result, Err(CircuitError::Open)));
        assert_eq!(cb.rejected_total, 1);
    }

    #[test]
    fn half_open_to_closed_on_success_threshold() {
        let mut cb = CircuitBreaker::new(1, 60, 2, Duration::from_millis(1));
        err_call(&mut cb);
        assert_eq!(*cb.state(), CircuitState::Open);

        // Simulate timeout elapsed
        std::thread::sleep(Duration::from_millis(2));
        // Next call triggers maybe_recover → HalfOpen, then executes
        ok_call(&mut cb); // 1st success → HalfOpen (success_threshold = 2)
        assert_eq!(*cb.state(), CircuitState::HalfOpen);
        ok_call(&mut cb); // 2nd success → Closed
        assert_eq!(*cb.state(), CircuitState::Closed);
    }

    #[test]
    fn half_open_to_open_on_failure() {
        let mut cb = CircuitBreaker::new(1, 60, 2, Duration::from_millis(1));
        err_call(&mut cb);
        std::thread::sleep(Duration::from_millis(2));
        err_call(&mut cb); // triggers HalfOpen then immediately fails → Open again
        assert_eq!(*cb.state(), CircuitState::Open);
        assert_eq!(cb.trips_total, 2);
    }

    #[test]
    fn open_to_half_open_transition() {
        let mut cb = CircuitBreaker::new(1, 60, 1, Duration::from_millis(5));
        err_call(&mut cb);
        assert_eq!(*cb.state(), CircuitState::Open);
        std::thread::sleep(Duration::from_millis(6));
        // Call should trigger maybe_recover → HalfOpen, then run
        ok_call(&mut cb);
        assert_eq!(*cb.state(), CircuitState::Closed);
    }

    #[test]
    fn record_success_and_failure_external() {
        let mut cb = CircuitBreaker::new(3, 60, 2, Duration::from_secs(30));
        cb.record_failure();
        cb.record_failure();
        assert_eq!(*cb.state(), CircuitState::Closed);
        cb.record_failure(); // 3rd → Open
        assert_eq!(*cb.state(), CircuitState::Open);
    }

    #[test]
    fn as_str_values() {
        assert_eq!(CircuitState::Closed.as_str(), "closed");
        assert_eq!(CircuitState::Open.as_str(), "open");
        assert_eq!(CircuitState::HalfOpen.as_str(), "half_open");
    }
}
