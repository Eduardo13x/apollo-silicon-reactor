//! SignalHealthMonitor — detects numerically pathological values in filter outputs.
//!
//! Prevents silent propagation of NaN/Inf/subnormal through the signal pipeline.
//! Pattern mirrors `OdeDivergenceResilient` in `adversarial_probe.rs`.
//!
//! ## Motivation
//! [Goldberg 1991 "What Every Computer Scientist Should Know About FP"]:
//! subnormals cause 10–100× slowdowns on many microarchitectures AND produce
//! incorrect comparisons (`denormal < f64::MIN_POSITIVE` is true while
//! `denormal == 0.0` is false). NaN values silently corrupt all downstream
//! arithmetic through IEEE 754 NaN-propagation.
//!
//! Historical context: `fix: add NaN guards to Kalman/CUSUM` arrived *after*
//! the system had already been silently diverging. This monitor provides
//! continuous observability so divergence is detected the cycle it begins,
//! not weeks later.

/// Counts numerically pathological values seen by the signal pipeline.
///
/// A value is *pathological* if it is:
/// - `NaN` (Not-a-Number) — propagates through all IEEE 754 arithmetic.
/// - `±Inf` — indicates unbounded filter divergence.
/// - Subnormal (|v| ∈ (0, `f64::MIN_POSITIVE`)) — denormal FP causes
///   hardware slowdowns and silent comparison errors [Goldberg 1991].
///
/// Normal zero (`0.0`) is always considered healthy.
#[derive(Debug, Default, Clone)]
pub struct SignalHealthMonitor {
    /// Total count of pathological values detected since construction.
    /// Monotonically increasing — reset only on daemon restart.
    /// Expose in telemetry; a rising count in prod signals filter divergence.
    pub violations_total: u64,
}

impl SignalHealthMonitor {
    /// Create a fresh monitor with zero violations.
    pub fn new() -> Self {
        Self { violations_total: 0 }
    }

    /// Returns `true` if `v` is healthy (finite and not subnormal).
    ///
    /// Increments `violations_total` on the first pathological value detected
    /// per call — one violation per call, regardless of how many dimensions fail.
    /// Use [`check_slice`] for multi-dimensional outputs.
    pub fn check_f64(&mut self, v: f64) -> bool {
        if v.is_nan() || v.is_infinite() || (v != 0.0 && v.abs() < f64::MIN_POSITIVE) {
            self.violations_total += 1;
            false
        } else {
            true
        }
    }

    /// Checks a slice — returns `false` if *any* element is pathological.
    ///
    /// Each pathological element increments `violations_total` independently,
    /// so the total count reflects how many individual values were bad, not
    /// just how many calls failed.
    pub fn check_slice(&mut self, vals: &[f64]) -> bool {
        let mut all_ok = true;
        for &v in vals {
            if !self.check_f64(v) {
                all_ok = false;
            }
        }
        all_ok
    }

    /// Return `v` if healthy, otherwise record a violation and return `fallback`.
    ///
    /// Use at filter outputs where the caller must have *some* value to proceed:
    /// ```ignore
    /// let safe_pressure = signal_health.sanitize(kalman_output, last_known_good);
    /// ```
    pub fn sanitize(&mut self, v: f64, fallback: f64) -> f64 {
        if self.check_f64(v) { v } else { fallback }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nan_detected() {
        let mut m = SignalHealthMonitor::new();
        assert!(!m.check_f64(f64::NAN), "NaN must be detected as pathological");
        assert_eq!(m.violations_total, 1);
    }

    #[test]
    fn test_inf_detected() {
        let mut m = SignalHealthMonitor::new();
        assert!(!m.check_f64(f64::INFINITY));
        assert!(!m.check_f64(f64::NEG_INFINITY));
        assert_eq!(m.violations_total, 2, "+Inf and -Inf are both violations");
    }

    #[test]
    fn test_subnormal_detected() {
        let mut m = SignalHealthMonitor::new();
        // f64::MIN_POSITIVE / 2.0 is a subnormal (denormal) value.
        let subnormal = f64::MIN_POSITIVE / 2.0;
        assert!(subnormal != 0.0, "subnormal is not zero");
        assert!(subnormal.abs() < f64::MIN_POSITIVE, "subnormal is below MIN_POSITIVE");
        assert!(!m.check_f64(subnormal), "subnormal must be flagged");
        assert_eq!(m.violations_total, 1);
    }

    #[test]
    fn test_normal_passes() {
        let mut m = SignalHealthMonitor::new();
        assert!(m.check_f64(0.0), "exact zero is healthy");
        assert!(m.check_f64(0.75), "normal float passes");
        assert!(m.check_f64(-1.23e10), "large negative passes");
        assert!(m.check_f64(f64::MIN_POSITIVE), "MIN_POSITIVE itself passes");
        assert_eq!(m.violations_total, 0, "no violations for valid values");
    }

    #[test]
    fn test_violations_counted() {
        let mut m = SignalHealthMonitor::new();
        let slice = [0.5, f64::NAN, 0.3, f64::INFINITY, 0.1];
        let ok = m.check_slice(&slice);
        assert!(!ok, "slice with bad values must return false");
        assert_eq!(m.violations_total, 2, "NaN + Inf = 2 violations");
    }

    #[test]
    fn test_sanitize_replaces_bad() {
        let mut m = SignalHealthMonitor::new();
        let result = m.sanitize(f64::NAN, 0.5);
        assert_eq!(result, 0.5, "NaN replaced by fallback");
        assert_eq!(m.violations_total, 1);

        let result2 = m.sanitize(0.8, 0.0);
        assert_eq!(result2, 0.8, "healthy value returned unchanged");
        assert_eq!(m.violations_total, 1, "no extra violation for healthy value");
    }
}
