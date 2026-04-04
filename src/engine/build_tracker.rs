//! Build progress tracker — estimates cargo build completion from
//! rustc process count dynamics.
//!
//! cargo spawns one `rustc` process per compilation unit.  The count
//! peaks when all crates start compiling in parallel, then drops as
//! each unit finishes.  Monitoring the trajectory gives a coarse but
//! reliable progress estimate without instrumenting the build system.
//!
//! # Algorithm
//!
//! 1. **Detect start**: `cargo` process appears + at least 1 `rustc`.
//! 2. **Peak tracking**: `max_rustc` = highest rustc count seen this build.
//! 3. **Progress**: `(max_rustc - current_rustc) / max_rustc` once peak
//!    is established (current < max = compilations completing).
//! 4. **Fallback**: if rustc never parallelizes (max = 1 throughout),
//!    use cycle count / EXPECTED_BUILD_CYCLES as a time-based estimate.
//! 5. **Done**: `cargo` exits (0 rustc, 0 cargo) → reset.
//!
//! # Paper
//!
//! [McKenney 2004] "Exploiting Deferred Destruction" — tracking live
//! object count (rustc processes) as a proxy for in-flight work is
//! analogous to RCU grace-period detection.  The completion signal
//! arrives when the "live count" reaches zero.

// ── Configuration ────────────────────────────────────────────────────────────

/// Maximum cycles a build can run before progress is clamped to 1.0.
/// ~200 cycles × 500ms ≈ 100s — conservative for large workspaces.
const EXPECTED_BUILD_CYCLES: u32 = 200;

/// Minimum rustc count seen before we trust the peak estimate.
/// 1 = single-threaded build; still usable (fall back to cycle count).
const MIN_PEAK_FOR_ESTIMATE: u32 = 2;

// ── BuildTracker ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BuildPhase {
    /// No build in progress.
    Idle,
    /// Build just started (<20% progress) — cargo spawned, few rustc done yet.
    Starting,
    /// Build midway (20–80%) — most compilation in flight.
    Active,
    /// Build wrapping up (>80%) — rustc count declining fast.
    Finishing,
}

pub struct BuildTracker {
    /// Highest rustc process count seen during the current build.
    max_rustc_seen: u32,
    /// cargo process count in the current cycle.
    cargo_count: u32,
    /// rustc process count in the current cycle.
    rustc_count: u32,
    /// Cycles elapsed since the build started.
    build_cycles: u32,
    /// Whether a build is in progress.
    pub build_active: bool,
    /// Current estimated build progress [0.0, 1.0].
    pub build_progress: f32,
    /// Current build phase.
    pub phase: BuildPhase,
}

impl BuildTracker {
    pub fn new() -> Self {
        Self {
            max_rustc_seen: 0,
            cargo_count: 0,
            rustc_count: 0,
            build_cycles: 0,
            build_active: false,
            build_progress: 0.0,
            phase: BuildPhase::Idle,
        }
    }

    /// Update build state from the current process list.
    ///
    /// `procs`: slice of (pid, name) pairs — same format as `WindowSensor::tick`.
    pub fn tick(&mut self, procs: &[(u32, &str)]) {
        self.cargo_count = 0;
        self.rustc_count = 0;

        for &(_, name) in procs {
            let n = name.to_ascii_lowercase();
            if n == "cargo" || n.starts_with("cargo-") {
                self.cargo_count += 1;
            } else if n == "rustc" || n.starts_with("rustc-") {
                self.rustc_count += 1;
            }
        }

        let build_signal = self.cargo_count > 0 || self.rustc_count > 0;

        if build_signal {
            if !self.build_active {
                // Build just started — reset state.
                self.build_active = true;
                self.max_rustc_seen = self.rustc_count;
                self.build_cycles = 0;
            } else {
                self.build_cycles += 1;
                if self.rustc_count > self.max_rustc_seen {
                    self.max_rustc_seen = self.rustc_count;
                }
            }

            self.build_progress = self.estimate_progress();
            self.phase = self.classify_phase();
        } else if self.build_active {
            // Build just finished — mark complete for one cycle, then reset.
            self.build_progress = 1.0;
            self.phase = BuildPhase::Idle;
            self.build_active = false;
            self.max_rustc_seen = 0;
            self.build_cycles = 0;
        } else {
            self.phase = BuildPhase::Idle;
            self.build_progress = 0.0;
        }
    }

    fn estimate_progress(&self) -> f32 {
        // If we have a solid peak and rustc count is declining, use it.
        if self.max_rustc_seen >= MIN_PEAK_FOR_ESTIMATE
            && self.rustc_count < self.max_rustc_seen
        {
            let completed = (self.max_rustc_seen - self.rustc_count) as f32;
            return (completed / self.max_rustc_seen as f32).min(0.95);
        }

        // Fallback: time-based estimate normalized to EXPECTED_BUILD_CYCLES.
        let time_estimate =
            (self.build_cycles as f32 / EXPECTED_BUILD_CYCLES as f32).min(0.90);
        time_estimate
    }

    fn classify_phase(&self) -> BuildPhase {
        let p = self.build_progress;
        if p < 0.20 {
            BuildPhase::Starting
        } else if p <= 0.80 {
            BuildPhase::Active
        } else {
            BuildPhase::Finishing
        }
    }

    /// Whether the build just finished this cycle (progress == 1.0, not active).
    pub fn just_finished(&self) -> bool {
        !self.build_active && self.build_progress == 1.0
    }
}

impl Default for BuildTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn procs<'a>(names: &[&'a str]) -> Vec<(u32, &'a str)> {
        names.iter().enumerate().map(|(i, &n)| (i as u32, n)).collect()
    }

    #[test]
    fn idle_when_no_build_processes() {
        let mut bt = BuildTracker::new();
        bt.tick(&procs(&["Brave", "Warp", "node"]));
        assert!(!bt.build_active);
        assert_eq!(bt.phase, BuildPhase::Idle);
        assert_eq!(bt.build_progress, 0.0);
    }

    #[test]
    fn detects_build_start() {
        let mut bt = BuildTracker::new();
        bt.tick(&procs(&["cargo", "rustc"]));
        assert!(bt.build_active);
        assert_eq!(bt.phase, BuildPhase::Starting);
    }

    #[test]
    fn progress_increases_as_rustc_declines() {
        let mut bt = BuildTracker::new();
        // Peak: 8 rustc processes
        bt.tick(&procs(&["cargo", "rustc", "rustc", "rustc", "rustc", "rustc", "rustc", "rustc", "rustc"]));
        assert_eq!(bt.max_rustc_seen, 8);
        assert_eq!(bt.build_progress, 0.0); // none done yet (still at peak)

        // 4 rustc done → 50% progress
        bt.tick(&procs(&["cargo", "rustc", "rustc", "rustc", "rustc"]));
        assert!((bt.build_progress - 0.5).abs() < 0.01, "expected ~50% got {}", bt.build_progress);
        assert_eq!(bt.phase, BuildPhase::Active);
    }

    #[test]
    fn finishing_phase_when_mostly_done() {
        let mut bt = BuildTracker::new();
        // Peak 10 rustc
        let peak_names: Vec<&str> = std::iter::once("cargo")
            .chain(std::iter::repeat("rustc").take(10))
            .collect();
        bt.tick(&procs(&peak_names));
        // 1 rustc remaining → 90% → Finishing
        bt.tick(&procs(&["cargo", "rustc"]));
        assert_eq!(bt.phase, BuildPhase::Finishing);
        assert!(bt.build_progress >= 0.80, "expected ≥0.80 got {}", bt.build_progress);
    }

    #[test]
    fn resets_when_build_completes() {
        let mut bt = BuildTracker::new();
        bt.tick(&procs(&["cargo", "rustc"]));
        assert!(bt.build_active);
        // Build finishes
        bt.tick(&procs(&["Warp"]));
        assert!(!bt.build_active);
        assert_eq!(bt.phase, BuildPhase::Idle);
    }

    #[test]
    fn time_based_fallback_for_single_rustc() {
        let mut bt = BuildTracker::new();
        // Single-threaded build — max_rustc_seen = 1, never declines
        bt.tick(&procs(&["cargo", "rustc"]));
        // After many cycles, time-based estimate should be positive
        for _ in 0..50 {
            bt.tick(&procs(&["cargo", "rustc"]));
        }
        assert!(bt.build_progress > 0.0, "time-based estimate should be positive");
    }
}
