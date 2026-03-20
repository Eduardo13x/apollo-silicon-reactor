//! `thread_selfcounts` — undocumented XNU syscall 186.
//!
//! Returns per-thread CPU instructions and cycles for the calling thread.
//! Works **without root or entitlements** on macOS 10.15+/Apple Silicon.
//!
//! Discovered by the reverse engineering community; used by Google Benchmark
//! for `--benchmark_perf_counters=INSTRUCTIONS,CYCLES` on macOS.
//!
//! # What this gives Apollo
//!
//! - **Daemon self-monitoring**: Measure IPC of Apollo's own optimization cycle.
//!   IPC < 0.5 → daemon is memory-bound (too many allocations).
//!   IPC > 1.5 → daemon is compute-efficient.
//! - **Per-cycle cost tracking**: Instructions per cycle for regression detection.


/// Raw counters from `thread_selfcounts(1, ...)`.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct ThreadCounts {
    pub instructions: u64,
    pub cycles: u64,
}

/// Read the calling thread's instruction and cycle counters.
///
/// Cost: ~50ns (single syscall, no context switch).
/// Returns `None` if the syscall fails (unlikely on macOS 10.15+).
#[cfg(target_os = "macos")]
pub fn read_self_counts() -> Option<ThreadCounts> {
    let mut counts = ThreadCounts::default();
    // syscall 186 = thread_selfcounts
    // type=1: return (instructions, cycles) as two u64s
    let rc = unsafe {
        libc::syscall(
            186,
            1i32,
            &mut counts as *mut ThreadCounts as *mut libc::c_void,
            std::mem::size_of::<ThreadCounts>(),
        )
    };
    if rc == 0 {
        Some(counts)
    } else {
        None
    }
}

#[cfg(not(target_os = "macos"))]
pub fn read_self_counts() -> Option<ThreadCounts> {
    None
}

/// Measures IPC (instructions per cycle) of a code block.
///
/// ```ignore
/// let (result, ipc) = measure_ipc(|| {
///     expensive_computation();
/// });
/// println!("IPC: {:.2}", ipc);
/// ```
pub fn measure_ipc<F, R>(f: F) -> (R, f64)
where
    F: FnOnce() -> R,
{
    let before = read_self_counts();
    let result = f();
    let after = read_self_counts();

    let ipc = match (before, after) {
        (Some(b), Some(a)) => {
            let delta_insn = a.instructions.saturating_sub(b.instructions);
            let delta_cyc = a.cycles.saturating_sub(b.cycles);
            if delta_cyc > 0 {
                delta_insn as f64 / delta_cyc as f64
            } else {
                0.0
            }
        }
        _ => 0.0,
    };

    (result, ipc)
}

// ── Daemon cycle tracker ────────────────────────────────────────────────────

/// Tracks IPC of Apollo's own optimization cycles.
///
/// Updated once per daemon cycle. Exposes EMA-smoothed IPC for status reporting.
pub struct CycleIpcTracker {
    prev: Option<ThreadCounts>,
    ema_ipc: f64,
    _ema_instructions: f64,
    total_cycles_measured: u64,
}

impl CycleIpcTracker {
    pub fn new() -> Self {
        Self {
            prev: read_self_counts(),
            ema_ipc: 0.0,
            _ema_instructions: 0.0,
            total_cycles_measured: 0,
        }
    }

    /// Call at the end of each daemon cycle. Returns this cycle's IPC.
    pub fn tick(&mut self) -> f64 {
        let current = match read_self_counts() {
            Some(c) => c,
            None => return self.ema_ipc,
        };

        let ipc = if let Some(prev) = self.prev {
            let delta_insn = current.instructions.saturating_sub(prev.instructions);
            let delta_cyc = current.cycles.saturating_sub(prev.cycles);
            if delta_cyc > 100 {
                // Require minimum 100 cycles to avoid noise
                delta_insn as f64 / delta_cyc as f64
            } else {
                self.ema_ipc
            }
        } else {
            0.0
        };

        // EMA with alpha=0.1 (smooth over ~10 cycles)
        const ALPHA: f64 = 0.1;
        if self.total_cycles_measured == 0 {
            self.ema_ipc = ipc;
        } else {
            self.ema_ipc = ALPHA * ipc + (1.0 - ALPHA) * self.ema_ipc;
        }

        self.total_cycles_measured += 1;
        self.prev = Some(current);
        ipc
    }

    /// EMA-smoothed IPC of the daemon's optimization loop.
    pub fn ema_ipc(&self) -> f64 {
        self.ema_ipc
    }

    /// Total optimization cycles measured.
    pub fn cycles_measured(&self) -> u64 {
        self.total_cycles_measured
    }
}

impl Default for CycleIpcTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── IPC classification for throttle decisions ───────────────────────────────

/// Classify a process's IPC for throttle decisions.
///
/// Based on Apple M1 Firestorm/Icestorm microarchitecture:
/// - IPC < 0.3: heavily memory-bound (stalls on cache misses/TLB).
///   Safe to throttle — throttling won't make it slower.
/// - IPC 0.3–1.0: mixed workload. Default throttle policy.
/// - IPC > 1.0: compute-efficient (good cache behavior).
///   Throttling directly hurts throughput — be conservative.
/// - IPC > 2.0: highly optimized SIMD/NEON workload.
///   Do NOT throttle unless thermal emergency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpcClass {
    /// IPC < 0.3 — memory-bound, safe to throttle aggressively.
    MemoryBound,
    /// IPC 0.3–1.0 — mixed workload, default policy.
    Mixed,
    /// IPC > 1.0 — compute-efficient, throttle conservatively.
    ComputeBound,
    /// IPC > 2.0 — highly optimized, do not throttle.
    Optimized,
}

impl IpcClass {
    pub fn from_ipc(ipc: f64) -> Self {
        if ipc <= 0.0 {
            Self::Mixed // No data
        } else if ipc < 0.3 {
            Self::MemoryBound
        } else if ipc < 1.0 {
            Self::Mixed
        } else if ipc < 2.0 {
            Self::ComputeBound
        } else {
            Self::Optimized
        }
    }

    /// Whether throttling is safe for this IPC class.
    pub fn safe_to_throttle(&self) -> bool {
        matches!(self, Self::MemoryBound | Self::Mixed)
    }

    /// Whether aggressive throttling is safe.
    pub fn safe_to_throttle_aggressive(&self) -> bool {
        matches!(self, Self::MemoryBound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn selfcounts_reads_successfully() {
        let counts = read_self_counts();
        assert!(counts.is_some(), "thread_selfcounts should work on macOS");
        let c = counts.unwrap();
        assert!(c.instructions > 0, "should have executed some instructions");
        assert!(c.cycles > 0, "should have used some cycles");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn selfcounts_increases_with_work() {
        let before = read_self_counts().unwrap();
        // Do measurable work
        let mut x: f64 = 1.0;
        for _ in 0..1_000_000 {
            x = x * 1.0001 + 0.0001;
        }
        let _ = x; // prevent optimization
        let after = read_self_counts().unwrap();

        let delta_insn = after.instructions - before.instructions;
        let delta_cyc = after.cycles - before.cycles;

        assert!(delta_insn > 1_000_000, "should have retired >1M instructions, got {}", delta_insn);
        assert!(delta_cyc > 100_000, "should have used >100K cycles, got {}", delta_cyc);

        let ipc = delta_insn as f64 / delta_cyc as f64;
        assert!(ipc > 0.1 && ipc < 10.0, "IPC should be reasonable: {:.2}", ipc);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn measure_ipc_works() {
        let (_, ipc) = measure_ipc(|| {
            let mut x: f64 = 1.0;
            for _ in 0..1_000_000 {
                x = x * 1.0001 + 0.0001;
            }
            x
        });
        assert!(ipc > 0.1, "IPC should be measurable: {:.2}", ipc);
    }

    #[test]
    fn ipc_classification() {
        assert_eq!(IpcClass::from_ipc(0.1), IpcClass::MemoryBound);
        assert_eq!(IpcClass::from_ipc(0.5), IpcClass::Mixed);
        assert_eq!(IpcClass::from_ipc(1.5), IpcClass::ComputeBound);
        assert_eq!(IpcClass::from_ipc(2.5), IpcClass::Optimized);
        assert_eq!(IpcClass::from_ipc(0.0), IpcClass::Mixed);
        assert_eq!(IpcClass::from_ipc(-1.0), IpcClass::Mixed);
    }

    #[test]
    fn ipc_throttle_safety() {
        assert!(IpcClass::MemoryBound.safe_to_throttle());
        assert!(IpcClass::MemoryBound.safe_to_throttle_aggressive());
        assert!(IpcClass::Mixed.safe_to_throttle());
        assert!(!IpcClass::Mixed.safe_to_throttle_aggressive());
        assert!(!IpcClass::ComputeBound.safe_to_throttle());
        assert!(!IpcClass::Optimized.safe_to_throttle());
    }

    #[test]
    fn cycle_ipc_tracker_basics() {
        let mut tracker = CycleIpcTracker::new();
        assert_eq!(tracker.cycles_measured(), 0);
        let ipc = tracker.tick();
        assert!(ipc >= 0.0);
        assert_eq!(tracker.cycles_measured(), 1);
    }
}
