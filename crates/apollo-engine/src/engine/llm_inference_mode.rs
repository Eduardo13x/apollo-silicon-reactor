//! LLM Inference Mode — auto-detect local AI inference workloads.
//!
//! When a user runs Ollama, llama.cpp, MLX, LM Studio, or similar, the system
//! needs a fundamentally different resource allocation strategy:
//!
//! 1. **Prefer P-Cores**: The inference process is the highest-value compute
//!    on the system. Express that intent through Mach QoS while macOS retains
//!    final P/E-core placement authority.
//! 2. **Maximize unified memory headroom**: LLM weights can be 4–8 GB on an
//!    8 GB M1. File cache, inactive pages, and compressed memory all compete
//!    for the same pool. Aggressive reclaim during inference is critical.
//! 3. **Suppress interruptions**: Timer coalescing, background I/O, and daemon
//!    wakeups add latency variance to token generation. App-Nap all non-essential
//!    processes to minimize interference.
//! 4. **Disable Spotlight**: mds/mdworker scan new files and can cause 20-100ms
//!    I/O stalls that are directly felt as token-generation pauses.
//!
//! # Detection strategy
//!
//! We detect confirmed inference by process name (exact + prefix) since servers
//! tend to have stable names. A generic `python`/`python3` process can receive a
//! short QoS preference after sustained CPU, but never activates the global LLM
//! pressure/App-Nap mode without a stronger identity signal.
//!
//! # References
//!
//! - Apple Unified Memory Architecture (2020) — GPU and CPU share the same
//!   physical pool; memory pressure affects inference throughput directly.
//! - llama.cpp `README.md` — recommended flags for M1 (threads = P-core count).
//! - Dettmers et al. 2022, "LLM.int8()" — memory is the bottleneck, not FLOPS.

use std::time::{Duration, Instant};

// ── Process names ─────────────────────────────────────────────────────────────

/// Exact-match names that always indicate active LLM inference.
const LLM_EXACT_NAMES: &[&str] = &[
    "ollama",
    "ollama_llama_server",
    "llama-server",
    "llama-cli",
    "llama-run",
    "llamafile",
    "koboldcpp",
    "koboldcpp.py",
    "lm_studio",
    "LM Studio",
    "jan",
    "jan-main",
    "mlx_lm",
    "mlx_lm.generate",
];

/// Prefix matches — process names starting with these strings.
const LLM_PREFIX_NAMES: &[&str] = &[
    "ollama_", // ollama sub-processes
    "llama",   // llama.cpp variants
    "mlx_",    // Apple MLX variants
];

/// Minimum sustained CPU% for python/python3 to receive a local QoS preference.
const PYTHON_ML_QOS_CPU_THRESHOLD: f32 = 50.0;

/// Number of consecutive cycles python must exceed the threshold.
const PYTHON_SUSTAINED_CYCLES: u32 = 3;

// ── State ─────────────────────────────────────────────────────────────────────

/// A detected LLM inference process.
#[derive(Debug, Clone)]
pub struct LlmProcess {
    pub pid: u32,
    pub name: String,
    pub cpu_usage: f32,
}

/// Active LLM inference state.
#[derive(Debug, Clone)]
pub struct LlmInferenceState {
    /// The primary inference process (highest CPU).
    pub primary: LlmProcess,
    /// How long inference has been active.
    pub active_since: Instant,
    /// Pressure boost to add to memory_pressure when in this mode.
    /// Raises all gates so Apollo reacts more aggressively.
    pub pressure_boost: f64,
}

// ── Detector ─────────────────────────────────────────────────────────────────

/// Detects active LLM inference workloads from the running process list.
pub struct LlmInferenceDetector {
    /// Consecutive cycles python has been over threshold.
    python_high_cycles: u32,
    /// Current active state.
    active: Option<LlmInferenceState>,
    /// How long to keep state active after all LLM processes disappear.
    /// Prevents thrashing when inference briefly pauses between prompts.
    cooldown: Duration,
    /// When the last LLM process was seen.
    last_seen: Option<Instant>,
    /// Process positively observed in the current cycle. Unlike `active`, this
    /// is cleared during the cooldown so callers never steer a stale PID.
    observed_primary: Option<LlmProcess>,
}

impl LlmInferenceDetector {
    pub fn new() -> Self {
        Self {
            python_high_cycles: 0,
            active: None,
            cooldown: Duration::from_secs(30),
            last_seen: None,
            observed_primary: None,
        }
    }

    /// Scan the process list and update detection state.
    ///
    /// `processes`: iterator of (pid, name, cpu_usage_percent).
    /// Returns the current `LlmInferenceState` if active, None otherwise.
    pub fn observe<'a>(
        &mut self,
        processes: impl Iterator<Item = (u32, &'a str, f32)>,
    ) -> Option<&LlmInferenceState> {
        let mut best: Option<LlmProcess> = None;
        let mut hottest_python: Option<LlmProcess> = None;

        for (pid, name, cpu) in processes {
            let is_llm = Self::is_llm_by_name(name);

            // Python heuristic: sustained high CPU.
            let is_python = name == "python" || name == "python3" || name.starts_with("python3.");
            if is_python && cpu >= PYTHON_ML_QOS_CPU_THRESHOLD {
                if hottest_python
                    .as_ref()
                    .is_none_or(|current| cpu > current.cpu_usage)
                {
                    hottest_python = Some(LlmProcess {
                        pid,
                        name: name.to_string(),
                        cpu_usage: cpu,
                    });
                }
            }

            if is_llm {
                match &best {
                    None => {
                        best = Some(LlmProcess {
                            pid,
                            name: name.to_string(),
                            cpu_usage: cpu,
                        })
                    }
                    Some(b) if cpu > b.cpu_usage => {
                        best = Some(LlmProcess {
                            pid,
                            name: name.to_string(),
                            cpu_usage: cpu,
                        })
                    }
                    _ => {}
                }
            }
        }

        // Python sustained-CPU heuristic.
        if hottest_python.is_some() {
            self.python_high_cycles = self.python_high_cycles.saturating_add(1);
        } else {
            self.python_high_cycles = 0;
        }
        let python_qos_candidate = (self.python_high_cycles >= PYTHON_SUSTAINED_CYCLES)
            .then_some(hottest_python)
            .flatten();
        self.observed_primary = best.clone().or(python_qos_candidate);

        // Update state.
        if let Some(proc) = best {
            self.last_seen = Some(Instant::now());
            let now = Instant::now();
            if let Some(ref mut state) = self.active {
                state.primary = proc;
            } else {
                self.active = Some(LlmInferenceState {
                    active_since: now,
                    pressure_boost: 0.20, // +20pp raises all gates significantly
                    primary: proc,
                });
                println!(
                    "[llm-mode] Inference detected: {} (cpu={:.0}%) — activating aggressive mode",
                    self.active.as_ref().unwrap().primary.name,
                    self.active.as_ref().unwrap().primary.cpu_usage,
                );
            }
        } else {
            // No LLM process — check cooldown before clearing.
            let expired = self
                .last_seen
                .map(|t| t.elapsed() > self.cooldown)
                .unwrap_or(true);
            if expired {
                if self.active.is_some() {
                    println!("[llm-mode] Inference ended — restoring normal mode");
                }
                self.active = None;
            }
        }

        self.active.as_ref()
    }

    /// Whether LLM inference is currently active.
    pub fn is_active(&self) -> bool {
        self.active.is_some()
    }

    /// Pressure boost to add when active (0.0 when not active).
    pub fn pressure_boost(&self) -> f64 {
        self.active
            .as_ref()
            .map(|s| s.pressure_boost)
            .unwrap_or(0.0)
    }

    /// Current-cycle inference process. This deliberately excludes the
    /// detector's cooldown-only state so kernel actuation never follows a PID
    /// that has disappeared or been recycled.
    pub fn observed_primary(&self) -> Option<&LlmProcess> {
        self.observed_primary.as_ref()
    }

    fn is_llm_by_name(name: &str) -> bool {
        // OnceLock pre-lowered exact set — O(1) lookup vs O(N) per-call lowercase chain.
        static EXACT_LC: std::sync::OnceLock<std::collections::HashSet<String>> =
            std::sync::OnceLock::new();
        let exact = EXACT_LC.get_or_init(|| {
            LLM_EXACT_NAMES
                .iter()
                .map(|n| n.to_ascii_lowercase())
                .collect()
        });
        let name_lc = name.to_ascii_lowercase();
        if exact.contains(&name_lc) {
            return true;
        }
        LLM_PREFIX_NAMES.iter().any(|&p| name_lc.starts_with(p))
    }
}

impl Default for LlmInferenceDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_ollama() {
        let mut d = LlmInferenceDetector::new();
        let procs = vec![(1234u32, "ollama", 45.0f32)];
        let state = d.observe(procs.iter().map(|(p, n, c)| (*p, *n, *c)));
        assert!(state.is_some());
        assert_eq!(state.unwrap().primary.name, "ollama");
    }

    #[test]
    fn detects_llama_server() {
        let mut d = LlmInferenceDetector::new();
        let procs = vec![(999u32, "llama-server", 80.0f32)];
        let state = d.observe(procs.iter().map(|(p, n, c)| (*p, *n, *c)));
        assert!(state.is_some());
    }

    #[test]
    fn detects_mlx_prefix() {
        let mut d = LlmInferenceDetector::new();
        let procs = vec![(555u32, "mlx_lm.generate", 60.0f32)];
        let state = d.observe(procs.iter().map(|(p, n, c)| (*p, *n, *c)));
        assert!(state.is_some());
    }

    #[test]
    fn no_false_positive_on_background_python() {
        let mut d = LlmInferenceDetector::new();
        // Low-CPU python should not trigger
        let procs = vec![(100u32, "python3", 0.5f32)];
        let state = d.observe(procs.iter().map(|(p, n, c)| (*p, *n, *c)));
        assert!(state.is_none());
    }

    #[test]
    fn sustained_hot_python_selects_qos_target_without_global_llm_mode() {
        let mut d = LlmInferenceDetector::new();
        for _ in 0..(PYTHON_SUSTAINED_CYCLES - 1) {
            let state = d.observe(std::iter::once((7331, "python3", 75.0)));
            assert!(state.is_none());
            assert!(d.observed_primary().is_none());
        }

        let state = d.observe(std::iter::once((7331, "python3", 75.0)));
        assert!(
            state.is_none(),
            "generic Python must not trigger global LLM mode"
        );
        assert_eq!(d.observed_primary().map(|p| p.pid), Some(7331));
    }

    #[test]
    fn cooldown_state_does_not_expose_a_stale_primary_for_actuation() {
        let mut d = LlmInferenceDetector::new();
        d.observe(std::iter::once((4444, "ollama", 30.0)));
        assert_eq!(d.observed_primary().map(|p| p.pid), Some(4444));

        d.observe(std::iter::empty());
        assert!(d.is_active(), "cooldown keeps the behavioral mode active");
        assert!(
            d.observed_primary().is_none(),
            "cooldown must not keep a kernel-actuation target"
        );
    }

    #[test]
    fn cooldown_keeps_active_briefly_after_process_dies() {
        let mut d = LlmInferenceDetector::new();
        // Activate
        let procs = vec![(1234u32, "ollama", 50.0f32)];
        d.observe(procs.iter().map(|(p, n, c)| (*p, *n, *c)));
        assert!(d.is_active());
        // Process disappears but cooldown hasn't expired
        let empty: Vec<(u32, &str, f32)> = vec![];
        let state = d.observe(empty.iter().map(|(p, n, c)| (*p, *n, *c)));
        // Still active due to cooldown
        assert!(state.is_some());
    }

    #[test]
    fn pressure_boost_is_nonzero_when_active() {
        let mut d = LlmInferenceDetector::new();
        let procs = vec![(1234u32, "ollama", 50.0f32)];
        d.observe(procs.iter().map(|(p, n, c)| (*p, *n, *c)));
        assert!(d.pressure_boost() > 0.0);
    }

    #[test]
    fn pressure_boost_is_zero_when_idle() {
        let d = LlmInferenceDetector::new();
        assert_eq!(d.pressure_boost(), 0.0);
    }

    #[test]
    fn llama_prefix_match() {
        assert!(LlmInferenceDetector::is_llm_by_name("llama-cli"));
        assert!(LlmInferenceDetector::is_llm_by_name("llama-server"));
        assert!(LlmInferenceDetector::is_llm_by_name("llamafile"));
        assert!(!LlmInferenceDetector::is_llm_by_name("launchd"));
        assert!(!LlmInferenceDetector::is_llm_by_name("python3"));
    }
}
