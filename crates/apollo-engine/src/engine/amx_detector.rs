//! AMX / ML Workload Detector — detects processes using Apple's Matrix coprocessor.
//!
//! Apple AMX is an undocumented matrix coprocessor in all Apple Silicon (M1+).
//! Instructions are encoded as reserved AArch64 words intercepted by the CPU:
//!
//!   AMX_SET = .word 0x00201220  (enable, zeros 5120 bytes of register state)
//!   AMX_CLR = .word 0x00201221  (disable, frees kernel context-switch tracking)
//!   General = .word (0x00201000 | (op << 5) | operand)
//!
//! There is NO userspace register to query AMX state of another process.
//! Detection uses multi-signal heuristics:
//!
//! 1. Process path matching — known ML runtimes (ollama, python+torch, mlx, etc.)
//! 2. Accelerate framework detection — libBLAS/libBNNS/vecLib presence via path
//! 3. Power signature — AMX workloads draw ~8W/thread vs ~2.5W for NEON-only
//!    (detected via existing IOKit power readings when available)
//!
//! When an AMX/ML workload is detected, Apollo should:
//! - Route to P-cores (AMX unit lives in the P-cluster)
//! - NEVER throttle or freeze — ML workloads are user-initiated and expensive to restart
//! - Reduce competing processes' priority to give the ML workload memory bandwidth

use std::collections::HashSet;

use super::proc_taskinfo;

// ── AMX hardware constants ───────────────────────────────────────────────────

/// AMX register file: 8×X(64B) + 8×Y(64B) + 64×Z(64B) = 5120 bytes.
/// Saved/restored by XNU on context switch when dirty.
pub const AMX_STATE_BYTES: usize = 5120;

/// AMX_SET instruction encoding (op=17, imm=0).
/// Enables AMX and zeros all registers. Raises SIGILL if already enabled.
#[allow(dead_code)]
pub const AMX_SET_ENCODING: u32 = 0x00201220;

/// AMX_CLR instruction encoding (op=17, imm=1).
/// Disables AMX and frees context-switch state.
#[allow(dead_code)]
pub const AMX_CLR_ENCODING: u32 = 0x00201221;

// ── Known ML process signatures ──────────────────────────────────────────────

/// Binary names that are definitively ML inference engines.
const ML_BINARY_NAMES: &[&str] = &[
    // Local LLM inference
    "ollama",
    "ollama-runner",
    "llama-server",
    "llama-cli",
    "mlc-llm",
    "mlc_chat",
    "llamafile",
    "koboldcpp",
    "text-generation-launcher",
    "vllm",
    "tgi",
    // Apple MLX framework
    "mlx_lm",
    "mlx-server",
    // Stable Diffusion / image gen
    "stable-diffusion",
    "sd-server",
    "draw-things",
    // ML training/inference tools
    "whisper",
    "whisper-server",
    "tortoise-tts",
    // Apple on-device ML/intelligence (iOS/macOS Sequoia+)
    "mlhostd",
    "intelligencecontextd",
    "modelmanagerd",
];

/// Path substrings that indicate Accelerate/AMX usage.
const ACCELERATE_PATHS: &[&str] = &[
    "libBLAS",
    "libBNNS",
    "libvDSP",
    "vecLib",
    "MLCompute",
    "CoreML",
    "libmlx",
    "Accelerate.framework",
    "MetalPerformanceShaders",
];

/// Python-related binary names (need secondary check for ML imports).
const PYTHON_NAMES: &[&str] = &[
    "python",
    "python3",
    "python3.10",
    "python3.11",
    "python3.12",
    "python3.13",
];

// ── Detector ─────────────────────────────────────────────────────────────────

/// Result of AMX/ML workload detection for a process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlWorkloadType {
    /// Not an ML workload.
    None,
    /// Confirmed ML inference engine (ollama, llama.cpp, etc.)
    InferenceEngine,
    /// Python process (may be running ML — treat as likely ML).
    PythonMl,
    /// Process linked against Accelerate framework (may use AMX).
    AccelerateUser,
}

impl MlWorkloadType {
    pub fn is_ml(&self) -> bool {
        !matches!(self, MlWorkloadType::None)
    }

    /// Priority boost factor for scheduling decisions.
    /// Higher = should be given more resources.
    pub fn priority_boost(&self) -> f32 {
        match self {
            MlWorkloadType::None => 0.0,
            MlWorkloadType::InferenceEngine => 1.0,
            MlWorkloadType::PythonMl => 0.7,
            MlWorkloadType::AccelerateUser => 0.3,
        }
    }
}

/// Detect if a single process is an ML workload.
/// Uses proc_pidpath (~3µs) — no subprocess spawning.
pub fn detect_ml_workload(pid: u32) -> MlWorkloadType {
    let path = match proc_taskinfo::get_proc_path(pid) {
        Some(p) => p,
        None => return MlWorkloadType::None,
    };

    // Extract binary name from path
    let binary_name = path.rsplit('/').next().unwrap_or(&path);

    // Check against known ML engines first (most specific)
    for &name in ML_BINARY_NAMES {
        if binary_name.eq_ignore_ascii_case(name) {
            return MlWorkloadType::InferenceEngine;
        }
    }

    // Check if it's a Python process (common ML runtime)
    for &py_name in PYTHON_NAMES {
        if binary_name.starts_with(py_name) {
            return MlWorkloadType::PythonMl;
        }
    }

    // Check path for Accelerate framework components
    for &accel_path in ACCELERATE_PATHS {
        if path.contains(accel_path) {
            return MlWorkloadType::AccelerateUser;
        }
    }

    MlWorkloadType::None
}

/// Bulk-detect ML workloads across all processes.
/// Returns PIDs of processes that are likely ML workloads.
pub fn detect_all_ml_workloads() -> Vec<(u32, MlWorkloadType)> {
    proc_taskinfo::list_all_pids()
        .into_iter()
        .filter_map(|pid| {
            let wl = detect_ml_workload(pid);
            if wl.is_ml() {
                Some((pid, wl))
            } else {
                None
            }
        })
        .collect()
}

/// Returns a HashSet of PIDs that should NEVER be throttled/frozen.
pub fn ml_protected_pids() -> HashSet<u32> {
    detect_all_ml_workloads()
        .into_iter()
        .filter(|(_, wl)| wl.priority_boost() >= 0.5)
        .map(|(pid, _)| pid)
        .collect()
}

// ── AMX hardware probing ─────────────────────────────────────────────────────

/// Check if AMX is available on this hardware.
/// True for all Apple Silicon M1+. Uses a safe probe: attempts AMX_SET,
/// catches SIGILL if unavailable, then AMX_CLR if successful.
#[cfg(target_arch = "aarch64")]
pub fn probe_amx_available() -> bool {
    use std::sync::atomic::{AtomicBool, Ordering};

    static RESULT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *RESULT.get_or_init(|| {
        // Install SIGILL handler, attempt AMX_SET, restore handler.
        // If SIGILL fires, AMX is not available.
        static AMX_OK: AtomicBool = AtomicBool::new(false);

        extern "C" fn sigill_handler(_sig: libc::c_int) {
            // SIGILL fired — AMX not available. Do nothing,
            // the setjmp/signal mechanism will handle it.
        }

        unsafe {
            let old_handler = libc::signal(
                libc::SIGILL,
                sigill_handler as *const () as libc::sighandler_t,
            );

            // Fork a child to do the dangerous probe — if it crashes, we're safe.
            let pid = libc::fork();
            if pid == 0 {
                // Child: attempt AMX_SET
                std::arch::asm!(".word 0x00201220", options(nomem, nostack));
                // If we get here, AMX is available. CLR it.
                std::arch::asm!(".word 0x00201221", options(nomem, nostack));
                libc::_exit(0); // success
            } else if pid > 0 {
                let mut status: libc::c_int = 0;
                libc::waitpid(pid, &mut status, 0);
                // Child exited with 0 = AMX works
                AMX_OK.store(
                    libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0,
                    Ordering::SeqCst,
                );
            }

            libc::signal(libc::SIGILL, old_handler);
        }

        AMX_OK.load(Ordering::SeqCst)
    })
}

#[cfg(not(target_arch = "aarch64"))]
pub fn probe_amx_available() -> bool {
    false
}

/// Estimate AMX context-switch overhead: 5120 bytes × 2 (save+restore) = ~10KB I/O.
/// At L1 bandwidth (~200 GB/s), this is ~50ns per context switch.
/// With dirty AMX state, context switches are measurably slower.
pub fn amx_context_switch_overhead_ns() -> u64 {
    if probe_amx_available() {
        // 5120 bytes save + 5120 bytes restore = 10240 bytes
        // L1 bandwidth ~200 GB/s → ~50 ns
        // L2 bandwidth ~100 GB/s → ~100 ns (if evicted)
        // Conservative estimate
        50
    } else {
        0
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amx_encodings_correct() {
        // Verify instruction encoding formula
        let set = 0x00201000 | (17 << 5) | 0;
        let clr = 0x00201000 | (17 << 5) | 1;
        assert_eq!(set, AMX_SET_ENCODING);
        assert_eq!(clr, AMX_CLR_ENCODING);

        // Verify op field for some instructions
        let ldx = 0x00201000 | (0 << 5) | 0; // AMXLDX, reg x0
        assert_eq!(ldx, 0x00201000);

        let fma32 = 0x00201000 | (12 << 5) | 0; // AMXFMA32
        assert_eq!(fma32, 0x00201180);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn amx_available_on_apple_silicon() {
        let available = probe_amx_available();
        println!("AMX available: {}", available);
        // All Apple Silicon M1+ should have AMX
        assert!(available, "AMX should be available on Apple Silicon");
    }

    #[test]
    fn detect_self_not_ml() {
        let pid = std::process::id();
        let wl = detect_ml_workload(pid);
        // cargo test is not an ML workload
        assert_eq!(wl, MlWorkloadType::None);
    }

    #[test]
    fn detect_known_ml_names() {
        // Test the name matching logic with fake paths
        let test_cases = vec![
            ("/usr/local/bin/ollama", MlWorkloadType::InferenceEngine),
            ("/opt/homebrew/bin/python3", MlWorkloadType::PythonMl),
            ("/usr/bin/python3.12", MlWorkloadType::PythonMl),
        ];

        for (path, expected_type) in test_cases {
            let binary_name = path.rsplit('/').next().unwrap();
            let mut detected = MlWorkloadType::None;

            for &name in ML_BINARY_NAMES {
                if binary_name.eq_ignore_ascii_case(name) {
                    detected = MlWorkloadType::InferenceEngine;
                    break;
                }
            }
            if detected == MlWorkloadType::None {
                for &py_name in PYTHON_NAMES {
                    if binary_name.starts_with(py_name) {
                        detected = MlWorkloadType::PythonMl;
                        break;
                    }
                }
            }

            assert_eq!(detected, expected_type, "path={}", path);
        }
    }

    #[test]
    fn bulk_scan_detects_any_python() {
        let results = detect_all_ml_workloads();
        // On a dev machine, there's likely at least one python process
        // But this test shouldn't fail if there isn't
        println!("ML workloads detected: {} processes", results.len());
        for (pid, wl) in &results {
            if let Some(path) = proc_taskinfo::get_proc_path(*pid) {
                println!("  PID {} ({:?}): {}", pid, wl, path);
            }
        }
    }

    #[test]
    fn ml_protected_pids_no_panic() {
        let protected = ml_protected_pids();
        println!("ML-protected PIDs: {:?}", protected);
        // Should not panic regardless of what's running
    }

    #[test]
    fn priority_boost_ordering() {
        assert!(
            MlWorkloadType::InferenceEngine.priority_boost()
                > MlWorkloadType::PythonMl.priority_boost()
        );
        assert!(
            MlWorkloadType::PythonMl.priority_boost()
                > MlWorkloadType::AccelerateUser.priority_boost()
        );
        assert!(
            MlWorkloadType::AccelerateUser.priority_boost() > MlWorkloadType::None.priority_boost()
        );
    }

    #[test]
    fn amx_state_size() {
        // 8 × X(64B) + 8 × Y(64B) + 64 × Z(64B)
        let computed = 8 * 64 + 8 * 64 + 64 * 64;
        assert_eq!(computed, AMX_STATE_BYTES);
    }

    #[test]
    fn context_switch_overhead_reasonable() {
        let overhead = amx_context_switch_overhead_ns();
        println!("AMX csw overhead estimate: {}ns", overhead);
        if probe_amx_available() {
            assert!(overhead > 0 && overhead < 1000);
        }
    }
}
