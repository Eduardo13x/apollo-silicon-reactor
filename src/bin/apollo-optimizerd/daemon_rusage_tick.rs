//! # Daemon Rusage Tick
//!
//! EMA interactivity classifier — cpu_wall_ratio computation per cycle, extracted
//! from main.rs (Wave 33). [Fowler 2004] Strangler Fig — pure move.
//!
//! ## Responsibilities
//! - Query proc_pid_rusage for each top_process
//! - Compute delta_cpu / delta_wall ratio (CPU-bound = high, I/O-bound = low)
//! - Guard against PID recycling via proc_start_abstime sentinel
//! - Update rusage_cpu_prev for the next cycle
//!
//! ## Ordering invariant
//! Must run AFTER snapshot is collected and BEFORE llm_daemon::usage_learning_tick
//! which consumes cpu_wall_ratios.

use std::collections::HashMap;
use std::time::Instant;

use apollo_engine::collector::SystemSnapshot;
use apollo_engine::engine::proc_taskinfo;

/// Compute per-process cpu_wall_ratio from rusage deltas.
///
/// # Parameters
/// - `snapshot` — system snapshot providing `top_processes` list
/// - `last_rusage_at` — mutable timestamp of last rusage sample (updated in-place)
/// - `rusage_cpu_prev` — previous cycle (user_ns, system_ns, start_abstime) per PID
///
/// Returns a map from process name → cpu_wall_ratio ∈ [0.0, 1.0].
pub fn compute_cpu_wall_ratios(
    snapshot: &SystemSnapshot,
    last_rusage_at: &mut Instant,
    rusage_cpu_prev: &mut HashMap<u32, (u64, u64, u64)>,
) -> HashMap<String, f32> {
    let elapsed_rusage = last_rusage_at.elapsed();
    *last_rusage_at = Instant::now();

    let mut cpu_wall_ratios: HashMap<String, f32> = HashMap::new();
    let mut new_rusage_prev: HashMap<u32, (u64, u64, u64)> = HashMap::new();

    for p in &snapshot.top_processes {
        if let Some(ri) = proc_taskinfo::get_rusage_info(p.pid) {
            let total_cpu = ri.user_time_ns + ri.system_time_ns;
            if let Some(&(prev_user, prev_system, prev_start)) = rusage_cpu_prev.get(&p.pid) {
                // PID recycling guard: proc_start_abstime change = different process.
                if ri.proc_start_abstime == prev_start {
                    let prev_total = prev_user + prev_system;
                    if total_cpu >= prev_total {
                        let delta_cpu = total_cpu - prev_total;
                        let delta_wall = elapsed_rusage.as_nanos() as u64;
                        if delta_wall > 0 {
                            let ratio = (delta_cpu as f64 / delta_wall as f64).min(1.0) as f32;
                            cpu_wall_ratios.insert(p.name.clone(), ratio);
                        }
                    }
                }
            }
            new_rusage_prev.insert(
                p.pid,
                (ri.user_time_ns, ri.system_time_ns, ri.proc_start_abstime),
            );
        }
    }
    *rusage_cpu_prev = new_rusage_prev;
    cpu_wall_ratios
}
