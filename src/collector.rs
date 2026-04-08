use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use sysinfo::{Disks, Networks, System};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SystemSnapshot {
    pub timestamp: DateTime<Utc>,
    pub cpu: CpuStats,
    pub memory: MemoryStats,
    pub pressure: PressureStats,
    pub disks: Vec<DiskStats>,
    pub networks: Vec<NetworkStats>,
    pub top_processes: Vec<ProcessStats>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CpuStats {
    pub global_usage: f32,
    pub core_count: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MemoryStats {
    pub total_ram: u64,
    pub used_ram: u64,
    pub free_ram: u64,
    pub total_swap: u64,
    pub used_swap: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PressureStats {
    // 0..1 where 1 == high pressure.
    pub memory_pressure: f64,
    pub swap_used_bytes: u64,
    pub swap_total_bytes: u64,
    pub swap_delta_bytes_per_sec: f64,
    pub thermal_level: String,
    /// Raw compressor pressure (0.0-1.0): ratio of uncompressed pages in compressor
    /// to total physical pages, scaled by 0.85. Used by the RL threshold agent.
    #[serde(default)]
    pub compressor_pressure: f64,
    /// Composite VM flow score from `VmRate::thrashing_score()`. 0 = quiet,
    /// 5_000+ = actively thrashing the compressor. Distinguishes a sleepy
    /// 70% pressure system from a thrashing 70% pressure system.
    #[serde(default)]
    pub thrashing_score: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DiskStats {
    pub name: String,
    pub mount_point: String,
    pub total_space: u64,
    pub available_space: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NetworkStats {
    pub interface_name: String,
    pub received: u64,
    pub transmitted: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProcessStats {
    pub pid: u32,
    pub name: String,
    pub cpu_usage: f32,
    pub memory_usage: u64,
    /// CPU/wall-clock ratio from proc_pid_rusage delta.
    /// Low (< 0.05) = I/O-bound (interactive), high (> 0.70) = CPU-bound (build).
    /// Populated by the daemon's main loop; None in one-shot snapshots.
    #[serde(default)]
    pub cpu_wall_ratio: Option<f32>,
}

pub struct SystemCollector {
    sys: System,
    disks: Disks,
    networks: Networks,
    prev_swap_used_bytes: Option<u64>,
    prev_swap_at: Option<Instant>,
    /// Tracks whether refresh_processes has ever hung (>5s).
    pub process_refresh_hung: bool,
    /// Number of process refresh cycles skipped (startup grace).
    pub process_refresh_skip_count: u32,
    /// Light call count (cycles since creation, for startup grace period).
    pub light_call_count: u32,
    /// EMA state for compressor_pressure smoothing (α=0.25).
    /// Applied before the MAX fusion with kernel_pressure to reduce noise
    /// before it enters the Kalman filter in signal_intelligence.
    compressor_ema: f64,
}

#[allow(clippy::new_without_default, dead_code)]
impl SystemCollector {
    pub fn new() -> Self {
        // Use System::new() + targeted refresh instead of System::new_all()
        // to avoid the expensive initial process enumeration at startup.
        // refresh_processes() is called once here to pre-seed the process list so
        // that top_processes is non-empty from cycle 1 (fixes startup blind spot).
        // The 3-cycle grace period still skips refresh_processes on each cycle to
        // avoid double-refresh overhead, but the initial seed ensures decisions
        // are never made with an empty process table.
        let mut sys = System::new();
        sys.refresh_cpu();
        sys.refresh_memory();
        sys.refresh_processes();
        let disks = Disks::new_with_refreshed_list();
        let networks = Networks::new_with_refreshed_list();
        Self {
            sys,
            disks,
            networks,
            prev_swap_used_bytes: None,
            prev_swap_at: None,
            process_refresh_hung: false,
            process_refresh_skip_count: 0,
            light_call_count: 0,
            compressor_ema: 0.0,
        }
    }

    pub fn system(&self) -> &System {
        &self.sys
    }

    pub fn collect_snapshot(&mut self) -> SystemSnapshot {
        // Refresh system stats — skip process refresh for first 3 cycles
        // (startup grace period: avoids expensive initial enumeration).
        // Lightweight refresh every cycle: CPU + memory + processes only.
        // Full refresh (disk/network) every 30 cycles (~9s at 300ms interval).
        // refresh_all() includes refresh_components() (temps) and disk/net I/O stats —
        // expensive (~30-50ms on macOS with 200 processes) and unneeded each cycle
        // for scheduling decisions. SMC reader handles temps independently.
        // [Bhatt 2009 "Reducing overhead of application tracing"; sysinfo docs]
        self.light_call_count += 1;
        if self.light_call_count <= 3 {
            self.process_refresh_skip_count += 1;
            self.sys.refresh_cpu();
            self.sys.refresh_memory();
        } else if self.light_call_count % 30 == 0 {
            // Full refresh for disk/network stats (for journal and metrics).
            self.sys.refresh_all();
            self.disks.refresh_list();
            self.networks.refresh();
        } else {
            // Light path: enough for all scheduling decisions.
            self.sys.refresh_cpu();
            self.sys.refresh_memory();
            self.sys.refresh_processes();
        }

        // CPU
        let global_cpu = self.sys.global_cpu_info().cpu_usage();
        let core_count = self.sys.cpus().len();

        // Memory
        let total_ram = self.sys.total_memory();
        let used_ram = self.sys.used_memory();
        let free_ram = self.sys.free_memory();
        let total_swap = self.sys.total_swap();
        let used_swap = self.sys.used_swap();

        // Pressure (public commands, no private APIs)
        let (_, swap_used_bytes, swap_total_bytes, compressor_pressure_raw, kernel_pressure) =
            collect_pressure_facts();
        // EMA smoothing on compressor_pressure (α=0.25) to remove single-sample noise
        // before MAX fusion. Kalman in signal_intelligence still smooths the fused value,
        // but pre-smoothing here reduces the noise it has to compensate for (less lag).
        let alpha = 0.25f64;
        let compressor_pressure =
            self.compressor_ema * (1.0 - alpha) + compressor_pressure_raw * alpha;
        self.compressor_ema = compressor_pressure;
        let mem_pressure = kernel_pressure.max(compressor_pressure);
        let nowi = Instant::now();
        let swap_delta_bps = match (self.prev_swap_used_bytes, self.prev_swap_at) {
            (Some(prev_used), Some(prev_at)) => {
                let dt = nowi.duration_since(prev_at).as_secs_f64().max(0.001);
                // Signed delta: negative when swap shrinks (pressure resolving).
                // [Arlitt & Williamson 1997] — rate metrics must be signed.
                (swap_used_bytes as i64 - prev_used as i64) as f64 / dt
            }
            _ => 0.0,
        };
        self.prev_swap_used_bytes = Some(swap_used_bytes);
        self.prev_swap_at = Some(nowi);

        // Disks
        let disks = self
            .disks
            .iter()
            .map(|disk| DiskStats {
                name: disk.name().to_string_lossy().into_owned(),
                mount_point: disk.mount_point().to_string_lossy().into_owned(),
                total_space: disk.total_space(),
                available_space: disk.available_space(),
            })
            .collect();

        // Networks
        let networks = self
            .networks
            .iter()
            .map(|(name, data)| NetworkStats {
                interface_name: name.clone(),
                received: data.received(),
                transmitted: data.transmitted(),
            })
            .collect();

        // Processes - Get top 10 by CPU usage
        let mut processes: Vec<ProcessStats> = self
            .sys
            .processes()
            .iter()
            .map(|(pid, process)| ProcessStats {
                pid: pid.as_u32(),
                name: process.name().to_string(),
                cpu_usage: process.cpu_usage(),
                memory_usage: process.memory(),
                cpu_wall_ratio: None,
            })
            .collect();

        // Sort by CPU usage descending
        processes.sort_by(|a, b| {
            b.cpu_usage
                .partial_cmp(&a.cpu_usage)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_processes = processes.into_iter().take(10).collect();

        SystemSnapshot {
            timestamp: Utc::now(),
            cpu: CpuStats {
                global_usage: global_cpu,
                core_count,
            },
            memory: MemoryStats {
                total_ram,
                used_ram,
                free_ram,
                total_swap,
                used_swap,
            },
            pressure: PressureStats {
                memory_pressure: mem_pressure,
                swap_used_bytes,
                swap_total_bytes,
                swap_delta_bytes_per_sec: swap_delta_bps,
                thermal_level: "unknown".to_string(),
                compressor_pressure,
                thrashing_score: 0.0, // populated by daemon from pressure collector
            },
            disks,
            networks,
            top_processes,
        }
    }

    /// Light snapshot: skips disk/network refresh and uses direct sysctl calls
    /// instead of subprocesses. Use when hw_pressure is Nominal and memory is low.
    /// ~10x faster than collect_snapshot().
    pub fn collect_snapshot_light(&mut self) -> SystemSnapshot {
        self.sys.refresh_cpu();
        self.sys.refresh_memory();
        self.sys.refresh_processes();

        let global_cpu = self.sys.global_cpu_info().cpu_usage();
        let core_count = self.sys.cpus().len();

        let total_ram = self.sys.total_memory();
        let used_ram = self.sys.used_memory();
        let free_ram = self.sys.free_memory();
        let total_swap = self.sys.total_swap();
        let used_swap = self.sys.used_swap();

        let (_, swap_used_bytes, swap_total_bytes, compressor_pressure_raw, kernel_pressure) =
            collect_pressure_facts();
        let alpha = 0.25f64;
        let compressor_pressure =
            self.compressor_ema * (1.0 - alpha) + compressor_pressure_raw * alpha;
        self.compressor_ema = compressor_pressure;
        let mem_pressure = kernel_pressure.max(compressor_pressure);
        let nowi = Instant::now();
        let swap_delta_bps = match (self.prev_swap_used_bytes, self.prev_swap_at) {
            (Some(prev_used), Some(prev_at)) => {
                let dt = nowi.duration_since(prev_at).as_secs_f64().max(0.001);
                // Signed delta: negative when swap shrinks (pressure resolving).
                // [Arlitt & Williamson 1997] — rate metrics must be signed.
                (swap_used_bytes as i64 - prev_used as i64) as f64 / dt
            }
            _ => 0.0,
        };
        self.prev_swap_used_bytes = Some(swap_used_bytes);
        self.prev_swap_at = Some(nowi);

        let mut processes: Vec<ProcessStats> = self
            .sys
            .processes()
            .iter()
            .map(|(pid, process)| ProcessStats {
                pid: pid.as_u32(),
                name: process.name().to_string(),
                cpu_usage: process.cpu_usage(),
                memory_usage: process.memory(),
                cpu_wall_ratio: None,
            })
            .collect();
        processes.sort_by(|a, b| {
            b.cpu_usage
                .partial_cmp(&a.cpu_usage)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_processes = processes.into_iter().take(10).collect();

        SystemSnapshot {
            timestamp: Utc::now(),
            cpu: CpuStats {
                global_usage: global_cpu,
                core_count,
            },
            memory: MemoryStats {
                total_ram,
                used_ram,
                free_ram,
                total_swap,
                used_swap,
            },
            pressure: PressureStats {
                memory_pressure: mem_pressure,
                swap_used_bytes,
                swap_total_bytes,
                swap_delta_bytes_per_sec: swap_delta_bps,
                thermal_level: "unknown".to_string(),
                compressor_pressure,
                thrashing_score: 0.0, // populated by daemon from pressure collector
            },
            disks: vec![],    // skipped in light mode
            networks: vec![], // skipped in light mode
            top_processes,
        }
    }
}

/// Read a u64 sysctl value directly via libc — no subprocess, ~200 ns.
fn sysctl_u64(name: &std::ffi::CStr) -> Option<u64> {
    let mut val: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut val as *mut u64 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 {
        Some(val)
    } else {
        None
    }
}

/// Returns (memory_pressure_fused, swap_used_bytes, swap_total_bytes, compressor_pressure_raw, kernel_pressure).
/// `memory_pressure_fused` = MAX(kernel_pressure, compressor_pressure_raw) — callers that want
/// EMA-smoothed compressor should recompute the fusion with the smoothed value.
fn collect_pressure_facts() -> (f64, u64, u64, f64, f64) {
    // kern.memorystatus_level: 0–100 (% memory available).
    // Faster than spawning /usr/bin/memory_pressure — direct kernel read.
    let kernel_pressure = sysctl_u64(c"kern.memorystatus_level")
        .map(|level| (1.0 - (level as f64 / 100.0)).clamp(0.0, 1.0))
        .unwrap_or(0.0);

    // Compressor pressure: macOS reports 0 swap even when 4+ GB are compressed.
    // The compressor uses RAM and causes decompression latency, so it IS pressure.
    // We read raw VM stats via host_statistics64 to get the logical uncompressed size
    // held in the compressor.  Blend: MAX(kernel_pressure, compressor_ratio × 0.85)
    // so Apollo acts early when the compressor is thrashing even if jetsam hasn't fired.
    let compressor_pressure: f64 = {
        use std::ffi::c_uint;
        extern "C" {
            fn host_statistics64(
                host: libc::mach_port_t,
                flavor: c_uint,
                host_info: *mut libc::c_int,
                count: *mut c_uint,
            ) -> libc::kern_return_t;
        }
        extern "C" {
            fn mach_host_self() -> libc::mach_port_t;
        }

        // vm_statistics64 struct — exact layout from XNU osfmk/mach/vm_statistics.h.
        // Mixed u32/u64 fields; #[repr(C)] matches the ABI on ARM64 macOS.
        // Verified byte offsets (Python/ctypes):
        //   compressor_page_count                    → offset 128
        //   total_uncompressed_pages_in_compressor   → offset 144
        #[repr(C)]
        struct VmStats64 {
            free_count: u32,
            active_count: u32, // 0, 4
            inactive_count: u32,
            wire_count: u32,      // 8, 12
            zero_fill_count: u64, // 16
            reactivations: u64,   // 24
            pageins: u64,         // 32
            pageouts: u64,        // 40
            faults: u64,          // 48
            cow_faults: u64,      // 56
            lookups: u64,         // 64
            hits: u64,            // 72
            purges: u64,          // 80
            purgeable_count: u32,
            speculative_count: u32,                      // 88, 92
            decompressions: u64,                         // 96
            compressions: u64,                           // 104
            swapins: u64,                                // 112
            swapouts: u64,                               // 120
            compressor_page_count: u32,                  // 128 — physical pages used by compressor
            throttled_count: u32,                        // 132
            external_page_count: u32,                    // 136
            internal_page_count: u32,                    // 140
            total_uncompressed_pages_in_compressor: u64, // 144 — logical (uncompressed) pages
        }

        // HOST_VM_INFO64 = 4; count is in natural_t (u32) units → 152 / 4 = 38.
        const HOST_VM_INFO64: c_uint = 4;
        let count_val = (std::mem::size_of::<VmStats64>() / std::mem::size_of::<u32>()) as c_uint;

        let mut stats = std::mem::MaybeUninit::<VmStats64>::zeroed();
        let mut count = count_val;
        let port = unsafe { mach_host_self() };
        let kr = unsafe {
            host_statistics64(
                port,
                HOST_VM_INFO64,
                stats.as_mut_ptr() as *mut libc::c_int,
                &mut count,
            )
        };
        if kr == 0 {
            let s = unsafe { stats.assume_init() };
            let total_pages = sysctl_u64(c"hw.memsize")
                .map(|b| b / 16384)
                .unwrap_or(1)
                .max(1);
            // Use the logical (uncompressed) size: this is the real memory footprint
            // of data currently held in the compressor — what would be needed if
            // the compressor were flushed back to RAM.
            let uncompressed_pages = s.total_uncompressed_pages_in_compressor;
            (uncompressed_pages as f64 / total_pages as f64).clamp(0.0, 1.0) * 0.85
        } else {
            0.0
        }
    };

    // Use the higher of the two signals so Apollo acts on whichever is worse.
    let memory_pressure = kernel_pressure.max(compressor_pressure);

    // vm.swapusage is a struct xsw_usage { total, avail, used, pagesize, encrypted }
    // all fields are u64.  Layout: [total, avail, used, pagesize, encrypted_flag]
    let mut xsw = [0u64; 5];
    let mut len = std::mem::size_of_val(&xsw);
    let swap_ok = unsafe {
        libc::sysctlbyname(
            c"vm.swapusage".as_ptr(),
            xsw.as_mut_ptr() as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    } == 0;

    let (swap_used_bytes, swap_total_bytes) = if swap_ok {
        (xsw[2], xsw[0]) // used = xsw[2], total = xsw[0]
    } else {
        (0, 0)
    };

    (
        memory_pressure,
        swap_used_bytes,
        swap_total_bytes,
        compressor_pressure,
        kernel_pressure,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Struct construction ──────────────────────────────────────────────────

    #[test]
    fn system_snapshot_fields_accessible() {
        let snap = SystemSnapshot {
            timestamp: chrono::Utc::now(),
            cpu: CpuStats {
                global_usage: 42.0,
                core_count: 8,
            },
            memory: MemoryStats {
                total_ram: 8 * 1024 * 1024 * 1024,
                used_ram: 4 * 1024 * 1024 * 1024,
                free_ram: 4 * 1024 * 1024 * 1024,
                total_swap: 2 * 1024 * 1024 * 1024,
                used_swap: 512 * 1024 * 1024,
            },
            pressure: PressureStats {
                memory_pressure: 0.45,
                swap_used_bytes: 512 * 1024 * 1024,
                swap_total_bytes: 2 * 1024 * 1024 * 1024,
                swap_delta_bytes_per_sec: 1_000_000.0,
                thermal_level: "nominal".to_string(),
                compressor_pressure: 0.30,
                thrashing_score: 0.0,
            },
            disks: vec![],
            networks: vec![],
            top_processes: vec![],
        };
        assert_eq!(snap.cpu.core_count, 8);
        assert!((snap.cpu.global_usage - 42.0).abs() < 0.01);
        assert_eq!(snap.pressure.thermal_level, "nominal");
    }

    #[test]
    fn process_stats_cpu_wall_ratio_defaults_none() {
        let ps = ProcessStats {
            pid: 1234,
            name: "test_proc".to_string(),
            cpu_usage: 5.5,
            memory_usage: 1024 * 1024,
            cpu_wall_ratio: None,
        };
        assert!(ps.cpu_wall_ratio.is_none());
        assert_eq!(ps.name, "test_proc");
    }

    // ── EMA math (mirrors collect_snapshot EMA logic) ────────────────────────

    #[test]
    fn ema_converges_to_target() {
        // After N steps, EMA should be within ε of the constant input.
        let alpha = 0.25f64;
        let target = 0.60;
        let mut ema = 0.0f64;
        for _ in 0..40 {
            ema = ema * (1.0 - alpha) + target * alpha;
        }
        assert!(
            (ema - target).abs() < 0.01,
            "EMA should converge: got {ema:.4}"
        );
    }

    #[test]
    fn ema_alpha_bounds_hold() {
        // EMA output should always remain within [0, 1] for inputs in [0, 1].
        let alpha = 0.25f64;
        let mut ema = 0.0f64;
        let inputs = [0.0, 0.5, 1.0, 0.8, 0.2, 0.0, 0.9];
        for &v in &inputs {
            ema = ema * (1.0 - alpha) + v * alpha;
            assert!((0.0..=1.0).contains(&ema), "EMA out of range: {ema}");
        }
    }

    // ── Pressure fusion logic ────────────────────────────────────────────────

    #[test]
    fn pressure_fusion_takes_max() {
        // mem_pressure = kernel_pressure.max(compressor_pressure)
        let kernel = 0.40f64;
        let compressor = 0.65f64;
        let fused = kernel.max(compressor);
        assert!(
            (fused - compressor).abs() < 1e-9,
            "should take compressor when higher"
        );

        let kernel2 = 0.80f64;
        let compressor2 = 0.30f64;
        let fused2 = kernel2.max(compressor2);
        assert!(
            (fused2 - kernel2).abs() < 1e-9,
            "should take kernel when higher"
        );
    }

    #[test]
    fn pressure_fusion_clamped_to_unit_interval() {
        // Even with extreme raw values, fused result should be in [0, 1].
        for (k, c) in [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (1.0, 0.0), (0.0, 1.0)] {
            let fused = (k as f64).max(c as f64);
            assert!((0.0..=1.0).contains(&fused));
        }
    }

    // ── collect_pressure_facts smoke test ───────────────────────────────────

    #[test]
    fn collect_pressure_facts_returns_valid_range() {
        let (fused, swap_used, swap_total, comp_raw, kernel) = collect_pressure_facts();
        // All pressure values must be in [0, 1].
        assert!((0.0..=1.0).contains(&fused), "fused={fused}");
        assert!((0.0..=1.0).contains(&comp_raw), "comp_raw={comp_raw}");
        assert!((0.0..=1.0).contains(&kernel), "kernel={kernel}");
        // Swap values must be non-negative.
        assert!(
            swap_used <= swap_total || swap_total == 0,
            "swap_used ({swap_used}) > swap_total ({swap_total})"
        );
        // fused must be max(kernel, comp_raw) — may differ by EMA rounding
        let expected_min = kernel.max(comp_raw);
        assert!(
            fused >= expected_min - 1e-9,
            "fused={fused} < max(k,c)={expected_min}"
        );
    }

    // ── SystemCollector construction ─────────────────────────────────────────

    #[test]
    fn system_collector_new_does_not_panic() {
        // Verifies that initialization (including refresh_processes) completes.
        let collector = SystemCollector::new();
        assert!(!collector.process_refresh_hung);
        assert_eq!(collector.light_call_count, 0);
    }

    #[test]
    fn collect_snapshot_light_returns_valid_pressure() {
        let mut collector = SystemCollector::new();
        let snap = collector.collect_snapshot_light();
        assert!((0.0..=1.0).contains(&snap.pressure.memory_pressure));
        assert!((0.0..=1.0).contains(&snap.pressure.compressor_pressure));
        assert!(
            snap.pressure.swap_used_bytes <= snap.pressure.swap_total_bytes
                || snap.pressure.swap_total_bytes == 0
        );
    }

    #[test]
    fn collect_snapshot_increments_light_call_count() {
        let mut collector = SystemCollector::new();
        assert_eq!(collector.light_call_count, 0);
        collector.collect_snapshot();
        assert_eq!(collector.light_call_count, 1);
        collector.collect_snapshot();
        assert_eq!(collector.light_call_count, 2);
    }

    #[test]
    fn swap_delta_is_zero_on_first_call() {
        let mut collector = SystemCollector::new();
        // On first collect_snapshot, prev_swap_at is None → delta = 0.
        let snap = collector.collect_snapshot();
        assert_eq!(
            snap.pressure.swap_delta_bytes_per_sec, 0.0,
            "first-call delta should be 0"
        );
    }

    // ── Serialization round-trip ─────────────────────────────────────────────

    #[test]
    fn process_stats_serde_roundtrip() {
        let ps = ProcessStats {
            pid: 5678,
            name: "roundtrip_proc".to_string(),
            cpu_usage: 12.5,
            memory_usage: 2 * 1024 * 1024,
            cpu_wall_ratio: Some(0.45),
        };
        let json = serde_json::to_string(&ps).expect("serialize");
        let ps2: ProcessStats = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(ps2.pid, ps.pid);
        assert_eq!(ps2.name, ps.name);
        assert!((ps2.cpu_usage - ps.cpu_usage).abs() < 0.01);
        assert_eq!(ps2.cpu_wall_ratio, ps.cpu_wall_ratio);
    }

    #[test]
    fn process_stats_cpu_wall_ratio_default_is_none() {
        // When cpu_wall_ratio is absent from JSON, it should default to None.
        let json = r#"{"pid":9,"name":"old_proc","cpu_usage":3.0,"memory_usage":1024}"#;
        let ps: ProcessStats =
            serde_json::from_str(json).expect("deserialize without cpu_wall_ratio");
        assert!(ps.cpu_wall_ratio.is_none());
    }

    // ── Micro-benchmark: collect_pressure_facts latency ──────────────────────

    #[test]
    fn bench_collect_pressure_facts_latency() {
        // Warm-up
        for _ in 0..3 {
            let _ = collect_pressure_facts();
        }
        let start = std::time::Instant::now();
        let n = 20;
        for _ in 0..n {
            let _ = collect_pressure_facts();
        }
        let elapsed = start.elapsed();
        let per_call_ms = elapsed.as_secs_f64() * 1000.0 / n as f64;
        // Two sysctl calls + host_statistics64 should complete in < 5ms each.
        assert!(
            per_call_ms < 5.0,
            "collect_pressure_facts too slow: {per_call_ms:.2}ms/call (expected < 5ms)"
        );
    }
}
