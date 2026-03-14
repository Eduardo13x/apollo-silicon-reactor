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
}

pub struct SystemCollector {
    sys: System,
    disks: Disks,
    networks: Networks,
    prev_swap_used_bytes: Option<u64>,
    prev_swap_at: Option<Instant>,
}

#[allow(clippy::new_without_default, dead_code)]
impl SystemCollector {
    pub fn new() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();
        let disks = Disks::new_with_refreshed_list();
        let networks = Networks::new_with_refreshed_list();
        Self {
            sys,
            disks,
            networks,
            prev_swap_used_bytes: None,
            prev_swap_at: None,
        }
    }

    pub fn system(&self) -> &System {
        &self.sys
    }

    pub fn collect_snapshot(&mut self) -> SystemSnapshot {
        // Refresh system stats
        self.sys.refresh_all();
        self.disks.refresh_list();
        self.networks.refresh();

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
        let (mem_pressure, swap_used_bytes, swap_total_bytes) = collect_pressure_facts();
        let nowi = Instant::now();
        let swap_delta_bps = match (self.prev_swap_used_bytes, self.prev_swap_at) {
            (Some(prev_used), Some(prev_at)) => {
                let dt = nowi.duration_since(prev_at).as_secs_f64().max(0.001);
                (swap_used_bytes.saturating_sub(prev_used) as f64) / dt
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

        let (mem_pressure, swap_used_bytes, swap_total_bytes) = collect_pressure_facts();
        let nowi = Instant::now();
        let swap_delta_bps = match (self.prev_swap_used_bytes, self.prev_swap_at) {
            (Some(prev_used), Some(prev_at)) => {
                let dt = nowi.duration_since(prev_at).as_secs_f64().max(0.001);
                (swap_used_bytes.saturating_sub(prev_used) as f64) / dt
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

fn collect_pressure_facts() -> (f64, u64, u64) {
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

    (memory_pressure, swap_used_bytes, swap_total_bytes)
}
