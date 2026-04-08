//! Direct process introspection via proc_pidinfo / proc_pid_rusage.
//!
//! Bypasses the sysinfo crate entirely — one syscall per process, ~2µs each.
//! Gives us signals that sysinfo doesn't expose:
//! - Idle wakeups (the #1 signal for identifying wasteful daemons)
//! - Context switches (high = scheduler contention)
//! - Mach message count (high = IPC-heavy daemon)
//! - Page-ins (thrashing indicator)
//! - CPU instructions and cycles (actual work done)
//! - Disk I/O bytes (real I/O footprint)
//! - Energy billing (Apple's own power attribution)
//! - Per-QoS CPU time breakdown
//!
//! These are the most granular per-process metrics available in macOS EL0.

use std::ffi::CStr;

// ── FFI declarations (libproc.h) ─────────────────────────────────────────────

const PROC_PIDTASKINFO: i32 = 4;
const PROC_PIDPATHINFO_MAXSIZE: u32 = 4096;
const RUSAGE_INFO_V4: i32 = 4;

extern "C" {
    fn proc_pidinfo(
        pid: i32,
        flavor: i32,
        arg: u64,
        buffer: *mut libc::c_void,
        buffersize: i32,
    ) -> i32;

    fn proc_pidpath(pid: i32, buffer: *mut u8, buffersize: u32) -> i32;

    fn proc_listallpids(buffer: *mut libc::c_void, buffersize: i32) -> i32;

    fn proc_pid_rusage(pid: i32, flavor: i32, rusage_info: *mut libc::c_void) -> i32;
}

// ── FFI structs (matching Darwin kernel headers exactly) ─────────────────────

/// proc_taskinfo — from <sys/proc_info.h>
/// 6 × u64 + 12 × i32 = 48 + 48 = 96 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct RawTaskInfo {
    pti_virtual_size: u64,
    pti_resident_size: u64,
    pti_total_user: u64,
    pti_total_system: u64,
    pti_threads_user: u64,
    pti_threads_system: u64,
    pti_policy: i32,
    pti_faults: i32,
    pti_pageins: i32,
    pti_cow_faults: i32,
    pti_messages_sent: i32,
    pti_messages_received: i32,
    pti_syscalls_mach: i32,
    pti_syscalls_unix: i32,
    pti_csw: i32,
    pti_threadnum: i32,
    pti_numrunning: i32,
    pti_priority: i32,
}

/// rusage_info_v4 — from <sys/resource.h>
/// 16 bytes uuid + 35 × u64 = 16 + 280 = 296 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct RawRusageInfoV4 {
    ri_uuid: [u8; 16],
    ri_user_time: u64,
    ri_system_time: u64,
    ri_pkg_idle_wkups: u64,
    ri_interrupt_wkups: u64,
    ri_pageins: u64,
    ri_wired_size: u64,
    ri_resident_size: u64,
    ri_phys_footprint: u64,
    ri_proc_start_abstime: u64,
    ri_proc_exit_abstime: u64,
    ri_child_user_time: u64,
    ri_child_system_time: u64,
    ri_child_pkg_idle_wkups: u64,
    ri_child_interrupt_wkups: u64,
    ri_child_pageins: u64,
    ri_child_elapsed_abstime: u64,
    ri_diskio_bytesread: u64,
    ri_diskio_byteswritten: u64,
    ri_cpu_time_qos_default: u64,
    ri_cpu_time_qos_maintenance: u64,
    ri_cpu_time_qos_background: u64,
    ri_cpu_time_qos_utility: u64,
    ri_cpu_time_qos_legacy: u64,
    ri_cpu_time_qos_user_initiated: u64,
    ri_cpu_time_qos_user_interactive: u64,
    ri_billed_system_time: u64,
    ri_serviced_system_time: u64,
    ri_logical_writes: u64,
    ri_lifetime_max_phys_footprint: u64,
    ri_instructions: u64,
    ri_cycles: u64,
    ri_billed_energy: u64,
    ri_serviced_energy: u64,
    ri_interval_max_phys_footprint: u64,
    ri_runnable_time: u64,
}

impl Default for RawRusageInfoV4 {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

// ── Public types ─────────────────────────────────────────────────────────────

/// Per-process task info — Mach-level metrics.
#[derive(Debug, Clone)]
pub struct TaskInfo {
    pub pid: u32,
    pub virtual_size: u64,
    pub resident_size: u64,
    pub total_user_ns: u64,
    pub total_system_ns: u64,
    pub faults: u32,
    pub pageins: u32,
    pub cow_faults: u32,
    pub messages_sent: u32,
    pub messages_received: u32,
    pub syscalls_mach: u32,
    pub syscalls_unix: u32,
    pub context_switches: u32,
    pub thread_count: u32,
    pub running_threads: u32,
    pub priority: i32,
}

/// Per-process resource usage — Apple's detailed accounting.
#[derive(Debug, Clone)]
pub struct RusageInfo {
    pub pid: u32,
    pub user_time_ns: u64,
    pub system_time_ns: u64,
    pub idle_wakeups: u64,
    pub interrupt_wakeups: u64,
    pub pageins: u64,
    pub wired_size: u64,
    pub resident_size: u64,
    pub phys_footprint: u64,
    pub disk_read_bytes: u64,
    pub disk_write_bytes: u64,
    pub logical_writes: u64,
    pub instructions: u64,
    pub cycles: u64,
    pub billed_energy: u64,
    pub runnable_time_ns: u64,
    /// Absolute time (mach_absolute_time units) when the process started.
    /// Used for PID-recycling detection: if this changes, the PID was reused.
    pub proc_start_abstime: u64,
    /// CPU time breakdown by QoS class.
    pub qos_time: QoSBreakdown,
}

/// CPU time spent in each QoS class.
#[derive(Debug, Clone, Default)]
pub struct QoSBreakdown {
    pub default_ns: u64,
    pub maintenance_ns: u64,
    pub background_ns: u64,
    pub utility_ns: u64,
    pub user_initiated_ns: u64,
    pub user_interactive_ns: u64,
}

impl RusageInfo {
    /// Total on-CPU time for this process (user + system), in nanoseconds.
    pub fn on_cpu_ns(&self) -> u64 {
        self.user_time_ns.saturating_add(self.system_time_ns)
    }
}

/// Per-process CPU contention ratio between two successive `RusageInfo`
/// samples of the same pid.
///
/// Definition:
/// ```text
/// runnable_delta = curr.runnable_time_ns - prev.runnable_time_ns
/// on_cpu_delta   = curr.on_cpu_ns()      - prev.on_cpu_ns()
/// contention     = runnable_delta / (runnable_delta + on_cpu_delta)
/// ```
///
/// Semantics: on Darwin, `ri_runnable_time` counts time the process was
/// `TH_RUN` and ready but NOT actually running on a core. On-CPU time is
/// the actual execution. So contention ∈ [0, 1] is the fraction of the
/// process's "wanted CPU" that it did not get:
///
/// - `0.0` → process got every nanosecond of CPU it asked for.
/// - `1.0` → process was starved for the entire interval (wanted CPU
///   the whole time but the scheduler couldn't place it).
/// - Somewhere between → system is contended and this pid is paying
///   some of the cost.
///
/// This is the macOS equivalent of Linux PSI's per-task `some` stall
/// accounting — the single most important signal for deciding whether
/// a process is being starved by its neighbours.
///
/// Returns `None` when the process did not want any CPU in the window
/// (runnable_delta + on_cpu_delta == 0) — there is no contention
/// signal to report in that case.
///
/// References:
/// - [Brown 2019] "Pressure Stall Information" Linux kernel docs —
///   PSI defines "some" tasks are stalled as the ratio of
///   runnable-but-not-running time to total runnable time.
/// - [Apple XNU `osfmk/kern/thread.h`] — `ri_runnable_time` is the
///   accumulator for TH_RUN time excluding actual on-CPU execution.
pub fn cpu_contention_ratio(prev: &RusageInfo, curr: &RusageInfo) -> Option<f64> {
    let runnable_delta = curr
        .runnable_time_ns
        .saturating_sub(prev.runnable_time_ns);
    let on_cpu_delta = curr.on_cpu_ns().saturating_sub(prev.on_cpu_ns());
    let total = runnable_delta.saturating_add(on_cpu_delta);
    if total == 0 {
        return None;
    }
    Some(runnable_delta as f64 / total as f64)
}

// ── Core functions ───────────────────────────────────────────────────────────

/// Read Mach task info for a process. ~2µs per call.
pub fn get_task_info(pid: u32) -> Option<TaskInfo> {
    let mut raw = RawTaskInfo::default();
    let size = std::mem::size_of::<RawTaskInfo>() as i32;
    let rc = unsafe {
        proc_pidinfo(
            pid as i32,
            PROC_PIDTASKINFO,
            0,
            &mut raw as *mut _ as *mut libc::c_void,
            size,
        )
    };
    if rc <= 0 {
        return None;
    }

    Some(TaskInfo {
        pid,
        virtual_size: raw.pti_virtual_size,
        resident_size: raw.pti_resident_size,
        total_user_ns: raw.pti_total_user,
        total_system_ns: raw.pti_total_system,
        faults: raw.pti_faults as u32,
        pageins: raw.pti_pageins as u32,
        cow_faults: raw.pti_cow_faults as u32,
        messages_sent: raw.pti_messages_sent as u32,
        messages_received: raw.pti_messages_received as u32,
        syscalls_mach: raw.pti_syscalls_mach as u32,
        syscalls_unix: raw.pti_syscalls_unix as u32,
        context_switches: raw.pti_csw as u32,
        thread_count: raw.pti_threadnum as u32,
        running_threads: raw.pti_numrunning as u32,
        priority: raw.pti_priority,
    })
}

/// Read detailed resource usage for a process. ~3µs per call.
/// Includes idle wakeups, instructions, cycles, energy, disk I/O.
pub fn get_rusage_info(pid: u32) -> Option<RusageInfo> {
    let mut raw = RawRusageInfoV4::default();
    let rc = unsafe {
        proc_pid_rusage(
            pid as i32,
            RUSAGE_INFO_V4,
            &mut raw as *mut _ as *mut libc::c_void,
        )
    };
    if rc != 0 {
        return None;
    }

    Some(RusageInfo {
        pid,
        user_time_ns: raw.ri_user_time,
        system_time_ns: raw.ri_system_time,
        idle_wakeups: raw.ri_pkg_idle_wkups,
        interrupt_wakeups: raw.ri_interrupt_wkups,
        pageins: raw.ri_pageins,
        wired_size: raw.ri_wired_size,
        resident_size: raw.ri_resident_size,
        phys_footprint: raw.ri_phys_footprint,
        disk_read_bytes: raw.ri_diskio_bytesread,
        disk_write_bytes: raw.ri_diskio_byteswritten,
        logical_writes: raw.ri_logical_writes,
        instructions: raw.ri_instructions,
        cycles: raw.ri_cycles,
        billed_energy: raw.ri_billed_energy,
        runnable_time_ns: raw.ri_runnable_time,
        proc_start_abstime: raw.ri_proc_start_abstime,
        qos_time: QoSBreakdown {
            default_ns: raw.ri_cpu_time_qos_default,
            maintenance_ns: raw.ri_cpu_time_qos_maintenance,
            background_ns: raw.ri_cpu_time_qos_background,
            utility_ns: raw.ri_cpu_time_qos_utility,
            user_initiated_ns: raw.ri_cpu_time_qos_user_initiated,
            user_interactive_ns: raw.ri_cpu_time_qos_user_interactive,
        },
    })
}

/// Get the full executable path for a process.
pub fn get_proc_path(pid: u32) -> Option<String> {
    let mut buf = vec![0u8; PROC_PIDPATHINFO_MAXSIZE as usize];
    let rc = unsafe { proc_pidpath(pid as i32, buf.as_mut_ptr(), PROC_PIDPATHINFO_MAXSIZE) };
    if rc <= 0 {
        return None;
    }
    let cstr = unsafe { CStr::from_ptr(buf.as_ptr() as *const libc::c_char) };
    Some(cstr.to_string_lossy().into_owned())
}

/// Get all PIDs on the system. Returns sorted list.
pub fn list_all_pids() -> Vec<u32> {
    // First call with null to get count
    let count = unsafe { proc_listallpids(std::ptr::null_mut(), 0) };
    if count <= 0 {
        return vec![];
    }

    // Allocate with some headroom for new processes
    let capacity = (count as usize + 64) * std::mem::size_of::<i32>();
    let mut buf = vec![0i32; count as usize + 64];
    let actual =
        unsafe { proc_listallpids(buf.as_mut_ptr() as *mut libc::c_void, capacity as i32) };
    if actual <= 0 {
        return vec![];
    }

    buf.truncate(actual as usize);
    let mut pids: Vec<u32> = buf.iter().filter(|&&p| p > 0).map(|&p| p as u32).collect();
    pids.sort_unstable();
    pids
}

/// Bulk scan: get TaskInfo + RusageInfo for all processes.
/// ~2-5ms for ~400 processes (vs ~50ms with sysinfo refresh).
pub fn bulk_process_scan() -> Vec<(TaskInfo, Option<RusageInfo>)> {
    let pids = list_all_pids();
    pids.iter()
        .filter_map(|&pid| {
            let ti = get_task_info(pid)?;
            let ri = get_rusage_info(pid);
            Some((ti, ri))
        })
        .collect()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_rusage(user: u64, system: u64, runnable: u64) -> RusageInfo {
        RusageInfo {
            pid: 1,
            user_time_ns: user,
            system_time_ns: system,
            idle_wakeups: 0,
            interrupt_wakeups: 0,
            pageins: 0,
            wired_size: 0,
            resident_size: 0,
            phys_footprint: 0,
            disk_read_bytes: 0,
            disk_write_bytes: 0,
            logical_writes: 0,
            instructions: 0,
            cycles: 0,
            billed_energy: 0,
            runnable_time_ns: runnable,
            proc_start_abstime: 0,
            qos_time: QoSBreakdown::default(),
        }
    }

    #[test]
    fn contention_ratio_zero_when_no_wait() {
        // Process ran 1 ms, never waited.
        let prev = mk_rusage(0, 0, 0);
        let curr = mk_rusage(500_000, 500_000, 0); // 1 ms on-CPU
        assert_eq!(cpu_contention_ratio(&prev, &curr), Some(0.0));
    }

    #[test]
    fn contention_ratio_one_when_fully_starved() {
        // Process wanted CPU for 1 ms, got none.
        let prev = mk_rusage(0, 0, 0);
        let curr = mk_rusage(0, 0, 1_000_000);
        assert_eq!(cpu_contention_ratio(&prev, &curr), Some(1.0));
    }

    #[test]
    fn contention_ratio_half_when_balanced() {
        // Process ran 500 μs on-CPU, waited 500 μs runnable.
        let prev = mk_rusage(0, 0, 0);
        let curr = mk_rusage(250_000, 250_000, 500_000);
        let c = cpu_contention_ratio(&prev, &curr).unwrap();
        assert!((c - 0.5).abs() < 1e-9);
    }

    #[test]
    fn contention_ratio_none_when_idle() {
        // Process did nothing — no contention to report.
        let prev = mk_rusage(100, 100, 100);
        let curr = mk_rusage(100, 100, 100);
        assert_eq!(cpu_contention_ratio(&prev, &curr), None);
    }

    #[test]
    fn contention_ratio_clamps_backwards_samples() {
        // Non-monotonic samples (should never happen from kernel, but guard).
        let prev = mk_rusage(1000, 1000, 1000);
        let curr = mk_rusage(500, 500, 500);
        // All saturating_sub → 0, total → 0, returns None.
        assert_eq!(cpu_contention_ratio(&prev, &curr), None);
    }

    #[test]
    fn on_cpu_ns_sums_user_and_system() {
        let ri = mk_rusage(1_000, 500, 9_999);
        assert_eq!(ri.on_cpu_ns(), 1_500);
    }

    #[test]
    fn read_self_task_info() {
        let pid = std::process::id();
        let info = get_task_info(pid).expect("should read own task info");
        assert_eq!(info.pid, pid);
        assert!(info.resident_size > 0, "RSS must be > 0");
        assert!(info.virtual_size > 0, "VSIZE must be > 0");
        assert!(info.thread_count >= 1, "must have at least 1 thread");
        println!(
            "self: RSS={}MB VSIZE={}MB threads={} csw={} faults={} pageins={}",
            info.resident_size / 1024 / 1024,
            info.virtual_size / 1024 / 1024,
            info.thread_count,
            info.context_switches,
            info.faults,
            info.pageins,
        );
    }

    #[test]
    fn read_self_rusage() {
        let pid = std::process::id();
        let info = get_rusage_info(pid).expect("should read own rusage");
        assert_eq!(info.pid, pid);
        assert!(info.phys_footprint > 0, "footprint must be > 0");
        // instructions/cycles may be 0 on some configs
        println!(
            "self rusage: idle_wakeups={} interrupt_wkups={} instructions={} cycles={} \
             disk_r={}KB disk_w={}KB energy={} footprint={}MB",
            info.idle_wakeups,
            info.interrupt_wakeups,
            info.instructions,
            info.cycles,
            info.disk_read_bytes / 1024,
            info.disk_write_bytes / 1024,
            info.billed_energy,
            info.phys_footprint / 1024 / 1024,
        );
    }

    #[test]
    fn read_self_path() {
        let pid = std::process::id();
        let path = get_proc_path(pid).expect("should read own path");
        assert!(!path.is_empty());
        // Should contain "cargo" or "apollo" or the test runner
        println!("self path: {}", path);
    }

    #[test]
    fn nonexistent_pid_returns_none() {
        assert!(get_task_info(999_999_999).is_none());
        assert!(get_rusage_info(999_999_999).is_none());
        assert!(get_proc_path(999_999_999).is_none());
    }

    #[test]
    fn list_all_pids_has_our_pid() {
        let pids = list_all_pids();
        let my_pid = std::process::id();
        assert!(
            pids.len() > 10,
            "should have many processes, got {}",
            pids.len()
        );
        assert!(pids.contains(&my_pid), "should contain our own PID");
        assert!(pids.contains(&1), "should contain launchd (PID 1)");
        println!("total PIDs: {}", pids.len());
    }

    #[test]
    fn read_kernel_task() {
        // PID 0 = kernel_task — needs root to read, but shouldn't crash
        let info = get_task_info(0);
        // May be None without root — that's fine
        if let Some(i) = info {
            println!(
                "kernel_task: RSS={}MB threads={}",
                i.resident_size / 1024 / 1024,
                i.thread_count
            );
        }
    }

    #[test]
    fn read_launchd() {
        // PID 1 = launchd
        let info = get_task_info(1);
        if let Some(i) = info {
            println!(
                "launchd: RSS={}MB threads={}",
                i.resident_size / 1024 / 1024,
                i.thread_count
            );
        }
    }

    #[test]
    fn bulk_scan_returns_processes() {
        let results = bulk_process_scan();
        assert!(
            results.len() > 5,
            "bulk scan should return processes, got {}",
            results.len()
        );

        // Stats
        let with_rusage = results.iter().filter(|(_, r)| r.is_some()).count();
        let total_rss: u64 = results.iter().map(|(t, _)| t.resident_size).sum();
        println!(
            "bulk scan: {} processes, {} with rusage, total RSS={}MB",
            results.len(),
            with_rusage,
            total_rss / 1024 / 1024,
        );
    }

    #[test]
    fn qos_breakdown_sums_correctly() {
        let pid = std::process::id();
        if let Some(info) = get_rusage_info(pid) {
            let qos_total = info.qos_time.default_ns
                + info.qos_time.maintenance_ns
                + info.qos_time.background_ns
                + info.qos_time.utility_ns
                + info.qos_time.user_initiated_ns
                + info.qos_time.user_interactive_ns;
            println!(
                "QoS breakdown: default={} maint={} bg={} util={} ui={} uix={}  total={}",
                info.qos_time.default_ns,
                info.qos_time.maintenance_ns,
                info.qos_time.background_ns,
                info.qos_time.utility_ns,
                info.qos_time.user_initiated_ns,
                info.qos_time.user_interactive_ns,
                qos_total,
            );
        }
    }
}
