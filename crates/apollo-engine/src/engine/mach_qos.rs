//! Mach QoS and Task Policy — M1-native process and thread scheduling
//!
//! On M1/macOS, frequency scaling is not accessible from userspace.
//! The correct lever is the Mach scheduler's Quality of Service (QoS):
//!
//!  QOS_CLASS_USER_INTERACTIVE  → P-Cores (Firestorm), highest throughput
//!  QOS_CLASS_USER_INITIATED    → P-Cores, slightly lower priority
//!  QOS_CLASS_DEFAULT           → scheduler decides
//!  QOS_CLASS_UTILITY           → mix, reduced energy impact flag
//!  QOS_CLASS_BACKGROUND        → E-Cores (Icestorm) ONLY + throttled I/O
//!
//! Additionally, task_policy_set(TASK_CATEGORY_POLICY) can mark an entire
//! process as "background", which forces all its threads to E-Cores.
//!
//! Phase 1 adds thread-level scheduling: enumerate threads within a process
//! and apply per-thread QoS policies to route hot threads to P-cores and
//! cold threads to E-cores within the same process.
//!
//! Phase 2 adds direct Mach syscalls for latency/throughput QoS tiers,
//! replacing fork/exec of `/usr/sbin/taskpolicy` (~5ms → ~50µs per call).
//!
//! Requirements:
//!   - The daemon must run as root to call task_for_pid() on other processes.
//!   - SIP does NOT block task_policy_set for root daemons.

use std::collections::{HashMap, HashSet};

// ── Low-level FFI ─────────────────────────────────────────────────────────────

/// Subset of Mach constants used here.
// Made `pub` (was private) to expose AFFINITY_TAG_* constants for
// downstream consumers (decide_actions.rs Phase B 2026-05-06).
// All other constants in this module are equally Apple-ABI public values.
pub mod mach_sys {
    #![allow(non_upper_case_globals, dead_code)]

    pub const KERN_SUCCESS: i32 = 0;
    pub const MACH_PORT_NULL: u32 = 0;

    /// task_category_policy role values
    pub const TASK_UNSPECIFIED: i32 = 0;
    pub const TASK_FOREGROUND_APPLICATION: i32 = 1;
    pub const TASK_BACKGROUND_APPLICATION: i32 = 2;
    pub const TASK_CONTROL_APPLICATION: i32 = 3;
    pub const TASK_GRAPHICS_SERVER: i32 = 4;
    pub const TASK_THROTTLE_APPLICATION: i32 = 5;
    pub const TASK_NONUI_APPLICATION: i32 = 6;

    pub const TASK_CATEGORY_POLICY: i32 = 1;
    pub const TASK_CATEGORY_POLICY_COUNT: u32 = 1;

    // Thread policy flavors
    pub const THREAD_LATENCY_QOS_POLICY: i32 = 7;
    pub const THREAD_THROUGHPUT_QOS_POLICY: i32 = 5;
    pub const THREAD_LATENCY_QOS_POLICY_COUNT: u32 = 1;
    pub const THREAD_THROUGHPUT_QOS_POLICY_COUNT: u32 = 1;

    // Thread affinity (P/E cluster routing on Apple Silicon).
    // Threads with the same nonzero affinity_tag are clustered together;
    // tag=0 means "no preference". macOS treats this as a hint, not a hard binding.
    // Convention used in Apollo:
    //   AFFINITY_TAG_P_CLUSTER (1) — latency-sensitive, route to Firestorm/Avalanche.
    //   AFFINITY_TAG_E_CLUSTER (2) — battery/throughput, route to Icestorm/Blizzard.
    pub const THREAD_AFFINITY_POLICY: i32 = 4;
    pub const THREAD_AFFINITY_POLICY_COUNT: u32 = 1;
    pub const AFFINITY_TAG_NONE: u32 = 0;
    pub const AFFINITY_TAG_P_CLUSTER: u32 = 1;
    pub const AFFINITY_TAG_E_CLUSTER: u32 = 2;

    // Task-level QoS flavors (for direct latency/throughput QoS)
    pub const TASK_POLICY_QOS: i32 = 9;
    pub const TASK_BASE_QOS_POLICY: i32 = 8;
    pub const TASK_QOS_POLICY_COUNT: u32 = 1;

    // Latency QoS tiers
    pub const LATENCY_QOS_TIER_UNSPECIFIED: i32 = 0;
    pub const LATENCY_QOS_TIER_0: i32 = 0xFF; // Interactive
    pub const LATENCY_QOS_TIER_1: i32 = 0xFE;
    pub const LATENCY_QOS_TIER_2: i32 = 0xFD;
    pub const LATENCY_QOS_TIER_3: i32 = 0xFC; // Background
    pub const LATENCY_QOS_TIER_4: i32 = 0xFB;
    pub const LATENCY_QOS_TIER_5: i32 = 0xFA;

    // Throughput QoS tiers
    pub const THROUGHPUT_QOS_TIER_UNSPECIFIED: i32 = 0;
    pub const THROUGHPUT_QOS_TIER_0: i32 = 0xFF; // High throughput
    pub const THROUGHPUT_QOS_TIER_1: i32 = 0xFE;
    pub const THROUGHPUT_QOS_TIER_2: i32 = 0xFD;
    pub const THROUGHPUT_QOS_TIER_3: i32 = 0xFC;
    pub const THROUGHPUT_QOS_TIER_4: i32 = 0xFB;
    pub const THROUGHPUT_QOS_TIER_5: i32 = 0xFA;

    // Task suppression (App Nap)
    pub const TASK_SUPPRESSION_POLICY: i32 = 3;
    pub const TASK_SUPPRESSION_POLICY_COUNT: u32 = 9;

    // Real-time thread scheduling
    pub const THREAD_TIME_CONSTRAINT_POLICY: i32 = 2;
    pub const THREAD_TIME_CONSTRAINT_POLICY_COUNT: u32 = 4;

    // Thread info flavors
    pub const THREAD_BASIC_INFO: u32 = 3;
    pub const THREAD_BASIC_INFO_COUNT: u32 = 10; // sizeof(thread_basic_info) / sizeof(i32)

    // Thread run states
    pub const TH_STATE_RUNNING: i32 = 1;
    pub const TH_STATE_STOPPED: i32 = 2;
    pub const TH_STATE_WAITING: i32 = 3;
    pub const TH_STATE_UNINTERRUPTIBLE: i32 = 4;
    pub const TH_STATE_HALTED: i32 = 5;
}

#[cfg(target_os = "macos")]
mod ffi {
    use libc::{c_int, c_uint, c_void, pid_t};

    pub type MachPortT = c_uint;
    pub type KernReturnT = c_int;
    pub type TaskPolicyFlavorT = c_int;
    pub type MachMsgTypeNumberT = c_uint;

    #[repr(C)]
    pub struct TaskCategoryPolicy {
        pub role: c_int,
    }

    /// task_qos_policy for latency/throughput QoS.
    #[repr(C)]
    pub struct TaskQosPolicy {
        pub task_latency_qos_tier: c_int,
        pub task_throughput_qos_tier: c_int,
    }

    /// thread_affinity_policy_data_t — single u32 affinity tag.
    /// Threads with same tag → kernel clusters them on the same cluster
    /// (best-effort hint, not enforced). [Apple TN: ARM big.LITTLE QoS]
    #[repr(C)]
    pub struct ThreadAffinityPolicy {
        pub affinity_tag: c_uint,
    }

    /// thread_basic_info structure for thread_info().
    #[repr(C)]
    pub struct ThreadBasicInfo {
        pub user_time_seconds: c_int,
        pub user_time_microseconds: c_int,
        pub system_time_seconds: c_int,
        pub system_time_microseconds: c_int,
        pub cpu_usage: c_int, // 0–1000 fixed-point per-core
        pub policy: c_int,
        pub run_state: c_int,
        pub flags: c_int,
        pub suspend_count: c_int,
        pub sleep_time: c_int,
    }

    /// thread_latency_qos_policy for per-thread latency QoS.
    #[repr(C)]
    pub struct ThreadLatencyQosPolicy {
        pub thread_latency_qos_tier: c_int,
    }

    /// thread_throughput_qos_policy for per-thread throughput QoS.
    #[repr(C)]
    pub struct ThreadThroughputQosPolicy {
        pub thread_throughput_qos_tier: c_int,
    }

    /// task_suppression_policy for App Nap enforcement.
    /// COUNT = 9 (sizeof / sizeof(int)).
    #[repr(C)]
    pub struct TaskSuppressionPolicy {
        pub active: libc::c_int,
        pub lowpri_cpu: libc::c_int,
        pub timer_throttle: libc::c_int,
        pub disk_throttle: libc::c_int,
        pub cpu_limit: libc::c_int,
        pub suspend: libc::c_int,
        pub throughput_qos: libc::c_int,
        pub suppressed_cpu: libc::c_int,
        pub background_sockets: libc::c_int,
    }

    /// thread_time_constraint_policy for real-time UI thread scheduling.
    /// COUNT = 4 (4 × uint32_t).
    #[repr(C)]
    pub struct ThreadTimeConstraintPolicy {
        pub period: u32,              // nanoseconds between periods
        pub computation: u32,         // nanoseconds of CPU per period
        pub constraint: u32,          // maximum scheduling delay (ns)
        pub preemptible: libc::c_int, // 0=non-preemptible, 1=preemptible
    }

    #[allow(clashing_extern_declarations)]
    extern "C" {
        pub fn mach_task_self() -> MachPortT;
        pub fn task_for_pid(target_tport: MachPortT, pid: pid_t, t: *mut MachPortT) -> KernReturnT;
        pub fn task_policy_set(
            task: MachPortT,
            flavor: TaskPolicyFlavorT,
            policy_info: *const c_void,
            count: MachMsgTypeNumberT,
        ) -> KernReturnT;
        pub fn proc_pidpath(pid: pid_t, buffer: *mut u8, buffersize: u32) -> c_int;
        pub fn mach_port_deallocate(target_task: MachPortT, name: MachPortT) -> KernReturnT;

        // Thread enumeration
        pub fn task_threads(
            task: MachPortT,
            thread_list: *mut *mut MachPortT,
            thread_count: *mut MachMsgTypeNumberT,
        ) -> KernReturnT;

        // Thread info
        pub fn thread_info(
            thread: MachPortT,
            flavor: u32,
            thread_info_out: *mut c_int,
            count: *mut MachMsgTypeNumberT,
        ) -> KernReturnT;

        // Per-thread policy
        pub fn thread_policy_set(
            thread: MachPortT,
            flavor: c_int,
            policy_info: *const c_void,
            count: MachMsgTypeNumberT,
        ) -> KernReturnT;

        // VM deallocation for thread_list cleanup.
        // On macOS, mach_vm_deallocate uses mach_vm_address_t (u64) and
        // mach_vm_size_t (u64) but the target is mach_port_t (u32).
        pub fn vm_deallocate(target_task: MachPortT, address: usize, size: usize) -> KernReturnT;

        // Host + processor set enumeration (for batch task enumeration).
        pub fn mach_host_self() -> MachPortT;
        pub fn host_processor_sets(
            host: MachPortT,
            pset_list: *mut *mut MachPortT,
            pset_count: *mut MachMsgTypeNumberT,
        ) -> KernReturnT;
        pub fn host_processor_set_priv(
            host_priv: MachPortT,
            set_name: MachPortT,
            pset: *mut MachPortT,
        ) -> KernReturnT;
        pub fn processor_set_tasks_with_flavor(
            pset: MachPortT,
            flavor: i32,
            task_list: *mut *mut MachPortT,
            task_count: *mut MachMsgTypeNumberT,
        ) -> KernReturnT;
        pub fn pid_for_task(task: MachPortT, pid: *mut i32) -> KernReturnT;

        // Mach port enumeration — returns all port names + types for a task.
        // Used for Mach port accounting: excessive port count = IPC leak.
        pub fn mach_port_names(
            target_task: MachPortT,
            names: *mut *mut MachPortT,
            names_cnt: *mut MachMsgTypeNumberT,
            types: *mut *mut MachMsgTypeNumberT,
            types_cnt: *mut MachMsgTypeNumberT,
        ) -> KernReturnT;
    }
}

// ── QoS / scheduling tier ────────────────────────────────────────────────────

/// The tier we want to enforce for a process in the Mach scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingTier {
    /// Interactive foreground — routed to P-Cores (Firestorm).
    Foreground,
    /// Normal background — scheduler decides core assignment.
    Normal,
    /// Throttled background — routed to E-Cores (Icestorm) + reduced I/O.
    Background,
}

/// Thread activity pattern detected within a process.
/// Used by decide_actions for differentiated scheduling decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadPattern {
    /// One thread consuming >80% CPU while most others are waiting.
    /// Possible infinite loop, regex backtracking, or spinlock.
    Runaway,
    /// All (or nearly all) threads are actively consuming CPU.
    /// Legitimate CPU-bound workload (compilation, rendering).
    Saturated,
    /// Most threads are waiting (I/O-bound / interactive behavior).
    /// Process is waiting on user input, network, or disk.
    IoBound,
    /// Mixed or not enough threads to classify.
    Normal,
}

/// Result of thread analysis within a process.
#[derive(Debug, Clone)]
pub struct ThreadAnalysis {
    /// Indices of hot threads (high CPU delta).
    pub hot: Vec<u32>,
    /// Indices of cold threads (waiting, low CPU).
    pub cold: Vec<u32>,
    /// Detected pattern.
    pub pattern: ThreadPattern,
    /// Total thread count.
    pub thread_count: usize,
    /// Number of threads actively running (not waiting).
    pub active_count: usize,
}

/// Per-thread QoS tier for big.LITTLE thread-level scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ThreadTier {
    /// Hot thread → P-core routing (low latency, high throughput).
    Interactive,
    /// Utility thread → scheduler decides.
    Utility,
    /// Cold thread → E-core routing (background throughput).
    Background,
}

/// Latency QoS tier for direct Mach syscall (replaces `taskpolicy -l`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyTier {
    /// Interactive: lowest latency, highest priority.
    Interactive,
    /// Default: scheduler decides.
    Default,
    /// Background: highest latency tolerance.
    Background,
}

/// Throughput QoS tier for direct Mach syscall (replaces `taskpolicy -t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThroughputTier {
    /// High throughput (foreground, compilation).
    High,
    /// Default throughput.
    Default,
    /// Low throughput (background daemons).
    Low,
}

/// Result of applying a QoS policy change.
///
/// `success` reflects no-error outcomes including silent skips (cache-hit,
/// SIP-blocked, permanently-blocked) where NO syscall ran. `mutated` is true
/// only when `apply_task_policy` actually executed task_policy_set and got
/// KERN_SUCCESS. Use `mutated` for effect_decay enrollment to avoid phantom
/// observations.
#[derive(Debug, Clone)]
pub struct QoSOutcome {
    pub pid: u32,
    pub tier: SchedulingTier,
    pub success: bool,
    pub mutated: bool,
    pub error: Option<String>,
}

/// Snapshot of a single thread's state within a process.
#[derive(Debug, Clone)]
pub struct ThreadSnapshot {
    /// Index into the thread list (stable within a single enumeration).
    pub thread_index: u32,
    /// Accumulated user-mode CPU time in microseconds.
    pub user_time_us: u64,
    /// Accumulated system-mode CPU time in microseconds.
    pub system_time_us: u64,
    /// CPU usage in kernel fixed-point (0–1000 per core).
    pub cpu_usage_raw: i32,
    /// Thread run state (TH_STATE_RUNNING=1, WAITING=3, etc.).
    pub run_state: i32,
}

// ── Manager ───────────────────────────────────────────────────────────────────

/// Staleness window for the `current_tier` cache. Within it a same-tier
/// request is a true no-op skip; past it Apollo re-applies to reconcile
/// with external mutations (app self-QoS, runningboard).
const CURRENT_TIER_TTL: std::time::Duration = std::time::Duration::from_secs(60);

pub struct MachQoSManager {
    /// Current tier per PID — we only issue a syscall when it changes.
    /// pid → (last tier Apollo applied, when). Fight-hunt fix (2026-06-10):
    /// the cache previously had no timestamps — a hit skipped the syscall
    /// FOREVER, but apps re-set their own QoS and runningboard re-stomps
    /// priorities on app state transitions, so Apollo's belief went stale
    /// and needed re-applies were silently skipped. Hits older than
    /// CURRENT_TIER_TTL now fall through to a real re-apply (~50µs, at
    /// most one per pid per TTL window).
    current_tier: HashMap<u32, (SchedulingTier, std::time::Instant)>,
    /// PIDs permanently skipped (SIP, hardened runtime, or entitlement-protected).
    ///
    /// A7 fix (round-3): paired with `permanently_blocked_since` so GC can
    /// expire entries after a TTL, preventing a recycled PID from inheriting
    /// the "blocked" verdict of a long-dead process.
    permanently_blocked: HashSet<u32>,
    /// Insertion timestamps for `permanently_blocked`, used for TTL-based
    /// expiry alongside the liveness-based `gc_dead_pids`.
    permanently_blocked_since: HashMap<u32, std::time::Instant>,
    /// Previous thread CPU times for delta tracking: (pid, thread_idx) → total_cpu_us.
    prev_thread_cpu: HashMap<(u32, u32), u64>,
    /// PIDs currently in App Nap suppression mode.
    app_napped: HashSet<u32>,
    /// Cached IO tier per PID — skip task_for_pid when tier unchanged.
    io_tier_cache: HashMap<u32, i32>,
}

impl MachQoSManager {
    pub fn new() -> Self {
        Self {
            current_tier: HashMap::new(),
            permanently_blocked: HashSet::new(),
            permanently_blocked_since: HashMap::new(),
            prev_thread_cpu: HashMap::new(),
            app_napped: HashSet::new(),
            io_tier_cache: HashMap::new(),
        }
    }

    /// Record `pid` as permanently blocked, stamping insertion time for
    /// TTL-based GC (A7).
    #[inline]
    fn mark_blocked(&mut self, pid: u32) {
        if self.permanently_blocked.insert(pid) {
            self.permanently_blocked_since
                .insert(pid, std::time::Instant::now());
        }
    }

    /// Apply `tier` to `pid`.  Skips the syscall if already at that tier
    /// or if the PID is permanently blocked.
    pub fn set_tier(&mut self, pid: u32, tier: SchedulingTier) -> QoSOutcome {
        // Permanently blocked PIDs (SIP-protected) — never retry.
        if self.permanently_blocked.contains(&pid) {
            return QoSOutcome {
                pid,
                tier,
                success: true,
                mutated: false,
                error: None,
            };
        }

        // Check cache FIRST: any pid already in current_tier has been
        // successfully task_for_pid'd before, so by construction it is not
        // SIP-protected. Short-circuiting here skips the proc_pidpath()
        // syscall that is_sip_protected() would otherwise cost on every
        // single set_tier call — which matters on the unfreeze hot path
        // where we promote every pid to Foreground. Steady state drops
        // from 2 syscalls per set_tier (proc_pidpath + task_policy) to 1.
        //
        // [Hennessy & Patterson 2017] §2.1 — "make the common case fast";
        // the common case during unfreeze is a pid already classified.
        if let Some(&(cached, stamped_at)) = self.current_tier.get(&pid) {
            if cached == tier && stamped_at.elapsed() < CURRENT_TIER_TTL {
                return QoSOutcome {
                    pid,
                    tier,
                    success: true,
                    mutated: false,
                    error: None,
                };
            }
            // Same tier but stale stamp: fall through to a real re-apply —
            // external writers may have changed the live policy underneath us.
            // Cached with a different tier — skip is_sip_protected (already
            // proven non-SIP) and go straight to apply_task_policy below.
            let result = self.apply_task_policy(pid, tier);
            if result.success {
                self.current_tier.insert(pid, (tier, std::time::Instant::now()));
            } else {
                self.mark_blocked(pid);
                self.current_tier.remove(&pid);
                return QoSOutcome {
                    pid,
                    tier,
                    success: true,
                    mutated: false,
                    error: None,
                };
            }
            return result;
        }

        // First encounter with this pid — pay the SIP classification cost.
        // Pre-filter: if the executable is in a SIP-protected path or
        // proc_pidpath fails, block permanently without attempting task_for_pid.
        if Self::is_sip_protected(pid) {
            self.mark_blocked(pid);
            return QoSOutcome {
                pid,
                tier,
                success: true,
                mutated: false,
                error: None,
            };
        }

        let result = self.apply_task_policy(pid, tier);

        if result.success {
            self.current_tier.insert(pid, (tier, std::time::Instant::now()));
        } else {
            // task_for_pid failed — block permanently and report as silent skip.
            self.mark_blocked(pid);
            self.current_tier.remove(&pid);
            return QoSOutcome {
                pid,
                tier,
                success: true,
                mutated: false,
                error: None,
            };
        }

        result
    }

    /// Purge dead PIDs from all tracking maps.
    /// Call periodically (e.g. every 30 cycles) to prevent unbounded growth
    /// and to handle PID recycling — a recycled PID must be re-evaluated.
    ///
    /// A7 fix (round-3): also TTL-expire `permanently_blocked` entries after
    /// 60s.  `kill(pid, 0) == 0` alone cannot distinguish "original blocked
    /// process still running" from "PID was recycled to a fresh process that
    /// we should actually evaluate".  The TTL forces re-evaluation so a new
    /// occupant isn't inheriting a zombie decision.
    pub fn gc_dead_pids(&mut self) {
        const PERM_BLOCK_TTL: std::time::Duration = std::time::Duration::from_secs(60);
        let now = std::time::Instant::now();
        let blocked_since = &self.permanently_blocked_since;
        self.permanently_blocked.retain(|&pid| {
            let alive = (unsafe { libc::kill(pid as i32, 0) }) == 0;
            if !alive {
                return false;
            }
            // Alive: keep only if the block was recorded recently.  Expired
            // entries are re-evaluated next set_tier call (which will re-add
            // them via mark_blocked on subsequent failure).
            match blocked_since.get(&pid) {
                Some(t) => now.duration_since(*t) < PERM_BLOCK_TTL,
                // Unknown insertion time (upgrade from older state) — drop
                // so it gets re-stamped if it really is still blocked.
                None => false,
            }
        });
        self.permanently_blocked_since
            .retain(|&pid, _| self.permanently_blocked.contains(&pid));
        self.current_tier
            .retain(|&pid, _| (unsafe { libc::kill(pid as i32, 0) }) == 0);
        self.prev_thread_cpu
            .retain(|&(pid, _), _| (unsafe { libc::kill(pid as i32, 0) }) == 0);
        self.app_napped
            .retain(|&pid| (unsafe { libc::kill(pid as i32, 0) }) == 0);
        self.io_tier_cache
            .retain(|&pid, _| (unsafe { libc::kill(pid as i32, 0) }) == 0);
    }

    /// Apply QoS changes to many processes in one pass.
    pub fn apply_batch(&mut self, changes: &[(u32, SchedulingTier)]) -> Vec<QoSOutcome> {
        changes
            .iter()
            .map(|(pid, tier)| self.set_tier(*pid, *tier))
            .collect()
    }

    /// Remove tracking for a PID (process has exited).
    pub fn remove(&mut self, pid: u32) {
        self.current_tier.remove(&pid);
    }

    /// Current tier for a PID, or None if not tracked.
    pub fn current_tier(&self, pid: u32) -> Option<SchedulingTier> {
        self.current_tier.get(&pid).map(|(t, _)| *t)
    }

    /// Return all currently tracked (pid, tier) pairs.
    pub fn current_tier_keys(&self) -> Vec<(u32, SchedulingTier)> {
        self.current_tier.iter().map(|(k, (t, _))| (*k, *t)).collect()
    }

    /// All PIDs currently App-Napped by Apollo. Fight-hunt fix
    /// (2026-06-10): the release sweep previously iterated
    /// `current_tier_keys()` and filtered by nap state — but `app_napped`
    /// is an independent set; a napped pid with no tier entry was
    /// invisible to the sweep and stayed throttled forever.
    pub fn app_napped_pids(&self) -> Vec<u32> {
        self.app_napped.iter().copied().collect()
    }

    /// Count of processes currently pushed to E-Cores.
    pub fn background_count(&self) -> usize {
        self.current_tier
            .values()
            .filter(|(t, _)| *t == SchedulingTier::Background)
            .count()
    }

    // ── Phase 1: Thread-level scheduling ────────────────────────────────

    /// Enumerate all threads in a process and return their CPU/state snapshots.
    /// Returns `None` if the process is SIP-protected or task_for_pid fails.
    #[cfg(target_os = "macos")]
    pub fn enumerate_threads(&self, pid: u32) -> Option<Vec<ThreadSnapshot>> {
        if self.permanently_blocked.contains(&pid) {
            return None;
        }

        unsafe {
            use self::ffi::*;
            use self::mach_sys::*;

            let mut task_port: MachPortT = MACH_PORT_NULL;
            let kr = task_for_pid(mach_task_self(), pid as i32, &mut task_port);
            if kr != KERN_SUCCESS {
                return None;
            }

            let mut thread_list: *mut MachPortT = std::ptr::null_mut();
            let mut thread_count: MachMsgTypeNumberT = 0;

            let kr = task_threads(task_port, &mut thread_list, &mut thread_count);
            // Release task port early — thread ports are independent.
            mach_port_deallocate(mach_task_self(), task_port);

            if kr != KERN_SUCCESS || thread_list.is_null() || thread_count == 0 {
                return None;
            }

            let mut snapshots = Vec::with_capacity(thread_count as usize);

            for i in 0..thread_count {
                let thread_port = *thread_list.add(i as usize);
                let mut info = std::mem::zeroed::<ThreadBasicInfo>();
                let mut count = THREAD_BASIC_INFO_COUNT;

                let kr = thread_info(
                    thread_port,
                    THREAD_BASIC_INFO,
                    &mut info as *mut ThreadBasicInfo as *mut i32,
                    &mut count,
                );

                if kr == KERN_SUCCESS {
                    let user_us = info.user_time_seconds as u64 * 1_000_000
                        + info.user_time_microseconds as u64;
                    let sys_us = info.system_time_seconds as u64 * 1_000_000
                        + info.system_time_microseconds as u64;

                    snapshots.push(ThreadSnapshot {
                        thread_index: i,
                        user_time_us: user_us,
                        system_time_us: sys_us,
                        cpu_usage_raw: info.cpu_usage,
                        run_state: info.run_state,
                    });
                }

                // Release thread port to avoid Mach port leaks.
                mach_port_deallocate(mach_task_self(), thread_port);
            }

            // Deallocate the thread list array allocated by task_threads.
            let list_size = (thread_count as u64) * (std::mem::size_of::<MachPortT>() as u64);
            vm_deallocate(mach_task_self(), thread_list as usize, list_size as usize);

            Some(snapshots)
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn enumerate_threads(&self, _pid: u32) -> Option<Vec<ThreadSnapshot>> {
        None
    }

    /// Classify threads as hot or cold based on CPU delta tracking.
    ///
    /// A thread is "hot" if its CPU delta (since last enumeration) exceeds
    /// 5% of wall-clock time. A thread is "cold" if it spent >90% in
    /// TH_STATE_WAITING.
    ///
    /// Returns `(hot_indices, cold_indices)`.
    pub fn classify_threads(
        &mut self,
        pid: u32,
        threads: &[ThreadSnapshot],
    ) -> (Vec<u32>, Vec<u32>) {
        let analysis = self.analyze_threads(pid, threads);
        (analysis.hot, analysis.cold)
    }

    /// Full thread analysis: hot/cold classification + pattern detection.
    ///
    /// Pattern detection (based on thread state distribution):
    /// - Runaway: 1 thread >80% CPU raw while >75% of threads are WAITING
    /// - Saturated: >75% of threads are hot (legitimate CPU-bound workload)
    /// - IoBound: >80% of threads are cold/waiting
    /// - Normal: mixed or insufficient data
    pub fn analyze_threads(&mut self, pid: u32, threads: &[ThreadSnapshot]) -> ThreadAnalysis {
        let mut hot = Vec::new();
        let mut cold = Vec::new();
        let mut active_count = 0usize;

        for t in threads {
            let key = (pid, t.thread_index);
            let total_cpu = t.user_time_us + t.system_time_us;

            let is_waiting = t.run_state == mach_sys::TH_STATE_WAITING;
            if !is_waiting {
                active_count += 1;
            }

            if let Some(&prev_cpu) = self.prev_thread_cpu.get(&key) {
                let delta = total_cpu.saturating_sub(prev_cpu);
                // Hot: >50ms of CPU in the last cycle (roughly 5% of a 1s cycle).
                if delta > 50_000 || t.cpu_usage_raw > 50 {
                    hot.push(t.thread_index);
                } else if is_waiting && t.cpu_usage_raw < 5 {
                    cold.push(t.thread_index);
                }
            }

            self.prev_thread_cpu.insert(key, total_cpu);
        }

        let n = threads.len();
        let pattern = if n < 2 {
            ThreadPattern::Normal
        } else {
            // Runaway: exactly 1 very hot thread + most others waiting.
            let very_hot = threads.iter().filter(|t| t.cpu_usage_raw > 800).count();
            let waiting = threads
                .iter()
                .filter(|t| t.run_state == mach_sys::TH_STATE_WAITING)
                .count();

            if very_hot == 1 && waiting * 4 >= n * 3 {
                // 1 thread at >80% CPU, ≥75% threads waiting → runaway
                ThreadPattern::Runaway
            } else if hot.len() * 4 >= n * 3 {
                // ≥75% threads are hot → CPU-bound saturation
                ThreadPattern::Saturated
            } else if cold.len() * 5 >= n * 4 {
                // ≥80% threads are cold → I/O-bound
                ThreadPattern::IoBound
            } else {
                ThreadPattern::Normal
            }
        };

        ThreadAnalysis {
            hot,
            cold,
            pattern,
            thread_count: n,
            active_count,
        }
    }

    /// Apply a per-thread QoS tier to a specific thread within a process.
    /// Returns true on success, false if the thread or process is unavailable.
    #[cfg(target_os = "macos")]
    pub fn set_thread_qos(&self, pid: u32, thread_idx: u32, tier: ThreadTier) -> bool {
        if self.permanently_blocked.contains(&pid) {
            return true; // silently skip
        }

        unsafe {
            use self::ffi::*;
            use self::mach_sys::*;

            let mut task_port: MachPortT = MACH_PORT_NULL;
            let kr = task_for_pid(mach_task_self(), pid as i32, &mut task_port);
            if kr != KERN_SUCCESS {
                return false;
            }

            let mut thread_list: *mut MachPortT = std::ptr::null_mut();
            let mut thread_count: MachMsgTypeNumberT = 0;
            let kr = task_threads(task_port, &mut thread_list, &mut thread_count);
            mach_port_deallocate(mach_task_self(), task_port);

            if kr != KERN_SUCCESS || thread_list.is_null() || thread_idx >= thread_count {
                if !thread_list.is_null() && thread_count > 0 {
                    // Deallocate thread ports + list
                    for i in 0..thread_count {
                        mach_port_deallocate(mach_task_self(), *thread_list.add(i as usize));
                    }
                    let list_size =
                        (thread_count as u64) * (std::mem::size_of::<MachPortT>() as u64);
                    vm_deallocate(mach_task_self(), thread_list as usize, list_size as usize);
                }
                return false;
            }

            let target_thread = *thread_list.add(thread_idx as usize);

            // Apply latency QoS
            let latency_tier = match tier {
                ThreadTier::Interactive => LATENCY_QOS_TIER_0,
                ThreadTier::Utility => LATENCY_QOS_TIER_2,
                ThreadTier::Background => LATENCY_QOS_TIER_5,
            };
            let latency_policy = ThreadLatencyQosPolicy {
                thread_latency_qos_tier: latency_tier,
            };
            let kr_lat = thread_policy_set(
                target_thread,
                THREAD_LATENCY_QOS_POLICY,
                &latency_policy as *const _ as *const std::ffi::c_void,
                THREAD_LATENCY_QOS_POLICY_COUNT,
            );

            // Apply throughput QoS
            let throughput_tier = match tier {
                ThreadTier::Interactive => THROUGHPUT_QOS_TIER_0,
                ThreadTier::Utility => THROUGHPUT_QOS_TIER_2,
                ThreadTier::Background => THROUGHPUT_QOS_TIER_5,
            };
            let throughput_policy = ThreadThroughputQosPolicy {
                thread_throughput_qos_tier: throughput_tier,
            };
            let kr_thr = thread_policy_set(
                target_thread,
                THREAD_THROUGHPUT_QOS_POLICY,
                &throughput_policy as *const _ as *const std::ffi::c_void,
                THREAD_THROUGHPUT_QOS_POLICY_COUNT,
            );

            // Deallocate all thread ports and the list.
            for i in 0..thread_count {
                mach_port_deallocate(mach_task_self(), *thread_list.add(i as usize));
            }
            let list_size = (thread_count as u64) * (std::mem::size_of::<MachPortT>() as u64);
            vm_deallocate(mach_task_self(), thread_list as usize, list_size as usize);

            // Success if at least one policy applied (thread may have died mid-call).
            kr_lat == KERN_SUCCESS || kr_thr == KERN_SUCCESS
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn set_thread_qos(&self, _pid: u32, _thread_idx: u32, _tier: ThreadTier) -> bool {
        false
    }

    /// Hint a thread's preferred cluster (P-core or E-core) on Apple Silicon.
    ///
    /// Uses THREAD_AFFINITY_POLICY which the macOS scheduler treats as a hint.
    /// `tag` should be one of `mach_sys::AFFINITY_TAG_P_CLUSTER`,
    /// `AFFINITY_TAG_E_CLUSTER`, or `AFFINITY_TAG_NONE`. Threads sharing the
    /// same nonzero tag are coalesced onto the same cluster best-effort.
    ///
    /// Returns true on successful policy application; false if the thread or
    /// process is unavailable (e.g., PID died mid-call). On non-macOS targets
    /// this is a no-op that returns false.
    ///
    /// **Downstream wiring**: callers should consume `p_core_count` /
    /// `e_core_count` from `CapabilityReport` to skip affinity hints when the
    /// hardware lacks heterogeneous clusters (i.e., return early if either
    /// count is None or 0).
    ///
    /// [ARM big.LITTLE 2013 §3] thread-level affinity hints reduce migration
    /// cost when threads cooperate on shared data within a cluster.
    #[cfg(target_os = "macos")]
    pub fn set_thread_affinity_tag(&self, pid: u32, thread_idx: u32, tag: u32) -> bool {
        if self.permanently_blocked.contains(&pid) {
            return true; // silently skip
        }

        unsafe {
            use self::ffi::*;
            use self::mach_sys::*;

            let mut task_port: MachPortT = MACH_PORT_NULL;
            let kr = task_for_pid(mach_task_self(), pid as i32, &mut task_port);
            if kr != KERN_SUCCESS {
                return false;
            }

            let mut thread_list: *mut MachPortT = std::ptr::null_mut();
            let mut thread_count: MachMsgTypeNumberT = 0;
            let kr = task_threads(task_port, &mut thread_list, &mut thread_count);
            mach_port_deallocate(mach_task_self(), task_port);

            if kr != KERN_SUCCESS || thread_list.is_null() || thread_idx >= thread_count {
                if !thread_list.is_null() && thread_count > 0 {
                    for i in 0..thread_count {
                        mach_port_deallocate(mach_task_self(), *thread_list.add(i as usize));
                    }
                    let list_size =
                        (thread_count as u64) * (std::mem::size_of::<MachPortT>() as u64);
                    vm_deallocate(mach_task_self(), thread_list as usize, list_size as usize);
                }
                return false;
            }

            let target_thread = *thread_list.add(thread_idx as usize);
            let affinity_policy = ThreadAffinityPolicy { affinity_tag: tag };
            let kr_aff = thread_policy_set(
                target_thread,
                THREAD_AFFINITY_POLICY,
                &affinity_policy as *const _ as *const std::ffi::c_void,
                THREAD_AFFINITY_POLICY_COUNT,
            );

            for i in 0..thread_count {
                mach_port_deallocate(mach_task_self(), *thread_list.add(i as usize));
            }
            let list_size = (thread_count as u64) * (std::mem::size_of::<MachPortT>() as u64);
            vm_deallocate(mach_task_self(), thread_list as usize, list_size as usize);

            kr_aff == KERN_SUCCESS
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn set_thread_affinity_tag(&self, _pid: u32, _thread_idx: u32, _tag: u32) -> bool {
        false
    }

    // ── Phase 2: Direct Mach QoS syscalls ───────────────────────────────

    /// Set task-level latency QoS directly via task_policy_set (replaces `taskpolicy -l`).
    /// ~50µs vs ~5ms for fork/exec.
    pub fn set_latency_qos(&mut self, pid: u32, tier: LatencyTier) -> QoSOutcome {
        self.apply_qos_tier(pid, tier, ThroughputTier::Default)
    }

    /// Set task-level throughput QoS directly via task_policy_set (replaces `taskpolicy -t`).
    pub fn set_throughput_qos(&mut self, pid: u32, tier: ThroughputTier) -> QoSOutcome {
        self.apply_qos_tier(pid, LatencyTier::Default, tier)
    }

    /// Set both latency and throughput QoS in a single task_for_pid call.
    pub fn set_latency_and_throughput(
        &mut self,
        pid: u32,
        latency: LatencyTier,
        throughput: ThroughputTier,
    ) -> QoSOutcome {
        self.apply_qos_tier(pid, latency, throughput)
    }

    #[cfg(target_os = "macos")]
    fn apply_qos_tier(
        &mut self,
        pid: u32,
        latency: LatencyTier,
        throughput: ThroughputTier,
    ) -> QoSOutcome {
        if self.permanently_blocked.contains(&pid) || Self::is_sip_protected(pid) {
            self.mark_blocked(pid);
            return QoSOutcome {
                pid,
                tier: SchedulingTier::Normal,
                success: true,
                mutated: false,
                error: None,
            };
        }

        unsafe {
            use self::ffi::*;
            use self::mach_sys::*;

            let mut task_port: MachPortT = MACH_PORT_NULL;
            let kr = task_for_pid(mach_task_self(), pid as i32, &mut task_port);

            if kr != KERN_SUCCESS {
                self.mark_blocked(pid);
                return QoSOutcome {
                    pid,
                    tier: SchedulingTier::Normal,
                    success: true,
                    mutated: false,
                    error: None,
                };
            }

            let lat_tier = match latency {
                LatencyTier::Interactive => LATENCY_QOS_TIER_0,
                LatencyTier::Default => LATENCY_QOS_TIER_UNSPECIFIED,
                LatencyTier::Background => LATENCY_QOS_TIER_5,
            };
            let thr_tier = match throughput {
                ThroughputTier::High => THROUGHPUT_QOS_TIER_0,
                ThroughputTier::Default => THROUGHPUT_QOS_TIER_UNSPECIFIED,
                ThroughputTier::Low => THROUGHPUT_QOS_TIER_5,
            };

            let policy = TaskQosPolicy {
                task_latency_qos_tier: lat_tier,
                task_throughput_qos_tier: thr_tier,
            };

            let kr2 = task_policy_set(
                task_port,
                TASK_POLICY_QOS,
                &policy as *const _ as *const std::ffi::c_void,
                TASK_QOS_POLICY_COUNT,
            );

            mach_port_deallocate(mach_task_self(), task_port);

            if kr2 != KERN_SUCCESS {
                return QoSOutcome {
                    pid,
                    tier: SchedulingTier::Normal,
                    success: false,
                    mutated: false,
                    error: Some(format!("task_policy_set(QOS) failed: kern_return={}", kr2)),
                };
            }
        }

        QoSOutcome {
            pid,
            tier: SchedulingTier::Normal,
            success: true,
            mutated: true,
            error: None,
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn apply_qos_tier(
        &mut self,
        pid: u32,
        _latency: LatencyTier,
        _throughput: ThroughputTier,
    ) -> QoSOutcome {
        QoSOutcome {
            pid,
            tier: SchedulingTier::Normal,
            success: false,
            error: Some("QoS tiers only available on macOS".into()),
        }
    }

    /// Apply I/O tier directly via task_policy_set (replaces `taskpolicy -d`).
    /// Falls back to CLI when task_for_pid fails.
    #[cfg(target_os = "macos")]
    pub fn set_io_tier(&mut self, pid: u32, io_tier: i32) -> bool {
        if self.permanently_blocked.contains(&pid) || Self::is_sip_protected(pid) {
            self.mark_blocked(pid);
            return false;
        }
        // Skip task_for_pid syscall when IO tier unchanged — same idea as
        // current_tier cache for QoS tier. Eliminates the dominant source of
        // unnecessary Mach port lookups when process count exceeds IoShaper's
        // MAX_TRACKED_PIDS cache.
        if self.io_tier_cache.get(&pid) == Some(&io_tier) {
            return true;
        }

        unsafe {
            use self::ffi::*;
            use self::mach_sys::*;

            let mut task_port: MachPortT = MACH_PORT_NULL;
            let kr = task_for_pid(mach_task_self(), pid as i32, &mut task_port);

            if kr != KERN_SUCCESS {
                self.mark_blocked(pid);
                return false;
            }

            // TASK_CATEGORY_POLICY with the appropriate role handles I/O priority
            // routing. The io_tier value maps to task role:
            // 0 = Foreground (Interactive I/O)
            // 1-2 = Normal/Utility
            // 3-4 = Background (Throttled/Passive)
            let role = match io_tier {
                0 => TASK_FOREGROUND_APPLICATION,
                1 | 2 => TASK_UNSPECIFIED,
                _ => TASK_BACKGROUND_APPLICATION,
            };

            let policy = TaskCategoryPolicy { role };
            let kr2 = task_policy_set(
                task_port,
                TASK_CATEGORY_POLICY,
                &policy as *const _ as *const std::ffi::c_void,
                TASK_CATEGORY_POLICY_COUNT,
            );

            mach_port_deallocate(mach_task_self(), task_port);
            let ok = kr2 == KERN_SUCCESS;
            if ok {
                self.io_tier_cache.insert(pid, io_tier);
            }
            ok
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn set_io_tier(&mut self, _pid: u32, _io_tier: i32) -> bool {
        false
    }

    // ── App Nap (TASK_SUPPRESSION_POLICY) ───────────────────────────────

    /// Apply or remove App Nap suppression for a process.
    ///
    /// When `suppressed=true`, the process runs at severely reduced priority:
    /// low CPU, throttled timers, throttled disk I/O — identical to macOS App Nap.
    /// Unlike SIGSTOP, the process continues running; transitions are invisible.
    ///
    /// Returns true if the syscall succeeded.
    #[cfg(target_os = "macos")]
    pub fn set_app_nap(&mut self, pid: u32, suppressed: bool) -> bool {
        if self.permanently_blocked.contains(&pid) || Self::is_sip_protected(pid) {
            self.mark_blocked(pid);
            return false;
        }
        // Skip if already in the target state.
        if suppressed == self.app_napped.contains(&pid) {
            return true;
        }

        unsafe {
            use self::ffi::*;
            use self::mach_sys::*;

            let mut task_port: MachPortT = MACH_PORT_NULL;
            let kr = task_for_pid(mach_task_self(), pid as i32, &mut task_port);
            if kr != KERN_SUCCESS {
                self.mark_blocked(pid);
                return false;
            }

            let policy = if suppressed {
                TaskSuppressionPolicy {
                    active: 1,
                    lowpri_cpu: 1,
                    timer_throttle: LATENCY_QOS_TIER_5,
                    disk_throttle: 1,
                    cpu_limit: 0,
                    suspend: 0,
                    throughput_qos: THROUGHPUT_QOS_TIER_5,
                    suppressed_cpu: 1,
                    background_sockets: 1,
                }
            } else {
                TaskSuppressionPolicy {
                    active: 0,
                    lowpri_cpu: 0,
                    timer_throttle: LATENCY_QOS_TIER_UNSPECIFIED,
                    disk_throttle: 0,
                    cpu_limit: 0,
                    suspend: 0,
                    throughput_qos: THROUGHPUT_QOS_TIER_UNSPECIFIED,
                    suppressed_cpu: 0,
                    background_sockets: 0,
                }
            };

            let kr2 = task_policy_set(
                task_port,
                TASK_SUPPRESSION_POLICY,
                &policy as *const _ as *const std::ffi::c_void,
                TASK_SUPPRESSION_POLICY_COUNT,
            );
            mach_port_deallocate(mach_task_self(), task_port);

            if kr2 == KERN_SUCCESS {
                if suppressed {
                    self.app_napped.insert(pid);
                } else {
                    self.app_napped.remove(&pid);
                }
                true
            } else {
                false
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn set_app_nap(&mut self, _pid: u32, _suppressed: bool) -> bool {
        false
    }

    /// Whether a PID is currently App-Napped.
    pub fn is_app_napped(&self, pid: u32) -> bool {
        self.app_napped.contains(&pid)
    }

    /// Release App Nap from all tracked PIDs (e.g. on wake from sleep).
    pub fn release_all_app_nap(&mut self) {
        let pids: Vec<u32> = self.app_napped.iter().copied().collect();
        for pid in pids {
            self.set_app_nap(pid, false);
        }
    }

    // ── Real-Time UI Thread Boost (THREAD_TIME_CONSTRAINT_POLICY) ────────

    /// Apply real-time scheduling constraint to the foreground app's main thread.
    ///
    /// Guarantees `computation` ns of CPU within every `period` ns window.
    /// This prevents UI hitches when P-cores are saturated by background work
    /// (e.g., LLM inference + browser open simultaneously).
    ///
    /// Uses conservative values: 2ms/10ms, preemptible — safe for any app.
    ///
    /// Returns true if applied successfully.
    #[cfg(target_os = "macos")]
    pub fn set_realtime_boost(&mut self, pid: u32) -> bool {
        if self.permanently_blocked.contains(&pid) || Self::is_sip_protected(pid) {
            return false;
        }

        unsafe {
            use self::ffi::*;
            use self::mach_sys::*;

            let mut task_port: MachPortT = MACH_PORT_NULL;
            let kr = task_for_pid(mach_task_self(), pid as i32, &mut task_port);
            if kr != KERN_SUCCESS {
                self.mark_blocked(pid);
                return false;
            }

            let mut thread_list: *mut MachPortT = std::ptr::null_mut();
            let mut thread_count: MachMsgTypeNumberT = 0;
            let kr2 = task_threads(task_port, &mut thread_list, &mut thread_count);
            mach_port_deallocate(mach_task_self(), task_port);

            if kr2 != KERN_SUCCESS || thread_list.is_null() || thread_count == 0 {
                return false;
            }

            // Apply RT constraint to thread 0 (main/UI thread).
            let main_thread = *thread_list;
            let policy = ThreadTimeConstraintPolicy {
                period: 10_000_000,     // 10 ms
                computation: 2_000_000, // 2 ms guaranteed per period
                constraint: 5_000_000,  // must be scheduled within 5 ms
                preemptible: 1,         // allow preemption (safe)
            };
            let kr3 = thread_policy_set(
                main_thread,
                THREAD_TIME_CONSTRAINT_POLICY,
                &policy as *const _ as *const std::ffi::c_void,
                THREAD_TIME_CONSTRAINT_POLICY_COUNT,
            );

            // Deallocate all thread ports.
            for i in 0..thread_count {
                mach_port_deallocate(mach_task_self(), *thread_list.add(i as usize));
            }
            let list_size = (thread_count as u64) * (std::mem::size_of::<MachPortT>() as u64);
            vm_deallocate(mach_task_self(), thread_list as usize, list_size as usize);

            kr3 == KERN_SUCCESS
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn set_realtime_boost(&mut self, _pid: u32) -> bool {
        false
    }

    /// Remove RT constraint from a process's main thread (restore default scheduling).
    #[cfg(target_os = "macos")]
    pub fn clear_realtime_boost(&mut self, pid: u32) -> bool {
        if self.permanently_blocked.contains(&pid) {
            return false;
        }

        unsafe {
            use self::ffi::*;
            use self::mach_sys::*;

            let mut task_port: MachPortT = MACH_PORT_NULL;
            let kr = task_for_pid(mach_task_self(), pid as i32, &mut task_port);
            if kr != KERN_SUCCESS {
                return false;
            }

            let mut thread_list: *mut MachPortT = std::ptr::null_mut();
            let mut thread_count: MachMsgTypeNumberT = 0;
            let kr2 = task_threads(task_port, &mut thread_list, &mut thread_count);
            mach_port_deallocate(mach_task_self(), task_port);

            if kr2 != KERN_SUCCESS || thread_list.is_null() || thread_count == 0 {
                return false;
            }

            // Clear RT constraint on thread 0: zero period resets to default.
            let main_thread = *thread_list;
            let policy = ThreadTimeConstraintPolicy {
                period: 0,
                computation: 0,
                constraint: 0,
                preemptible: 1,
            };
            let kr3 = thread_policy_set(
                main_thread,
                THREAD_TIME_CONSTRAINT_POLICY,
                &policy as *const _ as *const std::ffi::c_void,
                THREAD_TIME_CONSTRAINT_POLICY_COUNT,
            );

            for i in 0..thread_count {
                mach_port_deallocate(mach_task_self(), *thread_list.add(i as usize));
            }
            let list_size = (thread_count as u64) * (std::mem::size_of::<MachPortT>() as u64);
            vm_deallocate(mach_task_self(), thread_list as usize, list_size as usize);

            kr3 == KERN_SUCCESS
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn clear_realtime_boost(&mut self, _pid: u32) -> bool {
        false
    }

    // ── Private ───────────────────────────────────────────────────────────

    /// Check if a PID's executable lives in a SIP-protected path.
    #[cfg(target_os = "macos")]
    fn is_sip_protected(pid: u32) -> bool {
        let mut buf = [0u8; 1024];
        let ret = unsafe { ffi::proc_pidpath(pid as i32, buf.as_mut_ptr(), buf.len() as u32) };
        if ret <= 0 || ret as usize > buf.len() {
            return true;
        }
        let path = &buf[..ret as usize];
        path.starts_with(b"/System/")
            || path.starts_with(b"/usr/libexec/")
            || path.starts_with(b"/usr/sbin/")
            || path.starts_with(b"/usr/bin/")
            || path.starts_with(b"/sbin/")
            || path.starts_with(b"/Library/PrivilegedHelperTools/")
            || path.starts_with(b"/usr/local/")
    }

    #[cfg(not(target_os = "macos"))]
    fn is_sip_protected(_pid: u32) -> bool {
        false
    }

    #[cfg(target_os = "macos")]
    fn apply_task_policy(&self, pid: u32, tier: SchedulingTier) -> QoSOutcome {
        use self::ffi::*;
        use self::mach_sys::*;

        let role = match tier {
            SchedulingTier::Foreground => TASK_FOREGROUND_APPLICATION,
            SchedulingTier::Normal => TASK_UNSPECIFIED,
            SchedulingTier::Background => TASK_BACKGROUND_APPLICATION,
        };

        unsafe {
            let mut task_port: MachPortT = MACH_PORT_NULL;
            let kr = task_for_pid(mach_task_self(), pid as i32, &mut task_port);

            if kr != KERN_SUCCESS {
                return QoSOutcome {
                    pid,
                    tier,
                    success: false,
                    mutated: false,
                    error: Some(format!("task_for_pid failed: kern_return={}", kr)),
                };
            }

            let policy = TaskCategoryPolicy { role };
            let kr2 = task_policy_set(
                task_port,
                TASK_CATEGORY_POLICY,
                &policy as *const _ as *const std::ffi::c_void,
                TASK_CATEGORY_POLICY_COUNT,
            );

            mach_port_deallocate(mach_task_self(), task_port);

            if kr2 != KERN_SUCCESS {
                return QoSOutcome {
                    pid,
                    tier,
                    success: false,
                    mutated: false,
                    error: Some(format!("task_policy_set failed: kern_return={}", kr2)),
                };
            }
        }

        QoSOutcome {
            pid,
            tier,
            success: true,
            mutated: true,
            error: None,
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn apply_task_policy(&self, pid: u32, tier: SchedulingTier) -> QoSOutcome {
        QoSOutcome {
            pid,
            tier,
            success: false,
            mutated: false,
            error: Some("task_policy_set only available on macOS".into()),
        }
    }

    // ── Mach port accounting ─────────────────────────────────────────────

    /// Count the number of Mach port rights held by a process.
    ///
    /// Processes with >5000 ports are typically leaking or flooding IPC.
    /// Normal healthy apps hold 50-500 ports.  Browsers/Electron can reach
    /// 2000-3000 legitimately.  Anything above 5000 is suspicious.
    ///
    /// Requires root (calls `task_for_pid`).  Returns `None` if the process
    /// is dead or inaccessible.
    #[cfg(target_os = "macos")]
    pub fn get_mach_port_count(&self, pid: u32) -> Option<u32> {
        use ffi::*;
        let self_task = unsafe { mach_task_self() };
        let mut task: MachPortT = mach_sys::MACH_PORT_NULL;
        let kr = unsafe { task_for_pid(self_task, pid as libc::pid_t, &mut task) };
        if kr != mach_sys::KERN_SUCCESS || task == mach_sys::MACH_PORT_NULL {
            return None;
        }

        let mut names: *mut MachPortT = std::ptr::null_mut();
        let mut names_cnt: MachMsgTypeNumberT = 0;
        let mut types: *mut MachMsgTypeNumberT = std::ptr::null_mut();
        let mut types_cnt: MachMsgTypeNumberT = 0;

        let kr = unsafe {
            mach_port_names(task, &mut names, &mut names_cnt, &mut types, &mut types_cnt)
        };

        // Deallocate the task port ASAP.
        unsafe { mach_port_deallocate(self_task, task) };

        if kr != mach_sys::KERN_SUCCESS {
            return None;
        }

        // Free the kernel-allocated arrays.
        if !names.is_null() && names_cnt > 0 {
            unsafe {
                vm_deallocate(
                    self_task,
                    names as usize,
                    names_cnt as usize * std::mem::size_of::<MachPortT>(),
                );
            }
        }
        if !types.is_null() && types_cnt > 0 {
            unsafe {
                vm_deallocate(
                    self_task,
                    types as usize,
                    types_cnt as usize * std::mem::size_of::<MachMsgTypeNumberT>(),
                );
            }
        }

        Some(names_cnt)
    }

    #[cfg(not(target_os = "macos"))]
    pub fn get_mach_port_count(&self, _pid: u32) -> Option<u32> {
        None
    }
}

impl Default for MachQoSManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helper: map our ProcessTier → SchedulingTier ──────────────────────────────

use crate::engine::process_classifier::ProcessTier;

/// Map a classifier tier to the correct Mach scheduling tier.
pub fn tier_for_process(process_tier: ProcessTier) -> SchedulingTier {
    match process_tier {
        ProcessTier::ActiveForeground => SchedulingTier::Foreground,
        ProcessTier::BackgroundVisible => SchedulingTier::Normal,
        ProcessTier::AppHelper => SchedulingTier::Normal,
        ProcessTier::SystemEssential => SchedulingTier::Foreground,
        ProcessTier::SilentDaemon => SchedulingTier::Background,
        ProcessTier::Stale => SchedulingTier::Background,
        ProcessTier::ZombieOrphan => SchedulingTier::Background,
        ProcessTier::Telemetry => SchedulingTier::Background,
    }
}

// ── Batch task enumeration via processor_set_tasks_with_flavor ─────────────

/// TASK_FLAVOR_READ: read-only task port (sufficient for mach_port_names / thread enumeration).
#[cfg(target_os = "macos")]
const TASK_FLAVOR_READ: i32 = 1;

/// Enumerate all tasks in a single Mach call.
///
/// Returns `Vec<(task_port, pid)>`.
/// **Caller must deallocate the task ports** via `mach_port_deallocate`
/// after use (or use `with_all_tasks` for automatic cleanup).
///
/// Requires root (host_processor_set_priv needs host_priv port).
/// Returns an empty Vec without root or on error.
#[cfg(target_os = "macos")]
pub fn enumerate_all_tasks() -> Vec<(u32, i32)> {
    use self::ffi::*;
    use self::mach_sys::*;

    unsafe {
        let host = mach_host_self();

        // Step 1: Get processor set name(s).
        let mut pset_list: *mut MachPortT = std::ptr::null_mut();
        let mut pset_count: MachMsgTypeNumberT = 0;
        let kr = host_processor_sets(host, &mut pset_list, &mut pset_count);
        if kr != KERN_SUCCESS || pset_list.is_null() || pset_count == 0 {
            return Vec::new();
        }

        // Step 2: Get privileged port for first (and usually only) pset.
        let pset_name = *pset_list;
        let mut pset_priv: MachPortT = MACH_PORT_NULL;
        let kr = host_processor_set_priv(host, pset_name, &mut pset_priv);

        // Deallocate pset_list array.
        vm_deallocate(
            mach_task_self(),
            pset_list as usize,
            pset_count as usize * std::mem::size_of::<MachPortT>(),
        );
        // Deallocate pset name port.
        mach_port_deallocate(mach_task_self(), pset_name);

        if kr != KERN_SUCCESS || pset_priv == MACH_PORT_NULL {
            return Vec::new();
        }

        // Step 3: Get all task ports via processor_set_tasks_with_flavor.
        let mut task_list: *mut MachPortT = std::ptr::null_mut();
        let mut task_count: MachMsgTypeNumberT = 0;
        let kr = processor_set_tasks_with_flavor(
            pset_priv,
            TASK_FLAVOR_READ,
            &mut task_list,
            &mut task_count,
        );
        mach_port_deallocate(mach_task_self(), pset_priv);

        if kr != KERN_SUCCESS || task_list.is_null() || task_count == 0 {
            return Vec::new();
        }

        // Step 4: Map task ports → PIDs.
        let mut result = Vec::with_capacity(task_count as usize);
        for i in 0..task_count {
            let task = *task_list.add(i as usize);
            let mut pid: i32 = 0;
            let kr = pid_for_task(task, &mut pid);
            if kr == KERN_SUCCESS {
                result.push((task, pid));
            } else {
                // Can't identify — deallocate and skip.
                mach_port_deallocate(mach_task_self(), task);
            }
        }

        // Deallocate the task list array (but NOT the individual task ports —
        // caller owns them via the result Vec).
        vm_deallocate(
            mach_task_self(),
            task_list as usize,
            task_count as usize * std::mem::size_of::<MachPortT>(),
        );

        result
    }
}

#[cfg(not(target_os = "macos"))]
pub fn enumerate_all_tasks() -> Vec<(u32, i32)> {
    Vec::new()
}

/// Convenience: enumerate all tasks, call `f(task_port, pid)` for each,
/// then deallocate all task ports automatically.
#[cfg(target_os = "macos")]
pub fn with_all_tasks<F: FnMut(u32, i32)>(mut f: F) {
    let tasks = enumerate_all_tasks();
    for &(task, pid) in &tasks {
        f(task, pid);
    }
    // Deallocate all task ports.
    for &(task, _) in &tasks {
        unsafe {
            ffi::mach_port_deallocate(ffi::mach_task_self(), task);
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub fn with_all_tasks<F: FnMut(u32, i32)>(_f: F) {}

/// Batch Mach port counting: get port counts for all accessible processes
/// in a single `processor_set_tasks_with_flavor` call instead of N×`task_for_pid`.
///
/// Returns `Vec<(pid, port_count)>`.
#[cfg(target_os = "macos")]
pub fn batch_mach_port_counts() -> Vec<(i32, u32)> {
    use self::ffi::*;
    use self::mach_sys::*;

    let tasks = enumerate_all_tasks();
    let mut result = Vec::with_capacity(tasks.len());

    let self_task = unsafe { mach_task_self() };

    for &(task, pid) in &tasks {
        // Count ports on this task.
        let mut names: *mut MachPortT = std::ptr::null_mut();
        let mut names_cnt: MachMsgTypeNumberT = 0;
        let mut types: *mut MachMsgTypeNumberT = std::ptr::null_mut();
        let mut types_cnt: MachMsgTypeNumberT = 0;

        let kr = unsafe {
            mach_port_names(task, &mut names, &mut names_cnt, &mut types, &mut types_cnt)
        };

        if kr == KERN_SUCCESS {
            // Free the kernel-allocated arrays.
            if !names.is_null() && names_cnt > 0 {
                unsafe {
                    vm_deallocate(
                        self_task,
                        names as usize,
                        names_cnt as usize * std::mem::size_of::<MachPortT>(),
                    );
                }
            }
            if !types.is_null() && types_cnt > 0 {
                unsafe {
                    vm_deallocate(
                        self_task,
                        types as usize,
                        types_cnt as usize * std::mem::size_of::<MachMsgTypeNumberT>(),
                    );
                }
            }
            result.push((pid, names_cnt));
        }

        // Deallocate the task port.
        unsafe { mach_port_deallocate(self_task, task) };
    }

    result
}

#[cfg(not(target_os = "macos"))]
pub fn batch_mach_port_counts() -> Vec<(i32, u32)> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    #[test]
    fn app_napped_pids_visible_without_tier_entry() {
        // Fight-hunt fix (2026-06-10): a napped pid with NO current_tier
        // entry must still be visible to the release sweep.
        let mut mgr = MachQoSManager::new();
        let pid = std::process::id();
        mgr.app_napped.insert(pid);
        assert!(!mgr.current_tier.contains_key(&pid));
        assert!(
            mgr.app_napped_pids().contains(&pid),
            "nap set must be sweepable independently of tier cache"
        );
    }

    #[test]
    fn current_tier_cache_ttl_falls_through_when_stale() {
        // Fight-hunt fix (2026-06-10): a fresh same-tier hit skips the
        // syscall and leaves the cache entry intact; a STALE hit must fall
        // through to a real re-apply. We can't run task_policy_set in
        // unprivileged tests, but the fall-through is observable: the
        // apply fails (no privileges) → mark_blocked removes the cache
        // entry. A fresh skip would never touch the cache.
        let mut mgr = MachQoSManager::new();
        let pid = std::process::id(); // self — alive, but task_for_pid fails unprivileged

        // Fresh entry → same-tier call is a pure cache skip.
        mgr.current_tier
            .insert(pid, (SchedulingTier::Normal, std::time::Instant::now()));
        let out = mgr.set_tier(pid, SchedulingTier::Normal);
        assert!(!out.mutated, "fresh same-tier hit must not claim mutation");
        assert!(
            mgr.current_tier.contains_key(&pid),
            "fresh skip must leave the cache entry intact"
        );

        // Backdate past TTL → same-tier call must fall through (re-apply).
        mgr.current_tier.insert(
            pid,
            (
                SchedulingTier::Normal,
                std::time::Instant::now() - CURRENT_TIER_TTL - std::time::Duration::from_secs(1),
            ),
        );
        let _ = mgr.set_tier(pid, SchedulingTier::Normal);
        assert!(
            !mgr.current_tier.contains_key(&pid),
            "stale hit must fall through to a real apply (observable via              the unprivileged-failure cleanup path)"
        );
    }

    use super::*;

    #[test]
    fn affinity_constants_match_apple_abi() {
        // THREAD_AFFINITY_POLICY = 4 per Apple <mach/thread_policy.h>.
        assert_eq!(mach_sys::THREAD_AFFINITY_POLICY, 4);
        assert_eq!(mach_sys::THREAD_AFFINITY_POLICY_COUNT, 1);
        assert_eq!(mach_sys::AFFINITY_TAG_NONE, 0);
        // Convention: nonzero tags map to clusters; semantics enforced by Apollo,
        // not the kernel (kernel only requires "same tag = same cluster" hint).
        assert_ne!(mach_sys::AFFINITY_TAG_P_CLUSTER, 0);
        assert_ne!(mach_sys::AFFINITY_TAG_E_CLUSTER, 0);
        assert_ne!(
            mach_sys::AFFINITY_TAG_P_CLUSTER,
            mach_sys::AFFINITY_TAG_E_CLUSTER,
            "P and E cluster tags must differ"
        );
    }

    #[test]
    fn affinity_policy_struct_size() {
        // ThreadAffinityPolicy is a single u32 affinity_tag; Apple ABI is 4 bytes.
        assert_eq!(
            std::mem::size_of::<ffi::ThreadAffinityPolicy>(),
            std::mem::size_of::<u32>()
        );
    }

    #[test]
    fn affinity_helper_blocks_known_pids() {
        // permanently_blocked PIDs (e.g., system processes that reject task_for_pid)
        // should silently succeed (return true) without attempting the FFI call.
        let mut mgr = MachQoSManager::new();
        mgr.permanently_blocked.insert(0); // PID 0 is kernel_task — always blocked
        assert!(
            mgr.set_thread_affinity_tag(0, 0, mach_sys::AFFINITY_TAG_P_CLUSTER),
            "blocked PIDs return true to silence callers"
        );
    }

    #[test]
    fn enumerate_no_crash() {
        // Should return an empty Vec without root, but never panic.
        let tasks = enumerate_all_tasks();
        // On CI / non-root: empty. On root macOS: contains our PID.
        let _ = tasks.len();
    }

    #[test]
    fn enumerate_returns_self_pid() {
        let tasks = enumerate_all_tasks();
        if !tasks.is_empty() {
            let my_pid = std::process::id() as i32;
            let found = tasks.iter().any(|&(_, pid)| pid == my_pid);
            assert!(found, "our own PID should be in the task list");
            // Clean up task ports.
            #[cfg(target_os = "macos")]
            for &(task, _) in &tasks {
                unsafe {
                    ffi::mach_port_deallocate(ffi::mach_task_self(), task);
                }
            }
        }
    }

    #[test]
    fn with_all_tasks_no_leak() {
        // Call with_all_tasks multiple times — should not leak ports.
        for _ in 0..5 {
            let mut count = 0u32;
            with_all_tasks(|_task, _pid| {
                count += 1;
            });
            let _ = count;
        }
    }

    /// Verify ThreadBasicInfo is exactly 40 bytes to match kernel ABI.
    ///
    /// The kernel's THREAD_BASIC_INFO_COUNT = 10 (in units of natural_t = i32 = 4 bytes),
    /// so the struct must be 10 × 4 = 40 bytes.
    #[test]
    #[cfg(target_os = "macos")]
    fn thread_basic_info_is_40_bytes() {
        assert_eq!(
            std::mem::size_of::<ffi::ThreadBasicInfo>(),
            40,
            "ThreadBasicInfo must be 40 bytes to match kernel THREAD_BASIC_INFO_COUNT=10"
        );
    }

    /// Verify THREAD_BASIC_INFO_COUNT is exactly 10 (natural_t units).
    #[test]
    fn thread_basic_info_count_is_10() {
        assert_eq!(
            mach_sys::THREAD_BASIC_INFO_COUNT,
            10,
            "THREAD_BASIC_INFO_COUNT must equal 10 to match the kernel ABI"
        );
    }

    // ── analyze_threads: ThreadPattern classification ─────────────────
    // These tests exercise the pure classification logic without FFI.
    // analyze_threads populates hot/cold vecs only when prev_thread_cpu
    // has a prior entry for the (pid, thread_idx) key — so "Saturated"
    // and "IoBound" cases need two calls (first seeds deltas).

    fn mk_thread(idx: u32, cpu_us: u64, cpu_raw: i32, running: bool) -> ThreadSnapshot {
        ThreadSnapshot {
            thread_index: idx,
            user_time_us: cpu_us,
            system_time_us: 0,
            cpu_usage_raw: cpu_raw,
            run_state: if running {
                mach_sys::TH_STATE_RUNNING
            } else {
                mach_sys::TH_STATE_WAITING
            },
        }
    }

    #[test]
    fn analyze_empty_returns_normal() {
        let mut m = MachQoSManager::new();
        let a = m.analyze_threads(1, &[]);
        assert_eq!(a.pattern, ThreadPattern::Normal);
        assert_eq!(a.thread_count, 0);
        assert_eq!(a.active_count, 0);
        assert!(a.hot.is_empty() && a.cold.is_empty());
    }

    #[test]
    fn analyze_runaway_one_hot_rest_waiting() {
        // 1 thread at >80% (raw=900), 3 waiting with low raw → Runaway.
        // Does NOT require delta seeding: Runaway condition only reads raw + run_state.
        let mut m = MachQoSManager::new();
        let threads = vec![
            mk_thread(0, 100_000, 900, true), // very hot
            mk_thread(1, 0, 0, false),        // waiting
            mk_thread(2, 0, 0, false),        // waiting
            mk_thread(3, 0, 0, false),        // waiting
        ];
        let a = m.analyze_threads(42, &threads);
        assert_eq!(a.pattern, ThreadPattern::Runaway);
        assert_eq!(a.thread_count, 4);
        assert_eq!(a.active_count, 1);
    }

    #[test]
    fn analyze_saturated_most_hot() {
        // All 4 threads hot → Saturated. Needs delta seeding (first call
        // establishes baseline, second call sees delta > 50ms).
        let mut m = MachQoSManager::new();
        let baseline: Vec<_> = (0..4).map(|i| mk_thread(i, 0, 0, true)).collect();
        let _ = m.analyze_threads(7, &baseline);
        let hot: Vec<_> = (0..4).map(|i| mk_thread(i, 200_000, 100, true)).collect(); // +200ms delta each
        let a = m.analyze_threads(7, &hot);
        assert_eq!(a.pattern, ThreadPattern::Saturated);
        assert_eq!(a.hot.len(), 4);
        assert_eq!(a.active_count, 4);
    }

    #[test]
    fn analyze_iobound_most_cold() {
        // 5 threads, all waiting with cpu_raw < 5 → IoBound.
        let mut m = MachQoSManager::new();
        let baseline: Vec<_> = (0..5).map(|i| mk_thread(i, 0, 0, false)).collect();
        let _ = m.analyze_threads(9, &baseline);
        let cold: Vec<_> = (0..5).map(|i| mk_thread(i, 100, 1, false)).collect();
        let a = m.analyze_threads(9, &cold);
        assert_eq!(a.pattern, ThreadPattern::IoBound);
        assert!(a.cold.len() >= 4, "expected ≥4 cold, got {}", a.cold.len());
        assert_eq!(a.active_count, 0);
    }

    #[test]
    fn analyze_normal_mixed_not_classified() {
        // 4 threads: 1 hot, 1 cold, 2 in-between → neither runaway
        // (needs 1 very_hot + ≥75% waiting) nor saturated/iobound → Normal.
        let mut m = MachQoSManager::new();
        let baseline = vec![
            mk_thread(0, 0, 0, true),
            mk_thread(1, 0, 0, true),
            mk_thread(2, 0, 0, false),
            mk_thread(3, 0, 0, false),
        ];
        let _ = m.analyze_threads(11, &baseline);
        let mixed = vec![
            mk_thread(0, 100_000, 60, true), // hot
            mk_thread(1, 10_000, 20, true),  // not hot, not cold
            mk_thread(2, 200, 10, false),    // not cold (raw≥5)
            mk_thread(3, 50, 1, false),      // cold
        ];
        let a = m.analyze_threads(11, &mixed);
        assert_eq!(a.pattern, ThreadPattern::Normal);
    }

    #[test]
    fn analyze_invariant_hot_cold_disjoint() {
        // hot and cold sets must never overlap (classifications are exclusive).
        let mut m = MachQoSManager::new();
        let baseline: Vec<_> = (0..6).map(|i| mk_thread(i, 0, 0, true)).collect();
        let _ = m.analyze_threads(13, &baseline);
        let next = vec![
            mk_thread(0, 100_000, 100, true),
            mk_thread(1, 100, 1, false),
            mk_thread(2, 50_000, 60, true),
            mk_thread(3, 200, 2, false),
            mk_thread(4, 10, 0, false),
            mk_thread(5, 0, 0, false),
        ];
        let a = m.analyze_threads(13, &next);
        for h in &a.hot {
            assert!(
                !a.cold.contains(h),
                "thread {} appears in both hot and cold",
                h
            );
        }
    }

    #[test]
    fn analyze_single_thread_is_normal() {
        // thread_count < 2 → Normal regardless of state.
        let mut m = MachQoSManager::new();
        let t = vec![mk_thread(0, 500_000, 999, true)];
        let a = m.analyze_threads(99, &t);
        assert_eq!(a.pattern, ThreadPattern::Normal);
        assert_eq!(a.thread_count, 1);
    }

    #[test]
    fn classify_threads_matches_analyze() {
        // classify_threads is a thin wrapper — results must align with analyze_threads.
        let mut m = MachQoSManager::new();
        let baseline: Vec<_> = (0..3).map(|i| mk_thread(i, 0, 0, true)).collect();
        let _ = m.analyze_threads(21, &baseline);
        let next = vec![
            mk_thread(0, 200_000, 100, true),
            mk_thread(1, 0, 0, false),
            mk_thread(2, 0, 1, false),
        ];
        let a = m.analyze_threads(21, &next);
        let (hot, cold) = m.classify_threads(21, &next);
        assert_eq!(hot, a.hot);
        assert_eq!(cold, a.cold);
    }
}
