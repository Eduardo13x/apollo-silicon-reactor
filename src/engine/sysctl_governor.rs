//! Reactive sysctl decision engine
//!
//! Monitors TCP health, memory pressure, filesystem pressure, and IPC load,
//! then emits `RootAction::SetSysctl` actions to tune the kernel in response.
//!
//! Four domains are managed independently, each with its own hysteresis
//! counters and cooldown tracking:
//!
//!   1. **TCP** — send/recv buffers, delayed_ack
//!   2. **IPC** — `kern.ipc.somaxconn` (listen backlog)
//!   3. **VM**  — compressor poll interval and sample minimum
//!   4. **FS**  — `kern.maxvnodes`
//!
//! All emitted actions use keys from the safety module's allowlist.  The
//! governor only emits actions when running as root.

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::engine::network_monitor::NetworkMonitor;
use crate::engine::swap_predictor::SwapTrend;
use crate::engine::types::{HardPath, RootAction};

// ── Input bundle ─────────────────────────────────────────────────────────────

/// All inputs the governor needs for one decision cycle.
pub struct SysctlGovernorInput<'a> {
    /// The `NetworkMonitor` itself (for EMA rates and throughput).
    pub net_monitor: &'a NetworkMonitor,
    /// Current swap trend from `SwapPredictor`.
    pub swap_trend: SwapTrend,
    /// System-wide memory pressure in [0.0, 1.0].
    pub memory_pressure: f64,
    /// Workload name from `WorkloadType` debug format (e.g. "Coding").
    pub workload: &'a str,
    /// Whether the machine is on battery power.
    pub on_battery: bool,
    /// Whether the daemon is running as root.
    pub is_root: bool,
}

// ── Observability ────────────────────────────────────────────────────────────

/// Snapshot of the governor's internal state for status reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct SysctlGovernorStatus {
    /// Whether the governor is active (only true when running as root).
    pub active: bool,
    /// Current tuned values for each managed sysctl key.
    pub current_values: HashMap<String, String>,
    /// Captured defaults (values read at init).
    pub defaults: HashMap<String, String>,
    /// Total number of sysctl writes emitted since creation.
    pub total_writes: u64,
    /// Number of sysctls currently changed from their defaults.
    pub active_tunings: usize,
    /// EMA retransmission rate from the network monitor (per 1000 segments).
    pub retransmission_rate: f64,
    /// EMA listen-drop rate from the network monitor (drops/sec).
    pub listen_drop_rate: f64,
    /// Last tune timestamps per key (seconds ago).
    pub last_tune_secs_ago: HashMap<String, u64>,
    /// TCP domain hysteresis state.
    pub tcp_consecutive_high: u32,
    pub tcp_consecutive_low: u32,
    /// IPC domain hysteresis state.
    pub ipc_consecutive_drops: u32,
    pub ipc_consecutive_clean: u32,
    /// VM domain hysteresis state.
    pub vm_consecutive_high: u32,
    pub vm_consecutive_low: u32,
    /// FS domain hysteresis state.
    pub fs_consecutive_high: u32,
    pub fs_consecutive_low: u32,
}

// ── Per-domain tuning state ──────────────────────────────────────────────────

/// TCP buffer and ACK tuning state.
struct TcpTuningState {
    /// Consecutive ticks with retransmission_rate > 5 per 1000.
    consecutive_high: u32,
    /// Consecutive ticks with retransmission_rate < 0.5 per 1000.
    consecutive_low: u32,
    /// Current sendspace value.
    sendspace: u64,
    /// Current recvspace value.
    recvspace: u64,
    /// Current delayed_ack value.
    delayed_ack: u32,
}

/// IPC (somaxconn) tuning state.
struct IpcTuningState {
    /// Consecutive ticks with listen_drops > 0.
    consecutive_drops: u32,
    /// Consecutive ticks with listen_drops == 0.
    consecutive_clean: u32,
    /// Current somaxconn value.
    somaxconn: u64,
}

/// VM compressor tuning state.
struct VmTuningState {
    /// Consecutive ticks with memory_pressure > 0.75.
    consecutive_high: u32,
    /// Consecutive ticks with memory_pressure < 0.40.
    consecutive_low: u32,
    /// Current compressor_poll_interval.
    poll_interval: u64,
    /// Current compressor_sample_min.
    sample_min: u64,
}

/// Filesystem (maxvnodes) tuning state.
struct FsTuningState {
    /// Consecutive ticks with vnode_usage > 80%.
    consecutive_high: u32,
    /// Consecutive ticks with vnode_usage < 30%.
    consecutive_low: u32,
    /// Current maxvnodes value.
    maxvnodes: u64,
}

// ── Constants ────────────────────────────────────────────────────────────────

// ── Persistence paths ────────────────────────────────────────────────────────

const DEFAULTS_PATH_ROOT: &str = "/var/lib/apollo/sysctl_defaults.json";
const DEFAULTS_PATH_USER: &str = "/tmp/apollo-sysctl_defaults.json";

const TCP_BUFFER_MIN: u64 = 131_072; // 128 KB
const TCP_BUFFER_MAX: u64 = 4_194_304; // 4 MB
const SOMAXCONN_MIN: u64 = 1_024;
const SOMAXCONN_MAX: u64 = 8_192;
const MAXVNODES_MIN: u64 = 100_000;
const MAXVNODES_MAX: u64 = 500_000;
const COOLDOWN: Duration = Duration::from_secs(60);

// ── SysctlGovernor ───────────────────────────────────────────────────────────

/// Reactive sysctl tuning engine.
///
/// Call `tick()` every daemon cycle with the current system state.  The
/// governor will return a (possibly empty) list of `RootAction::SetSysctl`
/// actions that should be executed.
pub struct SysctlGovernor {
    tcp: TcpTuningState,
    vm: VmTuningState,
    fs: FsTuningState,
    ipc: IpcTuningState,
    /// Minimum time between tuning the same key.
    cooldown: Duration,
    /// Last time each key was tuned.
    last_tuning: HashMap<String, Instant>,
    /// Default values captured at init (for revert).
    defaults: HashMap<String, String>,
    /// Current tuned values for observability.
    current_values: HashMap<String, String>,
    /// Whether the governor is active (requires root).
    active: bool,
    /// Whether the daemon is running as root (determines persistence path).
    is_root: bool,
    /// Total number of sysctl writes emitted since creation.
    total_writes: u64,
    /// Sysctl keys that could not be read during `capture_defaults()`.
    /// The governor will skip emitting actions for these keys.
    unavailable_keys: Vec<String>,
    /// Timestamp of the last successful `tick()` execution, used to guard
    /// against double-tick from reactor fast-tick.
    last_tick: Option<Instant>,
}

impl SysctlGovernor {
    /// Create a new governor.  Captures current sysctl values as defaults.
    ///
    /// If `is_root` is false the governor will be inactive and `tick()` will
    /// always return an empty vec.
    pub fn new(is_root: bool) -> Self {
        let (defaults, unavailable_keys) = if is_root {
            capture_defaults(is_root)
        } else {
            (HashMap::new(), Vec::new())
        };

        let sendspace = defaults
            .get("net.inet.tcp.sendspace")
            .and_then(|v| v.parse().ok())
            .unwrap_or(131_072);
        let recvspace = defaults
            .get("net.inet.tcp.recvspace")
            .and_then(|v| v.parse().ok())
            .unwrap_or(131_072);
        let delayed_ack = defaults
            .get("net.inet.tcp.delayed_ack")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let somaxconn = defaults
            .get("kern.ipc.somaxconn")
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048);
        #[cfg(target_os = "macos")]
        let poll_interval = defaults
            .get("vm.compressor_eval_period_in_msecs")
            .and_then(|v| v.parse().ok())
            .unwrap_or(250);
        #[cfg(not(target_os = "macos"))]
        let poll_interval = defaults
            .get("vm.compressor_poll_interval")
            .and_then(|v| v.parse().ok())
            .unwrap_or(20);

        #[cfg(target_os = "macos")]
        let sample_min = defaults
            .get("vm.compressor_sample_min_in_msecs")
            .and_then(|v| v.parse().ok())
            .unwrap_or(500);
        #[cfg(not(target_os = "macos"))]
        let sample_min = defaults
            .get("vm.compressor_sample_min")
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        let maxvnodes = defaults
            .get("kern.maxvnodes")
            .and_then(|v| v.parse().ok())
            .unwrap_or(200_000);

        let mut current_values = HashMap::new();
        current_values.insert("net.inet.tcp.sendspace".into(), sendspace.to_string());
        current_values.insert("net.inet.tcp.recvspace".into(), recvspace.to_string());
        current_values.insert("net.inet.tcp.delayed_ack".into(), delayed_ack.to_string());
        current_values.insert("kern.ipc.somaxconn".into(), somaxconn.to_string());
        #[cfg(target_os = "macos")]
        {
            current_values.insert(
                "vm.compressor_eval_period_in_msecs".into(),
                poll_interval.to_string(),
            );
            current_values.insert(
                "vm.compressor_sample_min_in_msecs".into(),
                sample_min.to_string(),
            );
        }
        #[cfg(not(target_os = "macos"))]
        {
            current_values.insert(
                "vm.compressor_poll_interval".into(),
                poll_interval.to_string(),
            );
            current_values.insert("vm.compressor_sample_min".into(), sample_min.to_string());
        }
        current_values.insert("kern.maxvnodes".into(), maxvnodes.to_string());

        Self {
            tcp: TcpTuningState {
                consecutive_high: 0,
                consecutive_low: 0,
                sendspace,
                recvspace,
                delayed_ack,
            },
            vm: VmTuningState {
                consecutive_high: 0,
                consecutive_low: 0,
                poll_interval,
                sample_min,
            },
            fs: FsTuningState {
                consecutive_high: 0,
                consecutive_low: 0,
                maxvnodes,
            },
            ipc: IpcTuningState {
                consecutive_drops: 0,
                consecutive_clean: 0,
                somaxconn,
            },
            cooldown: COOLDOWN,
            last_tuning: HashMap::new(),
            defaults,
            current_values,
            active: is_root,
            is_root,
            total_writes: 0,
            unavailable_keys,
            last_tick: None,
        }
    }

    /// Run one decision cycle.  Returns sysctl actions to apply.
    pub fn tick(&mut self, inputs: &SysctlGovernorInput) -> Vec<RootAction> {
        if !self.active || !inputs.is_root {
            return Vec::new();
        }

        // Ignore ticks that arrive within 2 seconds of each other to prevent
        // double-counting of hysteresis counters from reactor fast-tick.
        let now = Instant::now();
        if let Some(last) = self.last_tick {
            if now.checked_duration_since(last).unwrap_or(Duration::ZERO) < Duration::from_secs(2) {
                return Vec::new();
            }
        }
        self.last_tick = Some(now);

        let mut actions = Vec::new();

        self.tick_tcp(inputs, &mut actions, now);
        self.tick_ipc(inputs, &mut actions, now);
        self.tick_vm(inputs, &mut actions, now);
        self.tick_fs(&mut actions, now);

        actions
    }

    /// Generate actions to revert all managed sysctls to their captured defaults.
    ///
    /// For keys that were not captured at startup (listed in `unavailable_keys`),
    /// the function attempts to read the current value as a best-effort fallback.
    /// Keys that still cannot be read are logged and skipped.
    pub fn revert_to_defaults(&self) -> Vec<RootAction> {
        if !self.active {
            return Vec::new();
        }

        let mut actions: Vec<RootAction> = self
            .defaults
            .iter()
            .map(|(key, value)| RootAction::SetSysctl {
                key: key.clone(),
                value: value.clone(),
                reason: "sysctl-governor: reverting to default".to_string(),
            })
            .collect();

        // For keys that were unavailable at init, try reading the current value
        // as a fallback so we at least attempt a revert.
        for key in &self.unavailable_keys {
            if let Some(current) = read_sysctl(key) {
                eprintln!(
                    "sysctl-governor: key '{}' had no captured default; \
                     using current value '{}' as fallback for revert",
                    key, current
                );
                actions.push(RootAction::SetSysctl {
                    key: key.clone(),
                    value: current,
                    reason: format!(
                        "sysctl-governor: reverting '{}' using fallback (no default captured)",
                        key
                    ),
                });
            } else {
                // Key not available on this macOS version — skip silently.
            }
        }

        actions
    }

    /// Clear the persisted defaults file after a successful revert.
    ///
    /// Call this from the daemon after the revert actions returned by
    /// `revert_to_defaults()` have been successfully executed.  This ensures
    /// the next startup will capture fresh OS defaults instead of reusing
    /// stale persisted values.
    pub fn mark_reverted(&self) {
        clear_persisted_defaults(defaults_path(self.is_root));
    }

    /// Apply aggressive performance tuning for all managed sysctls.
    ///
    /// This replaces the old `SysctlTuner::apply_performance_tuning()`.
    /// Defaults are already captured at `new()`, so these values are safe to
    /// revert later via `revert_to_defaults()`.
    pub fn apply_initial_tuning(&self) -> Vec<RootAction> {
        if !self.active {
            return Vec::new();
        }

        let tunings: &[(&str, &str, &str)] = &[
            // I/O throttling — disable to allow full SSD speed
            ("debug.lowpri_throttle_enabled", "0", "disable I/O throttle"),
            // TCP buffers — 1 MB
            ("net.inet.tcp.sendspace", "1048576", "TCP send buffer 1 MB"),
            ("net.inet.tcp.recvspace", "1048576", "TCP recv buffer 1 MB"),
            // Disable delayed ACKs for lower latency
            ("net.inet.tcp.delayed_ack", "0", "disable delayed ACK"),
            // Local low-latency (LLM APIs)
            (
                "net.inet.tcp.min_iaj_win",
                "4",
                "TCP min inter-arrival jitter window",
            ),
            // High bandwidth scaling
            (
                "net.inet.tcp.win_scale_factor",
                "8",
                "TCP window scale factor",
            ),
            // 32 MB auto-tune max buffers
            (
                "net.inet.tcp.autorcvbufmax",
                "33554432",
                "TCP auto-recv max 32 MB",
            ),
            (
                "net.inet.tcp.autosndbufmax",
                "33554432",
                "TCP auto-send max 32 MB",
            ),
            // Memory compression
            #[cfg(target_os = "macos")]
            (
                "vm.compressor_eval_period_in_msecs",
                "20",
                "compressor poll interval",
            ),
            #[cfg(target_os = "macos")]
            (
                "vm.compressor_sample_min_in_msecs",
                "10",
                "compressor sample min",
            ),
            #[cfg(not(target_os = "macos"))]
            (
                "vm.compressor_poll_interval",
                "20",
                "compressor poll interval",
            ),
            #[cfg(not(target_os = "macos"))]
            ("vm.compressor_sample_min", "10", "compressor sample min"),
            // Filesystem cache
            ("kern.maxvnodes", "300000", "VNode cache for dev workloads"),
            ("kern.maxfiles", "100000", "max open files system-wide"),
            (
                "kern.maxfilesperproc",
                "50000",
                "max open files per process",
            ),
            // IPC
            ("kern.ipc.somaxconn", "2048", "listen backlog"),
            ("kern.ipc.maxsockbuf", "4194304", "max socket buffer 4 MB"),
            // GPU VRAM
            ("iogpu.wired_limit_mb", "12288", "GPU wired memory limit"),
            (
                "debug.iogpu.wired_limit",
                "12288",
                "GPU wired limit (Ventura)",
            ),
        ];

        tunings
            .iter()
            .filter(|(key, _, _)| !self.unavailable_keys.contains(&key.to_string()))
            .map(|(key, value, reason)| RootAction::SetSysctl {
                key: key.to_string(),
                value: value.to_string(),
                reason: format!("sysctl-governor: initial tuning — {}", reason),
            })
            .collect()
    }

    /// Check if macOS Server Performance Mode is enabled.
    pub fn check_server_mode() {
        println!("Checking Server Performance Mode...");
        let output = Command::new("/usr/sbin/serverinfo")
            .arg("--perfmode")
            .output();
        match output {
            Ok(out) => {
                let status = String::from_utf8_lossy(&out.stdout);
                if status.contains("enabled") {
                    println!("Server Performance Mode: ENABLED");
                } else {
                    println!("Server Performance Mode: DISABLED");
                }
            }
            Err(_) => {
                eprintln!("'serverinfo' not found — standard macOS install");
            }
        }
    }

    /// Apply initial tuning directly via sysctl commands (for CLI one-shot use).
    ///
    /// Unlike `apply_initial_tuning()` which returns `RootAction`s for the daemon
    /// pipeline, this method executes sysctl writes immediately and logs results.
    /// Defaults are already captured and persisted at `new()`.
    pub fn apply_tuning_direct(&self) {
        if !self.active {
            eprintln!("sysctl-governor: not root, skipping sysctl tuning");
            return;
        }
        println!("Applying kernel performance tuning...");
        for action in self.apply_initial_tuning() {
            if let RootAction::SetSysctl { key, value, .. } = &action {
                let output = Command::new("/usr/sbin/sysctl")
                    .args(["-w", &format!("{}={}", key, value)])
                    .output();
                match output {
                    Ok(out) if out.status.success() => {
                        println!("  {} = {}", key, value);
                    }
                    Ok(out) => {
                        let err = String::from_utf8_lossy(&out.stderr);
                        eprintln!("  WARN: failed to set {}: {}", key, err.trim());
                    }
                    Err(e) => {
                        eprintln!("  WARN: system error setting {}: {}", key, e);
                    }
                }
            }
        }
    }

    /// Snapshot of internal state for observability / status reporting.
    ///
    /// Accepts a reference to the `NetworkMonitor` to fill in the real
    /// EMA retransmission and listen-drop rates.
    pub fn status(&self, net_monitor: &NetworkMonitor) -> SysctlGovernorStatus {
        let now = Instant::now();
        let last_tune_secs_ago: HashMap<String, u64> = self
            .last_tuning
            .iter()
            .map(|(k, t)| {
                let secs = now
                    .checked_duration_since(*t)
                    .unwrap_or(Duration::from_secs(0))
                    .as_secs();
                (k.clone(), secs)
            })
            .collect();

        // Count how many sysctls have been changed from their defaults.
        let active_tunings = self
            .current_values
            .iter()
            .filter(|(key, val)| {
                self.defaults
                    .get(key.as_str())
                    .is_some_and(|def| def != *val)
            })
            .count();

        SysctlGovernorStatus {
            active: self.active,
            current_values: self.current_values.clone(),
            defaults: self.defaults.clone(),
            total_writes: self.total_writes,
            active_tunings,
            retransmission_rate: net_monitor.retransmission_rate(),
            listen_drop_rate: net_monitor.listen_drop_rate(),
            last_tune_secs_ago,
            tcp_consecutive_high: self.tcp.consecutive_high,
            tcp_consecutive_low: self.tcp.consecutive_low,
            ipc_consecutive_drops: self.ipc.consecutive_drops,
            ipc_consecutive_clean: self.ipc.consecutive_clean,
            vm_consecutive_high: self.vm.consecutive_high,
            vm_consecutive_low: self.vm.consecutive_low,
            fs_consecutive_high: self.fs.consecutive_high,
            fs_consecutive_low: self.fs.consecutive_low,
        }
    }

    // ── TCP domain ───────────────────────────────────────────────────────────

    fn tick_tcp(
        &mut self,
        inputs: &SysctlGovernorInput,
        actions: &mut Vec<RootAction>,
        now: Instant,
    ) {
        let retx_rate = inputs.net_monitor.retransmission_rate();
        let (send_bps, _recv_bps) = inputs.net_monitor.throughput_bps();

        // -- Hysteresis counters --
        if retx_rate > 50.0 {
            // > 5% = 50 per 1000 segments
            self.tcp.consecutive_high += 1;
            self.tcp.consecutive_low = 0;
        } else if retx_rate < 5.0 {
            // < 0.5% = 5 per 1000 segments
            self.tcp.consecutive_low += 1;
            self.tcp.consecutive_high = 0;
        } else {
            // In the middle band: reset both.
            self.tcp.consecutive_high = 0;
            self.tcp.consecutive_low = 0;
        }

        // -- Scale UP: retransmission_rate > 5% for 3 consecutive cycles --
        if self.tcp.consecutive_high >= 3 {
            let new_send = ((self.tcp.sendspace as f64 * 1.25) as u64).min(TCP_BUFFER_MAX);
            let new_recv = ((self.tcp.recvspace as f64 * 1.25) as u64).min(TCP_BUFFER_MAX);

            if new_send != self.tcp.sendspace && self.cooldown_ok("net.inet.tcp.sendspace", now) {
                self.emit_sysctl(
                    "net.inet.tcp.sendspace",
                    &new_send.to_string(),
                    "sysctl-governor: high retransmissions, scaling send buffer +25%",
                    actions,
                    now,
                );
                self.tcp.sendspace = new_send;
            }
            if new_recv != self.tcp.recvspace && self.cooldown_ok("net.inet.tcp.recvspace", now) {
                self.emit_sysctl(
                    "net.inet.tcp.recvspace",
                    &new_recv.to_string(),
                    "sysctl-governor: high retransmissions, scaling recv buffer +25%",
                    actions,
                    now,
                );
                self.tcp.recvspace = new_recv;
            }
            self.tcp.consecutive_high = 0; // Reset after action.
        }

        // -- Scale DOWN: retransmission_rate < 0.5% for 6 cycles AND throughput low --
        // Scale-down when throughput is below 1% of the current buffer capacity.
        // This avoids the ratchet effect where buffers grow but never shrink.
        let buffer_threshold = (self.tcp.sendspace / 100).max(65_536); // at least 64KB/s
        let throughput_low = send_bps < buffer_threshold;
        if self.tcp.consecutive_low >= 6 && throughput_low {
            let new_send = ((self.tcp.sendspace as f64 * 0.75) as u64).max(TCP_BUFFER_MIN);
            let new_recv = ((self.tcp.recvspace as f64 * 0.75) as u64).max(TCP_BUFFER_MIN);

            if new_send != self.tcp.sendspace && self.cooldown_ok("net.inet.tcp.sendspace", now) {
                self.emit_sysctl(
                    "net.inet.tcp.sendspace",
                    &new_send.to_string(),
                    "sysctl-governor: low retransmissions + low throughput, scaling send buffer -25%",
                    actions,
                    now,
                );
                self.tcp.sendspace = new_send;
            }
            if new_recv != self.tcp.recvspace && self.cooldown_ok("net.inet.tcp.recvspace", now) {
                self.emit_sysctl(
                    "net.inet.tcp.recvspace",
                    &new_recv.to_string(),
                    "sysctl-governor: low retransmissions + low throughput, scaling recv buffer -25%",
                    actions,
                    now,
                );
                self.tcp.recvspace = new_recv;
            }
            self.tcp.consecutive_low = 0;
        }

        // -- delayed_ack: workload-aware --
        let desired_ack = if inputs.on_battery {
            3 // Combine ACKs to reduce CPU wakes.
        } else if inputs.workload == "coding" || inputs.workload == "commandline" {
            0 // No delayed ACKs for interactive work.
        } else if inputs.workload == "mediaplayback" || send_bps > 100_000_000 {
            3 // High throughput or streaming: combine ACKs.
        } else {
            self.tcp.delayed_ack // Keep current.
        };

        if desired_ack != self.tcp.delayed_ack && self.cooldown_ok("net.inet.tcp.delayed_ack", now)
        {
            self.emit_sysctl(
                "net.inet.tcp.delayed_ack",
                &desired_ack.to_string(),
                &format!(
                    "sysctl-governor: adjusting delayed_ack for workload={} battery={}",
                    inputs.workload, inputs.on_battery
                ),
                actions,
                now,
            );
            self.tcp.delayed_ack = desired_ack;
        }
    }

    // ── IPC domain ───────────────────────────────────────────────────────────

    fn tick_ipc(
        &mut self,
        inputs: &SysctlGovernorInput,
        actions: &mut Vec<RootAction>,
        now: Instant,
    ) {
        let drop_rate = inputs.net_monitor.listen_drop_rate();

        if drop_rate > 0.0 {
            self.ipc.consecutive_drops += 1;
            self.ipc.consecutive_clean = 0;
        } else {
            self.ipc.consecutive_clean += 1;
            self.ipc.consecutive_drops = 0;
        }

        // Scale UP: listen_drops > 0 for 2 consecutive cycles.
        if self.ipc.consecutive_drops >= 2 {
            let new_val = ((self.ipc.somaxconn as f64 * 1.5) as u64).min(SOMAXCONN_MAX);
            if new_val != self.ipc.somaxconn && self.cooldown_ok("kern.ipc.somaxconn", now) {
                self.emit_sysctl(
                    "kern.ipc.somaxconn",
                    &new_val.to_string(),
                    "sysctl-governor: listen queue drops detected, scaling somaxconn +50%",
                    actions,
                    now,
                );
                self.ipc.somaxconn = new_val;
            }
            self.ipc.consecutive_drops = 0;
        }

        // Scale DOWN: listen_drops == 0 for 30 consecutive cycles.
        if self.ipc.consecutive_clean >= 30 {
            let new_val = ((self.ipc.somaxconn as f64 * 0.75) as u64).max(SOMAXCONN_MIN);
            if new_val != self.ipc.somaxconn && self.cooldown_ok("kern.ipc.somaxconn", now) {
                self.emit_sysctl(
                    "kern.ipc.somaxconn",
                    &new_val.to_string(),
                    "sysctl-governor: no listen drops for 30 cycles, scaling somaxconn -25%",
                    actions,
                    now,
                );
                self.ipc.somaxconn = new_val;
            }
            self.ipc.consecutive_clean = 0;
        }
    }

    // ── VM domain ────────────────────────────────────────────────────────────

    fn tick_vm(
        &mut self,
        inputs: &SysctlGovernorInput,
        actions: &mut Vec<RootAction>,
        now: Instant,
    ) {
        // Sanitize memory_pressure: NaN must be caught BEFORE clamp because
        // f64::NAN.clamp(0.0, 1.0) returns NaN.  Treat NaN as 0.5 (moderate).
        let pressure = if inputs.memory_pressure.is_nan() {
            0.5
        } else {
            inputs.memory_pressure.clamp(0.0, 1.0)
        };

        if pressure > 0.75 {
            self.vm.consecutive_high += 1;
            self.vm.consecutive_low = 0;
        } else if pressure < 0.40 {
            self.vm.consecutive_low += 1;
            self.vm.consecutive_high = 0;
        } else {
            // Default range: reset both, apply moderate settings.
            self.vm.consecutive_high = 0;
            self.vm.consecutive_low = 0;
        }

        let swap_growing = matches!(
            inputs.swap_trend,
            SwapTrend::Increasing | SwapTrend::Critical
        );

        // High pressure + swap growing for 3 cycles: aggressive compressor.
        if self.vm.consecutive_high >= 3 && swap_growing {
            #[cfg(target_os = "macos")]
            {
                if self.vm.poll_interval != 100
                    && self.cooldown_ok("vm.compressor_eval_period_in_msecs", now)
                {
                    self.emit_sysctl(
                        "vm.compressor_eval_period_in_msecs",
                        "100",
                        "sysctl-governor: high memory pressure + swap growing, aggressive compressor (100ms)",
                        actions,
                        now,
                    );
                    self.vm.poll_interval = 100;
                }
                if self.vm.sample_min != 250
                    && self.cooldown_ok("vm.compressor_sample_min_in_msecs", now)
                {
                    self.emit_sysctl(
                        "vm.compressor_sample_min_in_msecs",
                        "250",
                        "sysctl-governor: high memory pressure + swap growing, lower sample min (250ms)",
                        actions,
                        now,
                    );
                    self.vm.sample_min = 250;
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                if self.vm.poll_interval != 10
                    && self.cooldown_ok("vm.compressor_poll_interval", now)
                {
                    self.emit_sysctl(
                        "vm.compressor_poll_interval",
                        "10",
                        "sysctl-governor: high memory pressure + swap growing, aggressive compressor",
                        actions,
                        now,
                    );
                    self.vm.poll_interval = 10;
                }
                if self.vm.sample_min != 5 && self.cooldown_ok("vm.compressor_sample_min", now) {
                    self.emit_sysctl(
                        "vm.compressor_sample_min",
                        "5",
                        "sysctl-governor: high memory pressure + swap growing, lower sample min",
                        actions,
                        now,
                    );
                    self.vm.sample_min = 5;
                }
            }
            self.vm.consecutive_high = 0;
        }

        // Low pressure for 6 cycles: relaxed compressor.
        if self.vm.consecutive_low >= 6 {
            #[cfg(target_os = "macos")]
            {
                if self.vm.poll_interval != 500
                    && self.cooldown_ok("vm.compressor_eval_period_in_msecs", now)
                {
                    self.emit_sysctl(
                        "vm.compressor_eval_period_in_msecs",
                        "500",
                        "sysctl-governor: low memory pressure, relaxing compressor (500ms)",
                        actions,
                        now,
                    );
                    self.vm.poll_interval = 500;
                }
                if self.vm.sample_min != 1000
                    && self.cooldown_ok("vm.compressor_sample_min_in_msecs", now)
                {
                    self.emit_sysctl(
                        "vm.compressor_sample_min_in_msecs",
                        "1000",
                        "sysctl-governor: low memory pressure, relaxing sample min (1000ms)",
                        actions,
                        now,
                    );
                    self.vm.sample_min = 1000;
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                if self.vm.poll_interval != 40
                    && self.cooldown_ok("vm.compressor_poll_interval", now)
                {
                    self.emit_sysctl(
                        "vm.compressor_poll_interval",
                        "40",
                        "sysctl-governor: low memory pressure, relaxing compressor",
                        actions,
                        now,
                    );
                    self.vm.poll_interval = 40;
                }
                if self.vm.sample_min != 20 && self.cooldown_ok("vm.compressor_sample_min", now) {
                    self.emit_sysctl(
                        "vm.compressor_sample_min",
                        "20",
                        "sysctl-governor: low memory pressure, relaxing sample min",
                        actions,
                        now,
                    );
                    self.vm.sample_min = 20;
                }
            }
            self.vm.consecutive_low = 0;
        }

        // Default range (0.40 <= pressure <= 0.75): moderate settings.
        if self.vm.consecutive_high == 0
            && self.vm.consecutive_low == 0
            && (0.40..=0.75).contains(&pressure)
        {
            #[cfg(target_os = "macos")]
            {
                if self.vm.poll_interval != 250
                    && self.cooldown_ok("vm.compressor_eval_period_in_msecs", now)
                {
                    self.emit_sysctl(
                        "vm.compressor_eval_period_in_msecs",
                        "250",
                        "sysctl-governor: moderate memory pressure, balanced compressor (250ms)",
                        actions,
                        now,
                    );
                    self.vm.poll_interval = 250;
                }
                if self.vm.sample_min != 500
                    && self.cooldown_ok("vm.compressor_sample_min_in_msecs", now)
                {
                    self.emit_sysctl(
                        "vm.compressor_sample_min_in_msecs",
                        "500",
                        "sysctl-governor: moderate memory pressure, balanced sample min (500ms)",
                        actions,
                        now,
                    );
                    self.vm.sample_min = 500;
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                if self.vm.poll_interval != 20
                    && self.cooldown_ok("vm.compressor_poll_interval", now)
                {
                    self.emit_sysctl(
                        "vm.compressor_poll_interval",
                        "20",
                        "sysctl-governor: moderate memory pressure, balanced compressor",
                        actions,
                        now,
                    );
                    self.vm.poll_interval = 20;
                }
                if self.vm.sample_min != 10 && self.cooldown_ok("vm.compressor_sample_min", now) {
                    self.emit_sysctl(
                        "vm.compressor_sample_min",
                        "10",
                        "sysctl-governor: moderate memory pressure, balanced sample min",
                        actions,
                        now,
                    );
                    self.vm.sample_min = 10;
                }
            }
        }
    }

    // ── FS domain ────────────────────────────────────────────────────────────

    fn tick_fs(&mut self, actions: &mut Vec<RootAction>, now: Instant) {
        let vnode_usage = estimate_vnode_usage(self.fs.maxvnodes);

        if vnode_usage > 0.80 {
            self.fs.consecutive_high += 1;
            self.fs.consecutive_low = 0;
        } else if vnode_usage < 0.30 {
            self.fs.consecutive_low += 1;
            self.fs.consecutive_high = 0;
        } else {
            self.fs.consecutive_high = 0;
            self.fs.consecutive_low = 0;
        }

        // Scale UP: vnode_usage > 80% for 3 cycles.
        if self.fs.consecutive_high >= 3 {
            let new_val = ((self.fs.maxvnodes as f64 * 1.25) as u64).min(MAXVNODES_MAX);
            if new_val != self.fs.maxvnodes && self.cooldown_ok("kern.maxvnodes", now) {
                self.emit_sysctl(
                    "kern.maxvnodes",
                    &new_val.to_string(),
                    "sysctl-governor: high vnode usage, scaling maxvnodes +25%",
                    actions,
                    now,
                );
                self.fs.maxvnodes = new_val;
            }
            self.fs.consecutive_high = 0;
        }

        // Scale DOWN: vnode_usage < 30% for 30 cycles.
        if self.fs.consecutive_low >= 30 {
            let new_val = ((self.fs.maxvnodes as f64 * 0.85) as u64).max(MAXVNODES_MIN);
            if new_val != self.fs.maxvnodes && self.cooldown_ok("kern.maxvnodes", now) {
                self.emit_sysctl(
                    "kern.maxvnodes",
                    &new_val.to_string(),
                    "sysctl-governor: low vnode usage for 30 cycles, scaling maxvnodes -15%",
                    actions,
                    now,
                );
                self.fs.maxvnodes = new_val;
            }
            self.fs.consecutive_low = 0;
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Check whether the cooldown period has elapsed for a given key.
    fn cooldown_ok(&self, key: &str, now: Instant) -> bool {
        match self.last_tuning.get(key) {
            // Use checked_duration_since to handle clock going backwards
            // during NTP adjustments or sleep/wake transitions.
            Some(last) => {
                now.checked_duration_since(*last)
                    .unwrap_or(Duration::from_secs(0))
                    >= self.cooldown
            }
            None => true,
        }
    }

    /// Emit a `SetSysctl` action and record metadata.
    ///
    /// Skips keys that were unavailable during `capture_defaults()` to avoid
    /// writing to sysctls we cannot revert.
    fn emit_sysctl(
        &mut self,
        key: &str,
        value: &str,
        reason: &str,
        actions: &mut Vec<RootAction>,
        now: Instant,
    ) {
        if self.unavailable_keys.iter().any(|k| k == key) {
            return;
        }
        actions.push(RootAction::SetSysctl {
            key: key.to_string(),
            value: value.to_string(),
            reason: reason.to_string(),
        });
        self.last_tuning.insert(key.to_string(), now);
        self.current_values
            .insert(key.to_string(), value.to_string());
        self.total_writes += 1;
    }
}

impl Default for SysctlGovernor {
    fn default() -> Self {
        Self::new(false)
    }
}

// ── System helpers ───────────────────────────────────────────────────────────

/// Read a sysctl value via `sysctl -n <key>`.  Returns `None` on failure.
fn read_sysctl(key: &str) -> Option<String> {
    Command::new("/usr/sbin/sysctl")
        .args(["-n", key])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Managed sysctl keys.
const MANAGED_KEYS: &[&str] = &[
    "net.inet.tcp.sendspace",
    "net.inet.tcp.recvspace",
    "net.inet.tcp.delayed_ack",
    "net.inet.tcp.min_iaj_win",
    "net.inet.tcp.win_scale_factor",
    "net.inet.tcp.autorcvbufmax",
    "net.inet.tcp.autosndbufmax",
    "kern.ipc.somaxconn",
    "kern.ipc.maxsockbuf",
    "kern.maxfiles",
    "kern.maxfilesperproc",
    "kern.maxvnodes",
    "debug.lowpri_throttle_enabled",
    "iogpu.wired_limit_mb",
    "debug.iogpu.wired_limit",
    #[cfg(target_os = "macos")]
    "vm.compressor_eval_period_in_msecs",
    #[cfg(target_os = "macos")]
    "vm.compressor_sample_min_in_msecs",
    #[cfg(not(target_os = "macos"))]
    "vm.compressor_poll_interval",
    #[cfg(not(target_os = "macos"))]
    "vm.compressor_sample_min",
];

/// Return the persistence path for sysctl defaults based on privilege level.
fn defaults_path(is_root: bool) -> &'static str {
    if is_root {
        DEFAULTS_PATH_ROOT
    } else {
        DEFAULTS_PATH_USER
    }
}

/// Persist sysctl defaults to disk as pretty-printed JSON.
///
/// Best-effort: failures are logged but do not panic.
fn persist_defaults(path: &str, defaults: &HashMap<String, String>) {
    crate::engine::llm::write_json(std::path::Path::new(path), defaults, None);
}

/// Load previously persisted sysctl defaults from disk.
///
/// Returns `None` if the file does not exist or cannot be parsed.
fn load_persisted_defaults(path: &str) -> Option<HashMap<String, String>> {
    if !Path::new(path).exists() {
        return None;
    }
    match HardPath::read_to_string_limited(Path::new(path), 1024 * 1024) {
        Ok(contents) => match serde_json::from_str::<HashMap<String, String>>(&contents) {
            Ok(map) => Some(map),
            Err(e) => {
                eprintln!(
                    "sysctl-governor: WARNING: failed to parse persisted defaults from '{}': {}",
                    path, e
                );
                None
            }
        },
        Err(e) => {
            eprintln!(
                "sysctl-governor: WARNING: failed to read persisted defaults from '{}': {}",
                path, e
            );
            None
        }
    }
}

/// Clear the persisted defaults file from disk.
///
/// Best-effort: failures are logged but do not panic.
fn clear_persisted_defaults(path: &str) {
    if Path::new(path).exists() {
        if let Err(e) = std::fs::remove_file(path) {
            eprintln!(
                "sysctl-governor: WARNING: failed to remove persisted defaults '{}': {}",
                path, e
            );
        }
    }
}

/// Capture current values of all managed sysctl keys.
///
/// If a persisted defaults file exists on disk (indicating a previous crash
/// before revert), those values are used instead of reading from the kernel.
/// This prevents capturing already-tuned values as "defaults" after a crash.
///
/// Returns `(defaults, unavailable_keys)` where `unavailable_keys` lists
/// the keys that could not be read (e.g., permission denied, key does not
/// exist on this kernel version).
fn capture_defaults(is_root: bool) -> (HashMap<String, String>, Vec<String>) {
    let path = defaults_path(is_root);

    // Try to load persisted defaults first (crash recovery).
    if let Some(persisted) = load_persisted_defaults(path) {
        // Filter out stale keys that are no longer in MANAGED_KEYS.
        // This handles key renames across binary versions (e.g.
        // vm.compressor_poll_interval → vm.compressor_eval_period_in_msecs).
        let managed_set: std::collections::HashSet<&str> = MANAGED_KEYS.iter().copied().collect();
        let persisted: HashMap<String, String> = persisted
            .into_iter()
            .filter(|(k, _)| managed_set.contains(k.as_str()))
            .collect();
        if persisted.is_empty() {
            // All persisted keys were stale — discard the file and read fresh.
            clear_persisted_defaults(path);
        } else {
            eprintln!(
                "sysctl-governor: recovered {} persisted defaults from '{}' (previous crash detected)",
                persisted.len(),
                path
            );
            let unavailable: Vec<String> = MANAGED_KEYS
                .iter()
                .filter(|k| !persisted.contains_key(**k))
                .map(|k| k.to_string())
                .collect();
            return (persisted, unavailable);
        }
    }

    // No persisted defaults — read fresh from the kernel.
    let mut defaults = HashMap::new();
    let mut unavailable = Vec::new();
    for key in MANAGED_KEYS {
        if let Some(value) = read_sysctl(key) {
            defaults.insert(key.to_string(), value);
        } else {
            // Key not available on this macOS version — skip silently.
            unavailable.push(key.to_string());
        }
    }

    // Persist to disk so they survive a crash.
    if !defaults.is_empty() {
        persist_defaults(path, &defaults);
    }

    (defaults, unavailable)
}

/// Estimate vnode usage as a fraction of `maxvnodes`.
///
/// macOS does not expose a direct "current vnodes" counter via sysctl.
/// We approximate by reading `kern.num_vnodes` (available on macOS 12+)
/// and falling back to `kern.openfiles` as a rough proxy.
///
/// **Limitation:** `kern.openfiles` counts open file descriptors, which is
/// a subset of active vnodes.  A 1.5x multiplier is applied as a heuristic
/// but the true vnode count may differ significantly.
///
/// If both sysctls fail, returns a conservative estimate of 0.5 (50%)
/// rather than 0.0 which would never trigger the scale-up logic.
fn estimate_vnode_usage(maxvnodes: u64) -> f64 {
    if maxvnodes == 0 {
        return 0.5; // Conservative: assume moderate usage when max is unknown.
    }

    // Try kern.num_vnodes first (available on macOS 12+).
    if let Some(val) = read_sysctl("kern.num_vnodes") {
        if let Ok(current) = val.parse::<u64>() {
            return (current as f64 / maxvnodes as f64).clamp(0.0, 1.0);
        }
    }

    // Fallback: use kern.openfiles as a rough proxy (vnodes >= open files).
    // NOTE: This undercounts real vnode usage; see doc comment above.
    if let Some(val) = read_sysctl("kern.openfiles") {
        if let Ok(open_files) = val.parse::<u64>() {
            // Open files undercount vnodes; apply a 1.5x multiplier heuristic.
            let estimated = (open_files as f64 * 1.5) as u64;
            return (estimated as f64 / maxvnodes as f64).clamp(0.0, 1.0);
        }
    }

    // Both sysctls failed — return a conservative 50% estimate so the
    // governor does not assume zero usage and skip scaling entirely.
    0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::network_monitor::TcpStats;

    /// Helper: create a minimal NetworkMonitor for testing.
    fn test_monitor(retx_rate: f64, drop_rate: f64, send_bps: u64) -> NetworkMonitor {
        let mut monitor = NetworkMonitor::new();
        // Inject EMA values directly for deterministic tests.
        monitor.ema_retransmission_rate = retx_rate;
        monitor.ema_listen_drop_rate = drop_rate;
        // Push a synthetic history entry for throughput calculation.
        let stats = TcpStats {
            bytes_sent: send_bps,
            elapsed: std::time::Duration::from_secs(1),
            ..Default::default()
        };
        monitor.history.push_back(stats);
        monitor
    }

    /// Helper: simulate a tick that passes the minimum-interval guard.
    /// In tests, ticks execute instantly so we reset `last_tick` to allow
    /// consecutive calls without the 2-second debounce rejecting them.
    fn tick_ok(gov: &mut SysctlGovernor, inputs: &SysctlGovernorInput) -> Vec<RootAction> {
        gov.last_tick = None;
        gov.tick(inputs)
    }

    fn default_inputs(net_monitor: &NetworkMonitor) -> SysctlGovernorInput<'_> {
        SysctlGovernorInput {
            net_monitor,
            swap_trend: SwapTrend::Stable,
            memory_pressure: 0.50,
            workload: "General",
            on_battery: false,
            is_root: true,
        }
    }

    #[test]
    fn inactive_when_not_root() {
        let mut gov = SysctlGovernor::new(false);
        let monitor = NetworkMonitor::new();
        let inputs = SysctlGovernorInput {
            net_monitor: &monitor,
            swap_trend: SwapTrend::Stable,
            memory_pressure: 0.90,
            workload: "General",
            on_battery: false,
            is_root: false,
        };
        let actions = gov.tick(&inputs);
        assert!(actions.is_empty());
    }

    #[test]
    fn tcp_scale_up_after_3_high_cycles() {
        let mut gov = SysctlGovernor::new(true);
        // retx_rate > 50 (= 5%) triggers high counter.
        let monitor = test_monitor(60.0, 0.0, 1000);

        let inputs = default_inputs(&monitor);

        // Cycles 1 and 2: no actions yet.
        assert!(tick_ok(&mut gov, &inputs).is_empty());
        assert!(tick_ok(&mut gov, &inputs).is_empty());
        // Cycle 3: should emit scale-up actions.
        let actions = tick_ok(&mut gov, &inputs);
        assert!(
            !actions.is_empty(),
            "expected TCP scale-up actions on cycle 3"
        );
        // Verify at least one SetSysctl for sendspace or recvspace.
        let has_buffer_tune = actions.iter().any(|a| {
            matches!(a, RootAction::SetSysctl { key, .. }
                if key == "net.inet.tcp.sendspace" || key == "net.inet.tcp.recvspace")
        });
        assert!(has_buffer_tune, "expected buffer scaling action");
    }

    #[test]
    fn ipc_scale_up_after_2_drop_cycles() {
        let mut gov = SysctlGovernor::new(true);
        // drop_rate > 0 triggers consecutive_drops counter.
        let monitor = test_monitor(0.0, 1.0, 0);

        let inputs = default_inputs(&monitor);

        // Cycle 1: no action.
        assert!(tick_ok(&mut gov, &inputs).is_empty());
        // Cycle 2: should emit somaxconn scale-up.
        let actions = tick_ok(&mut gov, &inputs);
        let has_somaxconn = actions
            .iter()
            .any(|a| matches!(a, RootAction::SetSysctl { key, .. } if key == "kern.ipc.somaxconn"));
        assert!(has_somaxconn, "expected somaxconn scale-up on cycle 2");
    }

    #[test]
    fn vm_aggressive_on_high_pressure_and_swap() {
        let mut gov = SysctlGovernor::new(true);
        // Clear unavailable_keys so the test works without root privileges
        // (vm.compressor_* sysctls are not readable as non-root).
        gov.unavailable_keys.clear();
        let monitor = test_monitor(0.0, 0.0, 0);

        let mut inputs = default_inputs(&monitor);
        inputs.memory_pressure = 0.85;
        inputs.swap_trend = SwapTrend::Increasing;

        // 3 cycles needed.
        tick_ok(&mut gov, &inputs);
        tick_ok(&mut gov, &inputs);
        let actions = tick_ok(&mut gov, &inputs);

        // On macOS the governor uses the _in_msecs variants; on other platforms the legacy names.
        let has_aggressive = actions.iter().any(|a| {
            matches!(a, RootAction::SetSysctl { key, .. }
                if key == "vm.compressor_poll_interval"
                    || key == "vm.compressor_eval_period_in_msecs"
                    || key == "vm.compressor_sample_min"
                    || key == "vm.compressor_sample_min_in_msecs")
        });
        assert!(
            has_aggressive,
            "expected aggressive VM tuning on high pressure + swap"
        );
    }

    #[test]
    fn revert_to_defaults_emits_actions() {
        let gov = SysctlGovernor::new(true);
        let reverts = gov.revert_to_defaults();
        // Should have one action per captured default.
        assert!(
            !reverts.is_empty(),
            "expected revert actions for captured defaults"
        );
        for action in &reverts {
            match action {
                RootAction::SetSysctl { reason, .. } => {
                    assert!(reason.contains("reverting"));
                }
                _ => panic!("expected SetSysctl actions only"),
            }
        }
    }

    #[test]
    fn status_snapshot() {
        let gov = SysctlGovernor::new(true);
        let monitor = NetworkMonitor::new();
        let status = gov.status(&monitor);
        assert!(status.active);
        assert!(!status.current_values.is_empty());
    }

    #[test]
    fn cooldown_prevents_rapid_changes() {
        let mut gov = SysctlGovernor::new(true);
        let monitor = test_monitor(60.0, 0.0, 1000);
        let inputs = default_inputs(&monitor);

        // Drive to 3 cycles to trigger TCP scale-up.
        tick_ok(&mut gov, &inputs);
        tick_ok(&mut gov, &inputs);
        let first_actions = tick_ok(&mut gov, &inputs);
        assert!(!first_actions.is_empty());

        // Immediately try again: should be blocked by cooldown.
        gov.tcp.consecutive_high = 3; // Force counter.
        let second_actions = tick_ok(&mut gov, &inputs);
        let has_sendspace = second_actions.iter().any(
            |a| matches!(a, RootAction::SetSysctl { key, .. } if key == "net.inet.tcp.sendspace"),
        );
        assert!(
            !has_sendspace,
            "cooldown should prevent immediate re-tuning of sendspace"
        );
    }

    #[test]
    fn delayed_ack_coding_workload() {
        let mut gov = SysctlGovernor::new(true);
        // Set initial delayed_ack to something other than 0 to detect the change.
        gov.tcp.delayed_ack = 3;
        gov.current_values
            .insert("net.inet.tcp.delayed_ack".into(), "3".into());

        let monitor = test_monitor(0.0, 0.0, 0);

        let mut inputs = default_inputs(&monitor);
        // Use lowercase to match daemon's format!("{:?}", WorkloadType::Coding).to_lowercase()
        inputs.workload = "coding";

        let actions = gov.tick(&inputs);
        let has_ack_change = actions.iter().any(|a| {
            matches!(a, RootAction::SetSysctl { key, value, .. }
                if key == "net.inet.tcp.delayed_ack" && value == "0")
        });
        assert!(has_ack_change, "expected delayed_ack=0 for coding workload");
    }

    #[test]
    fn delayed_ack_battery_mode() {
        let mut gov = SysctlGovernor::new(true);
        gov.tcp.delayed_ack = 0;
        gov.current_values
            .insert("net.inet.tcp.delayed_ack".into(), "0".into());

        let monitor = test_monitor(0.0, 0.0, 0);

        let mut inputs = default_inputs(&monitor);
        inputs.on_battery = true;

        let actions = gov.tick(&inputs);
        let has_ack_change = actions.iter().any(|a| {
            matches!(a, RootAction::SetSysctl { key, value, .. }
                if key == "net.inet.tcp.delayed_ack" && value == "3")
        });
        assert!(has_ack_change, "expected delayed_ack=3 on battery");
    }
}
