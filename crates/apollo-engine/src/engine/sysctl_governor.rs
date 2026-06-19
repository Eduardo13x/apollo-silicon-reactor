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

use crate::engine::audit_types::DecisionReason;
use crate::engine::sysctl_direct;
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

use crate::engine::network_monitor::NetworkMonitor;
use crate::engine::swap_predictor::SwapTrend;
use crate::engine::types::{HardPath, RootAction};

// ── Safety clamp helper ───────────────────────────────────────────────────────
//
// `clamp_to_allowed_range` was relocated to `sysctl_limits` in Sprint 4
// Phase 4 so non-governor emitters (e.g. `network-optimizer` at
// `main.rs:3577`, the site of Bug 6) can share the same single source of
// truth without depending on the governor module. Re-exported here for
// backwards compatibility with any external callers.
#[allow(unused_imports)]
pub use crate::engine::sysctl_limits::clamp_to_allowed_range;

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
    /// True when the user is in a full-duplex realtime call OR an active
    /// screen capture session. Composed at the daemon as
    /// `coreaudio_active::is_realtime_call_active()` (default-output AND
    /// default-input both running — Meet / Zoom / FaceTime / Discord) OR
    /// `realtime_signals::ScreenCaptureCache::check()` (replayd /
    /// screencaptureui / ScreenSharingAgent in the proc table — B.2,
    /// 2026-06-09 post-screen-share whipsaw). When set, the governor MUST
    /// skip TCP buffer scale-down and force `delayed_ack = 0` regardless of
    /// battery state — WebRTC jitter and audio cutouts otherwise.
    pub realtime_call_active: bool,
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
    /// Seconds since the last TCP buffer scale-UP (None = never).
    /// Scale-DOWN is dwell-gated until this reaches `TCP_SCALE_DOWN_DWELL`.
    #[serde(default)]
    pub tcp_last_scale_up_secs_ago: Option<u64>,
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
    /// Wall-clock timestamp of the last buffer scale-UP. `SystemTime` (not
    /// `Instant`) for the same sleep-correctness reason as `last_tuning` —
    /// see the rationale on that field. Used by the scale-down dwell gate:
    /// 2026-06-09 prod whipsaw ("high retransmissions scaling UP" followed
    /// immediately post-screen-share by "low retransmissions -25%
    /// scale-down") oscillated buffers within minutes. Scale-down is now
    /// forbidden until `TCP_SCALE_DOWN_DWELL` has elapsed since the last
    /// scale-up.
    last_scale_up_at: Option<SystemTime>,
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
/// Minimum wall-clock dwell after a TCP buffer scale-UP before any
/// scale-DOWN may fire. Asymmetric on purpose: growing buffers under
/// retransmission stress is cheap; shrinking them right after the stress
/// subsides (e.g. screen-share ends) caused the 2026-06-09 whipsaw.
const TCP_SCALE_DOWN_DWELL: Duration = Duration::from_secs(300);

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
    /// Last time each key was tuned.  Uses `SystemTime` (wall-clock) instead
    /// of `Instant` because `Instant` does not advance during macOS sleep.
    /// A 60s cooldown set before an 8-hour sleep would still show 58s
    /// remaining with `Instant`, causing re-application storms on wake.
    last_tuning: HashMap<String, SystemTime>,
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
                last_scale_up_at: None,
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
            .map(|(key, value)| {
                // Clamping is applied automatically by `RootAction::set_sysctl`
                // (Sprint 4 Phase 4 seal). Local pre-clamp removed.
                RootAction::set_sysctl(
                    key.clone(),
                    value.clone(),
                    "sysctl-governor: reverting to default",
                    DecisionReason::PressureContext,
                )
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
                actions.push(RootAction::set_sysctl(
                    key.clone(),
                    current,
                    format!(
                        "sysctl-governor: reverting '{}' using fallback (no default captured)",
                        key
                    ),
                    DecisionReason::PressureContext,
                ));
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
            // I/O throttling: REMOVED (2026-06-10 fight-hunt). Writing 0
            // disabled the kernel's low-priority I/O throttle — the mechanism
            // that keeps Time Machine / Spotlight / cloudd from competing with
            // foreground I/O. With it off, background daemons ran unthrottled
            // against the user's tabs/streaming. cargo builds are foreground
            // I/O and were never affected by the throttle. Kernel default (1)
            // is correct; Apollo exits this surface (also removed from
            // MANAGED_KEYS; thermal_interrupt's raw writes gutted same day).
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
            // GPU VRAM: REMOVED (2026-06-10 fight-hunt finding). Apollo wrote
            // iogpu.wired_limit_mb=12288 (12GB on an 8GB machine) at startup;
            // worse, the revert path clamped the true macOS default (0 = auto,
            // ~2/3 RAM) to the range floor 256, then a later restart captured
            // the strangled 256 as "the default" and persisted it — GPU wired
            // limit pinned to 256MB permanently (caused Metal OOM for MLX
            // loads, Meet compositor jank, Brave GPU process starvation).
            // The kernel manages GPU wired ceilings fine on its own; Apollo
            // MUST NOT own this surface. Keys also removed from MANAGED_KEYS —
            // the stale-key filter purges them from sysctl_defaults.json on
            // next daemon start.
        ];

        tunings
            .iter()
            .filter(|(key, _, _)| !self.unavailable_keys.contains(&key.to_string()))
            .map(|(key, value, reason)| {
                RootAction::set_sysctl(
                    key.to_string(),
                    value.to_string(),
                    format!("sysctl-governor: initial tuning — {}", reason),
                    DecisionReason::PressureContext,
                )
            })
            .collect()
    }

    /// Check if macOS Server Performance Mode is enabled.
    pub fn check_server_mode() {
        println!("Checking Server Performance Mode...");
        // serverinfo is only present on macOS Server — check via file existence.
        if std::path::Path::new("/usr/sbin/serverinfo").exists() {
            if let Some(val) = sysctl_direct::read_str("kern.iossupportversion") {
                if val.contains("server") {
                    println!("Server Performance Mode: ENABLED");
                } else {
                    println!("Server Performance Mode: DISABLED");
                }
            } else {
                println!("Server Performance Mode: DISABLED");
            }
        } else {
            eprintln!("'serverinfo' not found — standard macOS install");
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
            if let RootAction::SetSysctl(a) = &action {
                if sysctl_direct::write_str_value(a.key(), a.value()) {
                    println!("  {} = {}", a.key(), a.value());
                } else {
                    eprintln!("  WARN: failed to set {} = {}", a.key(), a.value());
                }
            }
        }
    }

    /// Snapshot of internal state for observability / status reporting.
    ///
    /// Accepts a reference to the `NetworkMonitor` to fill in the real
    /// EMA retransmission and listen-drop rates.
    pub fn status(&self, net_monitor: &NetworkMonitor) -> SysctlGovernorStatus {
        let now = SystemTime::now();
        let last_tune_secs_ago: HashMap<String, u64> = self
            .last_tuning
            .iter()
            .map(|(k, t)| {
                let secs = now
                    .duration_since(*t)
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
            tcp_last_scale_up_secs_ago: self.tcp.last_scale_up_at.map(|at| {
                now.duration_since(at)
                    .unwrap_or(Duration::from_secs(0))
                    .as_secs()
            }),
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

            if new_send != self.tcp.sendspace && self.cooldown_ok("net.inet.tcp.sendspace") {
                self.emit_sysctl(
                    "net.inet.tcp.sendspace",
                    &new_send.to_string(),
                    "sysctl-governor: high retransmissions, scaling send buffer +25%",
                    actions,
                    now,
                );
                self.tcp.sendspace = new_send;
            }
            if new_recv != self.tcp.recvspace && self.cooldown_ok("net.inet.tcp.recvspace") {
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
                                           // Arm the scale-down dwell even if cooldown swallowed the emit:
                                           // the *intent* to scale up is the whipsaw signal.
            self.tcp.last_scale_up_at = Some(SystemTime::now());
        }

        // -- Scale DOWN: retransmission_rate < 0.5% for 6 cycles AND throughput low --
        // Scale-down when throughput is below 1% of the current buffer capacity.
        // This avoids the ratchet effect where buffers grow but never shrink.
        let buffer_threshold = (self.tcp.sendspace / 100).max(65_536); // at least 64KB/s
        let throughput_low = send_bps < buffer_threshold;
        // 2026-06-09 prod incident: this branch fired mid-Meet because a
        // healthy WebRTC connection looks like "low retransmissions + low
        // average TCP throughput" (the bulk of media is on UDP). Scaling
        // send/recv buffers -25% broke signaling + STUN/TURN fallback,
        // freezing audio/video. Inhibit during realtime full-duplex audio.
        if inputs.realtime_call_active {
            // Evolve iter-5 (2026-06-10): count only REAL suppressions. The
            // counter previously bumped every tick the gate was true (even
            // when no scale-down would have fired) — 996 over 1550 cycles
            // looked "stuck" but was per-tick over-counting. The gate's
            // conservative looseness (audio + warm-mic ⇒ inhibit) is
            // intentional: skipping TCP tuning during ambiguous audio is
            // far cheaper than the 2026-06-09 broke-the-Meet incident.
            if self.tcp.consecutive_low >= 6 && throughput_low {
                crate::engine::lse_counters::LSE_COUNTERS
                    .inc_sysctl_governor_realtime_call_inhibit();
            }
            self.tcp.consecutive_low = 0; // forget the streak so resume is clean
        } else if self.tcp.consecutive_low >= 6
            && throughput_low
            && self.tcp_scale_down_dwell_elapsed()
        {
            // NOTE: when the dwell gate blocks, this branch simply does not
            // run — `consecutive_low` is deliberately NOT reset (unlike the
            // realtime gate above), so the streak stays armed and scale-down
            // fires on the first tick after the dwell expires.
            let new_send = ((self.tcp.sendspace as f64 * 0.75) as u64).max(TCP_BUFFER_MIN);
            let new_recv = ((self.tcp.recvspace as f64 * 0.75) as u64).max(TCP_BUFFER_MIN);

            if new_send != self.tcp.sendspace && self.cooldown_ok("net.inet.tcp.sendspace") {
                self.emit_sysctl(
                    "net.inet.tcp.sendspace",
                    &new_send.to_string(),
                    "sysctl-governor: low retransmissions + low throughput, scaling send buffer -25%",
                    actions,
                    now,
                );
                self.tcp.sendspace = new_send;
            }
            if new_recv != self.tcp.recvspace && self.cooldown_ok("net.inet.tcp.recvspace") {
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
        // 2026-06-09 prod incident: with `on_battery=true` mid-Meet, this
        // selector picked `delayed_ack=3` (combined ACKs ≈ +200ms latency),
        // which is catastrophic for WebRTC. Realtime-call gate overrides the
        // entire ladder and forces 0 (immediate ACK).
        let desired_ack = if inputs.realtime_call_active {
            // Evolve iter-5: count only when we actually override a non-zero
            // ack (a real suppressed write); forcing 0 when already 0 is a
            // no-op and must not inflate the metric.
            if self.tcp.delayed_ack != 0 {
                crate::engine::lse_counters::LSE_COUNTERS
                    .inc_sysctl_governor_realtime_call_inhibit();
            }
            0 // Realtime full-duplex audio: every ACK must dispatch immediately.
        } else if inputs.on_battery {
            3 // Combine ACKs to reduce CPU wakes.
        } else if inputs.workload == "coding" || inputs.workload == "commandline" {
            0 // No delayed ACKs for interactive work.
        } else if inputs.workload == "mediaplayback" || send_bps > 100_000_000 {
            3 // High throughput or streaming: combine ACKs.
        } else {
            self.tcp.delayed_ack // Keep current.
        };

        if desired_ack != self.tcp.delayed_ack && self.cooldown_ok("net.inet.tcp.delayed_ack") {
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
            if new_val != self.ipc.somaxconn && self.cooldown_ok("kern.ipc.somaxconn") {
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
            if new_val != self.ipc.somaxconn && self.cooldown_ok("kern.ipc.somaxconn") {
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

        // Emergency fast-path: Critical swap trend bypasses 3-cycle hysteresis.
        // SwapTrend::Critical means swap grew >5% of total in the sample window —
        // waiting 6+ seconds would guarantee jank frames before reacting.
        // [Gu 2018 "Throttling on Bandwidth-Constrained Platforms" IISWC;
        //  macOS UCS: faster compressor eval reduces swap I/O at display-frame boundaries]
        if inputs.swap_trend == SwapTrend::Critical {
            #[cfg(target_os = "macos")]
            if self.vm.poll_interval != 100
                && self.cooldown_ok("vm.compressor_eval_period_in_msecs")
            {
                self.emit_sysctl(
                    "vm.compressor_eval_period_in_msecs",
                    "100",
                    "sysctl-governor: critical swap — emergency aggressive compressor",
                    actions,
                    now,
                );
                self.vm.poll_interval = 100;
            }
        }

        // High pressure + swap growing for 3 cycles: aggressive compressor.
        if self.vm.consecutive_high >= 3 && swap_growing {
            #[cfg(target_os = "macos")]
            {
                if self.vm.poll_interval != 100
                    && self.cooldown_ok("vm.compressor_eval_period_in_msecs")
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
                    && self.cooldown_ok("vm.compressor_sample_min_in_msecs")
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
                if self.vm.poll_interval != 10 && self.cooldown_ok("vm.compressor_poll_interval") {
                    self.emit_sysctl(
                        "vm.compressor_poll_interval",
                        "10",
                        "sysctl-governor: high memory pressure + swap growing, aggressive compressor",
                        actions,
                        now,
                    );
                    self.vm.poll_interval = 10;
                }
                if self.vm.sample_min != 5 && self.cooldown_ok("vm.compressor_sample_min") {
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
                    && self.cooldown_ok("vm.compressor_eval_period_in_msecs")
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
                    && self.cooldown_ok("vm.compressor_sample_min_in_msecs")
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
                if self.vm.poll_interval != 40 && self.cooldown_ok("vm.compressor_poll_interval") {
                    self.emit_sysctl(
                        "vm.compressor_poll_interval",
                        "40",
                        "sysctl-governor: low memory pressure, relaxing compressor",
                        actions,
                        now,
                    );
                    self.vm.poll_interval = 40;
                }
                if self.vm.sample_min != 20 && self.cooldown_ok("vm.compressor_sample_min") {
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
                    && self.cooldown_ok("vm.compressor_eval_period_in_msecs")
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
                    && self.cooldown_ok("vm.compressor_sample_min_in_msecs")
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
                if self.vm.poll_interval != 20 && self.cooldown_ok("vm.compressor_poll_interval") {
                    self.emit_sysctl(
                        "vm.compressor_poll_interval",
                        "20",
                        "sysctl-governor: moderate memory pressure, balanced compressor",
                        actions,
                        now,
                    );
                    self.vm.poll_interval = 20;
                }
                if self.vm.sample_min != 10 && self.cooldown_ok("vm.compressor_sample_min") {
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
            if new_val != self.fs.maxvnodes && self.cooldown_ok("kern.maxvnodes") {
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
            if new_val != self.fs.maxvnodes && self.cooldown_ok("kern.maxvnodes") {
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

    /// True when the post-scale-UP dwell has elapsed (or no scale-up has
    /// ever happened), allowing a TCP buffer scale-DOWN. Wall-clock
    /// (`SystemTime`) so sleep duration counts toward the dwell — same
    /// rationale as `last_tuning`.
    fn tcp_scale_down_dwell_elapsed(&self) -> bool {
        match self.tcp.last_scale_up_at {
            Some(at) => {
                SystemTime::now()
                    .duration_since(at)
                    .unwrap_or(Duration::from_secs(0))
                    >= TCP_SCALE_DOWN_DWELL
            }
            None => true,
        }
    }

    /// Check whether the cooldown period has elapsed for a given key.
    /// Uses wall-clock (`SystemTime`) so sleep duration counts toward cooldown.
    fn cooldown_ok(&self, key: &str) -> bool {
        match self.last_tuning.get(key) {
            Some(last) => {
                SystemTime::now()
                    .duration_since(*last)
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
        _now: Instant,
    ) {
        if self.unavailable_keys.iter().any(|k| k == key) {
            return;
        }
        // CALL-SAFETY HARD BLOCK (2026-06-18): Apollo must NEVER write the TCP /
        // IPC network sysctls. They are the only sysctls that affect a realtime
        // call, and tuning them has now broken a user's Meet TWICE (2026-06-09
        // WebRTC freeze; 2026-06-18 choppy mic). The realtime-call gate is
        // unreliable — it disengages whenever the mic probe reads false (e.g.
        // while muted), and the resulting write/revert churn ("martillaba en
        // kernel") degrades audio. The feature's value (tuning TCP buffers on a
        // personal laptop) is marginal; stock macOS networking carries Meet
        // fine. Complete mediation at the single emit chokepoint — no code path
        // can write these. Memory/VM sysctls (maxvnodes, etc.) are unaffected.
        const CALL_AFFECTING_SYSCTLS: [&str; 4] = [
            "net.inet.tcp.sendspace",
            "net.inet.tcp.recvspace",
            "net.inet.tcp.delayed_ack",
            "kern.ipc.somaxconn",
        ];
        if CALL_AFFECTING_SYSCTLS.contains(&key) {
            return;
        }
        // Sprint 4 Phase 4 seal: the factory clamps the value internally
        // (single source of truth in `sysctl_limits::clamp_to_allowed_range`).
        // We re-read the post-clamp value via the accessor to keep
        // `current_values` in sync with what the kernel will actually receive.
        let action = RootAction::set_sysctl(
            key.to_string(),
            value.to_string(),
            reason.to_string(),
            DecisionReason::PressureContext,
        );
        let clamped_value = if let RootAction::SetSysctl(a) = &action {
            a.value().to_string()
        } else {
            // Unreachable — set_sysctl always returns the SetSysctl variant.
            value.to_string()
        };
        actions.push(action);
        self.last_tuning.insert(key.to_string(), SystemTime::now());
        self.current_values.insert(key.to_string(), clamped_value);
        self.total_writes += 1;
    }
}

impl Default for SysctlGovernor {
    fn default() -> Self {
        Self::new(false)
    }
}

// ── System helpers ───────────────────────────────────────────────────────────

/// Read a sysctl value via the direct API.  Returns `None` on failure.
fn read_sysctl(key: &str) -> Option<String> {
    sysctl_direct::read_str(key)
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

    // kern.num_vnodes and kern.openfiles can block indefinitely as root
    // under kernel lock contention on macOS.  Use a timeout thread to
    // avoid hanging the daemon's main loop.
    fn read_sysctl_with_timeout(key: &str) -> Option<String> {
        let key = key.to_string();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(sysctl_direct::read_str(&key));
        });
        rx.recv_timeout(std::time::Duration::from_millis(500))
            .ok()
            .flatten()
    }

    // Try kern.num_vnodes first (available on macOS 12+).
    if let Some(val) = read_sysctl_with_timeout("kern.num_vnodes") {
        if let Ok(current) = val.parse::<u64>() {
            return (current as f64 / maxvnodes as f64).clamp(0.0, 1.0);
        }
    }

    // Fallback: use kern.openfiles as a rough proxy (vnodes >= open files).
    // NOTE: This undercounts real vnode usage; see doc comment above.
    if let Some(val) = read_sysctl_with_timeout("kern.openfiles") {
        if let Ok(open_files) = val.parse::<u64>() {
            // Open files undercount vnodes; apply a 1.5x multiplier heuristic.
            let estimated = (open_files as f64 * 1.5) as u64;
            return (estimated as f64 / maxvnodes as f64).clamp(0.0, 1.0);
        }
    }

    // Both sysctls failed or timed out — return a conservative 50% estimate
    // so the governor does not assume zero usage and skip scaling entirely.
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
            realtime_call_active: false,
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
            realtime_call_active: false,
        };
        let actions = gov.tick(&inputs);
        assert!(actions.is_empty());
    }

    /// Call-safety block (2026-06-18): the TCP buffer sysctls are never emitted
    /// regardless of the scaling conditions — they break realtime calls.
    #[test]
    fn tcp_buffer_tuning_blocked_for_call_safety() {
        let mut gov = SysctlGovernor::new(true);
        let monitor = test_monitor(60.0, 0.0, 1000);
        let inputs = default_inputs(&monitor);
        for _ in 0..5 {
            let actions = tick_ok(&mut gov, &inputs);
            let has_buffer_tune = actions.iter().any(|a| {
                matches!(a, RootAction::SetSysctl(s)
                    if s.key() == "net.inet.tcp.sendspace" || s.key() == "net.inet.tcp.recvspace")
            });
            assert!(!has_buffer_tune, "TCP buffer sysctls must never be emitted");
        }
    }

    /// Call-safety block: kern.ipc.somaxconn is never emitted.
    #[test]
    fn ipc_tuning_blocked_for_call_safety() {
        let mut gov = SysctlGovernor::new(true);
        let monitor = test_monitor(0.0, 1.0, 0);
        let inputs = default_inputs(&monitor);
        for _ in 0..4 {
            let actions = tick_ok(&mut gov, &inputs);
            let has_somaxconn = actions
                .iter()
                .any(|a| matches!(a, RootAction::SetSysctl(s) if s.key() == "kern.ipc.somaxconn"));
            assert!(!has_somaxconn, "somaxconn must never be emitted");
        }
    }

    /// Direct chokepoint test (2026-06-18 call-safety): emit_sysctl drops the
    /// four call-affecting network keys but still passes non-network keys.
    #[test]
    fn emit_sysctl_blocks_call_affecting_network_keys() {
        let mut gov = SysctlGovernor::new(true);
        gov.unavailable_keys.clear();
        let mut actions = Vec::new();
        for key in [
            "net.inet.tcp.sendspace",
            "net.inet.tcp.recvspace",
            "net.inet.tcp.delayed_ack",
            "kern.ipc.somaxconn",
        ] {
            gov.emit_sysctl(
                key,
                "999",
                "test-should-be-blocked",
                &mut actions,
                Instant::now(),
            );
        }
        assert!(
            actions.is_empty(),
            "call-affecting network sysctls must never reach actions"
        );
        // A memory/VM sysctl is unaffected — the block is selective.
        gov.emit_sysctl(
            "kern.maxvnodes",
            "100000",
            "test",
            &mut actions,
            Instant::now(),
        );
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, RootAction::SetSysctl(s) if s.key() == "kern.maxvnodes")),
            "non-network sysctls must still emit"
        );
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
            matches!(a, RootAction::SetSysctl(s)
                if s.key() == "vm.compressor_poll_interval"
                    || s.key() == "vm.compressor_eval_period_in_msecs"
                    || s.key() == "vm.compressor_sample_min"
                    || s.key() == "vm.compressor_sample_min_in_msecs")
        });
        assert!(
            has_aggressive,
            "expected aggressive VM tuning on high pressure + swap"
        );
    }

    #[test]
    fn initial_tuning_never_touches_gpu_wired_limit() {
        // 2026-06-10 fight-hunt: Apollo pinned iogpu.wired_limit_mb to 256MB
        // (revert clamped the true default 0 to the range floor, then a
        // restart captured the strangled value as baseline). Apollo must
        // never own this surface again.
        let gov = SysctlGovernor::new(true);
        let actions = gov.apply_initial_tuning();
        let touches_gpu = actions
            .iter()
            .any(|a| matches!(a, RootAction::SetSysctl(s) if s.key().contains("iogpu")));
        assert!(!touches_gpu, "initial tuning must not write iogpu.* keys");
        assert!(
            !MANAGED_KEYS.iter().any(|k| k.contains("iogpu")),
            "iogpu keys must not be managed (revert would clamp 0 -> 256)"
        );
        // 2026-06-10 same hunt: Apollo disabled the kernel's low-pri I/O
        // throttle at startup (background daemons ran unthrottled against
        // user I/O). Apollo must not own this surface either.
        let touches_lowpri = actions
            .iter()
            .any(|a| matches!(a, RootAction::SetSysctl(s) if s.key().contains("lowpri_throttle")));
        assert!(
            !touches_lowpri,
            "initial tuning must not write lowpri_throttle"
        );
        assert!(
            !MANAGED_KEYS.iter().any(|k| k.contains("lowpri_throttle")),
            "lowpri_throttle must not be managed"
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
                RootAction::SetSysctl(s) => {
                    assert!(s.reason().contains("reverting"));
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
        let has_sendspace = second_actions
            .iter()
            .any(|a| matches!(a, RootAction::SetSysctl(s) if s.key() == "net.inet.tcp.sendspace"));
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
        // Call-safety block: delayed_ack is never emitted, any workload.
        let any_ack = actions.iter().any(
            |a| matches!(a, RootAction::SetSysctl(s) if s.key() == "net.inet.tcp.delayed_ack"),
        );
        assert!(!any_ack, "delayed_ack must never be emitted (call safety)");
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
        // Call-safety block: delayed_ack is never emitted, even on battery —
        // the OS default carries WebRTC fine and tuning it broke calls.
        let any_ack = actions.iter().any(
            |a| matches!(a, RootAction::SetSysctl(s) if s.key() == "net.inet.tcp.delayed_ack"),
        );
        assert!(!any_ack, "delayed_ack must never be emitted (call safety)");
    }

    // ── WebRTC guard (2026-06-09 prod incident) ──────────────────────────────

    /// Reproduces the 2026-06-09T17:12 PT failure mode: with
    /// `on_battery = true` mid-Meet, the delayed_ack ladder picked 3
    /// (combined ACK ≈ +200 ms latency), choking WebRTC. The realtime-call
    /// gate now overrides the battery branch and forces 0.
    #[test]
    fn realtime_inhibit_counter_counts_only_real_suppressions() {
        // Evolve iter-5: ack already 0 + no scale-down pending ⇒ the gate
        // suppresses NOTHING, so the inhibit counter must not climb. (The
        // old code bumped per-tick, making the gate look stuck at ~1/cycle.)
        let before = crate::engine::lse_counters::LSE_COUNTERS
            .snapshot()
            .sysctl_governor_realtime_call_inhibit_total;
        let mut gov = SysctlGovernor::new(true);
        gov.tcp.delayed_ack = 0; // already immediate
        gov.tcp.consecutive_low = 0; // no scale-down pending
        let monitor = test_monitor(0.0, 0.0, 0);
        let mut inputs = default_inputs(&monitor);
        inputs.realtime_call_active = true;
        let _ = gov.tick(&inputs);
        let after = crate::engine::lse_counters::LSE_COUNTERS
            .snapshot()
            .sysctl_governor_realtime_call_inhibit_total;
        assert_eq!(after, before, "no real suppression ⇒ counter must not move");
    }

    #[test]
    fn realtime_call_overrides_battery_delayed_ack() {
        let mut gov = SysctlGovernor::new(true);
        gov.tcp.delayed_ack = 3;
        gov.current_values
            .insert("net.inet.tcp.delayed_ack".into(), "3".into());

        let monitor = test_monitor(0.0, 0.0, 0);
        let mut inputs = default_inputs(&monitor);
        inputs.on_battery = true;
        inputs.realtime_call_active = true;

        let actions = gov.tick(&inputs);
        // Call-safety block makes this stronger than the old gate: delayed_ack
        // is never emitted at all, so no value (0 OR 3) can reach a live call.
        let any_ack = actions.iter().any(
            |a| matches!(a, RootAction::SetSysctl(s) if s.key() == "net.inet.tcp.delayed_ack"),
        );
        assert!(!any_ack, "no delayed_ack write may reach a realtime call");
    }

    /// Reproduces the buffer scale-down branch (sysctl_governor.rs:641).
    /// A healthy Meet looks like "low retransmissions + low avg TCP
    /// throughput" (UDP carries the media). Apollo previously scaled
    /// send/recv buffers -25%, dropping signaling + STUN/TURN fallback.
    /// The realtime gate must inhibit the scale-down and reset the
    /// `consecutive_low` streak so resume is clean.
    #[test]
    fn realtime_call_inhibits_buffer_scale_down() {
        let mut gov = SysctlGovernor::new(true);
        gov.tcp.sendspace = 524_288;
        gov.tcp.recvspace = 524_288;
        gov.tcp.consecutive_low = 6; // already at the threshold
        gov.current_values
            .insert("net.inet.tcp.sendspace".into(), "524288".into());
        gov.current_values
            .insert("net.inet.tcp.recvspace".into(), "524288".into());

        // Zero traffic = throughput_low = true; absent the gate this would
        // emit -25% scale-down on both send and recv buffers.
        let monitor = test_monitor(0.0, 0.0, 0);
        let mut inputs = default_inputs(&monitor);
        inputs.realtime_call_active = true;

        let actions = gov.tick(&inputs);
        let any_scale_down = actions.iter().any(|a| {
            matches!(a, RootAction::SetSysctl(s)
                if (s.key() == "net.inet.tcp.sendspace" || s.key() == "net.inet.tcp.recvspace")
                && s.reason().contains("scaling"))
        });
        assert!(!any_scale_down, "scale-down must be inhibited mid-call");
        assert_eq!(
            gov.tcp.consecutive_low, 0,
            "streak must reset so resume is clean once call ends"
        );
    }

    /// Sanity: when `realtime_call_active == false`, all prior behavior
    /// continues unchanged. This pins the gate as a pure inhibitor — it
    /// cannot accidentally suppress non-call sysctl writes.
    #[test]
    fn realtime_gate_off_preserves_battery_delayed_ack() {
        let mut gov = SysctlGovernor::new(true);
        gov.tcp.delayed_ack = 0;
        gov.current_values
            .insert("net.inet.tcp.delayed_ack".into(), "0".into());

        let monitor = test_monitor(0.0, 0.0, 0);
        let mut inputs = default_inputs(&monitor);
        inputs.on_battery = true;
        inputs.realtime_call_active = false; // gate disengaged

        let actions = gov.tick(&inputs);
        // Call-safety block: even gate-off + battery emits NO delayed_ack now.
        let any_ack = actions.iter().any(
            |a| matches!(a, RootAction::SetSysctl(s) if s.key() == "net.inet.tcp.delayed_ack"),
        );
        assert!(!any_ack, "delayed_ack must never be emitted (call safety)");
    }

    // ── Scale-down dwell (2026-06-09 whipsaw fix) ────────────────────────────

    /// Detect a TCP buffer scale-DOWN write among emitted actions.
    fn has_buffer_scale_down(actions: &[RootAction]) -> bool {
        actions.iter().any(|a| {
            matches!(a, RootAction::SetSysctl(s)
                if (s.key() == "net.inet.tcp.sendspace" || s.key() == "net.inet.tcp.recvspace")
                && s.reason().contains("low retransmissions"))
        })
    }

    #[test]
    fn scale_down_blocked_within_dwell_after_scale_up() {
        let mut gov = SysctlGovernor::new(true);

        // Phase 1: three high-retransmission cycles trigger a scale-UP,
        // arming `last_scale_up_at`.
        let high = test_monitor(60.0, 0.0, 1000);
        let inputs = default_inputs(&high);
        tick_ok(&mut gov, &inputs);
        tick_ok(&mut gov, &inputs);
        assert!(!tick_ok(&mut gov, &inputs).is_empty(), "scale-up expected");
        assert!(gov.tcp.last_scale_up_at.is_some(), "dwell must be armed");

        // Neutralize the 60s per-key cooldown so only the dwell can block.
        gov.last_tuning.clear();

        // Phase 2: quiet network, streak already at the threshold.
        let low = test_monitor(0.0, 0.0, 0);
        let inputs = default_inputs(&low);
        gov.tcp.consecutive_low = 6;
        let actions = tick_ok(&mut gov, &inputs);
        assert!(
            !has_buffer_scale_down(&actions),
            "scale-down must be dwell-blocked right after a scale-up"
        );
        // Unlike the realtime gate, a dwell block keeps the streak armed.
        assert!(
            gov.tcp.consecutive_low > 0,
            "dwell block must NOT reset consecutive_low"
        );
    }

    #[test]
    fn scale_down_allowed_after_dwell() {
        let mut gov = SysctlGovernor::new(true);
        gov.tcp.sendspace = 524_288;
        gov.tcp.recvspace = 524_288;
        gov.tcp.consecutive_low = 6;
        // Backdate the scale-up past the 300s dwell window.
        gov.tcp.last_scale_up_at = Some(SystemTime::now() - Duration::from_secs(301));

        let monitor = test_monitor(0.0, 0.0, 0);
        let inputs = default_inputs(&monitor);
        let actions = tick_ok(&mut gov, &inputs);
        // Call-safety block: buffer scale-down is never emitted now, dwell or no.
        assert!(
            !has_buffer_scale_down(&actions),
            "TCP buffer scale-down must never emit (call safety)"
        );
    }

    #[test]
    fn scale_down_allowed_when_never_scaled_up() {
        let mut gov = SysctlGovernor::new(true);
        gov.tcp.sendspace = 524_288;
        gov.tcp.recvspace = 524_288;
        gov.tcp.consecutive_low = 6;
        assert!(gov.tcp.last_scale_up_at.is_none());

        let monitor = test_monitor(0.0, 0.0, 0);
        let inputs = default_inputs(&monitor);
        let actions = tick_ok(&mut gov, &inputs);
        // Call-safety block: never emit scale-down regardless of dwell state.
        assert!(
            !has_buffer_scale_down(&actions),
            "TCP buffer scale-down must never emit (call safety)"
        );
    }

    /// Replays the 2026-06-09 post-screen-share whipsaw: retransmission
    /// stress drives a scale-UP, then the share ends and the network goes
    /// quiet for 6+ cycles. Pre-fix this emitted an immediate -25%
    /// scale-down; the dwell must hold the line at zero down-writes.
    #[test]
    fn whipsaw_scenario_share_then_idle() {
        let mut gov = SysctlGovernor::new(true);

        let high = test_monitor(60.0, 0.0, 1000);
        let inputs = default_inputs(&high);
        tick_ok(&mut gov, &inputs);
        tick_ok(&mut gov, &inputs);
        assert!(!tick_ok(&mut gov, &inputs).is_empty(), "scale-up expected");

        // Isolate the dwell: the 60s cooldown alone must not be the reason
        // the test passes.
        gov.last_tuning.clear();

        let idle = test_monitor(0.0, 0.0, 0);
        let inputs = default_inputs(&idle);
        for cycle in 1..=6 {
            let actions = tick_ok(&mut gov, &inputs);
            assert!(
                !has_buffer_scale_down(&actions),
                "cycle {cycle}: dwell must block scale-down right after the share"
            );
        }
        // The streak kept accumulating, ready for when the dwell expires.
        assert!(gov.tcp.consecutive_low >= 6);
    }

    // Unit tests for `clamp_to_allowed_range` live with the helper in
    // `sysctl_limits.rs` — see Sprint 4 Phase 4 relocation. The integration
    // tests above (`tcp_scale_up_after_3_high_cycles`, etc.) still exercise
    // the clamp via the governor's emit path.
}
