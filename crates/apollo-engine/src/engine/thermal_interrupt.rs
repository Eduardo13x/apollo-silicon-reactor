//! Resource Sentinel — sub-100ms interrupt handler for thermal, memory, and power emergencies.
//!
//! Runs as a dedicated thread ("resource-sentinel") that monitors the SmcReader
//! and PressureCollector caches plus reactor signals. When a resource emergency
//! is detected, it emits bounded typed proposals and wakes the daemon loop. The
//! observer never owns an effector and never performs kernel mutation.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::engine::activity_sensor::active_pids;
use crate::engine::background_collectors::PressureData;
use crate::engine::decision_ledger::{
    ActuatorDecisionEvent, ActuatorDecisionOutcome, CycleDecisionEvents,
};
use crate::engine::foreground::ForegroundDetector;
use crate::engine::iokit_sensors::HardwareSnapshot;
use crate::engine::lock_ext::LockRecover;
use crate::engine::mach_qos::MachQoSManager;
use crate::engine::process_identity::ProcessIdentity;
use crate::engine::types::{FreezeSource, FrozenEntry};

// ── Interrupt Phase ──────────────────────────────────────────────────────────

/// Severity phase of the resource interrupt handler.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptPhase {
    /// No resource pressure.
    Idle = 0,
    /// Moderate pressure: thermal ≥90°C OR memory pressure ≥0.80.
    Moderate = 1,
    /// Emergency: thermal ≥95°C OR memory critical + swap thrash.
    Emergency = 2,
    /// Super-emergency: thermal ≥100°C OR dangerous rate-of-rise.
    SuperEmergency = 3,
}

impl InterruptPhase {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Moderate,
            2 => Self::Emergency,
            3 => Self::SuperEmergency,
            _ => Self::Idle,
        }
    }
}

// ── Shared State (lock-free) ─────────────────────────────────────────────────

/// Lock-free shared state between the sentinel thread, reactor, and main loop.
pub struct ResourceInterruptState {
    /// Current interrupt phase (read by main loop, written by sentinel).
    pub phase: AtomicU8,
    /// Whether the sentinel thread is currently active/responding.
    pub active: AtomicBool,
    /// Monotonic sequence number incremented on each phase transition.
    pub sequence: AtomicU64,

    // Signals from reactor (set by reactor, read+cleared by sentinel).
    /// Thermal event ≥ serious detected by reactor.
    pub thermal_signal: AtomicBool,
    /// Memory pressure event detected by reactor.
    pub memory_signal: AtomicBool,
    /// Power source change detected by reactor.
    pub power_signal: AtomicBool,

    /// PIDs frozen by the interrupt handler (separate from main loop freezes).
    pub interrupt_frozen_pids: Mutex<HashSet<u32>>,
    /// Fight-hunt fix (2026-06-10): PIDs the resource-interrupt path migrated
    /// to E-cores/Darwin-BG during Moderate/Emergency phases. recover()
    /// previously only SIGCONT'd the frozen set — every migrated process
    /// stayed pinned to Background tier AFTER the thermal event ended
    /// (a Meet call heats the M1 → mass demotion → call ends → system
    /// permanently sluggish until reboot). recover() now restores these
    /// to Normal and clears the set.
    pub interrupt_migrated_pids: Mutex<HashSet<u32>>,

    // Observability counters.
    pub total_fires: AtomicU64,
    pub total_frozen: AtomicU64,
    pub total_migrated: AtomicU64,
    pub total_recoveries: AtomicU64,
    /// Latency of the last sentinel observation/proposal tick in microseconds.
    pub last_latency_us: AtomicU64,
}

impl ResourceInterruptState {
    pub fn new() -> Self {
        Self {
            phase: AtomicU8::new(0),
            active: AtomicBool::new(false),
            sequence: AtomicU64::new(0),
            thermal_signal: AtomicBool::new(false),
            memory_signal: AtomicBool::new(false),
            power_signal: AtomicBool::new(false),
            interrupt_frozen_pids: Mutex::new(HashSet::new()),
            interrupt_migrated_pids: Mutex::new(HashSet::new()),
            total_fires: AtomicU64::new(0),
            total_frozen: AtomicU64::new(0),
            total_migrated: AtomicU64::new(0),
            total_recoveries: AtomicU64::new(0),
            last_latency_us: AtomicU64::new(0),
        }
    }

    /// Read the current phase without locking.
    pub fn current_phase(&self) -> InterruptPhase {
        InterruptPhase::from_u8(self.phase.load(Ordering::Acquire))
    }
}

impl Default for ResourceInterruptState {
    fn default() -> Self {
        Self::new()
    }
}

const RESOURCE_INTERRUPT_PROPOSAL_CAPACITY: usize = 128;
const RESOURCE_SENTINEL_WAKE_CAPACITY: usize = 1;

/// Lifecycle handle retained by the loop so proposal production can be stopped
/// and observed without ever joining the background observer.
#[must_use = "retain the handle and quiesce the resource sentinel before shutdown"]
pub struct ResourceSentinelShutdownHandle {
    stop: Arc<AtomicBool>,
    wake: SyncSender<()>,
    quiesced: Receiver<()>,
    worker: Option<JoinHandle<()>>,
    acknowledged: bool,
    revoke_on_drop: bool,
}

impl ResourceSentinelShutdownHandle {
    /// Request observer shutdown and interrupt its poll sleep. This never waits.
    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::Release);
        let _ = self.wake.try_send(());
    }

    /// Wait at most `timeout` for the observer to finish proposal production.
    pub fn wait_for_quiescence(&mut self, timeout: Duration) -> bool {
        if self.acknowledged {
            return true;
        }
        self.acknowledged = match self.quiesced.recv_timeout(timeout) {
            Ok(()) | Err(RecvTimeoutError::Disconnected) => true,
            Err(RecvTimeoutError::Timeout) => false,
        };
        self.acknowledged
    }

    pub fn quiesce(&mut self, timeout: Duration) -> bool {
        self.request_stop();
        self.wait_for_quiescence(timeout)
    }

    pub fn worker_finished(&self) -> bool {
        self.worker.as_ref().is_none_or(JoinHandle::is_finished)
    }

    fn detach_legacy(mut self) {
        self.revoke_on_drop = false;
    }
}

impl Drop for ResourceSentinelShutdownHandle {
    fn drop(&mut self) {
        if self.revoke_on_drop {
            self.request_stop();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceInterruptUnfreezeReason {
    Foreground,
    Recovery,
}

/// Typed observation emitted by the resource sentinel. This type deliberately
/// has no execute method and carries no effector, broker, or kernel authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceInterruptProposal {
    MigrateToBackground {
        pid: u32,
        name: String,
        start_sec: u64,
        start_usec: u64,
    },
    Freeze {
        pid: u32,
        name: String,
        start_sec: u64,
        start_usec: u64,
    },
    RestoreScheduling {
        pid: u32,
        name: String,
        start_sec: u64,
        start_usec: u64,
    },
    Unfreeze {
        pid: u32,
        name: String,
        start_sec: u64,
        start_usec: u64,
        reason: ResourceInterruptUnfreezeReason,
    },
}

impl ResourceInterruptProposal {
    pub fn pid(&self) -> u32 {
        match self {
            Self::MigrateToBackground { pid, .. }
            | Self::Freeze { pid, .. }
            | Self::RestoreScheduling { pid, .. }
            | Self::Unfreeze { pid, .. } => *pid,
        }
    }

    pub fn action_key(&self) -> &'static str {
        match self {
            Self::MigrateToBackground { .. } => "thermal_interrupt:scheduling_background",
            Self::Freeze { .. } => "thermal_interrupt:freeze",
            Self::RestoreScheduling { .. } => "thermal_interrupt:qos_restore",
            Self::Unfreeze {
                reason: ResourceInterruptUnfreezeReason::Foreground,
                ..
            } => "thermal_interrupt:foreground_sigcont",
            Self::Unfreeze {
                reason: ResourceInterruptUnfreezeReason::Recovery,
                ..
            } => "thermal_interrupt:sigcont_recovery",
        }
    }

    pub fn event(
        &self,
        cycle: u64,
        outcome: ActuatorDecisionOutcome,
        detail: impl Into<String>,
    ) -> ActuatorDecisionEvent {
        ActuatorDecisionEvent::local(
            self.action_key(),
            format!("pid:{}", self.pid()),
            cycle,
            outcome,
            "resource-interrupt-main-loop",
            detail,
        )
    }
}

/// Non-blocking producer retained by the observation-only sentinel. Channel
/// loss is propagated into the loop's bounded overflow summary.
#[derive(Clone)]
pub struct ResourceInterruptProposalSender {
    sender: SyncSender<ResourceInterruptProposal>,
    dropped: Arc<AtomicU64>,
    accepting: Arc<AtomicBool>,
    cycle_wake: Option<Arc<(Mutex<bool>, Condvar)>>,
}

impl ResourceInterruptProposalSender {
    pub fn propose(&self, proposal: ResourceInterruptProposal) -> bool {
        if !self.accepting.load(Ordering::Acquire) {
            return false;
        }
        match self.sender.try_send(proposal) {
            Ok(()) => {
                if let Some(wake) = &self.cycle_wake {
                    let (triggered, condvar) = &**wake;
                    *triggered.lock_recover() = true;
                    condvar.notify_one();
                }
                true
            }
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }
}

/// Bounded loop-owned batch. Proposals become auditable decisions only when
/// this batch is resolved by the loop's actuation window.
pub struct ResourceInterruptProposalBatch {
    proposals: Vec<ResourceInterruptProposal>,
    dropped: u64,
}

impl ResourceInterruptProposalBatch {
    pub fn len(&self) -> usize {
        self.proposals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.proposals.is_empty()
    }

    pub fn into_proposals(self) -> Vec<ResourceInterruptProposal> {
        self.proposals
    }

    pub fn dropped(&self) -> u64 {
        self.dropped
    }
}

/// Loop-owned consumer for bounded sentinel observations.
pub struct ResourceInterruptProposalReceiver {
    receiver: Receiver<ResourceInterruptProposal>,
    dropped: Arc<AtomicU64>,
    accepting: Arc<AtomicBool>,
}

impl ResourceInterruptProposalReceiver {
    pub fn drain(&self) -> ResourceInterruptProposalBatch {
        let mut proposals = Vec::with_capacity(RESOURCE_INTERRUPT_PROPOSAL_CAPACITY);
        for _ in 0..RESOURCE_INTERRUPT_PROPOSAL_CAPACITY {
            let proposal = match self.receiver.try_recv() {
                Ok(proposal) => proposal,
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            };
            proposals.push(proposal);
        }
        ResourceInterruptProposalBatch {
            proposals,
            dropped: self.dropped.swap(0, Ordering::AcqRel),
        }
    }

    /// Stop accepting new observations before the shutdown-only expiry drain.
    pub fn close(&self) {
        self.accepting.store(false, Ordering::Release);
    }
}

pub fn resource_interrupt_proposal_channel(
    cycle_wake: Option<Arc<(Mutex<bool>, Condvar)>>,
) -> (
    ResourceInterruptProposalSender,
    ResourceInterruptProposalReceiver,
) {
    let (sender, receiver) = mpsc::sync_channel(RESOURCE_INTERRUPT_PROPOSAL_CAPACITY);
    let dropped = Arc::new(AtomicU64::new(0));
    let accepting = Arc::new(AtomicBool::new(true));
    (
        ResourceInterruptProposalSender {
            sender,
            dropped: dropped.clone(),
            accepting: accepting.clone(),
            cycle_wake,
        },
        ResourceInterruptProposalReceiver {
            receiver,
            dropped,
            accepting,
        },
    )
}

/// Main-loop-owned authority boundary. Because this value is not shared with
/// the observer, closing it and executing a proposal cannot race.
pub struct ResourceInterruptActuationWindow {
    open: bool,
}

impl ResourceInterruptActuationWindow {
    pub fn open() -> Self {
        Self { open: true }
    }

    pub fn close(&mut self) {
        self.open = false;
    }

    pub fn resolve(
        &self,
        batch: ResourceInterruptProposalBatch,
        cycle: u64,
        mut execute: impl FnMut(ResourceInterruptProposal) -> CycleDecisionEvents,
    ) -> CycleDecisionEvents {
        let mut events = CycleDecisionEvents::default();
        events.record_dropped(batch.dropped);
        for proposal in batch.proposals {
            if self.open {
                events.extend_buffer(&execute(proposal));
            } else {
                events.push(proposal.event(
                    cycle,
                    ActuatorDecisionOutcome::Expired,
                    "daemon loop closed before proposal execution",
                ));
            }
        }
        events
    }
}

// ── Configuration ────────────────────────────────────────────────────────────

/// Tunable parameters for the resource sentinel.
pub struct SentinelConfig {
    /// How often the sentinel polls caches (default: 500ms).
    pub poll_interval: Duration,
    /// Temperature threshold for Moderate phase.
    pub thermal_moderate_c: f32,
    /// Temperature threshold for Emergency phase.
    pub thermal_emergency_c: f32,
    /// Temperature threshold for SuperEmergency phase.
    pub thermal_super_emergency_c: f32,
    /// Memory pressure threshold for Moderate phase.
    pub memory_pressure_moderate: f64,
    /// Memory pressure threshold for Emergency phase.
    pub memory_pressure_emergency: f64,
    /// Hysteresis: must drop this many °C below threshold to downgrade phase.
    pub hysteresis_c: f32,
    /// Minimum time between phase escalations.
    pub debounce: Duration,
    /// Rate-of-rise threshold (°C/s) that triggers SuperEmergency.
    pub rate_of_rise_threshold: f32,
}

impl Default for SentinelConfig {
    fn default() -> Self {
        // Thresholds aligned with thermal_manager.rs throttle ramp:
        // throttle_threshold=90°C, shutdown_threshold=100°C
        Self {
            poll_interval: Duration::from_millis(500),
            thermal_moderate_c: 90.0,
            thermal_emergency_c: 95.0,
            thermal_super_emergency_c: 100.0,
            memory_pressure_moderate: 0.80,
            memory_pressure_emergency: 0.92,
            hysteresis_c: 5.0,
            debounce: Duration::from_secs(2),
            rate_of_rise_threshold: 1.0,
        }
    }
}

// ── Pre-allocated Buffers ────────────────────────────────────────────────────

/// Pre-allocated buffers to avoid allocations on the hot path.
struct SentinelBuffers {
    /// Ring buffer for temperature history (rate-of-rise calculation).
    temp_history: [(f32, Instant); 8],
    temp_idx: usize,
    /// Consecutive ticks where compute_phase would return SuperEmergency.
    /// Requires ≥2 before actually escalating (filters sensor glitches).
    consecutive_super: u8,
    /// Protected process names that must never be stopped.
    protected: HashSet<&'static str>,
    /// Essential system processes that must never be touched.
    essential: HashSet<&'static str>,
    /// Foreground detector: dynamically protects whatever app the user is using.
    fg_detector: Arc<ForegroundDetector>,
}

impl SentinelBuffers {
    fn new(fg_detector: Arc<ForegroundDetector>) -> Self {
        let now = Instant::now();
        let mut protected = HashSet::new();
        let mut essential = HashSet::new();

        // Essential: kernel, init, critical daemons.
        // Usar exact-match (is_essential usa ==) para evitar falsos positivos por substring.
        for name in [
            "kernel_task",
            "launchd",
            "logd",
            "notifyd",
            "WindowServer",
            "loginwindow",
            "opendirectoryd",
            "diskarbitrationd",
            "fseventsd",
            "mds",
            "mds_stores",
            "coreaudiod",
            "configd",
            "distnoted",
            "UserEventAgent",
            "SystemUIServer",
            "Dock",
            "Finder",
            // Seguridad y autenticación — freezarlos provoca deadlocks de UI
            "securityd",
            "secd",
            "trustd",
            "tccd",
            "syspolicyd",
            // Networking y resolución de nombres
            "mDNSResponder",
            "nsurlsessiond",
            "networkd",
            "configd",
            // Gestión de ventanas y accesibilidad
            "Dock",
            "SystemUIServer",
            "universalaccessd",
            "AXVisualSupportAgent",
            // I/O y filesystem
            "diskmanagementd",
            "homed",
            "containermanagerd",
            // Otros daemons críticos de sistema
            "runningboardd",
            "corebrightnessd",
            "powerd",
            "thermald",
            "syslogd",
            "aslmanager",
        ] {
            essential.insert(name);
        }

        // Protected: dev background workloads + ALL user-facing GUI apps.
        //
        // Critical insight: the sentinel has no access to CGWindowServer to know if a
        // process has a visible window. Instead we enumerate all known user-facing
        // apps explicitly. Any GUI app NOT in this list risks SIGSTOP when it has been
        // inactive > 5 min (the is_recently_active window), even while visible/minimized.
        //
        // [WWDC 2017 "Modernizing GCD Usage"] — user-interactive processes need
        // dedicated CPU; freezing them produces visible hangs and broken IPC.
        for name in [
            // Apollo itself
            "apollo-optimizerd",
            // Build tools — background but never safe to freeze mid-compile
            "node",
            "cargo",
            "rustc",
            "swift",
            "clang",
            "python3",
            "python",
            // Web browsers — all have multi-process architectures; freezing the main
            // process or a renderer hangs IPC and the OS may force-quit the app.
            "Brave Browser",
            "Brave Browser H", // Brave Helper (Renderer/GPU/Plugin)
            "Google Chrome",
            "Google Chrome H",
            "Safari",
            "SafariForWebKitDevel",
            "Firefox",
            "firefox",
            "Arc",
            "Microsoft Edge",
            // IDEs and editors
            "Xcode",
            "Code", // VS Code
            "Cursor",
            "Nova",
            "Zed",
            "RubyMine",
            "IntelliJ IDEA",
            // Terminals
            "Terminal",
            "iTerm2",
            "Warp",
            "Ghostty",
            "alacritty",
            "kitty",
            // Communication / collaboration
            "zoom.us",
            "Slack",
            "Teams",
            "Discord",
            "Telegram",
            "Signal",
            "FaceTime",
            // Media — active playback pipelines; SIGSTOP causes audio/video stutter
            "Spotify",
            "Music",
            "Podcasts",
            "QuickTime Player",
            // AI / LLM apps
            "Claude",
            "LM Studio",
            "Ollama",
            // Other common GUI apps
            "Finder",
            "Mail",
            "Calendar",
            "Notes",
            "Messages",
            "Antigravity",
        ] {
            protected.insert(name);
        }

        Self {
            temp_history: [(0.0, now); 8],
            temp_idx: 0,
            consecutive_super: 0,
            protected,
            essential,
            fg_detector,
        }
    }

    /// Record a temperature sample and return the rate-of-rise (°C/s).
    /// Rejects single-sample spikes >5°C as sensor glitches.
    fn record_temp(&mut self, temp_c: f32) -> f32 {
        let now = Instant::now();
        // Sensor sanity: reject discontinuities >5°C from the previous sample.
        // SMC sensors can spike on Apple Silicon; a real thermal event won't
        // jump 5°C in 500ms.
        let prev_idx = (self.temp_idx + 7) % 8; // previous sample
        let (prev_temp, _) = self.temp_history[prev_idx];
        let clamped = if prev_temp > 0.0 && (temp_c - prev_temp).abs() > 5.0 {
            prev_temp // ignore spike, reuse previous reading
        } else {
            temp_c
        };

        let oldest_idx = (self.temp_idx + 1) % 8;
        let (oldest_temp, oldest_time) = self.temp_history[oldest_idx];
        let dt = now.duration_since(oldest_time).as_secs_f32().max(0.01);
        let rate = (clamped - oldest_temp) / dt;

        self.temp_history[self.temp_idx] = (clamped, now);
        self.temp_idx = (self.temp_idx + 1) % 8;

        rate
    }

    /// Check if a process name is essential (never touch).
    ///
    /// Usa exact-match para evitar falsos positivos por substring (e.g. "mds" no debe
    /// proteger "tmds" ni "cmds"). Además, cualquier proceso cuyo nombre empiece con
    /// "com.apple." es un XPC service del sistema y nunca debe ser frozen.
    fn is_essential(&self, name: &str) -> bool {
        // Exact match contra la lista estática.
        if self.essential.contains(name) {
            return true;
        }
        // Guard adicional: XPC services de Apple (com.apple.WebKit.WebContent, etc.)
        // nunca deben ser frozen — son parte del sandbox de cualquier app con webview.
        if name.starts_with("com.apple.") {
            return true;
        }
        false
    }

    /// Check if a process name is protected (don't freeze, but may migrate).
    fn is_protected(&self, name: &str) -> bool {
        self.protected.iter().any(|p| name.contains(p))
    }
}

// ── Sentinel Thread ──────────────────────────────────────────────────────────

/// Spawn the resource sentinel thread.
///
/// The sentinel monitors the SmcReader and PressureCollector caches, emits a
/// bounded proposal, and wakes the daemon loop on resource emergencies.
pub fn spawn_resource_sentinel(
    hw_cache: Arc<Mutex<Option<HardwareSnapshot>>>,
    pressure_cache: Arc<Mutex<PressureData>>,
    interrupt_state: Arc<ResourceInterruptState>,
    main_frozen: Arc<Mutex<HashMap<u32, FrozenEntry>>>,
    stop: Arc<AtomicBool>,
    config: SentinelConfig,
    fg_detector: Arc<ForegroundDetector>,
    qos_mgr: Option<Arc<Mutex<MachQoSManager>>>,
    frozen_state_path: PathBuf,
) {
    let (proposals, _discarded) = resource_interrupt_proposal_channel(None);
    spawn_resource_sentinel_inner(
        hw_cache,
        pressure_cache,
        interrupt_state,
        main_frozen,
        stop,
        config,
        fg_detector,
        qos_mgr,
        frozen_state_path,
        proposals,
    )
    .detach_legacy();
}

/// Spawn the live observation-only sentinel with a bounded proposal handoff.
pub fn spawn_resource_sentinel_with_proposals(
    hw_cache: Arc<Mutex<Option<HardwareSnapshot>>>,
    pressure_cache: Arc<Mutex<PressureData>>,
    interrupt_state: Arc<ResourceInterruptState>,
    main_frozen: Arc<Mutex<HashMap<u32, FrozenEntry>>>,
    stop: Arc<AtomicBool>,
    config: SentinelConfig,
    fg_detector: Arc<ForegroundDetector>,
    qos_mgr: Option<Arc<Mutex<MachQoSManager>>>,
    frozen_state_path: PathBuf,
    proposals: ResourceInterruptProposalSender,
) -> ResourceSentinelShutdownHandle {
    spawn_resource_sentinel_inner(
        hw_cache,
        pressure_cache,
        interrupt_state,
        main_frozen,
        stop,
        config,
        fg_detector,
        qos_mgr,
        frozen_state_path,
        proposals,
    )
}

fn spawn_resource_sentinel_inner(
    hw_cache: Arc<Mutex<Option<HardwareSnapshot>>>,
    pressure_cache: Arc<Mutex<PressureData>>,
    interrupt_state: Arc<ResourceInterruptState>,
    main_frozen: Arc<Mutex<HashMap<u32, FrozenEntry>>>,
    stop: Arc<AtomicBool>,
    config: SentinelConfig,
    fg_detector: Arc<ForegroundDetector>,
    _qos_mgr: Option<Arc<Mutex<MachQoSManager>>>,
    _frozen_state_path: PathBuf,
    proposals: ResourceInterruptProposalSender,
) -> ResourceSentinelShutdownHandle {
    let (wake_tx, wake_rx) = mpsc::sync_channel(RESOURCE_SENTINEL_WAKE_CAPACITY);
    let wake_keepalive = wake_tx.clone();
    let (quiesced_tx, quiesced_rx) = mpsc::sync_channel(1);
    let worker_stop = stop.clone();
    let worker = match thread::Builder::new()
        .name("resource-sentinel".into())
        .spawn(move || {
            let _wake_keepalive = wake_keepalive;
            sentinel_loop(
                hw_cache,
                pressure_cache,
                interrupt_state,
                main_frozen,
                worker_stop,
                config,
                fg_detector,
                proposals,
                wake_rx,
            );

            // All proposal production happens-before this acknowledgement.
            let _ = quiesced_tx.try_send(());
        }) {
        Ok(worker) => Some(worker),
        Err(e) => {
            eprintln!("warning: failed to spawn resource-sentinel: {}", e);
            None
        }
    };

    ResourceSentinelShutdownHandle {
        stop,
        wake: wake_tx,
        quiesced: quiesced_rx,
        worker,
        acknowledged: false,
        revoke_on_drop: true,
    }
}

fn sentinel_loop(
    hw_cache: Arc<Mutex<Option<HardwareSnapshot>>>,
    pressure_cache: Arc<Mutex<PressureData>>,
    state: Arc<ResourceInterruptState>,
    main_frozen: Arc<Mutex<HashMap<u32, FrozenEntry>>>,
    stop: Arc<AtomicBool>,
    config: SentinelConfig,
    fg_detector: Arc<ForegroundDetector>,
    proposals: ResourceInterruptProposalSender,
    wake: Receiver<()>,
) {
    let mut bufs = SentinelBuffers::new(fg_detector);
    let mut last_escalation = Instant::now() - config.debounce;
    let mut prev_phase = InterruptPhase::Idle;
    let mut last_fg_pid: Option<u32> = None;

    while !stop.load(Ordering::Acquire) {
        let tick_start = Instant::now();

        // Read caches (lock-free reads via try_lock to never block).
        let hw_temp = hw_cache.try_lock().ok().and_then(|g| {
            g.as_ref()
                .filter(|hw| !hw.temps_estimated)
                .and_then(|hw| hw.temps.p_cluster_celsius)
        });

        let pressure = pressure_cache
            .try_lock()
            .ok()
            .map(|g| g.clone())
            .unwrap_or_default();

        // Check reactor signals.
        let thermal_signaled = state.thermal_signal.swap(false, Ordering::AcqRel);
        let memory_signaled = state.memory_signal.swap(false, Ordering::AcqRel);
        let _power_signaled = state.power_signal.swap(false, Ordering::AcqRel);

        // Determine current resource severity. Preserve `None` when the SMC
        // reader has not yet populated so compute_phase sees the unknown
        // state explicitly instead of a silent `0.0` sentinel.
        let rate_of_rise = match hw_temp {
            Some(t) if t > 0.0 => bufs.record_temp(t),
            _ => 0.0,
        };

        let new_phase = compute_phase(
            hw_temp,
            rate_of_rise,
            &pressure,
            thermal_signaled,
            memory_signaled,
            prev_phase,
            &config,
        );

        // Require 2 consecutive SuperEmergency readings to prevent sensor
        // glitch false positives from freezing the entire system.
        let new_phase = if new_phase == InterruptPhase::SuperEmergency {
            bufs.consecutive_super = bufs.consecutive_super.saturating_add(1);
            if bufs.consecutive_super >= 2 {
                InterruptPhase::SuperEmergency
            } else {
                InterruptPhase::Emergency // demote until confirmed
            }
        } else {
            bufs.consecutive_super = 0;
            new_phase
        };

        // Apply hysteresis: only downgrade if temp is well below threshold.
        // When temp is unknown (reader boot-edge or stuck), `below` returns
        // false so hysteresis keeps the higher phase — prefer over-mitigation
        // to under-mitigation when we have no thermal evidence.
        let below = |limit: f32| hw_temp.map(|t| t < limit).unwrap_or(false);
        let effective_phase = if new_phase < prev_phase {
            let hysteresis_ok = match prev_phase {
                InterruptPhase::SuperEmergency => {
                    below(config.thermal_super_emergency_c - config.hysteresis_c)
                        && pressure.memory_pressure < config.memory_pressure_emergency - 0.05
                }
                InterruptPhase::Emergency => {
                    below(config.thermal_emergency_c - config.hysteresis_c)
                        && pressure.memory_pressure < config.memory_pressure_moderate - 0.05
                }
                InterruptPhase::Moderate => {
                    below(config.thermal_moderate_c - config.hysteresis_c)
                        && pressure.memory_pressure < config.memory_pressure_moderate - 0.10
                }
                InterruptPhase::Idle => true,
            };
            if hysteresis_ok {
                new_phase
            } else {
                prev_phase
            }
        } else {
            new_phase
        };

        // Apply debounce for escalations.
        let debounced_phase = if effective_phase > prev_phase {
            if tick_start.duration_since(last_escalation) >= config.debounce {
                last_escalation = tick_start;
                effective_phase
            } else {
                prev_phase
            }
        } else {
            effective_phase
        };

        // Phase transition: take action.
        if debounced_phase != prev_phase {
            state.phase.store(debounced_phase as u8, Ordering::Release);
            state.sequence.fetch_add(1, Ordering::Release);

            if debounced_phase > prev_phase {
                // Escalation.
                state.active.store(true, Ordering::Release);
                state.total_fires.fetch_add(1, Ordering::Relaxed);
                respond_to_phase(debounced_phase, &main_frozen, &mut bufs, &proposals);
            } else {
                // De-escalation → recovery.
                if debounced_phase == InterruptPhase::Idle {
                    recover(&state, &main_frozen, &mut bufs, &proposals);
                    state.active.store(false, Ordering::Release);
                }
            }

            let latency = tick_start.elapsed().as_micros() as u64;
            state.last_latency_us.store(latency, Ordering::Relaxed);
        } else if debounced_phase >= InterruptPhase::Emergency {
            // Sustained emergency: keep checking for new processes.
            state.active.store(true, Ordering::Release);
        }

        prev_phase = debounced_phase;

        // Reactive recovery proposal when foreground changes to a process that
        // is currently frozen. The daemon loop owns the eventual SIGCONT.
        let fg_pid = bufs.fg_detector.detect().pid();
        if fg_pid != last_fg_pid {
            if let Some(pid) = fg_pid {
                let entry = main_frozen
                    .try_lock()
                    .ok()
                    .and_then(|mf| mf.get(&pid).cloned());
                let sentinel_owned = state
                    .interrupt_frozen_pids
                    .try_lock()
                    .ok()
                    .is_some_and(|sf| sf.contains(&pid));
                if entry.is_some() || sentinel_owned {
                    let identity = entry
                        .as_ref()
                        .map(|entry| {
                            (
                                entry.process_name.clone().unwrap_or_default(),
                                entry.start_sec,
                                0,
                            )
                        })
                        .or_else(|| {
                            ProcessIdentity::from_pid(pid).map(|identity| {
                                (identity.name, identity.start_sec, identity.start_usec)
                            })
                        })
                        .unwrap_or_default();
                    proposals.propose(ResourceInterruptProposal::Unfreeze {
                        pid,
                        name: identity.0,
                        start_sec: identity.1,
                        start_usec: identity.2,
                        reason: ResourceInterruptUnfreezeReason::Foreground,
                    });
                }
            }
            last_fg_pid = fg_pid;
        }

        // Sleep until next poll.
        let elapsed = tick_start.elapsed();
        if elapsed < config.poll_interval {
            match wake.recv_timeout(config.poll_interval - elapsed) {
                Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => {}
            }
        }
    }
}

/// Compute the target phase based on current sensor readings.
///
/// `temp_c` is `None` when the SMC reader has not yet produced a sample
/// (boot-edge) or is stuck. Previously this was collapsed to `0.0` at the
/// call site via `unwrap_or(0.0)`, which silently erased the "unknown"
/// state and could keep the sentinel in `Idle` even while the CPU was hot
/// if the reader had stalled. Option propagates the missing-sensor case
/// through the decision logic so the thermal branches are simply skipped
/// when we don't know the temperature — pressure and reactor signals can
/// still escalate the phase on their own.
fn compute_phase(
    temp_c: Option<f32>,
    rate_of_rise: f32,
    pressure: &PressureData,
    thermal_signaled: bool,
    memory_signaled: bool,
    _prev: InterruptPhase,
    config: &SentinelConfig,
) -> InterruptPhase {
    // Super-emergency: extreme temperature OR dangerous rate-of-rise.
    if let Some(t) = temp_c {
        if t >= config.thermal_super_emergency_c
            || (t >= config.thermal_emergency_c && rate_of_rise >= config.rate_of_rise_threshold)
        {
            return InterruptPhase::SuperEmergency;
        }
        if t >= config.thermal_emergency_c {
            return InterruptPhase::Emergency;
        }
    }

    // Emergency: critical memory + swap thrash (sensor-independent).
    if pressure.memory_pressure >= config.memory_pressure_emergency
        && pressure.swap_delta_bps >= 500_000.0
    {
        return InterruptPhase::Emergency;
    }

    // Moderate: warm OR memory pressure rising.
    if let Some(t) = temp_c {
        if t >= config.thermal_moderate_c {
            return InterruptPhase::Moderate;
        }
    }
    if pressure.memory_pressure >= config.memory_pressure_moderate {
        return InterruptPhase::Moderate;
    }

    // Reactor signals can trigger moderate for faster response.
    if thermal_signaled || (memory_signaled && pressure.memory_pressure >= 0.70) {
        return InterruptPhase::Moderate;
    }

    InterruptPhase::Idle
}

/// Translate an emergency phase into bounded, typed proposals. This function is
/// deliberately observation-only: the daemon loop remains the sole actuator.
fn respond_to_phase(
    phase: InterruptPhase,
    main_frozen: &Arc<Mutex<HashMap<u32, FrozenEntry>>>,
    bufs: &mut SentinelBuffers,
    proposals: &ResourceInterruptProposalSender,
) {
    match phase {
        InterruptPhase::Moderate => {
            propose_migrations(main_frozen, bufs, proposals);
        }
        InterruptPhase::Emergency => {
            propose_freezes(main_frozen, bufs, proposals);
            propose_migrations(main_frozen, bufs, proposals);
        }
        InterruptPhase::SuperEmergency => {
            propose_freezes(main_frozen, bufs, proposals);
            propose_migrations(main_frozen, bufs, proposals);
        }
        InterruptPhase::Idle => {}
    }
}

fn proposal_identity(pid: u32, fallback_name: &str) -> (String, u64, u64) {
    ProcessIdentity::from_pid(pid)
        .map(|identity| (identity.name, identity.start_sec, identity.start_usec))
        .unwrap_or_else(|| (fallback_name.to_string(), 0, 0))
}

/// Identify heavy non-protected processes that should move to background QoS.
fn propose_migrations(
    main_frozen: &Arc<Mutex<HashMap<u32, FrozenEntry>>>,
    bufs: &SentinelBuffers,
    proposals: &ResourceInterruptProposalSender,
) {
    let main_frozen_pids: HashSet<u32> = main_frozen
        .try_lock()
        .ok()
        .map(|g| g.keys().copied().collect())
        .unwrap_or_default();

    let sys = sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::new().with_processes(sysinfo::ProcessRefreshKind::new().with_cpu()),
    );
    // Snapshot foreground state once before the loop (cached, <1µs).
    let fg_state = bufs.fg_detector.detect();
    let fg_pid = fg_state.pid();
    let recently_active_window = std::time::Duration::from_secs(300);

    for (pid, proc_info) in sys.processes() {
        let pid_u32 = pid.as_u32();
        if pid_u32 <= 1 || main_frozen_pids.contains(&pid_u32) {
            continue;
        }
        // Skip foreground app and recently active apps.
        // Without this check any GUI app inactive > 5 min (Brave, VS Code, Slack)
        // gets demoted to E-cores, and the OS takes seconds to re-promote them
        // when the user switches back — perceived as system slowness.
        // [Apple QoS doc] — Background tier stays until explicitly promoted.
        if fg_pid == Some(pid_u32) {
            continue;
        }
        let name = proc_info.name();
        if bufs
            .fg_detector
            .is_recently_active(name, recently_active_window)
        {
            continue;
        }
        if bufs.is_essential(name) || bufs.is_protected(name) {
            continue;
        }
        if crate::engine::process_identity::is_apple_platform_process(pid_u32) {
            continue;
        }
        if proc_info.cpu_usage() < 5.0 {
            continue;
        }
        let (name, start_sec, start_usec) = proposal_identity(pid_u32, proc_info.name());
        proposals.propose(ResourceInterruptProposal::MigrateToBackground {
            pid: pid_u32,
            name,
            start_sec,
            start_usec,
        });
    }
}

/// Identify at most four non-critical processes for an emergency freeze.
fn propose_freezes(
    main_frozen: &Arc<Mutex<HashMap<u32, FrozenEntry>>>,
    bufs: &SentinelBuffers,
    proposals: &ResourceInterruptProposalSender,
) {
    let main_frozen_pids: HashSet<u32> = main_frozen
        .try_lock()
        .ok()
        .map(|g| g.keys().copied().collect())
        .unwrap_or_default();

    let sys = sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::new()
            .with_processes(sysinfo::ProcessRefreshKind::new().with_cpu().with_memory()),
    );

    // Snapshot once: PIDs doing active work (audio, downloads, active children).
    let busy_pids = active_pids(sys.processes());

    let mut candidates: Vec<(u32, String)> = Vec::with_capacity(4);

    // Snapshot foreground state once before the loop (cached, <1µs).
    let fg_state = bufs.fg_detector.detect();
    let fg_pid = fg_state.pid();
    let recently_active_window = std::time::Duration::from_secs(300);

    for (pid, proc_info) in sys.processes() {
        let pid_u32 = pid.as_u32();
        if pid_u32 <= 1 || main_frozen_pids.contains(&pid_u32) {
            continue;
        }
        if fg_pid == Some(pid_u32) {
            continue;
        }
        let name = proc_info.name();
        if bufs
            .fg_detector
            .is_recently_active(name, recently_active_window)
        {
            continue;
        }
        if bufs.is_essential(name) || bufs.is_protected(name) {
            continue;
        }
        // Complete mediation (2026-06-18 audit): the local is_essential/is_protected
        // lists drift out of sync with safety.rs — they omitted CVMServer, powerd,
        // sharingd (sharingd has a 173×-freeze production scar). Defer to the single
        // source of truth so no protected daemon can be SIGSTOP'd in a thermal spike.
        if crate::engine::safety::is_protected_name(name) {
            continue;
        }
        // Behavioural app-bundle detection: any binary inside a .app bundle
        // is a user-facing application (or its helper). Skip it from thermal
        // freeze — the user's apps must not be paused by a temperature spike.
        // This closes the gap where apps like Raycast, Bartender, 1Password
        // were not in the hardcoded protected list but ARE user-facing.
        if crate::engine::proc_taskinfo::is_user_app_bundle(pid_u32).unwrap_or(false) {
            continue;
        }
        // Never freeze processes with active power assertions or busy children:
        // música reproduciéndose, terminal con build corriendo, descarga activa, etc.
        if busy_pids.contains(&pid_u32) {
            continue;
        }
        // Solo congelar procesos que usan recursos significativos.
        // Umbral de memoria elevado a 400MB (era 200MB) para ser más conservador en 8GB RAM.
        // CPU threshold mantenido en 10% para evitar freezar procesos activos.
        if proc_info.cpu_usage() < 10.0 && proc_info.memory() < 400 * 1024 * 1024 {
            continue;
        }
        // Cap de seguridad: máximo 4 procesos congelados por invocación del sentinel.
        // Evita freezar en cascada durante emergencias con muchas ventanas abiertas.
        if candidates.len() >= 4 {
            break;
        }
        candidates.push((pid_u32, name.to_string()));
    }

    for (pid, fallback_name) in candidates {
        let (name, start_sec, start_usec) = proposal_identity(pid, &fallback_name);
        proposals.propose(ResourceInterruptProposal::Freeze {
            pid,
            name,
            start_sec,
            start_usec,
        });
    }
}

/// Propose reversal for every sentinel-owned scheduling and freeze effect.
fn recover(
    state: &ResourceInterruptState,
    main_frozen: &Arc<Mutex<HashMap<u32, FrozenEntry>>>,
    _bufs: &mut SentinelBuffers,
    proposals: &ResourceInterruptProposalSender,
) {
    let migrated: Vec<u32> = state
        .interrupt_migrated_pids
        .lock_recover()
        .iter()
        .copied()
        .collect();
    for pid in migrated {
        let (name, start_sec, start_usec) = proposal_identity(pid, "");
        proposals.propose(ResourceInterruptProposal::RestoreScheduling {
            pid,
            name,
            start_sec,
            start_usec,
        });
    }

    let mut frozen: HashSet<u32> = state
        .interrupt_frozen_pids
        .lock_recover()
        .iter()
        .copied()
        .collect();
    let entries: HashMap<u32, FrozenEntry> = main_frozen
        .try_lock()
        .ok()
        .map(|guard| {
            guard
                .iter()
                .filter(|(_, entry)| entry.source == FreezeSource::Sentinel)
                .map(|(pid, entry)| (*pid, entry.clone()))
                .collect()
        })
        .unwrap_or_default();
    frozen.extend(entries.keys().copied());

    for pid in frozen {
        let (name, start_sec, start_usec) = entries
            .get(&pid)
            .map(|entry| {
                (
                    entry.process_name.clone().unwrap_or_default(),
                    entry.start_sec,
                    0,
                )
            })
            .unwrap_or_else(|| proposal_identity(pid, ""));
        proposals.propose(ResourceInterruptProposal::Unfreeze {
            pid,
            name,
            start_sec,
            start_usec,
            reason: ResourceInterruptUnfreezeReason::Recovery,
        });
    }
}

// ── Comparison operators for InterruptPhase ──────────────────────────────────

impl PartialOrd for InterruptPhase {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InterruptPhase {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::decision_ledger::{ActuatorDecisionOutcome, CycleDecisionEvents};

    #[test]
    fn proposal_admitted_before_loop_close_cannot_mutate_after_close() {
        let (sender, receiver) = resource_interrupt_proposal_channel(None);
        let (proposed_tx, proposed_rx) = mpsc::sync_channel(0);
        let (release_tx, release_rx) = mpsc::sync_channel(0);
        let worker = thread::spawn(move || {
            assert!(sender.propose(ResourceInterruptProposal::Freeze {
                pid: 4242,
                name: "late-worker".to_string(),
                start_sec: 7,
                start_usec: 0,
            }));
            proposed_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });

        proposed_rx.recv().unwrap();
        let batch = receiver.drain();
        assert_eq!(batch.len(), 1);
        let mut window = ResourceInterruptActuationWindow::open();
        window.close();
        release_tx.send(()).unwrap();
        worker.join().unwrap();

        let mutations = AtomicU64::new(0);
        let mut events = window.resolve(batch, 99, |proposal| {
            mutations.fetch_add(1, Ordering::Relaxed);
            let mut events = CycleDecisionEvents::default();
            events.push(proposal.event(
                99,
                ActuatorDecisionOutcome::Applied,
                "test executor mutated",
            ));
            events
        });
        assert_eq!(mutations.load(Ordering::Relaxed), 0);
        assert_eq!(events.len(), 1);
        assert_eq!(
            events.as_slice()[0].outcome,
            ActuatorDecisionOutcome::Expired
        );
        assert_eq!(events.as_slice()[0].proposal.target, "pid:4242");
        assert!(events.as_slice()[0].detail.contains("loop closed"));
        assert_eq!(events.dropped_total(), 0);
        assert!(events.seal_overflow_summary(99) == false);
    }

    #[test]
    fn bounded_proposal_handoff_preserves_pid_effect_and_identity() {
        let (sender, receiver) = resource_interrupt_proposal_channel(None);
        let proposals = [
            ResourceInterruptProposal::MigrateToBackground {
                pid: 4100,
                name: "migrate".to_string(),
                start_sec: 10,
                start_usec: 11,
            },
            ResourceInterruptProposal::Freeze {
                pid: 4101,
                name: "freeze".to_string(),
                start_sec: 12,
                start_usec: 13,
            },
            ResourceInterruptProposal::RestoreScheduling {
                pid: 4102,
                name: "restore".to_string(),
                start_sec: 14,
                start_usec: 15,
            },
            ResourceInterruptProposal::Unfreeze {
                pid: 4103,
                name: "unfreeze".to_string(),
                start_sec: 16,
                start_usec: 17,
                reason: ResourceInterruptUnfreezeReason::Recovery,
            },
        ];

        for proposal in proposals.clone() {
            assert!(sender.propose(proposal));
        }

        let batch = receiver.drain();
        assert_eq!(batch.dropped(), 0);
        assert_eq!(batch.into_proposals(), proposals);
    }

    #[test]
    fn proposal_handoff_propagates_bounded_channel_overflow() {
        let (sender, receiver) = resource_interrupt_proposal_channel(None);
        for pid in 1..=RESOURCE_INTERRUPT_PROPOSAL_CAPACITY as u32 {
            assert!(sender.propose(ResourceInterruptProposal::Freeze {
                pid,
                name: "overflow".to_string(),
                start_sec: 1,
                start_usec: 0,
            }));
        }
        assert!(!sender.propose(ResourceInterruptProposal::Freeze {
            pid: 9999,
            name: "overflow".to_string(),
            start_sec: 1,
            start_usec: 0,
        }));

        let batch = receiver.drain();
        assert_eq!(batch.len(), RESOURCE_INTERRUPT_PROPOSAL_CAPACITY);
        assert_eq!(batch.dropped(), 1);
    }

    #[test]
    fn migrated_set_records_and_drains() {
        // Fight-hunt fix (2026-06-10): migrations must be tracked so
        // recover() can undo them. Pin the set's record/drain contract.
        let state = ResourceInterruptState::new();
        state.interrupt_migrated_pids.lock_recover().insert(4242);
        assert!(state.interrupt_migrated_pids.lock_recover().contains(&4242));
        let drained: Vec<u32> = state
            .interrupt_migrated_pids
            .lock_recover()
            .drain()
            .collect();
        assert_eq!(drained, vec![4242]);
        assert!(state.interrupt_migrated_pids.lock_recover().is_empty());
    }

    #[test]
    fn phase_ordering() {
        assert!(InterruptPhase::Idle < InterruptPhase::Moderate);
        assert!(InterruptPhase::Moderate < InterruptPhase::Emergency);
        assert!(InterruptPhase::Emergency < InterruptPhase::SuperEmergency);
    }

    #[test]
    fn phase_from_u8_roundtrip() {
        for val in 0..=3 {
            let phase = InterruptPhase::from_u8(val);
            assert_eq!(phase as u8, val);
        }
        // Out of range maps to Idle.
        assert_eq!(InterruptPhase::from_u8(42), InterruptPhase::Idle);
        assert_eq!(InterruptPhase::from_u8(255), InterruptPhase::Idle);
    }

    #[test]
    fn resource_interrupt_state_defaults() {
        let state = ResourceInterruptState::new();
        assert_eq!(state.current_phase(), InterruptPhase::Idle);
        assert!(!state.active.load(Ordering::Relaxed));
        assert_eq!(state.sequence.load(Ordering::Relaxed), 0);
        assert!(!state.thermal_signal.load(Ordering::Relaxed));
        assert!(!state.memory_signal.load(Ordering::Relaxed));
        assert!(!state.power_signal.load(Ordering::Relaxed));
        assert_eq!(state.total_fires.load(Ordering::Relaxed), 0);
        assert_eq!(state.total_frozen.load(Ordering::Relaxed), 0);
        assert_eq!(state.total_migrated.load(Ordering::Relaxed), 0);
        assert_eq!(state.total_recoveries.load(Ordering::Relaxed), 0);
        assert!(state.interrupt_frozen_pids.lock_recover().is_empty());
    }

    #[test]
    fn state_default_trait() {
        let state = ResourceInterruptState::default();
        assert_eq!(state.current_phase(), InterruptPhase::Idle);
    }

    #[test]
    fn sentinel_config_defaults() {
        let cfg = SentinelConfig::default();
        assert_eq!(cfg.poll_interval, Duration::from_millis(500));
        assert!((cfg.thermal_moderate_c - 90.0).abs() < f32::EPSILON);
        assert!((cfg.thermal_emergency_c - 95.0).abs() < f32::EPSILON);
        assert!((cfg.thermal_super_emergency_c - 100.0).abs() < f32::EPSILON);
        assert!((cfg.memory_pressure_moderate - 0.80).abs() < f64::EPSILON);
        assert!((cfg.memory_pressure_emergency - 0.92).abs() < f64::EPSILON);
        assert!((cfg.hysteresis_c - 5.0).abs() < f32::EPSILON);
        assert_eq!(cfg.debounce, Duration::from_secs(2));
        assert!((cfg.rate_of_rise_threshold - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn compute_phase_idle_when_cool_and_low_pressure() {
        let cfg = SentinelConfig::default();
        let pressure = PressureData {
            memory_pressure: 0.3,
            swap_delta_bps: 0.0,
            ..PressureData::default()
        };
        let phase = compute_phase(
            Some(50.0),
            0.0,
            &pressure,
            false,
            false,
            InterruptPhase::Idle,
            &cfg,
        );
        assert_eq!(phase, InterruptPhase::Idle);
    }

    #[test]
    fn compute_phase_moderate_on_thermal() {
        let cfg = SentinelConfig::default();
        let pressure = PressureData::default();
        let phase = compute_phase(
            Some(91.0),
            0.0,
            &pressure,
            false,
            false,
            InterruptPhase::Idle,
            &cfg,
        );
        assert_eq!(phase, InterruptPhase::Moderate);
    }

    #[test]
    fn compute_phase_moderate_on_memory_pressure() {
        let cfg = SentinelConfig::default();
        let pressure = PressureData {
            memory_pressure: 0.85,
            ..PressureData::default()
        };
        let phase = compute_phase(
            Some(50.0),
            0.0,
            &pressure,
            false,
            false,
            InterruptPhase::Idle,
            &cfg,
        );
        assert_eq!(phase, InterruptPhase::Moderate);
    }

    #[test]
    fn compute_phase_emergency_on_high_thermal() {
        let cfg = SentinelConfig::default();
        let pressure = PressureData::default();
        let phase = compute_phase(
            Some(96.0),
            0.0,
            &pressure,
            false,
            false,
            InterruptPhase::Idle,
            &cfg,
        );
        assert_eq!(phase, InterruptPhase::Emergency);
    }

    #[test]
    fn compute_phase_emergency_on_memory_critical_with_swap_thrash() {
        let cfg = SentinelConfig::default();
        let pressure = PressureData {
            memory_pressure: 0.95,
            swap_delta_bps: 1_000_000.0,
            ..PressureData::default()
        };
        let phase = compute_phase(
            Some(50.0),
            0.0,
            &pressure,
            false,
            false,
            InterruptPhase::Idle,
            &cfg,
        );
        assert_eq!(phase, InterruptPhase::Emergency);
    }

    #[test]
    fn compute_phase_super_emergency_on_extreme_temp() {
        let cfg = SentinelConfig::default();
        let pressure = PressureData::default();
        let phase = compute_phase(
            Some(101.0),
            0.0,
            &pressure,
            false,
            false,
            InterruptPhase::Idle,
            &cfg,
        );
        assert_eq!(phase, InterruptPhase::SuperEmergency);
    }

    #[test]
    fn compute_phase_super_emergency_on_rate_of_rise() {
        let cfg = SentinelConfig::default();
        let pressure = PressureData::default();
        // 96°C + 1.5°C/s rate-of-rise → super-emergency
        let phase = compute_phase(
            Some(96.0),
            1.5,
            &pressure,
            false,
            false,
            InterruptPhase::Idle,
            &cfg,
        );
        assert_eq!(phase, InterruptPhase::SuperEmergency);
    }

    #[test]
    fn compute_phase_reactor_thermal_signal_triggers_moderate() {
        let cfg = SentinelConfig::default();
        let pressure = PressureData::default();
        // Reader has not populated yet (None) — thermal signal from reactor
        // still escalates to Moderate because the decision is sensor-independent.
        let phase = compute_phase(
            None,
            0.0,
            &pressure,
            true,
            false,
            InterruptPhase::Idle,
            &cfg,
        );
        assert_eq!(phase, InterruptPhase::Moderate);
    }

    #[test]
    fn compute_phase_memory_signal_needs_pressure_above_threshold() {
        let cfg = SentinelConfig::default();
        let low_pressure = PressureData {
            memory_pressure: 0.5,
            ..PressureData::default()
        };
        // Memory signal but low pressure → still idle, temp unknown.
        let phase = compute_phase(
            None,
            0.0,
            &low_pressure,
            false,
            true,
            InterruptPhase::Idle,
            &cfg,
        );
        assert_eq!(phase, InterruptPhase::Idle);

        let high_pressure = PressureData {
            memory_pressure: 0.75,
            ..PressureData::default()
        };
        // Memory signal + pressure ≥ 0.70 → moderate, temp unknown.
        let phase = compute_phase(
            None,
            0.0,
            &high_pressure,
            false,
            true,
            InterruptPhase::Idle,
            &cfg,
        );
        assert_eq!(phase, InterruptPhase::Moderate);
    }

    #[test]
    fn sentinel_buffers_essential_detection() {
        let bufs = SentinelBuffers::new(Arc::new(ForegroundDetector::new()));
        assert!(bufs.is_essential("kernel_task"));
        assert!(bufs.is_essential("WindowServer"));
        assert!(bufs.is_essential("launchd"));
        assert!(!bufs.is_essential("my_random_app"));
    }

    #[test]
    fn sentinel_buffers_protected_detection() {
        let bufs = SentinelBuffers::new(Arc::new(ForegroundDetector::new()));
        // Build tools are statically protected.
        assert!(bufs.is_protected("apollo-optimizerd"));
        assert!(bufs.is_protected("cargo"));
        assert!(bufs.is_protected("rustc"));
        assert!(bufs.is_protected("node"));
        // User-facing GUI apps are now ALSO in the static protected set.
        // The sentinel cannot query CGWindowServer, so explicit enumeration
        // of known GUI apps is the only safe approach. Without this, any GUI
        // app inactive > 300s would receive SIGSTOP during thermal Emergency.
        assert!(
            bufs.is_protected("Google Chrome"),
            "browsers must be statically protected"
        );
        assert!(
            bufs.is_protected("Brave Browser"),
            "browsers must be statically protected"
        );
        assert!(
            bufs.is_protected("Safari"),
            "browsers must be statically protected"
        );
        assert!(
            bufs.is_protected("Slack"),
            "communication apps must be protected"
        );
        assert!(bufs.is_protected("Claude"), "AI apps must be protected");
        // Analytics/background daemons are still not protected.
        assert!(!bufs.is_protected("com.apple.photoanalysisd"));
        assert!(!bufs.is_protected("mlhostd"));
    }

    #[test]
    fn sentinel_buffers_temp_history_rate_of_rise() {
        let mut bufs = SentinelBuffers::new(Arc::new(ForegroundDetector::new()));
        // Simulate temperature readings ~1 second apart.
        // Start at 80°C, rise 1°C per iteration.
        for i in 0..8 {
            let temp = 80.0 + i as f32;
            bufs.record_temp(temp);
            std::thread::sleep(Duration::from_millis(10));
        }
        // After 8 samples the rate should be positive.
        let rate = bufs.record_temp(88.0);
        assert!(rate > 0.0, "rate of rise should be positive: {rate}");
    }

    #[test]
    fn atomic_phase_store_and_load() {
        let state = ResourceInterruptState::new();
        state
            .phase
            .store(InterruptPhase::Emergency as u8, Ordering::Release);
        assert_eq!(state.current_phase(), InterruptPhase::Emergency);

        state
            .phase
            .store(InterruptPhase::SuperEmergency as u8, Ordering::Release);
        assert_eq!(state.current_phase(), InterruptPhase::SuperEmergency);

        state
            .phase
            .store(InterruptPhase::Idle as u8, Ordering::Release);
        assert_eq!(state.current_phase(), InterruptPhase::Idle);
    }

    #[test]
    fn interrupt_frozen_pids_tracking() {
        let state = ResourceInterruptState::new();
        {
            let mut pids = state.interrupt_frozen_pids.lock_recover();
            pids.insert(100);
            pids.insert(200);
            pids.insert(300);
        }
        assert_eq!(state.interrupt_frozen_pids.lock_recover().len(), 3);

        // Drain simulates recovery.
        let drained: Vec<u32> = state.interrupt_frozen_pids.lock_recover().drain().collect();
        assert_eq!(drained.len(), 3);
        assert!(state.interrupt_frozen_pids.lock_recover().is_empty());
    }
}
