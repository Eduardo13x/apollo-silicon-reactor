//! Resource Sentinel — sub-100ms interrupt handler for thermal, memory, and power emergencies.
//!
//! Runs as a dedicated thread ("resource-sentinel") that monitors the SmcReader
//! and PressureCollector caches plus reactor signals.  When a resource emergency
//! is detected, it takes immediate action (SIGSTOP, taskpolicy migration, sysctl
//! hints) without waiting for the main daemon loop.
//!
//! Communication with the main loop is entirely lock-free via atomics, except for
//! `interrupt_frozen_pids` which uses a Mutex accessed with `try_lock` from the
//! main loop.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::engine::activity_sensor::active_pids;
use crate::engine::background_collectors::PressureData;
use crate::engine::foreground::ForegroundDetector;
use crate::engine::iokit_sensors::HardwareSnapshot;
use crate::engine::lock_ext::LockRecover;
use crate::engine::mach_qos::{MachQoSManager, SchedulingTier};
use crate::engine::types::{FreezeSource, FrozenEntry};
use chrono::Utc;

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
    /// Fight-hunt fix (2026-06-10): PIDs the sentinel migrated to
    /// E-cores/Darwin-BG during Moderate/Emergency phases. recover()
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
    /// Latency of last sentinel action in microseconds.
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
/// The sentinel monitors the SmcReader and PressureCollector caches and reacts
/// to resource emergencies in <100ms without waiting for the main loop.
pub fn spawn_resource_sentinel(
    hw_cache: Arc<Mutex<Option<HardwareSnapshot>>>,
    pressure_cache: Arc<Mutex<PressureData>>,
    interrupt_state: Arc<ResourceInterruptState>,
    main_frozen: Arc<Mutex<HashMap<u32, FrozenEntry>>>,
    stop: Arc<AtomicBool>,
    config: SentinelConfig,
    fg_detector: Arc<ForegroundDetector>,
    qos_mgr: Option<Arc<Mutex<MachQoSManager>>>,
) {
    if let Err(e) = thread::Builder::new()
        .name("resource-sentinel".into())
        .spawn(move || {
            // Pin to E-cores via QOS_CLASS_BACKGROUND so the sentinel never
            // competes with user workloads on P-cores.
            pin_to_ecores();

            sentinel_loop(
                hw_cache,
                pressure_cache,
                interrupt_state,
                main_frozen,
                stop,
                config,
                fg_detector,
                qos_mgr,
            );
        })
    {
        eprintln!("warning: failed to spawn resource-sentinel: {}", e);
    }
}

/// Pin the current thread to E-cores via pthread QOS_CLASS_BACKGROUND.
/// This is a best-effort hint to the macOS scheduler; failure is non-fatal.
fn pin_to_ecores() {
    // QOS_CLASS_BACKGROUND = 0x09
    const QOS_CLASS_BACKGROUND: libc::c_uint = 0x09;
    unsafe {
        // int pthread_set_qos_class_self_np(qos_class_t, int relative_priority)
        extern "C" {
            fn pthread_set_qos_class_self_np(
                qos_class: libc::c_uint,
                relative_priority: libc::c_int,
            ) -> libc::c_int;
        }
        let _ = pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
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
    qos_mgr: Option<Arc<Mutex<MachQoSManager>>>,
) {
    let mut bufs = SentinelBuffers::new(fg_detector);
    let mut last_escalation = Instant::now() - config.debounce;
    let mut prev_phase = InterruptPhase::Idle;
    let mut last_fg_pid: Option<u32> = None;

    while !stop.load(Ordering::Acquire) {
        let tick_start = Instant::now();

        // Read caches (lock-free reads via try_lock to never block).
        let hw_temp = hw_cache
            .try_lock()
            .ok()
            .and_then(|g| g.as_ref().and_then(|hw| hw.temps.p_cluster_celsius));

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
                respond_to_phase(debounced_phase, &state, &main_frozen, &mut bufs, &qos_mgr);
            } else {
                // De-escalation → recovery.
                if debounced_phase == InterruptPhase::Idle {
                    recover(&state, &main_frozen, &mut bufs, &qos_mgr);
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

        // Reactive unfreeze: si el foreground cambió y el nuevo proceso estaba
        // congelado por el sentinel, mandamos SIGCONT de inmediato (<500ms lag).
        let fg_pid = bufs.fg_detector.detect().pid();
        if fg_pid != last_fg_pid {
            if let Some(pid) = fg_pid {
                // Descongelar del frozen principal y del sentinel.
                if let Ok(mut mf) = main_frozen.try_lock() {
                    if mf.remove(&pid).is_some() {
                        unsafe {
                            libc::kill(pid as i32, libc::SIGCONT);
                        }
                    }
                }
                if let Ok(mut sf) = state.interrupt_frozen_pids.lock() {
                    if sf.remove(&pid) {
                        unsafe {
                            libc::kill(pid as i32, libc::SIGCONT);
                        }
                    }
                }
            }
            last_fg_pid = fg_pid;
        }

        // Sleep until next poll.
        let elapsed = tick_start.elapsed();
        if elapsed < config.poll_interval {
            thread::sleep(config.poll_interval - elapsed);
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

/// Take emergency action based on the current phase.
fn respond_to_phase(
    phase: InterruptPhase,
    state: &ResourceInterruptState,
    main_frozen: &Arc<Mutex<HashMap<u32, FrozenEntry>>>,
    bufs: &mut SentinelBuffers,
    qos_mgr: &Option<Arc<Mutex<MachQoSManager>>>,
) {
    match phase {
        InterruptPhase::Moderate => {
            // Migrate non-protected to E-cores via direct Mach syscall.
            migrate_to_ecores(state, main_frozen, bufs, qos_mgr);
        }
        InterruptPhase::Emergency => {
            // SIGSTOP non-critical + E-core migration + memory pressure hint.
            freeze_non_critical(state, main_frozen, bufs);
            migrate_to_ecores(state, main_frozen, bufs, qos_mgr);
            send_memory_pressure_hint();
        }
        InterruptPhase::SuperEmergency => {
            // Everything above + I/O throttle.
            freeze_non_critical(state, main_frozen, bufs);
            migrate_to_ecores(state, main_frozen, bufs, qos_mgr);
            send_memory_pressure_hint();
            enable_io_throttle();
        }
        InterruptPhase::Idle => {}
    }
}

/// Migrate heavy non-protected processes to E-cores (background QoS).
fn migrate_to_ecores(
    state: &ResourceInterruptState,
    main_frozen: &Arc<Mutex<HashMap<u32, FrozenEntry>>>,
    bufs: &SentinelBuffers,
    qos_mgr: &Option<Arc<Mutex<MachQoSManager>>>,
) {
    let main_frozen_pids: HashSet<u32> = main_frozen
        .try_lock()
        .ok()
        .map(|g| g.keys().copied().collect())
        .unwrap_or_default();

    let sys = sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::new().with_processes(sysinfo::ProcessRefreshKind::new().with_cpu()),
    );
    let mut migrated = 0_u64;

    // Try to use direct Mach QoS manager (Phase 2: ~50µs vs ~5ms per call).
    let mut qos_guard = qos_mgr.as_ref().and_then(|m| m.try_lock().ok());

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
        // Phase 2: direct Mach syscall for E-core migration.
        if let Some(ref mut mgr) = qos_guard {
            mgr.set_tier(pid_u32, SchedulingTier::Background);
        } else {
            // Fallback: PRIO_DARWIN_BG (turnstile-compatible background QoS).
            // Do NOT use PRIO_PROCESS+nice=20 — that breaks the Mach
            // priority-inheritance chain and causes WindowServer IPC hangs.
            const PRIO_DARWIN_BG: libc::c_int = 0x1000;
            unsafe {
                libc::setpriority(PRIO_DARWIN_BG, pid_u32, 1);
            }
        }
        // Anti-ratchet: remember the demotion so recover() can undo it.
        state
            .interrupt_migrated_pids
            .lock_recover()
            .insert(pid_u32);
        migrated += 1;
    }

    state.total_migrated.fetch_add(migrated, Ordering::Relaxed);
}

/// SIGSTOP non-critical processes during Emergency/SuperEmergency.
fn freeze_non_critical(
    state: &ResourceInterruptState,
    main_frozen: &Arc<Mutex<HashMap<u32, FrozenEntry>>>,
    bufs: &SentinelBuffers,
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

    // Candidates collected in the filter loop; SIGSTOP sent only after identity
    // verification (one batch re-snapshot post-loop).
    let mut newly_frozen: Vec<(u32, String)> = Vec::new();

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
        if newly_frozen.len() >= 4 {
            break;
        }
        // Collect candidates; identity verification and SIGSTOP happen after the loop
        // (one batch re-snapshot instead of one System per process → O(N) not O(N²)).
        newly_frozen.push((pid_u32, name.to_string()));
    }

    // Identity verification: one batch re-snapshot for all candidates.
    // Avoids PID recycling between the filter snapshot and the actual SIGSTOP.
    // One System refresh is O(N) shared; doing it per-process would be O(N²).
    // [Chen et al. 2002] — TOCTTOU: verify identity before acting on a PID.
    let mut confirmed_frozen: Vec<u32> = Vec::new();
    if !newly_frozen.is_empty() {
        let verify_sys = sysinfo::System::new_with_specifics(
            sysinfo::RefreshKind::new().with_processes(sysinfo::ProcessRefreshKind::new()),
        );
        for (pid_u32, expected_name) in &newly_frozen {
            let pid_key = sysinfo::Pid::from_u32(*pid_u32);
            let name_matches = verify_sys
                .process(pid_key)
                .map(|p| p.name() == expected_name.as_str())
                .unwrap_or(false); // process no longer exists → do not signal
            if !name_matches {
                // PID recycled or process exited between snapshot and now.
                continue;
            }
            // Apple platform check: skip CS_PLATFORM_BINARY processes even in
            // thermal emergency — freezing WindowServer helpers causes display hangs.
            if crate::engine::process_identity::is_apple_platform_process(*pid_u32) {
                continue;
            }
            unsafe {
                libc::kill(*pid_u32 as i32, libc::SIGSTOP);
            }
            confirmed_frozen.push(*pid_u32);
        }
    }

    if !confirmed_frozen.is_empty() {
        if let Ok(mut guard) = state.interrupt_frozen_pids.lock() {
            for &pid in &confirmed_frozen {
                guard.insert(pid);
            }
        }
        // Sync into main_frozen so frozen_state.json captures sentinel freezes.
        {
            let mut mf = main_frozen.lock_recover();
            let now = Utc::now();
            for pid in &confirmed_frozen {
                mf.entry(*pid).or_insert_with(|| FrozenEntry {
                    frozen_at: now,
                    source: FreezeSource::Sentinel,
                    pressure_at_freeze: 1.0,
                    process_name: None,
                    start_sec: 0,
                    original_jetsam_priority: None,
                });
            }
        }
    }

    state
        .total_frozen
        .fetch_add(confirmed_frozen.len() as u64, Ordering::Relaxed);
}

/// Send memory pressure hint via sysctl to trigger kernel-level page reclaim.
///
/// DISABLED: `kern.memorystatus_vm_pressure_send` takes a *target PID* as its
/// value — writing `1` means "send pressure to PID 1" = launchd, which causes
/// jetsam to kill arbitrary child processes (including Brave, Chrome, etc.).
/// The only safe use of this sysctl is with a specific non-critical daemon PID,
/// which the main loop already handles per-process. The sentinel must NOT trigger
/// a system-wide jetsam cascade by targeting the root of the process tree.
/// [Apple TN2416 / XNU memorystatus_kern_extended_info]
fn send_memory_pressure_hint() {
    // Intentionally a no-op. See doc comment above.
    // The kernel's own jetsam daemon manages system-wide pressure responses.
    // Per-process hints are emitted by execute_actions.rs with explicit target PIDs.
}

/// Intentionally a no-op (2026-06-10 fight-hunt). The old body raw-wrote
/// `debug.lowpri_throttle_enabled=1` via sysctl_direct — bypassing the
/// mediator/journal/clamps (complete-mediation violation) AND dual-writing
/// a key the governor's initial tuning also owned. With the governor's
/// 0-write removed, the kernel default (1 = throttle ON) reigns always and
/// SuperEmergency needs no write: the kernel is already polite.
fn enable_io_throttle() {}

/// Intentionally a no-op (2026-06-10). The old body wrote 0 on recovery —
/// re-DISABLING the kernel's low-priority I/O throttle after every
/// emergency, leaving Time Machine/Spotlight competing unthrottled with
/// foreground I/O until the next emergency. Kernel default stands.
fn disable_io_throttle() {}

/// Recover: SIGCONT all interrupt-frozen PIDs, disable I/O throttle, remove from tracking.
fn recover(
    state: &ResourceInterruptState,
    main_frozen: &Arc<Mutex<HashMap<u32, FrozenEntry>>>,
    _bufs: &mut SentinelBuffers,
    qos_mgr: &Option<Arc<Mutex<MachQoSManager>>>,
) {
    // Disable I/O throttle if it was enabled during SuperEmergency.
    disable_io_throttle();

    // Fight-hunt fix (2026-06-10): undo the Moderate/Emergency E-core
    // migrations. Restore Normal tier (the kernel/runningboard will
    // re-elevate genuinely-foreground work) and clear the Darwin-BG flag
    // for fallback-path victims. Runs BEFORE the frozen-set early-return —
    // Moderate phases migrate without freezing anything.
    {
        let migrated: Vec<u32> = state.interrupt_migrated_pids.lock_recover().drain().collect();
        if !migrated.is_empty() {
            const PRIO_DARWIN_BG: libc::c_int = 0x1000;
            let mut qos_guard = qos_mgr.as_ref().and_then(|m| m.try_lock().ok());
            for pid in &migrated {
                // Process may have exited — all calls are no-ops then.
                if let Some(ref mut mgr) = qos_guard {
                    mgr.set_tier(*pid, SchedulingTier::Normal);
                }
                unsafe {
                    libc::setpriority(PRIO_DARWIN_BG, *pid, 0);
                }
            }
            tracing::info!(
                count = migrated.len(),
                "thermal-recover: restored E-core-migrated processes to normal"
            );
        }
    }

    let pids_to_resume: Vec<u32> = state.interrupt_frozen_pids.lock_recover().drain().collect();

    if pids_to_resume.is_empty() {
        // Still clean up sentinel entries in main_frozen even if nothing to resume.
        main_frozen
            .lock_recover()
            .retain(|_, entry| entry.source != FreezeSource::Sentinel);
        return;
    }

    // Consolidate retain + snapshot into ONE lock acquisition to eliminate TOCTOU.
    // The original code acquired main_frozen twice: once for retain(), once for the
    // keys snapshot. Between the two releases the main loop could insert a new
    // FreezeSource::Apollo entry for a PID we just cleaned, causing the snapshot
    // to show that PID as "main-loop frozen" and skip SIGCONT — leaving the process
    // permanently stuck in SIGSTOP with no SIGCONT planned.
    // [Herlihy & Shavit 2012] — check-then-act on shared state must be atomic.
    let main_frozen_pids: HashSet<u32> = {
        let mut mf = main_frozen.lock_recover();
        mf.retain(|_, entry| entry.source != FreezeSource::Sentinel);
        mf.keys().copied().collect()
    };

    let mut recovered = 0_u64;
    for pid in pids_to_resume {
        // Don't SIGCONT if the main loop also froze this PID.
        if main_frozen_pids.contains(&pid) {
            continue;
        }
        unsafe {
            libc::kill(pid as i32, libc::SIGCONT);
        }
        recovered += 1;
    }

    state
        .total_recoveries
        .fetch_add(recovered, Ordering::Relaxed);
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
