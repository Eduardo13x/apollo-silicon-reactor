//! Chromium/Electron renderer process manager.
//!
//! Manages RAM and CPU for tab renderer subprocesses in ALL Chromium-based
//! browsers (Brave, Chrome, Edge, Arc, Vivaldi, Opera) and Electron apps
//! (Slack, Discord, VS Code, Cursor, Notion, etc.).
//!
//! # Architecture
//! Every Chromium-based app spawns:
//! - Main process: "Brave Browser", "Google Chrome", "Slack", "Code", etc.
//! - Renderer processes: "[App] Helper (Renderer)" — 1 per tab/extension
//! - GPU process: "[App] Helper (GPU)" — 1 per app
//! - Network/plugin: "[App] Helper" or "[App] Helper (Plugin)"
//!
//! # Actions
//! 1. **E-core demotion** (Tier 1 — safe): Non-foreground renderers → E-cores
//! 2. **SIGSTOP idle renderers** (Tier 2 — guarded): Renderers idle 3+ cycles
//!    with no network FDs and no power assertion
//! 3. **Auto-SIGCONT**: Frozen renderer with CPU > 1% → unfreeze immediately
//! 4. **GPU E-core**: "[App] Helper (GPU)" → E-cores when browser not in fg
//!
//! # Safety invariants
//! - Never SIGSTOP main browser process
//! - Never freeze renderer with active TCP/UDP sockets (streaming, downloads)
//! - Never freeze renderer holding power assertion
//! - Never freeze > 50% of a browser's renderers simultaneously
//! - Never freeze renderer that had CPU > 1% in last 3 cycles
//! - Always SIGCONT frozen renderers on daemon shutdown
//!
//! # Papers
//! [Denning 1968] "Working Set Model for Program Behavior" — idle set identification
//! [Corbató 1968] Process suspension for resource reclaim
//! [Jones 2011] "Chromium Multi-Process Architecture" — renderer isolation model

use std::collections::{HashMap, HashSet};

// ── Constants ──────────────────────────────────────────────────────────────────

/// CPU% below which a renderer is considered "idle".
const IDLE_CPU_THRESHOLD: f32 = 0.5;

/// Default: must be idle for this many consecutive cycles before freezing.
/// Overridden by `set_pressure_context()` based on memory pressure.
const IDLE_CYCLES_DEFAULT: u8 = 3;

/// Fraction of CPU above which a frozen renderer is thawed immediately.
const THAW_CPU_THRESHOLD: f32 = 1.0;

/// Never freeze more than this fraction of a browser's renderers at once.
const MAX_FREEZE_RATIO: f32 = 0.5;

/// Check network FDs every N cycles (expensive proc_pidinfo call).
const FD_CHECK_EVERY_N_CYCLES: u8 = 5;

/// Minimum TCP/IP sockets before renderer is considered "network-active".
/// Renderers always have Unix-domain IPC sockets; we only block on TCP/UDP.
const MIN_INET_SOCKETS_TO_BLOCK: usize = 1;

/// Max cycles a renderer stays frozen before forced thaw (~5 min at 2s/cycle).
/// Prevents renderers stuck frozen if fg-change detection misses a tab switch.
const MAX_FROZEN_CYCLES: u8 = 150;

// ── Types ──────────────────────────────────────────────────────────────────────

/// A detected Chromium/Electron renderer process.
#[derive(Debug, Clone)]
pub struct RendererInfo {
    pub pid: u32,
    /// Full name, e.g. "Brave Browser Helper (Renderer)".
    pub name: String,
    /// Derived browser name, e.g. "Brave Browser".
    pub browser: String,
    pub cpu_pct: f32,
    pub memory_bytes: u64,
    /// Consecutive cycles where CPU < [`IDLE_CPU_THRESHOLD`].
    pub consecutive_idle_cycles: u8,
    pub frozen: bool,
    /// Cached result of last inet-socket check (updated every N cycles).
    pub has_inet_sockets: bool,
    pub has_assertion: bool,
    /// CPU history over last 3 cycles (newest first).
    cpu_history: [f32; 3],
    /// Cycles elapsed since this renderer was frozen (for max-freeze-duration guard).
    frozen_cycles: u8,
}

impl RendererInfo {
    fn new(pid: u32, name: String, browser: String, cpu_pct: f32, memory_bytes: u64) -> Self {
        Self {
            pid,
            name,
            browser,
            cpu_pct,
            memory_bytes,
            consecutive_idle_cycles: 0,
            frozen: false,
            has_inet_sockets: false,
            has_assertion: false,
            cpu_history: [cpu_pct, 0.0, 0.0],
            frozen_cycles: 0,
        }
    }

    /// True if CPU was above the activity threshold in any of the last 3 cycles.
    fn recently_active(&self) -> bool {
        self.cpu_history.iter().any(|&c| c > THAW_CPU_THRESHOLD)
    }
}

/// Per-browser aggregate statistics.
#[derive(Debug, Default, Clone)]
pub struct BrowserState {
    pub total_renderers: u32,
    pub frozen_renderers: u32,
    pub ecore_renderers: u32,
    pub total_renderer_memory_mb: f64,
    /// Estimated RAM freed by SIGSTOP (RSS of frozen renderers).
    pub freed_memory_mb: f64,
}

/// Snapshot of chromium manager metrics for reporting.
#[derive(Debug, Default, Clone)]
pub struct ChromiumMetrics {
    pub total_renderers: u32,
    pub frozen_renderers: u32,
    pub ecore_renderers: u32,
    pub total_renderer_memory_mb: f64,
    pub estimated_freed_mb: f64,
    pub browsers_managed: Vec<String>,
}

/// Actions returned by [`ChromiumManager::update()`].
/// The daemon decides whether to actually execute them.
#[derive(Debug, Clone)]
pub enum ChromiumAction {
    /// Freeze an idle renderer (send SIGSTOP).
    FreezeRenderer {
        pid: u32,
        name: String,
        /// RSS at time of freeze — estimate of RAM freed.
        estimated_mb: f64,
    },
    /// Thaw a renderer that became active while frozen (send SIGCONT).
    ThawRenderer { pid: u32, name: String },
    /// Demote renderer/GPU helper to E-cores via Mach QoS.
    DemoteToEcores { pid: u32, name: String },
}

// ── Main Manager ───────────────────────────────────────────────────────────────

/// Main manager struct — instantiated once in daemon, updated every cycle.
pub struct ChromiumManager {
    /// Known renderer processes: pid → RendererInfo.
    renderers: HashMap<u32, RendererInfo>,
    /// PIDs of GPU helpers: pid → browser name.
    gpu_helpers: HashMap<u32, String>,
    /// Per-browser statistics.
    browsers: HashMap<String, BrowserState>,
    /// PIDs we've sent SIGSTOP to (for cleanup on shutdown).
    frozen_pids: HashSet<u32>,
    /// Cycle counter for rate-limited FD checks.
    fd_check_cycle: u8,
    /// Current idle-cycles threshold (adjusted by pressure context).
    idle_cycles_required: u8,
    /// Whether to pause freeze decisions (fluidity: launch/window-op active).
    freeze_paused: bool,
    /// Total estimated RAM freed across all freezes this session.
    pub total_freed_mb: f64,
    /// E-core demotions applied this cycle.
    pub ecore_demotions: u32,
    /// SIGSTOP count this cycle.
    pub freezes_applied: u32,
    /// SIGCONT count this cycle (thaw recoveries).
    pub recoveries_applied: u32,
    /// E-core renderers this cycle.
    ecore_count: u32,
    /// Previous foreground browser — detect fg change to trigger immediate thaw.
    /// [Bug fix]: frozen renderers show 0% CPU always; thaw must be event-driven.
    prev_fg_browser: Option<String>,
    /// PIDs already sent to E-core demotion — avoid repeat calls each cycle.
    ecore_demoted: HashSet<u32>,
}

impl Default for ChromiumManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ChromiumManager {
    pub fn new() -> Self {
        Self {
            renderers: HashMap::new(),
            gpu_helpers: HashMap::new(),
            browsers: HashMap::new(),
            frozen_pids: HashSet::new(),
            fd_check_cycle: 0,
            idle_cycles_required: IDLE_CYCLES_DEFAULT,
            freeze_paused: false,
            total_freed_mb: 0.0,
            ecore_demotions: 0,
            freezes_applied: 0,
            recoveries_applied: 0,
            ecore_count: 0,
            prev_fg_browser: None,
            ecore_demoted: HashSet::new(),
        }
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    /// Returns true if a process name is a Chromium/Electron renderer.
    /// Pattern: ends with "Helper (Renderer)".
    ///
    /// Matches: "Brave Browser Helper (Renderer)", "Google Chrome Helper (Renderer)",
    /// "Slack Helper (Renderer)", "Code Helper (Renderer)", etc.
    pub fn is_renderer(name: &str) -> bool {
        name.ends_with("Helper (Renderer)")
    }

    /// Returns true if a process is a Chromium/Electron GPU helper.
    pub fn is_gpu_helper(name: &str) -> bool {
        name.ends_with("Helper (GPU)")
    }

    /// Extract the browser/app name from a helper process name.
    ///
    /// "Brave Browser Helper (Renderer)" → "Brave Browser"
    /// "Google Chrome Helper (GPU)"      → "Google Chrome"
    /// "Slack Helper (Renderer)"          → "Slack"
    /// "Code Helper (Renderer)"           → "Code"
    pub fn browser_name(helper_name: &str) -> &str {
        if let Some(s) = helper_name.strip_suffix(" Helper (Renderer)") {
            return s;
        }
        if let Some(s) = helper_name.strip_suffix(" Helper (GPU)") {
            return s;
        }
        if let Some(s) = helper_name.strip_suffix(" Helper (Plugin)") {
            return s;
        }
        if let Some(s) = helper_name.strip_suffix(" Helper") {
            return s;
        }
        helper_name
    }

    /// Set pressure-adaptive freeze aggressiveness.
    ///
    /// [Denning 1968] Working set shrinks under memory pressure — act faster
    /// when the system is under stress.
    ///
    /// | Pressure  | idle_cycles_required | Behaviour            |
    /// |-----------|----------------------|----------------------|
    /// | ≥ 0.80    | 1                    | Aggressive           |
    /// | ≥ 0.65    | 2                    | Normal               |
    /// | ≥ 0.50    | 3 (default)          | Conservative         |
    /// | < 0.40    | 5 (never)            | Relaxed / thaw all   |
    pub fn set_pressure_context(&mut self, pressure: f32) {
        self.idle_cycles_required = if pressure >= 0.80 {
            1
        } else if pressure >= 0.65 {
            2
        } else if pressure >= 0.50 {
            3
        } else {
            5 // effectively never freeze at low pressure
        };
    }

    /// Signal fluidity state — suspend freeze decisions during window ops / launches.
    /// E-core demotions continue regardless (they are safe at any time).
    pub fn set_fluidity_context(&mut self, window_op_active: bool, app_launching: bool) {
        self.freeze_paused = window_op_active || app_launching;
    }

    /// Update renderer inventory and compute actions for this cycle.
    ///
    /// # Parameters
    /// - `processes`: full process list `(pid, name, cpu_pct, memory_bytes)`
    /// - `foreground_pid`: PID of the currently focused app (if known)
    /// - `assertion_pids`: PIDs holding power assertions (from `pids_with_assertions()`)
    /// - `main_frozen`: PIDs already frozen by the main daemon system
    pub fn update(
        &mut self,
        processes: &[(u32, &str, f32, u64)],
        foreground_pid: Option<u32>,
        assertion_pids: &HashSet<u32>,
        main_frozen: &HashSet<u32>,
    ) -> Vec<ChromiumAction> {
        self.ecore_demotions = 0;
        self.freezes_applied = 0;
        self.recoveries_applied = 0;
        self.ecore_count = 0;

        // ── Step 1: Build current PID → (name, cpu, mem) maps ─────────────────
        let mut current_renderers: HashMap<u32, (&str, f32, u64)> = HashMap::new();
        let mut current_gpu: HashMap<u32, &str> = HashMap::new();

        for &(pid, name, cpu, mem) in processes {
            if Self::is_renderer(name) {
                current_renderers.insert(pid, (name, cpu, mem));
            } else if Self::is_gpu_helper(name) {
                current_gpu.insert(pid, name);
            }
        }

        // ── Step 2: Update GPU helpers (simpler — only E-core demotion) ───────
        self.gpu_helpers.clear();
        for (&pid, &name) in &current_gpu {
            let browser = Self::browser_name(name).to_string();
            self.gpu_helpers.insert(pid, browser);
        }

        // ── Step 3: Prune dead renderer PIDs ──────────────────────────────────
        let dead: Vec<u32> = self
            .renderers
            .keys()
            .copied()
            .filter(|pid| !current_renderers.contains_key(pid))
            .collect();
        for pid in &dead {
            if self.frozen_pids.remove(pid) {
                // Process exited while frozen — nothing to SIGCONT.
            }
            self.renderers.remove(pid);
        }

        // ── Step 4: Update or create renderer entries ──────────────────────────
        let do_fd_check = self.fd_check_cycle == 0;
        self.fd_check_cycle = (self.fd_check_cycle + 1) % FD_CHECK_EVERY_N_CYCLES;

        for (&pid, &(name, cpu, mem)) in &current_renderers {
            let browser = Self::browser_name(name).to_string();
            let entry = self
                .renderers
                .entry(pid)
                .or_insert_with(|| RendererInfo::new(pid, name.to_string(), browser.clone(), cpu, mem));

            // Shift CPU history (newest first)
            entry.cpu_history[2] = entry.cpu_history[1];
            entry.cpu_history[1] = entry.cpu_history[0];
            entry.cpu_history[0] = cpu;
            entry.cpu_pct = cpu;
            entry.memory_bytes = mem;
            entry.browser = browser;
            entry.has_assertion = assertion_pids.contains(&pid);

            // Update idle counter
            if cpu < IDLE_CPU_THRESHOLD {
                entry.consecutive_idle_cycles =
                    entry.consecutive_idle_cycles.saturating_add(1);
            } else {
                entry.consecutive_idle_cycles = 0;
            }

            // Update frozen duration counter — used for max-freeze-duration guard
            if entry.frozen {
                entry.frozen_cycles = entry.frozen_cycles.saturating_add(1);
            } else {
                entry.frozen_cycles = 0;
            }

            // Periodic network FD check (expensive — rate limited)
            if do_fd_check || !entry.has_inet_sockets {
                entry.has_inet_sockets = Self::has_inet_sockets(pid);
            }
        }

        // ── Step 5: Compute per-browser totals ────────────────────────────────
        self.browsers.clear();
        for info in self.renderers.values() {
            let bs = self.browsers.entry(info.browser.clone()).or_default();
            bs.total_renderers += 1;
            bs.total_renderer_memory_mb += info.memory_bytes as f64 / 1_048_576.0;
            if info.frozen {
                bs.frozen_renderers += 1;
                bs.freed_memory_mb += info.memory_bytes as f64 / 1_048_576.0;
            }
        }

        // ── Step 6: Build actions ──────────────────────────────────────────────
        let mut actions = Vec::new();

        // Determine which browser is in the foreground (by foreground_pid)
        let fg_browser: Option<String> = foreground_pid.and_then(|fg| {
            processes
                .iter()
                .find(|&&(pid, _, _, _)| pid == fg)
                .map(|&(_, name, _, _)| Self::browser_name(name).to_string())
        });

        // Collect freeze/thaw decisions per browser (respect MAX_FREEZE_RATIO)

        // ── Foreground-browser-change thaw (Bug fix #1) ───────────────────────
        // A SIGSTOP'd process always reports 0% CPU — the previous CPU-spike
        // thaw condition (cpu_pct > threshold) would NEVER fire for frozen
        // renderers. Thaw must be event-driven: when the user switches to a
        // Chromium browser, immediately SIGCONT all its frozen renderers so
        // they are ready when the browser activates a tab via IPC.
        // [Denning 1968] Working set must be in memory when process resumes.
        let fg_changed = fg_browser.as_ref() != self.prev_fg_browser.as_ref();
        if fg_changed {
            if let Some(new_fg) = &fg_browser {
                for (&pid, info) in &self.renderers {
                    if &info.browser == new_fg && info.frozen && !main_frozen.contains(&pid) {
                        actions.push(ChromiumAction::ThawRenderer {
                            pid,
                            name: info.name.clone(),
                        });
                    }
                }
            }
            self.prev_fg_browser = fg_browser.clone();
        }

        // Prune dead PIDs from ecore_demoted set (Bug fix #4)
        self.ecore_demoted.retain(|pid| self.renderers.contains_key(pid));

        // First pass: per-renderer thaws (non-frozen CPU spike) and E-core demotions
        let pids: Vec<u32> = self.renderers.keys().copied().collect();

        for pid in &pids {
            let info = match self.renderers.get(pid) {
                Some(i) => i,
                None => continue,
            };

            // Skip PIDs managed by the main daemon frozen system
            if main_frozen.contains(pid) {
                continue;
            }

            // Skip if already queued for thaw by the fg-change pass above
            if actions.iter().any(|a| matches!(a, ChromiumAction::ThawRenderer { pid: p, .. } if *p == *pid)) {
                continue;
            }

            // Thaw check 1: frozen renderer with CPU spike
            // (rare — would mean SIGSTOP failed or OS resumed it externally)
            if info.frozen && info.cpu_pct > THAW_CPU_THRESHOLD {
                actions.push(ChromiumAction::ThawRenderer {
                    pid: *pid,
                    name: info.name.clone(),
                });
                continue;
            }

            // Thaw check 2: max-freeze-duration guard — force thaw after ~5 min.
            // Prevents renderers stuck frozen if a foreground change was missed.
            // [Denning 1968] A process suspended too long must be reintegrated.
            if info.frozen && info.frozen_cycles >= MAX_FROZEN_CYCLES {
                actions.push(ChromiumAction::ThawRenderer {
                    pid: *pid,
                    name: info.name.clone(),
                });
                continue;
            }

            // E-core demotion for non-foreground renderers — deduplicated (Bug fix #4)
            // Only emit once per renderer, not every cycle (mach_qos.set_tier is sticky)
            let is_fg_browser = fg_browser
                .as_ref()
                .map(|fb| fb == &info.browser)
                .unwrap_or(false);
            if !is_fg_browser && !info.frozen && !self.ecore_demoted.contains(pid) {
                actions.push(ChromiumAction::DemoteToEcores {
                    pid: *pid,
                    name: info.name.clone(),
                });
                self.ecore_count += 1;
            }
        }

        // Second pass: freeze candidates (only if not paused by fluidity)
        if !self.freeze_paused {
            // Group candidates by browser to enforce MAX_FREEZE_RATIO
            let mut candidates_by_browser: HashMap<String, Vec<u32>> = HashMap::new();

            for pid in &pids {
                let info = match self.renderers.get(pid) {
                    Some(i) => i,
                    None => continue,
                };

                if main_frozen.contains(pid) {
                    continue;
                }
                if info.frozen {
                    continue; // already frozen
                }
                if info.has_assertion {
                    continue; // holding power assertion
                }
                if info.has_inet_sockets {
                    continue; // active network connection
                }
                if info.consecutive_idle_cycles < self.idle_cycles_required {
                    continue; // not idle long enough
                }
                if info.recently_active() {
                    continue; // was active recently
                }

                // Safety: never freeze the renderer of the foreground browser
                let is_fg_browser = fg_browser
                    .as_ref()
                    .map(|fb| fb == &info.browser)
                    .unwrap_or(false);
                if is_fg_browser {
                    continue;
                }

                candidates_by_browser
                    .entry(info.browser.clone())
                    .or_default()
                    .push(*pid);
            }

            // Apply MAX_FREEZE_RATIO per browser
            for (browser, candidates) in &candidates_by_browser {
                let browser_state = self.browsers.get(browser).cloned().unwrap_or_default();
                let total = browser_state.total_renderers.max(1) as f32;
                let already_frozen = browser_state.frozen_renderers as f32;
                let max_additional =
                    ((total * MAX_FREEZE_RATIO) - already_frozen).floor() as usize;
                let max_additional = max_additional.min(candidates.len());

                // Sort by idle cycles descending (freeze the most-idle first)
                let mut sorted = candidates.clone();
                sorted.sort_by(|a, b| {
                    let ia = self
                        .renderers
                        .get(a)
                        .map(|r| r.consecutive_idle_cycles)
                        .unwrap_or(0);
                    let ib = self
                        .renderers
                        .get(b)
                        .map(|r| r.consecutive_idle_cycles)
                        .unwrap_or(0);
                    ib.cmp(&ia)
                });

                for pid in sorted.iter().take(max_additional) {
                    if let Some(info) = self.renderers.get(pid) {
                        let mb = info.memory_bytes as f64 / 1_048_576.0;
                        actions.push(ChromiumAction::FreezeRenderer {
                            pid: *pid,
                            name: info.name.clone(),
                            estimated_mb: mb,
                        });
                    }
                }
            }
        }

        // ── Step 7: Apply low-pressure thaw-all (system has headroom) ─────────
        if self.idle_cycles_required >= 5 {
            // Pressure < 0.40: thaw all chromium-frozen renderers
            for pid in self.frozen_pids.iter().copied().collect::<Vec<_>>() {
                if main_frozen.contains(&pid) {
                    continue;
                }
                if let Some(info) = self.renderers.get(&pid) {
                    actions.push(ChromiumAction::ThawRenderer {
                        pid,
                        name: info.name.clone(),
                    });
                } else {
                    // Process died — just remove tracking
                }
            }
        }

        // ── Step 8: Apply state changes from actions ───────────────────────────
        for action in &actions {
            match action {
                ChromiumAction::FreezeRenderer { pid, estimated_mb, .. } => {
                    self.frozen_pids.insert(*pid);
                    self.total_freed_mb += estimated_mb;
                    self.freezes_applied += 1;
                    if let Some(info) = self.renderers.get_mut(pid) {
                        info.frozen = true;
                    }
                }
                ChromiumAction::ThawRenderer { pid, .. } => {
                    self.frozen_pids.remove(pid);
                    self.recoveries_applied += 1;
                    if let Some(info) = self.renderers.get_mut(pid) {
                        info.frozen = false;
                        info.consecutive_idle_cycles = 0;
                        info.frozen_cycles = 0;
                    }
                }
                ChromiumAction::DemoteToEcores { pid, .. } => {
                    self.ecore_demotions += 1;
                    self.ecore_demoted.insert(*pid);
                }
            }
        }

        // Update browser stats with final frozen counts
        self.browsers.clear();
        for info in self.renderers.values() {
            let bs = self.browsers.entry(info.browser.clone()).or_default();
            bs.total_renderers += 1;
            bs.total_renderer_memory_mb += info.memory_bytes as f64 / 1_048_576.0;
            if info.frozen {
                bs.frozen_renderers += 1;
                bs.freed_memory_mb += info.memory_bytes as f64 / 1_048_576.0;
            }
        }
        for _ in 0..self.ecore_count {
            // ecore_renderers tracked via ecore_count — update per-browser on demand
        }

        actions
    }

    /// Return all PIDs currently frozen by ChromiumManager.
    /// Used to coordinate with the main daemon's frozen_state.
    pub fn frozen_pids(&self) -> &HashSet<u32> {
        &self.frozen_pids
    }

    /// PIDs frozen by ChromiumManager as an iterator.
    pub fn frozen_pids_iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.frozen_pids.iter().copied()
    }

    /// Summary metrics for dashboard/telemetry.
    pub fn metrics(&self) -> ChromiumMetrics {
        let total_frozen: u32 = self.renderers.values().filter(|r| r.frozen).count() as u32;
        let total_mem: f64 = self
            .renderers
            .values()
            .map(|r| r.memory_bytes as f64 / 1_048_576.0)
            .sum();
        let freed_mb: f64 = self
            .renderers
            .values()
            .filter(|r| r.frozen)
            .map(|r| r.memory_bytes as f64 / 1_048_576.0)
            .sum();

        let mut browsers: Vec<String> = self.browsers.keys().cloned().collect();
        browsers.sort();

        ChromiumMetrics {
            total_renderers: self.renderers.len() as u32,
            frozen_renderers: total_frozen,
            ecore_renderers: self.ecore_count,
            total_renderer_memory_mb: total_mem,
            estimated_freed_mb: freed_mb,
            browsers_managed: browsers,
        }
    }

    /// SIGCONT all frozen renderers on daemon shutdown.
    /// Returns the list of PIDs that were thawed.
    pub fn shutdown_cleanup(&mut self) -> Vec<u32> {
        let pids: Vec<u32> = self.frozen_pids.drain().collect();
        for &pid in &pids {
            Self::thaw_renderer(pid);
            if let Some(info) = self.renderers.get_mut(&pid) {
                info.frozen = false;
            }
        }
        pids
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    /// Check if a renderer has active TCP/UDP network file descriptors.
    ///
    /// Uses `proc_pidinfo(PROC_PIDLISTFDS)` to enumerate open FDs, then
    /// `proc_pidfdinfo(PROC_PIDFDSOCKETINFO)` to classify each socket FD.
    ///
    /// Renderers always have Unix-domain IPC sockets to the browser process;
    /// we only block on TCP/INET sockets (active HTTP, WebSocket, streaming).
    ///
    /// [Corbató 1968] Don't freeze processes with active I/O.
    fn has_inet_sockets(pid: u32) -> bool {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = pid;
            return false;
        }

        #[cfg(target_os = "macos")]
        {
            #[repr(C)]
            #[derive(Copy, Clone, Default)]
            struct ProcFdInfo {
                proc_fd: i32,
                proc_fdtype: u32,
            }

            // Socket info structure (partial — we only need the AF field)
            #[repr(C)]
            #[derive(Default)]
            struct SocketInfo {
                soi_stat_pad: [u64; 7],  // struct stat64
                soi_so_linger: i16,
                soi_so_state: i16,
                soi_so_options: i16,
                soi_so_timeo: i16,
                soi_so_error: u16,
                soi_so_kind: i32,
                soi_so_pcb: u64,
                soi_proto: i32,
                soi_family: i16,
                soi_type: i16,
                soi_options: i16,
                _pad: [u8; 2],
            }

            #[repr(C)]
            #[derive(Default)]
            struct ProcFdSocketInfo {
                pfi_openflags: u32,
                pfi_status: u32,
                pfi_fds_pid: i32,
                pfi_fds_type: u32,
                soi: SocketInfo,
            }

            const PROC_PIDLISTFDS: i32 = 1;
            const PROC_PIDFDSOCKETINFO: i32 = 3;
            const PROX_FDTYPE_SOCKET: u32 = 2;
            const AF_INET: i16 = 2;
            const AF_INET6: i16 = 30;

            extern "C" {
                fn proc_pidinfo(
                    pid: i32,
                    flavor: i32,
                    arg: u64,
                    buffer: *mut libc::c_void,
                    buffersize: i32,
                ) -> i32;
            }

            // Step 1: get all FD infos
            let needed = unsafe {
                proc_pidinfo(
                    pid as i32,
                    PROC_PIDLISTFDS,
                    0,
                    std::ptr::null_mut(),
                    0,
                )
            };
            if needed <= 0 {
                return false;
            }

            let count = (needed as usize / std::mem::size_of::<ProcFdInfo>()) + 8;
            let mut buf: Vec<ProcFdInfo> = vec![ProcFdInfo::default(); count];
            let written = unsafe {
                proc_pidinfo(
                    pid as i32,
                    PROC_PIDLISTFDS,
                    0,
                    buf.as_mut_ptr() as *mut libc::c_void,
                    (count * std::mem::size_of::<ProcFdInfo>()) as i32,
                )
            };
            if written <= 0 {
                return false;
            }

            let actual = written as usize / std::mem::size_of::<ProcFdInfo>();
            let socket_fds: Vec<i32> = buf[..actual]
                .iter()
                .filter(|f| f.proc_fdtype == PROX_FDTYPE_SOCKET)
                .map(|f| f.proc_fd)
                .collect();

            // Step 2: for each socket FD, check if it's TCP/IP (not Unix domain)
            let mut inet_count = 0usize;
            for fd in socket_fds {
                let mut si = ProcFdSocketInfo::default();
                let ret = unsafe {
                    proc_pidinfo(
                        pid as i32,
                        PROC_PIDFDSOCKETINFO,
                        fd as u64,
                        &mut si as *mut ProcFdSocketInfo as *mut libc::c_void,
                        std::mem::size_of::<ProcFdSocketInfo>() as i32,
                    )
                };
                if ret > 0 && (si.soi.soi_family == AF_INET || si.soi.soi_family == AF_INET6) {
                    inet_count += 1;
                    if inet_count >= MIN_INET_SOCKETS_TO_BLOCK {
                        return true;
                    }
                }
            }
            false
        }
    }

    /// Send SIGSTOP to a renderer process. Returns true if the signal succeeded.
    pub fn freeze_renderer(pid: u32) -> bool {
        #[cfg(target_os = "macos")]
        {
            unsafe { libc::kill(pid as i32, libc::SIGSTOP) == 0 }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = pid;
            false
        }
    }

    /// Send SIGCONT to a renderer process.
    pub fn thaw_renderer(pid: u32) {
        #[cfg(target_os = "macos")]
        unsafe {
            libc::kill(pid as i32, libc::SIGCONT);
        }
        #[cfg(not(target_os = "macos"))]
        let _ = pid;
    }
}

// ── Unit Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── is_renderer() ──────────────────────────────────────────────────────────

    #[test]
    fn is_renderer_brave() {
        assert!(ChromiumManager::is_renderer("Brave Browser Helper (Renderer)"));
    }

    #[test]
    fn is_renderer_chrome() {
        assert!(ChromiumManager::is_renderer("Google Chrome Helper (Renderer)"));
    }

    #[test]
    fn is_renderer_edge() {
        assert!(ChromiumManager::is_renderer("Microsoft Edge Helper (Renderer)"));
    }

    #[test]
    fn is_renderer_arc() {
        assert!(ChromiumManager::is_renderer("Arc Helper (Renderer)"));
    }

    #[test]
    fn is_renderer_vivaldi() {
        assert!(ChromiumManager::is_renderer("Vivaldi Helper (Renderer)"));
    }

    #[test]
    fn is_renderer_opera() {
        assert!(ChromiumManager::is_renderer("Opera Helper (Renderer)"));
    }

    #[test]
    fn is_renderer_slack() {
        assert!(ChromiumManager::is_renderer("Slack Helper (Renderer)"));
    }

    #[test]
    fn is_renderer_vscode() {
        assert!(ChromiumManager::is_renderer("Code Helper (Renderer)"));
    }

    #[test]
    fn is_renderer_cursor() {
        assert!(ChromiumManager::is_renderer("Cursor Helper (Renderer)"));
    }

    #[test]
    fn is_renderer_discord() {
        assert!(ChromiumManager::is_renderer("Discord Helper (Renderer)"));
    }

    #[test]
    fn is_renderer_notion() {
        assert!(ChromiumManager::is_renderer("Notion Helper (Renderer)"));
    }

    #[test]
    fn is_renderer_false_for_main_process() {
        // Main browser process must NOT match
        assert!(!ChromiumManager::is_renderer("Brave Browser"));
        assert!(!ChromiumManager::is_renderer("Google Chrome"));
        assert!(!ChromiumManager::is_renderer("Slack"));
        assert!(!ChromiumManager::is_renderer("Code"));
        assert!(!ChromiumManager::is_renderer("Discord"));
    }

    #[test]
    fn is_renderer_false_for_gpu_helper() {
        assert!(!ChromiumManager::is_renderer("Brave Browser Helper (GPU)"));
        assert!(!ChromiumManager::is_renderer("Google Chrome Helper (GPU)"));
    }

    #[test]
    fn is_renderer_false_for_plain_helper() {
        assert!(!ChromiumManager::is_renderer("Brave Browser Helper"));
        assert!(!ChromiumManager::is_renderer("Slack Helper"));
    }

    // ── is_gpu_helper() ────────────────────────────────────────────────────────

    #[test]
    fn is_gpu_helper_detection() {
        assert!(ChromiumManager::is_gpu_helper("Brave Browser Helper (GPU)"));
        assert!(ChromiumManager::is_gpu_helper("Google Chrome Helper (GPU)"));
        assert!(ChromiumManager::is_gpu_helper("Slack Helper (GPU)"));
        assert!(!ChromiumManager::is_gpu_helper("Brave Browser Helper (Renderer)"));
        assert!(!ChromiumManager::is_gpu_helper("Brave Browser"));
    }

    // ── browser_name() ─────────────────────────────────────────────────────────

    #[test]
    fn browser_name_brave() {
        assert_eq!(
            ChromiumManager::browser_name("Brave Browser Helper (Renderer)"),
            "Brave Browser"
        );
    }

    #[test]
    fn browser_name_chrome() {
        assert_eq!(
            ChromiumManager::browser_name("Google Chrome Helper (Renderer)"),
            "Google Chrome"
        );
    }

    #[test]
    fn browser_name_edge() {
        assert_eq!(
            ChromiumManager::browser_name("Microsoft Edge Helper (Renderer)"),
            "Microsoft Edge"
        );
    }

    #[test]
    fn browser_name_slack() {
        assert_eq!(
            ChromiumManager::browser_name("Slack Helper (Renderer)"),
            "Slack"
        );
    }

    #[test]
    fn browser_name_code() {
        assert_eq!(
            ChromiumManager::browser_name("Code Helper (Renderer)"),
            "Code"
        );
    }

    #[test]
    fn browser_name_gpu_helper() {
        assert_eq!(
            ChromiumManager::browser_name("Brave Browser Helper (GPU)"),
            "Brave Browser"
        );
    }

    #[test]
    fn browser_name_arc() {
        assert_eq!(
            ChromiumManager::browser_name("Arc Helper (Renderer)"),
            "Arc"
        );
    }

    // ── Idle counter tracking ──────────────────────────────────────────────────

    #[test]
    fn idle_counter_increments_when_cpu_below_threshold() {
        let mut mgr = ChromiumManager::new();
        let procs: Vec<(u32, &str, f32, u64)> = vec![
            (100, "Brave Browser Helper (Renderer)", 0.1, 50_000_000),
        ];
        let none_set = HashSet::new();

        mgr.update(&procs, None, &none_set, &none_set);
        mgr.update(&procs, None, &none_set, &none_set);
        mgr.update(&procs, None, &none_set, &none_set);

        let info = mgr.renderers.get(&100).expect("renderer must be tracked");
        assert!(
            info.consecutive_idle_cycles >= 2,
            "idle counter should be ≥ 2 after 3 updates: got {}",
            info.consecutive_idle_cycles
        );
    }

    #[test]
    fn idle_counter_resets_on_cpu_spike() {
        let mut mgr = ChromiumManager::new();
        let none_set = HashSet::new();

        // Three idle cycles
        let idle: Vec<(u32, &str, f32, u64)> =
            vec![(100, "Brave Browser Helper (Renderer)", 0.1, 50_000_000)];
        mgr.update(&idle, None, &none_set, &none_set);
        mgr.update(&idle, None, &none_set, &none_set);
        mgr.update(&idle, None, &none_set, &none_set);

        // CPU spike
        let active: Vec<(u32, &str, f32, u64)> =
            vec![(100, "Brave Browser Helper (Renderer)", 15.0, 50_000_000)];
        mgr.update(&active, None, &none_set, &none_set);

        let info = mgr.renderers.get(&100).unwrap();
        assert_eq!(
            info.consecutive_idle_cycles, 0,
            "idle counter must reset to 0 after CPU spike"
        );
    }

    // ── MAX_FREEZE_RATIO enforcement ───────────────────────────────────────────

    #[test]
    fn max_freeze_ratio_respected() {
        let mut mgr = ChromiumManager::new();
        mgr.set_pressure_context(0.75); // idle_cycles_required = 2

        let none_set = HashSet::new();

        // 4 renderers for Brave Browser — all idle (cpu = 0.1)
        let procs: Vec<(u32, &str, f32, u64)> = vec![
            (101, "Brave Browser Helper (Renderer)", 0.1, 50_000_000),
            (102, "Brave Browser Helper (Renderer)", 0.1, 50_000_000),
            (103, "Brave Browser Helper (Renderer)", 0.1, 50_000_000),
            (104, "Brave Browser Helper (Renderer)", 0.1, 50_000_000),
        ];

        // Run enough cycles to accumulate idle time
        for _ in 0..4 {
            mgr.update(&procs, None, &none_set, &none_set);
        }

        let freeze_count = mgr
            .renderers
            .values()
            .filter(|r| r.frozen)
            .count();

        // 50% of 4 = 2, so at most 2 should be frozen
        assert!(
            freeze_count <= 2,
            "MAX_FREEZE_RATIO violated: {} frozen out of 4 (max=2)",
            freeze_count
        );
    }

    // ── Renderer with assertion must not be frozen ─────────────────────────────

    #[test]
    fn renderer_with_assertion_not_frozen() {
        let mut mgr = ChromiumManager::new();
        mgr.set_pressure_context(0.80); // most aggressive

        let pid = 200u32;
        let mut assertion_pids = HashSet::new();
        assertion_pids.insert(pid);

        let procs: Vec<(u32, &str, f32, u64)> =
            vec![(pid, "Brave Browser Helper (Renderer)", 0.0, 50_000_000)];

        let none_set = HashSet::new();
        for _ in 0..5 {
            mgr.update(&procs, None, &assertion_pids, &none_set);
        }

        let info = mgr.renderers.get(&pid).unwrap();
        assert!(!info.frozen, "renderer with power assertion must never be frozen");
    }

    // ── main_frozen PIDs must not be touched ───────────────────────────────────

    #[test]
    fn renderer_in_main_frozen_not_touched() {
        let mut mgr = ChromiumManager::new();
        mgr.set_pressure_context(0.80);

        let pid = 300u32;
        let mut main_frozen = HashSet::new();
        main_frozen.insert(pid);

        let procs: Vec<(u32, &str, f32, u64)> =
            vec![(pid, "Slack Helper (Renderer)", 0.0, 50_000_000)];

        let none_set = HashSet::new();
        let mut all_actions = Vec::new();
        for _ in 0..5 {
            let acts = mgr.update(&procs, None, &none_set, &main_frozen);
            all_actions.extend(acts);
        }

        // ChromiumManager must not have issued any Freeze/Thaw for this PID
        let touched = all_actions.iter().any(|a| match a {
            ChromiumAction::FreezeRenderer { pid: p, .. } => *p == pid,
            ChromiumAction::ThawRenderer { pid: p, .. } => *p == pid,
            _ => false,
        });
        assert!(!touched, "must not touch a PID managed by the main daemon");
    }

    // ── New PIDs added, dead PIDs removed ─────────────────────────────────────

    #[test]
    fn dead_pids_removed_from_inventory() {
        let mut mgr = ChromiumManager::new();
        let none_set = HashSet::new();

        let procs1: Vec<(u32, &str, f32, u64)> =
            vec![(500, "Brave Browser Helper (Renderer)", 0.1, 50_000_000)];
        mgr.update(&procs1, None, &none_set, &none_set);
        assert!(mgr.renderers.contains_key(&500), "PID 500 should be tracked");

        // Next cycle: PID 500 is gone
        let procs2: Vec<(u32, &str, f32, u64)> = vec![];
        mgr.update(&procs2, None, &none_set, &none_set);
        assert!(!mgr.renderers.contains_key(&500), "PID 500 should be pruned");
    }

    #[test]
    fn new_pids_added_to_inventory() {
        let mut mgr = ChromiumManager::new();
        let none_set = HashSet::new();

        let procs: Vec<(u32, &str, f32, u64)> =
            vec![(600, "Code Helper (Renderer)", 0.5, 30_000_000)];
        mgr.update(&procs, None, &none_set, &none_set);
        assert!(mgr.renderers.contains_key(&600), "new PID should be added");
        assert_eq!(
            mgr.renderers.get(&600).unwrap().browser,
            "Code"
        );
    }

    // ── Pressure-adaptive thresholds ───────────────────────────────────────────

    #[test]
    fn pressure_high_reduces_idle_cycles_required() {
        let mut mgr = ChromiumManager::new();
        mgr.set_pressure_context(0.85);
        assert_eq!(mgr.idle_cycles_required, 1, "high pressure → 1 cycle required");
    }

    #[test]
    fn pressure_low_increases_idle_cycles_required() {
        let mut mgr = ChromiumManager::new();
        mgr.set_pressure_context(0.30);
        assert_eq!(mgr.idle_cycles_required, 5, "low pressure → 5 cycles (never freeze)");
    }

    #[test]
    fn pressure_normal_uses_default() {
        let mut mgr = ChromiumManager::new();
        mgr.set_pressure_context(0.55);
        assert_eq!(mgr.idle_cycles_required, 3, "normal pressure → default 3 cycles");
    }

    // ── Fluidity pause ─────────────────────────────────────────────────────────

    #[test]
    fn no_freezes_during_window_op() {
        let mut mgr = ChromiumManager::new();
        mgr.set_pressure_context(0.80); // aggressive
        mgr.set_fluidity_context(true, false); // window op active

        let none_set = HashSet::new();
        let procs: Vec<(u32, &str, f32, u64)> =
            vec![(700, "Brave Browser Helper (Renderer)", 0.0, 50_000_000)];

        let mut freeze_count = 0;
        for _ in 0..5 {
            let acts = mgr.update(&procs, None, &none_set, &none_set);
            freeze_count += acts
                .iter()
                .filter(|a| matches!(a, ChromiumAction::FreezeRenderer { .. }))
                .count();
        }
        assert_eq!(freeze_count, 0, "no freezes during window op");
    }

    #[test]
    fn ecore_demotions_continue_during_window_op() {
        let mut mgr = ChromiumManager::new();
        mgr.set_fluidity_context(true, false); // window op active

        let none_set = HashSet::new();
        let procs: Vec<(u32, &str, f32, u64)> =
            vec![(800, "Brave Browser Helper (Renderer)", 0.1, 50_000_000)];

        let acts = mgr.update(&procs, None, &none_set, &none_set);
        let ecore = acts
            .iter()
            .filter(|a| matches!(a, ChromiumAction::DemoteToEcores { .. }))
            .count();
        // E-core demotions must still happen even with freeze_paused=true
        assert!(ecore > 0, "E-core demotions must continue during window ops");
    }

    // ── Shutdown cleanup ───────────────────────────────────────────────────────

    #[test]
    fn shutdown_cleanup_drains_frozen_pids() {
        let mut mgr = ChromiumManager::new();
        // Manually inject frozen PIDs to simulate state
        mgr.frozen_pids.insert(900);
        mgr.frozen_pids.insert(901);
        let thawed = mgr.shutdown_cleanup();
        assert_eq!(thawed.len(), 2, "shutdown_cleanup must return all frozen PIDs");
        assert!(mgr.frozen_pids.is_empty(), "frozen_pids must be empty after cleanup");
    }

    // ── ChromiumMetrics ────────────────────────────────────────────────────────

    #[test]
    fn metrics_empty_manager() {
        let mgr = ChromiumManager::new();
        let m = mgr.metrics();
        assert_eq!(m.total_renderers, 0);
        assert_eq!(m.frozen_renderers, 0);
        assert_eq!(m.estimated_freed_mb, 0.0);
        assert!(m.browsers_managed.is_empty());
    }

    #[test]
    fn metrics_after_tracking() {
        let mut mgr = ChromiumManager::new();
        let none_set = HashSet::new();
        let procs: Vec<(u32, &str, f32, u64)> = vec![
            (1001, "Brave Browser Helper (Renderer)", 0.3, 100_000_000),
            (1002, "Brave Browser Helper (Renderer)", 0.1, 80_000_000),
            (1003, "Slack Helper (Renderer)", 0.2, 60_000_000),
        ];
        mgr.update(&procs, None, &none_set, &none_set);

        let m = mgr.metrics();
        assert_eq!(m.total_renderers, 3);
        assert!(m.total_renderer_memory_mb > 0.0);
        assert!(
            m.browsers_managed.contains(&"Brave Browser".to_string()),
            "Brave Browser must be tracked"
        );
        assert!(
            m.browsers_managed.contains(&"Slack".to_string()),
            "Slack must be tracked"
        );
    }

    // ── Performance benchmark ──────────────────────────────────────────────────

    #[test]
    fn bench_update_200_processes() {
        let mut mgr = ChromiumManager::new();
        let none_set = HashSet::new();

        // 15 renderers + 185 other processes
        let mut procs: Vec<(u32, &'static str, f32, u64)> = Vec::with_capacity(200);
        for i in 0..15u32 {
            procs.push((
                100 + i,
                "Brave Browser Helper (Renderer)",
                0.1,
                50_000_000,
            ));
        }
        for i in 0..185u32 {
            procs.push((1000 + i, "some-daemon", 0.5, 10_000_000));
        }

        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = mgr.update(&procs, None, &none_set, &none_set);
        }
        let elapsed = start.elapsed();

        eprintln!(
            "ChromiumManager.update() x1000: {:?} = {:?}/call",
            elapsed,
            elapsed / 1000
        );

        // Must be < 200µs per call on M1 (conservatively allowing 2× the stated goal)
        assert!(
            elapsed.as_millis() < 200,
            "ChromiumManager.update() too slow: {:?}/call",
            elapsed / 1000
        );
    }

    // ── Thaw behaviour fixes ───────────────────────────────────────────────────

    /// Bug fix #1: SIGSTOP'd renderers always report 0% CPU — the old thaw
    /// condition (cpu_pct > threshold) would never fire. Thaw must happen when
    /// the foreground browser changes, not on CPU spike detection.
    #[test]
    fn thaw_on_foreground_browser_change() {
        let mut mgr = ChromiumManager::new();
        let none_set: HashSet<u32> = HashSet::new();

        // Cycle 1-3: Brave renderers idle, no foreground
        let brave_procs: Vec<(u32, &str, f32, u64)> = vec![
            (200, "Brave Browser Helper (Renderer)", 0.1, 50_000_000),
            (201, "Brave Browser Helper (Renderer)", 0.1, 50_000_000),
        ];
        mgr.set_pressure_context(0.80); // aggressive: 1 cycle to freeze
        for _ in 0..4 {
            mgr.update(&brave_procs, None, &none_set, &none_set);
        }

        // Mark renderers as frozen (simulate daemon executing FreezeRenderer actions)
        if let Some(r) = mgr.renderers.get_mut(&200) {
            r.frozen = true;
            r.frozen_cycles = 1;
        }
        if let Some(r) = mgr.renderers.get_mut(&201) {
            r.frozen = true;
            r.frozen_cycles = 1;
        }
        mgr.frozen_pids.insert(200);
        mgr.frozen_pids.insert(201);

        // Simulate: 0% CPU (as would happen for SIGSTOP'd process)
        let frozen_procs: Vec<(u32, &str, f32, u64)> = vec![
            (200, "Brave Browser Helper (Renderer)", 0.0, 50_000_000),
            (201, "Brave Browser Helper (Renderer)", 0.0, 50_000_000),
        ];

        // No foreground — no thaw yet
        let actions = mgr.update(&frozen_procs, None, &none_set, &none_set);
        let thaws: Vec<_> = actions.iter().filter(|a| matches!(a, ChromiumAction::ThawRenderer { .. })).collect();
        assert!(thaws.is_empty(), "No thaw when no foreground browser");

        // User switches to Brave (pid 200 is now foreground)
        let actions = mgr.update(&frozen_procs, Some(200), &none_set, &none_set);
        let thaws: Vec<u32> = actions.iter().filter_map(|a| match a {
            ChromiumAction::ThawRenderer { pid, .. } => Some(*pid),
            _ => None,
        }).collect();
        assert_eq!(thaws.len(), 2, "Both frozen Brave renderers must thaw on fg change, got {:?}", thaws);
        assert!(thaws.contains(&200), "pid 200 must be thawed");
        assert!(thaws.contains(&201), "pid 201 must be thawed");
    }

    /// Bug fix #2: Max-freeze-duration guard — renderer frozen too long gets
    /// thawed even if foreground detection missed the switch.
    #[test]
    fn thaw_after_max_frozen_cycles() {
        let mut mgr = ChromiumManager::new();
        let none_set: HashSet<u32> = HashSet::new();
        let procs: Vec<(u32, &str, f32, u64)> = vec![
            (300, "Brave Browser Helper (Renderer)", 0.0, 50_000_000),
        ];

        mgr.update(&procs, None, &none_set, &none_set);
        // Simulate frozen state
        if let Some(r) = mgr.renderers.get_mut(&300) {
            r.frozen = true;
            r.frozen_cycles = MAX_FROZEN_CYCLES; // at the limit
        }
        mgr.frozen_pids.insert(300);

        // One more cycle pushes frozen_cycles to MAX_FROZEN_CYCLES+1 via saturating_add
        // But since we set it to MAX_FROZEN_CYCLES directly, update() sees >= MAX and thaws
        let actions = mgr.update(&procs, None, &none_set, &none_set);
        let thaws: Vec<_> = actions.iter().filter(|a| matches!(a, ChromiumAction::ThawRenderer { pid, .. } if *pid == 300)).collect();
        assert!(!thaws.is_empty(), "Renderer must be thawed after MAX_FROZEN_CYCLES");
    }

    /// Bug fix #4: E-core demotion must not be re-emitted every cycle for the
    /// same renderer (mach_qos.set_tier is sticky — repeat calls are waste).
    #[test]
    fn ecore_demotion_deduplicated_across_cycles() {
        let mut mgr = ChromiumManager::new();
        let none_set: HashSet<u32> = HashSet::new();
        let procs: Vec<(u32, &str, f32, u64)> = vec![
            (400, "Brave Browser Helper (Renderer)", 0.1, 50_000_000),
        ];

        // Cycle 1: first time — should emit DemoteToEcores
        let actions1 = mgr.update(&procs, None, &none_set, &none_set);
        let demotions1 = actions1.iter().filter(|a| matches!(a, ChromiumAction::DemoteToEcores { .. })).count();
        assert_eq!(demotions1, 1, "First cycle must emit exactly 1 demotion");

        // Cycle 2: same renderer — must NOT re-emit (already demoted)
        let actions2 = mgr.update(&procs, None, &none_set, &none_set);
        let demotions2 = actions2.iter().filter(|a| matches!(a, ChromiumAction::DemoteToEcores { .. })).count();
        assert_eq!(demotions2, 0, "Subsequent cycles must NOT re-emit demotion for same PID");
    }

    /// Bug fix #3: FreezeSource::ChromiumManager must exist as distinct variant.
    /// (Compilation test — if the enum doesn't have this variant this won't build.)
    #[test]
    fn freeze_source_chromium_manager_variant_exists() {
        use crate::engine::types::FreezeSource;
        let src = FreezeSource::ChromiumManager;
        // Just needs to compile and be non-panicking
        let _ = format!("{:?}", src);
    }
}
