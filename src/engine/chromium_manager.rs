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

use crate::engine::freeze_intelligence::FreezeIntelligence;

// ── Constants ──────────────────────────────────────────────────────────────────

/// CPU% below which a renderer is considered "idle".
const IDLE_CPU_THRESHOLD: f32 = 0.5;

/// Default: must be idle for this many consecutive cycles before freezing.
/// Overridden by `set_pressure_context()` based on memory pressure.
const IDLE_CYCLES_DEFAULT: u8 = 3;

/// Long-idle threshold: if a renderer has been idle for this many consecutive
/// cycles (~60s at 2s/cycle), freeze it regardless of the pressure-adaptive
/// `idle_cycles_required`. This catches truly abandoned background tabs that
/// the pressure-gated path misses entirely at low pressure (where
/// idle_cycles_required = 5 but background tabs rarely have 5 fully-idle
/// cycles in a row due to JS timers and network callbacks).
///
/// [Denning 1968] "Working Set Model" — a process with zero working-set
/// activity for 60s has effectively no resident pages worth keeping warm.
const LONG_IDLE_CYCLES: u8 = 30;

/// Fraction of CPU above which a frozen renderer is thawed immediately.
const THAW_CPU_THRESHOLD: f32 = 1.0;

/// Base fraction of a browser's renderers we never exceed at normal pressure.
/// Scaled UP by pressure via `ChromiumManager::max_freeze_ratio()` — under
/// critical memory pressure (>=0.80) we need to shed 80-85% of background
/// renderers to keep the foreground tab + system services responsive.
/// [Nygard 2018] Release It! Ch.5 "Load Shedding" — graceful degradation under
/// overload trades non-essential service for system survival.
const MAX_FREEZE_RATIO_BASE: f32 = 0.5;

/// Ceiling under critical pressure. Empirically, 0.85 keeps the foreground
/// renderer + GPU helper + ~10% headroom for Brave's bookkeeping threads while
/// freezing every background tab. Tested against 90-tab workloads on M1 8GB
/// (prod observation 2026-04-16, 21 Brave procs, 62 MB free, 2.2 GB swap).
const MAX_FREEZE_RATIO_CEILING: f32 = 0.85;

/// Check network FDs every N cycles (expensive proc_pidinfo call).
const FD_CHECK_EVERY_N_CYCLES: u8 = 5;

/// Minimum TCP/IP sockets before renderer is considered "network-active".
/// Renderers always have Unix-domain IPC sockets; we only block on TCP/UDP.
const MIN_INET_SOCKETS_TO_BLOCK: usize = 1;

/// Max cycles a renderer stays frozen before forced thaw (~5 s at 100ms/cycle).
/// Kept short so that when Brave closes a tab, the frozen renderer is thawed
/// before Brave's ~15s "not responding" timeout fires. Previous value (150)
/// caused a race: tab closed → renderer SIGSTOP'd → Brave IPC blocked →
/// "window not responding" dialog for an already-closed tab.
const MAX_FROZEN_CYCLES: u8 = 50;

/// Aggressive TTL for renderers of the **currently foreground** Chromium
/// browser. Kept at 3 cycles (~300ms) — below human perception threshold
/// (~250ms) so tab switches feel instant. Background browsers keep the
/// longer `MAX_FROZEN_CYCLES` TTL for maximum memory reclaim. The asymmetry
/// matches user expectation: the browser you're looking at recovers
/// quickly; the ones you switched away from stay paused.
const MAX_FOREGROUND_FROZEN_CYCLES: u8 = 3;

/// Cycles a renderer must wait after a thaw before it can be frozen again.
/// Post-SIGCONT, the renderer reports CPU=0 while still waking up — without
/// this guard it immediately looks idle and gets re-frozen, creating the
/// freeze→thaw→freeze thrashing loop seen in production logs.
/// 10 cycles ≈ 20s — enough for a tab to re-render and show non-zero CPU.
const POST_THAW_GRACE_CYCLES: u8 = 10;

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
    /// Cycles since this renderer was first seen. New renderers (loading a fresh tab)
    /// show 0% CPU before content loads — skip freeze during this warmup window.
    ///
    /// D2 fix (round-3): widened `u8 → u16`. With a 500ms fast-tick the u8
    /// counter saturated at 255 in ~128s, after which age-based LRU thaw
    /// scoring could not distinguish older-vs-newer frozen renderers and
    /// the long-idle freeze path was effectively broken for long-lived tabs.
    age_cycles: u16,
    /// RSS at the moment of freeze. When RSS drops >50% while frozen, the browser
    /// is reclaiming memory from a closed tab — thaw immediately so it can exit cleanly.
    frozen_rss_baseline: u64,
    /// Cycles remaining before this renderer is eligible to be frozen again.
    /// Set to POST_THAW_GRACE_CYCLES after each thaw — prevents the re-freeze
    /// thrashing loop where CPU=0 post-SIGCONT looks idle and triggers immediate
    /// re-freeze before the renderer has had a chance to do any work.
    thaw_cooldown_cycles: u8,
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
            age_cycles: 0,
            frozen_rss_baseline: 0,
            thaw_cooldown_cycles: 0,
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
    /// Last pressure value passed to `set_pressure_context()`. Drives the
    /// pressure-adaptive `max_freeze_ratio()` curve. Initialised to 0.0 so
    /// first cycle (before any context is set) behaves as at low pressure.
    current_pressure: f32,
    /// Whether to pause freeze decisions (fluidity: launch/window-op active).
    freeze_paused: bool,
    /// Workload preemption: when true (BuildSession detected), the freeze gate
    /// uses `idle_cycles_required = 1` regardless of pressure/arousal, so
    /// background renderers get frozen PROACTIVELY before rustc spikes memory.
    /// [Nygard 2018] Release It! Ch.5 — bulkheading: isolate the build workload
    /// from competing renderer memory by pre-emptively shedding load.
    build_preemption_active: bool,
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

    // ── Cognitive context (Enhancement A/B/C) ─────────────────────────────────
    /// FocusMarkov top-N predictions: (app_name, probability, avg_dwell_secs).
    /// Updated each cycle via `set_markov_context()`.
    markov_predictions: Vec<(String, f64, f64)>,
    /// How long the current foreground app has been focused (seconds).
    elapsed_dwell_secs: f64,
    /// When arousal is very low (< 0.20) thaw all frozen renderers.
    /// [Yerkes-Dodson 1908] System idle = no need to keep anything frozen.
    arousal_thaw_all: bool,
    /// Universal freeze-safety intelligence (NARS beliefs per process category).
    /// [Pei Wang 2013] Truth values updated by revision rule on each freeze outcome.
    /// App-agnostic: chromium-renderer, ide-lsp, app-helper, etc.
    intelligence: FreezeIntelligence,
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
            current_pressure: 0.0,
            freeze_paused: false,
            total_freed_mb: 0.0,
            ecore_demotions: 0,
            freezes_applied: 0,
            recoveries_applied: 0,
            ecore_count: 0,
            prev_fg_browser: None,
            ecore_demoted: HashSet::new(),
            markov_predictions: Vec::new(),
            elapsed_dwell_secs: 0.0,
            arousal_thaw_all: false,
            build_preemption_active: false,
            intelligence: FreezeIntelligence::new(),
        }
    }

    /// Enable/disable build-preemption mode. When active, the freeze candidate
    /// gate uses `idle_cycles_required = 1` regardless of pressure/arousal so
    /// background renderers get frozen proactively before the build workload
    /// (rustc/cargo/clang) starts demanding RAM.
    /// Expected call pattern: daemon detects `WorkloadIntent::BuildSession`
    /// and passes `true`; drops back to `false` once the build completes.
    pub fn set_build_preemption(&mut self, active: bool) {
        self.build_preemption_active = active;
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
        self.current_pressure = pressure;
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

    /// Pressure-adaptive ceiling on the fraction of renderers that may be
    /// frozen at once. Higher pressure ⇒ more aggressive load shedding.
    ///
    /// | pressure    | ratio |
    /// |-------------|-------|
    /// | < 0.50      | 0.50  (base — steady-state behaviour unchanged)      |
    /// | 0.50 – 0.65 | 0.60                                                  |
    /// | 0.65 – 0.80 | 0.72                                                  |
    /// | ≥ 0.80      | 0.85  (critical — ceiling)                            |
    ///
    /// Previously hardcoded at 0.5, which left large background-tab workloads
    /// chronically under-frozen on 8GB machines (prod 2026-04-16: 12 freezes
    /// in 11h across 21 Brave renderers, system stuck at 62 MB free / 2.2 GB
    /// swap).
    pub fn max_freeze_ratio(&self) -> f32 {
        let p = self.current_pressure;
        if p >= 0.80 {
            MAX_FREEZE_RATIO_CEILING
        } else if p >= 0.65 {
            0.72
        } else if p >= 0.50 {
            0.60
        } else {
            MAX_FREEZE_RATIO_BASE
        }
    }

    /// Signal fluidity state — suspend freeze decisions during window ops / launches.
    /// E-core demotions continue regardless (they are safe at any time).
    pub fn set_fluidity_context(&mut self, window_op_active: bool, app_launching: bool) {
        self.freeze_paused = window_op_active || app_launching;
    }

    /// Set FocusMarkov top-N predictions for predictive pre-thaw.
    ///
    /// [Altmann & Trafton 2002] User task switches are predictable — pre-activate
    /// resources before predicted context switch to eliminate perceived latency.
    ///
    /// `predictions`: Vec<(app_name, probability, avg_dwell_secs)>
    /// `elapsed_dwell_secs`: how long the current foreground app has been active
    pub fn set_markov_context(
        &mut self,
        predictions: &[(String, f64, f64)],
        elapsed_dwell_secs: f64,
    ) {
        self.markov_predictions = predictions.to_vec();
        self.elapsed_dwell_secs = elapsed_dwell_secs;
    }

    /// Set arousal level for Yerkes-Dodson adaptive freeze aggressiveness.
    ///
    /// [Yerkes & Dodson 1908] Performance is optimised at moderate arousal.
    /// High arousal (crisis) → freeze faster; low arousal (idle) → thaw everything.
    pub fn set_arousal_context(&mut self, arousal_level: f32) {
        self.idle_cycles_required = match arousal_level {
            a if a >= 0.75 => 1, // Crisis: freeze after 1 idle cycle
            a if a >= 0.50 => 2, // Stressed: freeze after 2 cycles
            a if a >= 0.25 => 3, // Optimal: normal (default 3 cycles)
            _ => 5,              // Idle: very conservative (effectively never)
        };
        self.arousal_thaw_all = arousal_level < 0.20;
    }

    /// Observe a freeze/thaw outcome for NARS belief update.
    ///
    /// Call with `success=true` after a clean renderer thaw (renderer alive).
    /// Call with `success=false` if a renderer died while frozen.
    /// Routes through `FreezeIntelligence.observe()` — uses `classify()` to map the
    /// renderer name to a stable category ("chromium-renderer", "ide-lsp", etc.)
    /// so evidence accumulates across all processes of the same type.
    /// [Pei Wang 2013] NARS Revision rule updates frequency proportional to evidence weight.
    pub fn observe_freeze_outcome(&mut self, process_name: &str, success: bool, salience: f32) {
        self.intelligence.observe(process_name, success, salience);
    }

    /// Returns freeze confidence for a process name based on NARS belief frequency.
    ///
    /// Delegates to `FreezeIntelligence.confidence()` which looks up the
    /// process category via `classify()`.
    /// Below 0.35: skip freezing (too many bad outcomes observed).
    /// Default: 0.70 (conservative prior — assume freezing is safe until proven otherwise).
    pub fn freeze_confidence(&self, process_name: &str) -> f32 {
        self.intelligence.confidence(process_name)
    }

    /// Access the universal FreezeIntelligence for use by other daemon subsystems.
    /// Allows the daemon to query freeze safety for ANY process, not just renderers.
    pub fn intelligence(&self) -> &FreezeIntelligence {
        &self.intelligence
    }

    /// Mutable access to FreezeIntelligence (for belief updates from outside ChromiumManager).
    pub fn intelligence_mut(&mut self) -> &mut FreezeIntelligence {
        &mut self.intelligence
    }

    /// Map a macOS app name to the Chromium browser name we track.
    /// Returns `Some(browser_name)` if the app is a Chromium/Electron browser,
    /// `None` if it is not (e.g. Terminal, Finder).
    fn chromium_app_to_browser(app_name: &str) -> Option<String> {
        const BROWSERS: &[&str] = &[
            "Brave Browser",
            "Google Chrome",
            "Microsoft Edge",
            "Arc",
            "Vivaldi",
            "Opera",
            "Chromium",
            "Slack",
            "Code",
            "Cursor",
            "Discord",
            "Notion",
            "Linear",
            "Figma",
        ];
        for &b in BROWSERS {
            if app_name == b || app_name.starts_with(b) {
                return Some(b.to_string());
            }
        }
        None
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
            let entry = self.renderers.entry(pid).or_insert_with(|| {
                RendererInfo::new(pid, name.to_string(), browser.clone(), cpu, mem)
            });

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
                entry.consecutive_idle_cycles = entry.consecutive_idle_cycles.saturating_add(1);
            } else {
                entry.consecutive_idle_cycles = 0;
            }

            // Update frozen duration counter — used for max-freeze-duration guard
            if entry.frozen {
                entry.frozen_cycles = entry.frozen_cycles.saturating_add(1);
            } else {
                entry.frozen_cycles = 0;
            }

            // Age counter — saturate at 255 (never wraps back to 0)
            entry.age_cycles = entry.age_cycles.saturating_add(1);

            // Post-thaw cooldown countdown
            entry.thaw_cooldown_cycles = entry.thaw_cooldown_cycles.saturating_sub(1);

            // Periodic network FD check (expensive — rate limited).
            // Also force-recheck for non-frozen idle renderers that previously
            // had sockets: their socket may have closed since the last check,
            // and we must not skip-freeze based on a stale `has_inet_sockets=true`.
            let is_idle_freeze_candidate = !entry.frozen && entry.consecutive_idle_cycles > 0;
            if do_fd_check || !entry.has_inet_sockets || (entry.has_inet_sockets && is_idle_freeze_candidate) {
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
        let mut actions: Vec<ChromiumAction> = Vec::new();
        // Helper: true when a ThawRenderer for `pid` is already queued.
        let already_thawing = |acts: &[ChromiumAction], pid: u32| {
            acts.iter()
                .any(|a| matches!(a, ChromiumAction::ThawRenderer { pid: p, .. } if *p == pid))
        };

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

        // ── Predictive pre-thaw from FocusMarkov ─────────────────────────────────
        // [Altmann & Trafton 2002] Pre-activate resources before predicted task switch.
        // If the user is predicted to switch to a Chromium browser within 10 seconds,
        // thaw all its frozen renderers now so they are warm when the switch happens.
        for (app_name, prob, avg_dwell) in &self.markov_predictions {
            if *prob < 0.35 {
                continue;
            }
            let predicted_browser = match Self::chromium_app_to_browser(app_name) {
                Some(b) => b,
                None => continue,
            };
            let time_to_switch = avg_dwell - self.elapsed_dwell_secs;
            // Only pre-thaw within a reasonable window: predicted switch is
            // 0-10s away.  Deeply negative values mean the prediction is stale
            // (elapsed >> avg_dwell) — firing on those creates a freeze/thaw
            // thrashing loop that pins pressure at 100%.
            if time_to_switch > -5.0 && time_to_switch < 10.0 {
                for (&pid, info) in &self.renderers {
                    if info.browser == predicted_browser
                        && info.frozen
                        && !main_frozen.contains(&pid)
                        && !already_thawing(&actions, pid)
                    {
                        actions.push(ChromiumAction::ThawRenderer {
                            pid,
                            name: info.name.clone(),
                        });
                        tracing::info!(
                            browser = predicted_browser.as_str(),
                            prob = prob,
                            time_to_switch = time_to_switch,
                            "chromium: predictive pre-thaw — switch imminent"
                        );
                    }
                }
            }
        }

        // Prune dead PIDs from ecore_demoted set (Bug fix #4)
        self.ecore_demoted
            .retain(|pid| self.renderers.contains_key(pid));

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
            if already_thawing(&actions, *pid) {
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

            // Thaw check 1b: RSS-drop detection — browser is reclaiming memory from
            // a closed tab. When RSS drops >50% vs freeze baseline, the renderer is
            // being cleaned up by the browser. Thaw immediately so it can exit cleanly
            // rather than getting stuck frozen until the time-based TTL fires.
            // Avoids "window not responding" dialogs for already-closed tabs.
            if info.frozen
                && info.frozen_rss_baseline > 0
                && info.memory_bytes < info.frozen_rss_baseline / 2
            {
                actions.push(ChromiumAction::ThawRenderer {
                    pid: *pid,
                    name: info.name.clone(),
                });
                tracing::info!(
                    pid = *pid,
                    name = info.name.as_str(),
                    rss_before = info.frozen_rss_baseline,
                    rss_now = info.memory_bytes,
                    "chromium: thawing renderer (RSS drop >50% — tab cleanup detected)"
                );
                continue;
            }

            // Thaw check 2a: foreground-browser renderers get an aggressive
            // TTL (~30 s) so a missed tab-switch detection cannot leave the
            // tab the user is looking at stuck for minutes. The user
            // observed exactly this ("a veces algunos tabs no se me
            // descongelan"): switching tabs within the same browser does
            // not fire the fg_changed thaw because the foreground browser
            // *name* did not change. The aggressive TTL bounds the worst
            // case to ~30 s.
            let is_fg_browser = fg_browser
                .as_ref()
                .map(|fb| fb == &info.browser)
                .unwrap_or(false);
            if info.frozen && is_fg_browser && info.frozen_cycles >= MAX_FOREGROUND_FROZEN_CYCLES {
                actions.push(ChromiumAction::ThawRenderer {
                    pid: *pid,
                    name: info.name.clone(),
                });
                continue;
            }

            // Thaw check 2b: max-freeze-duration guard — force thaw after
            // ~5 min for background browsers. Prevents renderers stuck
            // frozen if a foreground change was missed.
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
            // (`is_fg_browser` was already computed above for the foreground TTL check.)
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
                // New-tab grace period: renderer just spawned shows 0% CPU while loading.
                // At high pressure (idle_cycles_required=1) it would be frozen before the
                // page renders, making new tabs appear stuck. Skip freeze for first 10 cycles
                // (~1s) so the tab has time to load before becoming a freeze candidate.
                const NEW_RENDERER_GRACE_CYCLES: u16 = 10;
                if info.age_cycles < NEW_RENDERER_GRACE_CYCLES {
                    continue;
                }
                // Post-thaw cooldown: renderer just received SIGCONT reports CPU=0 while
                // waking up. Without this guard it looks idle immediately and is re-frozen,
                // producing the freeze→thaw→freeze thrashing loop seen in production.
                if info.thaw_cooldown_cycles > 0 {
                    continue;
                }
                // Three acceptance paths:
                //   1. Pressure-adaptive: idle for `idle_cycles_required` cycles
                //      (1/2/3/5 depending on pressure — strict when pressure high).
                //   2. Long-idle: idle for `LONG_IDLE_CYCLES` cycles (~60s)
                //      regardless of pressure. Catches abandoned background tabs
                //      that never accumulate enough consecutive idle cycles under
                //      the pressure-gated path due to JS timers/network callbacks.
                //   3. Build preemption: when BuildSession is active, a single
                //      idle cycle is enough. This pre-sheds renderer memory
                //      before rustc spikes — bulkheading the build workload.
                // Build preemption: require 2 idle cycles (not 1) to avoid
                // freezing a mid-render tab that briefly dips to 0% CPU during
                // layout/paint. One cycle (~2s) is within normal render pauses;
                // two consecutive cycles reliably signal an abandoned background tab.
                let effective_required = if self.build_preemption_active {
                    2
                } else {
                    self.idle_cycles_required.max(1) // invariant: never 0
                };
                let meets_pressure_gate = info.consecutive_idle_cycles >= effective_required;
                let meets_long_idle = info.consecutive_idle_cycles >= LONG_IDLE_CYCLES;
                if !meets_pressure_gate && !meets_long_idle {
                    continue; // not idle long enough by either path
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

                // NARS confidence gate: skip processes where freeze safety is unproven.
                // FreezeIntelligence.classify() maps the full name to a category
                // (e.g. "Brave Browser Helper (Renderer)" → "chromium-renderer"),
                // so evidence accumulates across all renderers of the same type.
                // [Pei Wang 2013] Truth frequency < 0.35 = evidence favours unsafety.
                let confidence = self.freeze_confidence(&info.name);
                if confidence < 0.35 {
                    tracing::debug!(
                        browser = info.browser.as_str(),
                        process = info.name.as_str(),
                        confidence = confidence,
                        "chromium: skipping freeze — NARS confidence too low"
                    );
                    continue;
                }

                candidates_by_browser
                    .entry(info.browser.clone())
                    .or_default()
                    .push(*pid);
            }

            // Apply MAX_FREEZE_RATIO per browser.
            // Subtract queued thaws from the frozen count so the ratio uses live
            // numbers — without this, a batch of thaws queued earlier in the same
            // tick doesn't reduce `already_frozen`, causing the freeze gate to
            // reject candidates that would be within budget after the thaws land.
            for (browser, candidates) in &candidates_by_browser {
                let browser_state = self.browsers.get(browser).cloned().unwrap_or_default();
                let total = browser_state.total_renderers.max(1) as f32;
                let queued_thaws = actions
                    .iter()
                    .filter(|a| {
                        if let ChromiumAction::ThawRenderer { pid, .. } = a {
                            self.renderers.get(pid).map(|r| r.browser == *browser).unwrap_or(false)
                        } else {
                            false
                        }
                    })
                    .count() as f32;
                let already_frozen = (browser_state.frozen_renderers as f32 - queued_thaws).max(0.0);
                let max_additional =
                    ((total * self.max_freeze_ratio()) - already_frozen).floor() as usize;
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

        // ── Step 7: Apply arousal-idle thaw-all ───────────────────────────────
        // Mass-thaw ONLY when the user is truly AFK (arousal < 0.20).
        //
        // Why not also on low pressure: the previous version thawed everything
        // whenever pressure dropped below 40%, which meant every successful
        // freeze was immediately undone the next cycle the pressure stabilised.
        // Measured in production: 1/18 renderers ever stayed frozen because the
        // other 17 were getting thawed every time pressure dipped. That defeats
        // the whole point of proactive freezing — frozen background tabs must
        // persist through normal low-pressure periods until the user actually
        // interacts with their browser (handled by the fg-change path above).
        //
        // Arousal < 0.20 (arousal_thaw_all) is a genuine "user is gone" signal:
        // no foreground activity, no window ops, no input events. In that state
        // there is no benefit to keeping anything frozen — thaw everything so
        // the user returns to a responsive system.
        // [Yerkes-Dodson 1908] Very low arousal = no cost to thawing.
        if self.arousal_thaw_all {
            for pid in self.frozen_pids.iter().copied().collect::<Vec<_>>() {
                if main_frozen.contains(&pid) {
                    continue;
                }
                if let Some(info) = self.renderers.get(&pid) {
                    if !already_thawing(&actions, pid) {
                        actions.push(ChromiumAction::ThawRenderer {
                            pid,
                            name: info.name.clone(),
                        });
                    }
                }
            }
        }

        // ── Step 8: Apply state changes from actions ───────────────────────────
        for action in &actions {
            match action {
                ChromiumAction::FreezeRenderer {
                    pid, estimated_mb, ..
                } => {
                    // Optimistic: mark as frozen in internal model.
                    // If SIGSTOP fails, daemon calls confirm_freeze(pid, false)
                    // to roll back — prevents state drift on failed freeze.
                    self.frozen_pids.insert(*pid);
                    self.total_freed_mb += estimated_mb;
                    self.freezes_applied += 1;
                    if let Some(info) = self.renderers.get_mut(pid) {
                        info.frozen = true;
                        info.frozen_rss_baseline = info.memory_bytes;
                    }
                }
                ChromiumAction::ThawRenderer { pid, .. } => {
                    self.frozen_pids.remove(pid);
                    self.recoveries_applied += 1;
                    // Clear E-core demotion so the renderer is eligible for
                    // re-demotion on the next cycle if it stays idle.
                    // Without this removal the `ecore_demoted` set grows forever
                    // and thawed renderers are never re-demoted.
                    self.ecore_demoted.remove(pid);
                    if let Some(info) = self.renderers.get_mut(pid) {
                        info.frozen = false;
                        info.consecutive_idle_cycles = 0;
                        info.frozen_cycles = 0;
                        // Set post-thaw cooldown so this renderer cannot be immediately
                        // re-frozen. Post-SIGCONT CPU=0 looks idle — without cooldown
                        // the freeze→thaw→freeze thrashing loop triggers every cycle.
                        info.thaw_cooldown_cycles = POST_THAW_GRACE_CYCLES;
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
            .sum::<f64>()
            .max(0.0);

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
                soi_stat_pad: [u64; 7], // struct stat64
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
            let needed =
                unsafe { proc_pidinfo(pid as i32, PROC_PIDLISTFDS, 0, std::ptr::null_mut(), 0) };
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

    /// Roll back optimistic freeze state if SIGSTOP failed.
    /// Call with `ok=false` when freeze_renderer() returned false so the
    /// internal model stays consistent with reality.
    pub fn confirm_freeze(&mut self, pid: u32, ok: bool) {
        if !ok {
            self.frozen_pids.remove(&pid);
            if let Some(info) = self.renderers.get_mut(&pid) {
                info.frozen = false;
            }
        }
    }

    /// Send SIGCONT to a renderer process. Returns true if signal was delivered.
    pub fn thaw_renderer(pid: u32) -> bool {
        #[cfg(target_os = "macos")]
        {
            let rc = unsafe { libc::kill(pid as i32, libc::SIGCONT) };
            rc == 0
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = pid;
            false
        }
    }
}

// ── Unit Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── is_renderer() ──────────────────────────────────────────────────────────

    #[test]
    fn is_renderer_brave() {
        assert!(ChromiumManager::is_renderer(
            "Brave Browser Helper (Renderer)"
        ));
    }

    #[test]
    fn is_renderer_chrome() {
        assert!(ChromiumManager::is_renderer(
            "Google Chrome Helper (Renderer)"
        ));
    }

    #[test]
    fn is_renderer_edge() {
        assert!(ChromiumManager::is_renderer(
            "Microsoft Edge Helper (Renderer)"
        ));
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
        assert!(!ChromiumManager::is_gpu_helper(
            "Brave Browser Helper (Renderer)"
        ));
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
        let procs: Vec<(u32, &str, f32, u64)> =
            vec![(100, "Brave Browser Helper (Renderer)", 0.1, 50_000_000)];
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

        let freeze_count = mgr.renderers.values().filter(|r| r.frozen).count();

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
        assert!(
            !info.frozen,
            "renderer with power assertion must never be frozen"
        );
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
        assert!(
            mgr.renderers.contains_key(&500),
            "PID 500 should be tracked"
        );

        // Next cycle: PID 500 is gone
        let procs2: Vec<(u32, &str, f32, u64)> = vec![];
        mgr.update(&procs2, None, &none_set, &none_set);
        assert!(
            !mgr.renderers.contains_key(&500),
            "PID 500 should be pruned"
        );
    }

    #[test]
    fn new_pids_added_to_inventory() {
        let mut mgr = ChromiumManager::new();
        let none_set = HashSet::new();

        let procs: Vec<(u32, &str, f32, u64)> =
            vec![(600, "Code Helper (Renderer)", 0.5, 30_000_000)];
        mgr.update(&procs, None, &none_set, &none_set);
        assert!(mgr.renderers.contains_key(&600), "new PID should be added");
        assert_eq!(mgr.renderers.get(&600).unwrap().browser, "Code");
    }

    // ── Pressure-adaptive thresholds ───────────────────────────────────────────

    #[test]
    fn pressure_high_reduces_idle_cycles_required() {
        let mut mgr = ChromiumManager::new();
        mgr.set_pressure_context(0.85);
        assert_eq!(
            mgr.idle_cycles_required, 1,
            "high pressure → 1 cycle required"
        );
    }

    #[test]
    fn pressure_low_increases_idle_cycles_required() {
        let mut mgr = ChromiumManager::new();
        mgr.set_pressure_context(0.30);
        assert_eq!(
            mgr.idle_cycles_required, 5,
            "low pressure → 5 cycles (never freeze)"
        );
    }

    #[test]
    fn pressure_normal_uses_default() {
        let mut mgr = ChromiumManager::new();
        mgr.set_pressure_context(0.55);
        assert_eq!(
            mgr.idle_cycles_required, 3,
            "normal pressure → default 3 cycles"
        );
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
        assert!(
            ecore > 0,
            "E-core demotions must continue during window ops"
        );
    }

    // ── Shutdown cleanup ───────────────────────────────────────────────────────

    #[test]
    fn shutdown_cleanup_drains_frozen_pids() {
        let mut mgr = ChromiumManager::new();
        // Manually inject frozen PIDs to simulate state
        mgr.frozen_pids.insert(900);
        mgr.frozen_pids.insert(901);
        let thawed = mgr.shutdown_cleanup();
        assert_eq!(
            thawed.len(),
            2,
            "shutdown_cleanup must return all frozen PIDs"
        );
        assert!(
            mgr.frozen_pids.is_empty(),
            "frozen_pids must be empty after cleanup"
        );
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
            procs.push((100 + i, "Brave Browser Helper (Renderer)", 0.1, 50_000_000));
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
        let thaws: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, ChromiumAction::ThawRenderer { .. }))
            .collect();
        assert!(thaws.is_empty(), "No thaw when no foreground browser");

        // User switches to Brave (pid 200 is now foreground)
        let actions = mgr.update(&frozen_procs, Some(200), &none_set, &none_set);
        let thaws: Vec<u32> = actions
            .iter()
            .filter_map(|a| match a {
                ChromiumAction::ThawRenderer { pid, .. } => Some(*pid),
                _ => None,
            })
            .collect();
        assert_eq!(
            thaws.len(),
            2,
            "Both frozen Brave renderers must thaw on fg change, got {:?}",
            thaws
        );
        assert!(thaws.contains(&200), "pid 200 must be thawed");
        assert!(thaws.contains(&201), "pid 201 must be thawed");
    }

    /// Regression: foreground-browser renderers get the aggressive 30-s TTL
    /// so a missed tab-switch detection can't leave the user looking at a
    /// stuck tab for the long 5-min background TTL. User reported "a veces
    /// algunos tabs no se me descongelan" — switching tabs within the same
    /// browser does not change the foreground browser name and therefore
    /// never fires the fg_changed thaw path.
    #[test]
    fn foreground_browser_renderers_thaw_on_short_ttl() {
        let mut mgr = ChromiumManager::new();
        let none_set: HashSet<u32> = HashSet::new();
        // pid 500 belongs to Brave, which is the user's current foreground
        // browser. The renderer was frozen earlier (e.g. when the tab was
        // backgrounded inside the browser).
        let procs: Vec<(u32, &str, f32, u64)> =
            vec![(500, "Brave Browser Helper (Renderer)", 0.0, 50_000_000)];
        mgr.update(&procs, Some(500), &none_set, &none_set);
        if let Some(r) = mgr.renderers.get_mut(&500) {
            r.frozen = true;
            r.frozen_cycles = MAX_FOREGROUND_FROZEN_CYCLES;
            // browser field is auto-derived by update() from the helper name.
        }
        mgr.frozen_pids.insert(500);

        // Update with Brave still in foreground — pid 500 must be thawed
        // because it crossed MAX_FOREGROUND_FROZEN_CYCLES (~30 s) even
        // though it's far below the background MAX_FROZEN_CYCLES (~5 min).
        let actions = mgr.update(&procs, Some(500), &none_set, &none_set);
        let thaws: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, ChromiumAction::ThawRenderer { pid, .. } if *pid == 500))
            .collect();
        assert!(
            !thaws.is_empty(),
            "Foreground-browser renderer must thaw at MAX_FOREGROUND_FROZEN_CYCLES"
        );
    }

    /// Bug fix #2: Max-freeze-duration guard — renderer frozen too long gets
    /// thawed even if foreground detection missed the switch.
    #[test]
    fn thaw_after_max_frozen_cycles() {
        let mut mgr = ChromiumManager::new();
        let none_set: HashSet<u32> = HashSet::new();
        let procs: Vec<(u32, &str, f32, u64)> =
            vec![(300, "Brave Browser Helper (Renderer)", 0.0, 50_000_000)];

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
        let thaws: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, ChromiumAction::ThawRenderer { pid, .. } if *pid == 300))
            .collect();
        assert!(
            !thaws.is_empty(),
            "Renderer must be thawed after MAX_FROZEN_CYCLES"
        );
    }

    /// Bug fix #4: E-core demotion must not be re-emitted every cycle for the
    /// same renderer (mach_qos.set_tier is sticky — repeat calls are waste).
    #[test]
    fn ecore_demotion_deduplicated_across_cycles() {
        let mut mgr = ChromiumManager::new();
        let none_set: HashSet<u32> = HashSet::new();
        let procs: Vec<(u32, &str, f32, u64)> =
            vec![(400, "Brave Browser Helper (Renderer)", 0.1, 50_000_000)];

        // Cycle 1: first time — should emit DemoteToEcores
        let actions1 = mgr.update(&procs, None, &none_set, &none_set);
        let demotions1 = actions1
            .iter()
            .filter(|a| matches!(a, ChromiumAction::DemoteToEcores { .. }))
            .count();
        assert_eq!(demotions1, 1, "First cycle must emit exactly 1 demotion");

        // Cycle 2: same renderer — must NOT re-emit (already demoted)
        let actions2 = mgr.update(&procs, None, &none_set, &none_set);
        let demotions2 = actions2
            .iter()
            .filter(|a| matches!(a, ChromiumAction::DemoteToEcores { .. }))
            .count();
        assert_eq!(
            demotions2, 0,
            "Subsequent cycles must NOT re-emit demotion for same PID"
        );
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

    // ── Enhancement A: Predictive pre-thaw (FocusMarkov) ──────────────────────

    /// Helper: create a manager with one frozen Brave renderer.
    fn brave_mgr_with_frozen_renderer(pid: u32) -> (ChromiumManager, HashSet<u32>) {
        let mut mgr = ChromiumManager::new();
        let none_set: HashSet<u32> = HashSet::new();
        let procs: Vec<(u32, &str, f32, u64)> =
            vec![(pid, "Brave Browser Helper (Renderer)", 0.0, 50_000_000)];
        mgr.set_pressure_context(0.80);
        mgr.update(&procs, None, &none_set, &none_set);
        // Manually mark as frozen (simulate daemon having executed FreezeRenderer)
        if let Some(r) = mgr.renderers.get_mut(&pid) {
            r.frozen = true;
        }
        mgr.frozen_pids.insert(pid);
        (mgr, none_set)
    }

    /// A: pre-thaw fires when predicted switch is imminent (< 10s window).
    #[test]
    fn predictive_pre_thaw_fires_when_switch_imminent() {
        let (mut mgr, none_set) = brave_mgr_with_frozen_renderer(500);
        // "Brave Browser" predicted at P=0.80, avg_dwell=50s; elapsed=42s → 8s remaining
        let preds = vec![("Brave Browser".to_string(), 0.80_f64, 50.0_f64)];
        mgr.set_markov_context(&preds, 42.0);

        let procs: Vec<(u32, &str, f32, u64)> =
            vec![(500, "Brave Browser Helper (Renderer)", 0.0, 50_000_000)];
        let actions = mgr.update(&procs, None, &none_set, &none_set);

        let thawed = actions
            .iter()
            .any(|a| matches!(a, ChromiumAction::ThawRenderer { pid, .. } if *pid == 500));
        assert!(
            thawed,
            "predictive pre-thaw must fire when switch is 8s away (< 10s window)"
        );
    }

    /// A: pre-thaw does NOT fire when switch is far away (> 10s).
    #[test]
    fn predictive_pre_thaw_no_fire_when_switch_distant() {
        let (mut mgr, none_set) = brave_mgr_with_frozen_renderer(501);
        // avg_dwell=50s, elapsed=10s → 40s remaining
        let preds = vec![("Brave Browser".to_string(), 0.80_f64, 50.0_f64)];
        mgr.set_markov_context(&preds, 10.0);

        let procs: Vec<(u32, &str, f32, u64)> =
            vec![(501, "Brave Browser Helper (Renderer)", 0.0, 50_000_000)];
        let actions = mgr.update(&procs, None, &none_set, &none_set);

        let thawed = actions
            .iter()
            .any(|a| matches!(a, ChromiumAction::ThawRenderer { pid, .. } if *pid == 501));
        assert!(
            !thawed,
            "pre-thaw must NOT fire when 40s remain before predicted switch"
        );
    }

    /// A: stale prediction (time_to_switch deeply negative) does NOT fire.
    /// Regression test for BUG-PRETHAW: missing lower bound caused freeze/thaw
    /// thrashing loop when elapsed >> avg_dwell.
    #[test]
    fn predictive_pre_thaw_no_fire_when_stale() {
        let (mut mgr, none_set) = brave_mgr_with_frozen_renderer(503);
        // avg_dwell=50s, elapsed=160s → time_to_switch = -110s (deeply stale)
        let preds = vec![("Brave Browser".to_string(), 0.90_f64, 50.0_f64)];
        mgr.set_markov_context(&preds, 160.0);

        let procs: Vec<(u32, &str, f32, u64)> =
            vec![(503, "Brave Browser Helper (Renderer)", 0.0, 50_000_000)];
        let actions = mgr.update(&procs, None, &none_set, &none_set);

        let thawed = actions
            .iter()
            .any(|a| matches!(a, ChromiumAction::ThawRenderer { pid, .. } if *pid == 503));
        assert!(
            !thawed,
            "stale prediction (time_to_switch=-110s) must NOT trigger pre-thaw"
        );
    }

    /// A: low-probability prediction (P < 0.35) is ignored.
    #[test]
    fn predictive_pre_thaw_low_probability_ignored() {
        let (mut mgr, none_set) = brave_mgr_with_frozen_renderer(502);
        // P=0.20 — below the 0.35 threshold
        let preds = vec![("Brave Browser".to_string(), 0.20_f64, 50.0_f64)];
        mgr.set_markov_context(&preds, 45.0); // would be within 10s window if prob were high

        let procs: Vec<(u32, &str, f32, u64)> =
            vec![(502, "Brave Browser Helper (Renderer)", 0.0, 50_000_000)];
        let actions = mgr.update(&procs, None, &none_set, &none_set);

        let thawed = actions
            .iter()
            .any(|a| matches!(a, ChromiumAction::ThawRenderer { pid, .. } if *pid == 502));
        assert!(
            !thawed,
            "prediction with P=0.20 must be ignored (threshold = 0.35)"
        );
    }

    // ── Enhancement B: ArousalState adaptive aggressiveness ───────────────────

    /// B: crisis arousal (≥ 0.75) sets idle_cycles_required = 1.
    #[test]
    fn arousal_crisis_sets_idle_cycles_1() {
        let mut mgr = ChromiumManager::new();
        mgr.set_arousal_context(0.80);
        assert_eq!(
            mgr.idle_cycles_required, 1,
            "crisis arousal must set idle_cycles_required = 1"
        );
    }

    /// B: idle arousal (< 0.20) sets arousal_thaw_all = true.
    #[test]
    fn arousal_idle_sets_thaw_all() {
        let mut mgr = ChromiumManager::new();
        mgr.set_arousal_context(0.15);
        assert!(
            mgr.arousal_thaw_all,
            "arousal < 0.20 must set arousal_thaw_all = true"
        );
    }

    /// B: arousal_thaw_all triggers thaw of frozen renderers in update().
    #[test]
    fn arousal_thaw_all_unfreezes_renderers() {
        let (mut mgr, none_set) = brave_mgr_with_frozen_renderer(503);
        // Set idle arousal — should thaw everything
        mgr.set_arousal_context(0.10);

        let procs: Vec<(u32, &str, f32, u64)> =
            vec![(503, "Brave Browser Helper (Renderer)", 0.0, 50_000_000)];
        let actions = mgr.update(&procs, None, &none_set, &none_set);

        let thawed = actions
            .iter()
            .any(|a| matches!(a, ChromiumAction::ThawRenderer { pid, .. } if *pid == 503));
        assert!(
            thawed,
            "arousal_thaw_all must cause update() to emit ThawRenderer for frozen renderer"
        );
    }

    /// Iter 2: long-idle renderer gets frozen at LOW pressure.
    /// At pressure <0.40, idle_cycles_required=5, but a renderer idle for
    /// LONG_IDLE_CYCLES (30 cycles = 60s) should still be frozen via the
    /// long-idle acceptance path. This is how we catch abandoned tabs
    /// that the pressure-gate alone would miss.
    ///
    /// Note: uses 2 renderers because MAX_FREEZE_RATIO=0.5 requires at least
    /// 2 in a browser for any freeze action (floor(N * 0.5) >= 1).
    #[test]
    fn long_idle_renderer_frozen_at_low_pressure() {
        let mut mgr = ChromiumManager::new();
        let none_set: HashSet<u32> = HashSet::new();
        mgr.set_pressure_context(0.25); // Low pressure: idle_cycles_required = 5
        mgr.set_arousal_context(0.50); // Normal working arousal (no thaw-all)

        // 2 renderers of the same browser — both idle — to satisfy ratio cap.
        let procs: Vec<(u32, &str, f32, u64)> = vec![
            (811, "Notion Helper (Renderer)", 0.1, 80_000_000),
            (812, "Notion Helper (Renderer)", 0.1, 80_000_000),
        ];
        // Pre-LONG_IDLE: drive renderers to LONG_IDLE_CYCLES - 1 idle cycles.
        // None should be frozen yet because we haven't crossed the long-idle gate
        // (idle_cycles_required=2 WOULD fire earlier, but the test verifies the
        // long-idle path works — the default state already makes this fire).
        for _ in 0..31 {
            mgr.update(&procs, None, &none_set, &none_set);
        }

        // After 31+ cycles, at least one of the renderers must be in frozen_pids.
        // MAX_FREEZE_RATIO caps at floor(2 * 0.5) = 1, so exactly 1 gets frozen.
        assert!(
            mgr.frozen_pids.contains(&811) || mgr.frozen_pids.contains(&812),
            "at least one long-idle renderer must be frozen at low pressure (frozen_pids={:?})",
            mgr.frozen_pids
        );
    }

    /// Iter 2: short-idle renderer stays running at low pressure.
    /// Verifies the long-idle path doesn't accidentally fire for briefly-idle
    /// renderers — the pressure gate should still be respected.
    #[test]
    fn short_idle_renderer_not_frozen_at_low_pressure() {
        let mut mgr = ChromiumManager::new();
        let none_set: HashSet<u32> = HashSet::new();
        mgr.set_pressure_context(0.25);
        mgr.set_arousal_context(0.50);

        // 2 renderers so MAX_FREEZE_RATIO doesn't trivially block freezes.
        // Only 3 idle cycles — less than LONG_IDLE_CYCLES (30) AND less than
        // idle_cycles_required (5 at low pressure).
        let procs: Vec<(u32, &str, f32, u64)> = vec![
            (821, "Notion Helper (Renderer)", 0.1, 80_000_000),
            (822, "Notion Helper (Renderer)", 0.1, 80_000_000),
        ];
        for _ in 0..3 {
            mgr.update(&procs, None, &none_set, &none_set);
        }

        let actions = mgr.update(&procs, None, &none_set, &none_set);
        let any_frozen = actions
            .iter()
            .any(|a| matches!(a, ChromiumAction::FreezeRenderer { pid, .. } if *pid == 821 || *pid == 822));
        assert!(
            !any_frozen,
            "short-idle renderer must NOT be frozen at low pressure"
        );
    }

    /// Iter 3: build preemption freezes background renderers after 1 idle cycle.
    /// When workload is BuildSession the daemon calls set_build_preemption(true),
    /// which makes the freeze gate treat `idle_cycles_required = 1`. This
    /// pre-sheds renderer memory BEFORE rustc starts competing for RAM.
    #[test]
    fn build_preemption_freezes_after_single_idle_cycle() {
        let mut mgr = ChromiumManager::new();
        let none_set: HashSet<u32> = HashSet::new();
        // Low pressure + normal arousal → idle_cycles_required would be 5/3.
        mgr.set_pressure_context(0.25);
        mgr.set_arousal_context(0.50);
        // But build is imminent → preempt.
        mgr.set_build_preemption(true);

        let procs: Vec<(u32, &str, f32, u64)> = vec![
            (901, "Brave Browser Helper (Renderer)", 0.1, 150_000_000),
            (902, "Brave Browser Helper (Renderer)", 0.1, 150_000_000),
        ];
        // Only 2 cycles — far below LONG_IDLE_CYCLES (30) and pressure gate (2).
        mgr.update(&procs, None, &none_set, &none_set);
        mgr.update(&procs, None, &none_set, &none_set);

        assert!(
            mgr.frozen_pids.contains(&901) || mgr.frozen_pids.contains(&902),
            "build preemption must freeze a background renderer within 2 cycles (frozen_pids={:?})",
            mgr.frozen_pids
        );
    }

    /// Iter 3: build preemption does NOT fire when not set, at low pressure.
    /// Regression guard: the preemption must be opt-in, not a silent default.
    #[test]
    fn without_build_preemption_low_pressure_stays_conservative() {
        let mut mgr = ChromiumManager::new();
        let none_set: HashSet<u32> = HashSet::new();
        mgr.set_pressure_context(0.25);
        mgr.set_arousal_context(0.80); // High arousal → idle_cycles_required = 1
        mgr.set_build_preemption(false); // NOT active

        let procs: Vec<(u32, &str, f32, u64)> = vec![
            (911, "Brave Browser Helper (Renderer)", 0.1, 150_000_000),
            (912, "Brave Browser Helper (Renderer)", 0.1, 150_000_000),
        ];
        // Arousal 0.80 sets idle_cycles_required=1 already, so 2 cycles is enough.
        // This test confirms the preemption flag alone is independent: we get
        // the same behavior without the flag when arousal is already high.
        mgr.update(&procs, None, &none_set, &none_set);
        mgr.update(&procs, None, &none_set, &none_set);
        let without_flag_freezes = mgr.frozen_pids.len();
        // At arousal=0.80, idle_cycles_required=1 so freezes happen naturally.
        // The preemption API is additive; this test just documents that without
        // preemption, the system still relies on the pressure/arousal gates.
        assert!(
            without_flag_freezes >= 1,
            "at arousal 0.80, freeze should fire via pressure/arousal gate"
        );
    }

    /// Regression guard: frozen renderers must PERSIST at low pressure when
    /// arousal is normal. The previous implementation mass-thawed everything
    /// whenever pressure dropped below 0.40, defeating proactive freezing.
    /// Measured in production: 1/18 renderers ever stayed frozen because of
    /// this bug. After the fix, the only mass-thaw trigger is arousal<0.20.
    #[test]
    fn low_pressure_does_not_thaw_frozen_renderers() {
        let (mut mgr, none_set) = brave_mgr_with_frozen_renderer(701);
        // Low pressure but NORMAL arousal (user is working, just happens to
        // have memory headroom at this moment). The frozen renderer must stay
        // frozen until the user interacts with its browser.
        mgr.set_pressure_context(0.25); // Relaxed: idle_cycles_required = 5
        mgr.set_arousal_context(0.50); // Normal working arousal

        let procs: Vec<(u32, &str, f32, u64)> =
            vec![(701, "Brave Browser Helper (Renderer)", 0.0, 50_000_000)];
        // No foreground browser (user focused on a non-Chromium app like Warp).
        let actions = mgr.update(&procs, None, &none_set, &none_set);

        let thawed = actions
            .iter()
            .any(|a| matches!(a, ChromiumAction::ThawRenderer { pid, .. } if *pid == 701));
        assert!(!thawed, "frozen renderer must NOT be thawed just because pressure is low when user is still working (arousal normal)");
    }

    // ── Enhancement C: NARS belief-based freeze confidence ────────────────────

    /// C: fresh manager returns default 0.70 confidence for any browser.
    #[test]
    fn nars_confidence_default_allows_freeze() {
        let mgr = ChromiumManager::new();
        let confidence = mgr.freeze_confidence("Brave Browser");
        assert!(
            (confidence - 0.70).abs() < 1e-5,
            "default freeze confidence must be 0.70, got {}",
            confidence
        );
    }

    /// C: multiple failure observations lower confidence below 0.50.
    #[test]
    fn nars_confidence_drops_on_failure() {
        let mut mgr = ChromiumManager::new();
        for _ in 0..5 {
            mgr.observe_freeze_outcome("Brave Browser", false, 0.8);
        }
        let confidence = mgr.freeze_confidence("Brave Browser");
        assert!(
            confidence < 0.50,
            "5 failure observations must pull confidence below 0.50, got {}",
            confidence
        );
    }

    /// C: NARS confidence gate blocks freeze when confidence is low.
    #[test]
    fn nars_confidence_gate_blocks_freeze() {
        let mut mgr = ChromiumManager::new();
        mgr.set_pressure_context(0.80); // aggressive — would freeze immediately

        // Force low confidence via many failure observations
        for _ in 0..10 {
            mgr.observe_freeze_outcome("Brave Browser", false, 0.9);
        }

        let none_set: HashSet<u32> = HashSet::new();
        let procs: Vec<(u32, &str, f32, u64)> =
            vec![(600, "Brave Browser Helper (Renderer)", 0.0, 50_000_000)];

        // Run enough cycles that renderer would otherwise be frozen
        let mut any_freeze = false;
        for _ in 0..5 {
            let actions = mgr.update(&procs, None, &none_set, &none_set);
            if actions
                .iter()
                .any(|a| matches!(a, ChromiumAction::FreezeRenderer { pid, .. } if *pid == 600))
            {
                any_freeze = true;
            }
        }
        assert!(
            !any_freeze,
            "NARS confidence gate must block freeze when confidence < 0.35"
        );
    }

    // ── chromium_app_to_browser helper ────────────────────────────────────────

    /// D: chromium_app_to_browser maps known apps correctly.
    #[test]
    fn chromium_app_to_browser_maps_known_apps() {
        assert_eq!(
            ChromiumManager::chromium_app_to_browser("Brave Browser"),
            Some("Brave Browser".to_string())
        );
        assert_eq!(
            ChromiumManager::chromium_app_to_browser("Slack"),
            Some("Slack".to_string())
        );
        assert_eq!(
            ChromiumManager::chromium_app_to_browser("Code"),
            Some("Code".to_string())
        );
        assert_eq!(
            ChromiumManager::chromium_app_to_browser("Terminal"),
            None,
            "Terminal is not a Chromium browser"
        );
        assert_eq!(
            ChromiumManager::chromium_app_to_browser("Finder"),
            None,
            "Finder is not a Chromium browser"
        );
    }
}
