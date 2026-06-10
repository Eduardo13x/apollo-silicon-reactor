//! Window and app lifecycle event sensor for Apollo daemon.
//!
//! Tracks process-list deltas between cycles to infer app lifecycle events AND
//! classify the user's current *session phase* and *workload intent* — making
//! Apollo predictive rather than purely reactive.
//!
//! # Signals produced
//!
//! | Signal | What it means |
//! |--------|---------------|
//! | `tab_delta` | Tabs opened (+) or closed (−) this cycle |
//! | `renderer_count` | Current open tab count proxy |
//! | `tab_velocity_ema` | Smoothed trend: positive = ramping, negative = winding down |
//! | `session_phase` | ColdStart / Ramping / Settled / WindingDown / Idle |
//! | `workload_intent` | Build / Research / AI / Media / General |
//! | `pressure_floor_correction` | How much to add to "expected normal" pressure |
//! | `freed_heavy_app` | Heavy app terminated this cycle |
//!
//! # Papers
//!
//! [Pirolli & Card 1999] "Information Foraging" Psych. Review — users alternate
//! between high-velocity "foraging" (opening tabs, switching contexts) and low-
//! velocity "exploitation" (settled deep work). SessionPhase maps these states.
//!
//! [Denning 1968] "The Working Set Model for Program Behavior" CACM — open tab
//! count is a proxy for the browser's active working set. More tabs → higher
//! expected resident memory → raise the pressure "floor" accordingly.
//!
//! [Altmann & Trafton 2002] "Memory for Goals" Cognitive Science — foreground
//! switch rate is a proxy for multi-task load. High switch rate = high cognitive
//! load = user needs low-latency foreground response.
//!
//! [GoF 1994] Observer Pattern — cycle-to-cycle diff avoids push notification
//! callbacks, which require a GUI session not available in root daemons.
//!
//! [Mogul & Heidemann 1997] USENIX — polling at bounded frequency is preferable
//! to interrupt-driven callbacks under burst conditions (tab close storms).

use std::collections::HashMap;

// ── Constants ─────────────────────────────────────────────────────────────────

const RENDERER_PATTERNS: &[&str] = &["Helper (Renderer)", "Helper (GPU)", "Helper (Plugin)"];

const HEAVY_APP_PATTERNS: &[&str] = &[
    "Brave Browser",
    "Google Chrome",
    "Firefox",
    "Safari",
    "Electron",
    "Slack",
    "Discord",
    "Notion",
    "Zoom",
    "Teams",
    "ollama",
    "ollama_llama_server",
];

/// Substrings identifying active compilation / build processes.
const BUILD_PATTERNS: &[&str] = &[
    // Keep only unambiguous patterns that won't substring-match system daemons.
    // Removed: "cc" (matches accessd, accountsd, etc.), "c++" (too rare as process name),
    //          "stable" (rust toolchain wrapper — rarely a standalone process name on macOS).
    "cargo", "rustc", "clang", "clang++", "make", "ninja", "gradle", "swift", "javac",
];

/// Substrings identifying ML inference / AI workloads.
const AI_PATTERNS: &[&str] = &[
    "ollama",
    "ollama_llama_server",
    "python",
    "python3",
    "TGOnDeviceInferenceProviderService",
];

/// Substrings identifying media playback processes.
const MEDIA_PATTERNS: &[&str] = &[
    "Spotify",
    "VLC",
    "mpv",
    "Stremio",
    "QuickTime",
    "IINA",
    "Music",
    "Podcasts",
];

/// EMA alpha for tab_velocity smoothing. α=0.3 → ~3-cycle memory.
const TAB_VELOCITY_ALPHA: f64 = 0.3;

/// renderer_count above which ResearchSession is detected (no build tools).
const RESEARCH_TAB_THRESHOLD: u32 = 8;

/// tab_velocity_ema threshold to enter Ramping phase.
const RAMP_THRESHOLD: f64 = 0.35;

/// tab_velocity_ema threshold to enter WindingDown phase.
const WIND_DOWN_THRESHOLD: f64 = -0.35;

/// Pressure per renderer process on 8 GB M1 (empirical: ~100 MB / 8192 MB ≈ 0.012).
const PRESSURE_PER_RENDERER: f64 = 0.012;

/// Max pressure floor correction (cap at 25% of range).
const MAX_FLOOR_CORRECTION: f64 = 0.25;

/// Cycles of trend data needed before exiting ColdStart.
const COLD_START_CYCLES: u32 = 4;

// ── Types ─────────────────────────────────────────────────────────────────────

/// The phase of the user's current work session, inferred from tab velocity
/// and renderer count. Maps to Pirolli & Card's foraging/exploitation cycle.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SessionPhase {
    /// No browser tabs open — user is idle or in a non-browser app.
    Idle,
    /// First N cycles after init or browser launch — insufficient trend data.
    #[default]
    ColdStart,
    /// Tab velocity trending positive — session intensifying, pressure incoming.
    /// Apollo should be more proactive (pre-position resources).
    Ramping,
    /// Stable tab count — user in deep work mode.
    /// Normal reactive behavior appropriate.
    Settled,
    /// Tab velocity trending negative — session winding down, pressure will drop.
    /// Apollo can relax thresholds (see also: window_relief_cycles in daemon).
    WindingDown,
}

/// The type of workload the user is currently running, inferred from process
/// names. Drives which processes Apollo should protect vs. relax.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum WorkloadIntent {
    /// No strong workload signal detected.
    #[default]
    General,
    /// Active compilation: cargo/rustc/clang/make detected.
    /// Apollo should protect build processes, relax browser renderers.
    BuildSession,
    /// Many browser tabs open, no build tools — user is researching.
    /// Apollo should protect browser renderers, be generous with tab RAM.
    ResearchSession,
    /// Ollama or Python inference running — protect AI process, relax rest.
    AISession,
    /// Media playback detected — low-latency response needed for UI thread.
    MediaSession,
}

/// Kind of app lifecycle event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppEventKind {
    Terminated,
    Launched,
    ForegroundGained,
}

/// A single app lifecycle event.
#[derive(Debug, Clone)]
pub struct AppEvent {
    pub pid: u32,
    pub name: String,
    pub kind: AppEventKind,
}

/// Full summary of app/session state for one daemon cycle.
#[derive(Debug, Clone)]
pub struct WindowDelta {
    /// All detected lifecycle events this cycle.
    pub events: Vec<AppEvent>,

    // ── Tab signals ──────────────────────────────────────────────────────────
    /// Net change in renderer processes this cycle (tabs opened/closed).
    pub tab_delta: i32,
    /// Current renderer count (open tab proxy).
    pub renderer_count: u32,
    /// EMA-smoothed tab velocity. Positive = opening trend, negative = closing.
    /// Range: typically ±3.0. Used to classify SessionPhase.
    pub tab_velocity_ema: f64,

    // ── Session intelligence ──────────────────────────────────────────────────
    /// Current session phase inferred from tab velocity trend.
    pub session_phase: SessionPhase,
    /// Current workload type inferred from process signatures.
    pub workload_intent: WorkloadIntent,
    /// How much to add to "normal expected" pressure for this session.
    /// Based on renderer count: 13 tabs → +0.156. Apollo uses this to avoid
    /// treating "busy browser session pressure" as an emergency.
    pub pressure_floor_correction: f64,

    // ── Heavy app events ──────────────────────────────────────────────────────
    /// True if a heavy (≥100 MB) app terminated this cycle.
    pub freed_heavy_app: bool,
    /// Conservative estimate of MB freed (200 MB per heavy app terminated).
    pub estimated_freed_mb: u32,

    // ── Foreground ───────────────────────────────────────────────────────────
    /// Name of current foreground app.
    pub foreground_name: String,
    /// True if the foreground app changed since last cycle.
    pub foreground_changed: bool,
    /// Foreground switch rate EMA (0..1). High = multi-task load.
    pub foreground_switch_ema: f64,
}

impl Default for WindowDelta {
    fn default() -> Self {
        Self {
            events: Vec::new(),
            tab_delta: 0,
            renderer_count: 0,
            tab_velocity_ema: 0.0,
            session_phase: SessionPhase::ColdStart,
            workload_intent: WorkloadIntent::General,
            pressure_floor_correction: 0.0,
            freed_heavy_app: false,
            estimated_freed_mb: 0,
            foreground_name: String::new(),
            foreground_changed: false,
            foreground_switch_ema: 0.0,
        }
    }
}

// ── Sensor ────────────────────────────────────────────────────────────────────

/// Tracks process-list deltas and infers session context each daemon cycle.
pub struct WindowSensor {
    prev_procs: HashMap<u32, String>,
    prev_renderer_count: u32,
    prev_foreground: String,
    initialized: bool,
    cycles_since_init: u32,
    /// EMA of tab_delta — the core "session velocity" signal.
    tab_velocity_ema: f64,
    /// EMA of foreground change (1.0 = changed, 0.0 = stable). Alpha=0.2.
    foreground_switch_ema: f64,
}

impl WindowSensor {
    pub fn new() -> Self {
        Self {
            prev_procs: HashMap::new(),
            prev_renderer_count: 0,
            prev_foreground: String::new(),
            initialized: false,
            cycles_since_init: 0,
            tab_velocity_ema: 0.0,
            foreground_switch_ema: 0.0,
        }
    }

    /// Run one sensor tick. Call once per daemon cycle after process collection.
    ///
    /// `procs` — all (pid, name) pairs from the current process table.
    /// `foreground` — name of the currently focused app (from ForegroundDetector).
    pub fn tick(&mut self, procs: &[(u32, &str)], foreground: Option<&str>) -> WindowDelta {
        let mut delta = WindowDelta::default();

        // ── Foreground ────────────────────────────────────────────────────────
        let fg_name = foreground.unwrap_or("").to_string();
        delta.foreground_changed = fg_name != self.prev_foreground;
        delta.foreground_name = fg_name.clone();

        // Foreground switch EMA: α=0.2 (5-cycle memory [Altmann & Trafton 2002])
        let fg_signal = if delta.foreground_changed { 1.0 } else { 0.0 };
        self.foreground_switch_ema = 0.2 * fg_signal + 0.8 * self.foreground_switch_ema;
        delta.foreground_switch_ema = self.foreground_switch_ema;

        self.prev_foreground = fg_name.clone();

        // ── Build current process map ─────────────────────────────────────────
        let mut current: HashMap<u32, String> = HashMap::with_capacity(procs.len());
        let mut renderer_count: u32 = 0;

        for &(pid, name) in procs {
            current.insert(pid, name.to_string());
            if is_renderer(name) {
                renderer_count += 1;
            }
        }

        delta.renderer_count = renderer_count;

        // ── Pressure floor correction [Denning 1968] ──────────────────────────
        // Each renderer ≈ 100 MB / 8192 MB = 1.2% of RAM pressure.
        // Cap at 25% to avoid over-compensating.
        delta.pressure_floor_correction =
            (renderer_count as f64 * PRESSURE_PER_RENDERER).min(MAX_FLOOR_CORRECTION);

        // ── Workload intent (process signature matching) ───────────────────────
        delta.workload_intent = classify_workload(procs, renderer_count);

        // ── First tick initialization ─────────────────────────────────────────
        if !self.initialized {
            self.prev_procs = current;
            self.prev_renderer_count = renderer_count;
            self.initialized = true;
            // Still return ColdStart on first tick (no trend data)
            return delta;
        }

        self.cycles_since_init += 1;

        // ── Detect terminations ───────────────────────────────────────────────
        for (pid, name) in &self.prev_procs {
            if !current.contains_key(pid) {
                if is_heavy_app(name) {
                    delta.freed_heavy_app = true;
                    delta.estimated_freed_mb += 200;
                }
                if !is_renderer(name) {
                    delta.events.push(AppEvent {
                        pid: *pid,
                        name: name.clone(),
                        kind: AppEventKind::Terminated,
                    });
                }
            }
        }

        // ── Detect launches ───────────────────────────────────────────────────
        for (pid, name) in &current {
            if !self.prev_procs.contains_key(pid) && !is_renderer(name) {
                delta.events.push(AppEvent {
                    pid: *pid,
                    name: name.clone(),
                    kind: AppEventKind::Launched,
                });
            }
        }

        // ── Tab delta + velocity EMA [Pirolli & Card 1999] ───────────────────
        delta.tab_delta = renderer_count as i32 - self.prev_renderer_count as i32;
        // EMA smooths burst open/close into a trend signal.
        // α=0.3: new observation weighted 30%, history 70%.
        self.tab_velocity_ema = TAB_VELOCITY_ALPHA * delta.tab_delta as f64
            + (1.0 - TAB_VELOCITY_ALPHA) * self.tab_velocity_ema;
        delta.tab_velocity_ema = self.tab_velocity_ema;

        // ── Session phase classification ──────────────────────────────────────
        delta.session_phase = self.classify_phase(renderer_count);

        // ── Foreground gained event ───────────────────────────────────────────
        if delta.foreground_changed && !fg_name.is_empty() {
            if let Some((&pid, _)) = current.iter().find(|(_, n)| n.as_str() == fg_name) {
                delta.events.push(AppEvent {
                    pid,
                    name: fg_name,
                    kind: AppEventKind::ForegroundGained,
                });
            }
        }

        // ── Update state ──────────────────────────────────────────────────────
        self.prev_procs = current;
        self.prev_renderer_count = renderer_count;

        delta
    }

    /// Classify the current session phase from velocity EMA + renderer count.
    fn classify_phase(&self, renderer_count: u32) -> SessionPhase {
        if renderer_count == 0 {
            return SessionPhase::Idle;
        }
        if self.cycles_since_init < COLD_START_CYCLES {
            return SessionPhase::ColdStart;
        }
        if self.tab_velocity_ema >= RAMP_THRESHOLD {
            SessionPhase::Ramping
        } else if self.tab_velocity_ema <= WIND_DOWN_THRESHOLD {
            SessionPhase::WindingDown
        } else {
            SessionPhase::Settled
        }
    }

    /// Remove stale PIDs. Call periodically (every 100 cycles).
    pub fn gc(&mut self, alive_pids: &std::collections::HashSet<u32>) {
        self.prev_procs.retain(|pid, _| alive_pids.contains(pid));
    }
}

impl Default for WindowSensor {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// OnceLock Aho-Corasick matchers — built once at first use, scanned in single
// pass per name instead of N substring iter loops. Called per-PID per-cycle via
// classify_workload + is_renderer hot paths. [Sprint 2026-06-03]
fn renderer_ac() -> &'static aho_corasick::AhoCorasick {
    static AC: std::sync::OnceLock<aho_corasick::AhoCorasick> = std::sync::OnceLock::new();
    AC.get_or_init(|| aho_corasick::AhoCorasick::new(RENDERER_PATTERNS).expect("renderer patterns"))
}
fn heavy_app_ac() -> &'static aho_corasick::AhoCorasick {
    static AC: std::sync::OnceLock<aho_corasick::AhoCorasick> = std::sync::OnceLock::new();
    AC.get_or_init(|| {
        aho_corasick::AhoCorasick::new(HEAVY_APP_PATTERNS).expect("heavy app patterns")
    })
}
fn build_ac() -> &'static aho_corasick::AhoCorasick {
    static AC: std::sync::OnceLock<aho_corasick::AhoCorasick> = std::sync::OnceLock::new();
    AC.get_or_init(|| aho_corasick::AhoCorasick::new(BUILD_PATTERNS).expect("build patterns"))
}
fn ai_ac() -> &'static aho_corasick::AhoCorasick {
    static AC: std::sync::OnceLock<aho_corasick::AhoCorasick> = std::sync::OnceLock::new();
    AC.get_or_init(|| aho_corasick::AhoCorasick::new(AI_PATTERNS).expect("ai patterns"))
}
fn media_ac() -> &'static aho_corasick::AhoCorasick {
    static AC: std::sync::OnceLock<aho_corasick::AhoCorasick> = std::sync::OnceLock::new();
    AC.get_or_init(|| aho_corasick::AhoCorasick::new(MEDIA_PATTERNS).expect("media patterns"))
}

fn is_renderer(name: &str) -> bool {
    renderer_ac().is_match(name)
}

fn is_heavy_app(name: &str) -> bool {
    heavy_app_ac().is_match(name)
}

/// Classify workload intent from full process list + renderer count.
fn classify_workload(procs: &[(u32, &str)], renderer_count: u32) -> WorkloadIntent {
    let mut has_build = false;
    let mut has_ai = false;
    let mut has_media = false;

    let (b_ac, a_ac, m_ac) = (build_ac(), ai_ac(), media_ac());
    for &(_, name) in procs {
        if !has_build && b_ac.is_match(name) {
            has_build = true;
        }
        if !has_ai && a_ac.is_match(name) {
            has_ai = true;
        }
        if !has_media && m_ac.is_match(name) {
            has_media = true;
        }
        if has_build && has_ai && has_media {
            break;
        }
    }

    // Priority: Build > AI > Media > Research > General
    if has_build {
        WorkloadIntent::BuildSession
    } else if has_ai {
        WorkloadIntent::AISession
    } else if has_media {
        WorkloadIntent::MediaSession
    } else if renderer_count >= RESEARCH_TAB_THRESHOLD {
        WorkloadIntent::ResearchSession
    } else {
        WorkloadIntent::General
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_procs<'a>(list: &'a [(u32, &'a str)]) -> Vec<(u32, &'a str)> {
        list.to_vec()
    }

    // ── Basic lifecycle ───────────────────────────────────────────────────────

    #[test]
    fn first_tick_no_events() {
        let mut s = WindowSensor::new();
        let procs = make_procs(&[(1, "launchd"), (100, "Brave Browser")]);
        let d = s.tick(&procs, None);
        assert!(d.events.is_empty());
        assert_eq!(d.tab_delta, 0);
        assert!(!d.freed_heavy_app);
    }

    #[test]
    fn detects_app_terminated() {
        let mut s = WindowSensor::new();
        s.tick(&make_procs(&[(1, "launchd"), (200, "Slack")]), None);
        let d = s.tick(&make_procs(&[(1, "launchd")]), None);
        let term: Vec<_> = d
            .events
            .iter()
            .filter(|e| e.kind == AppEventKind::Terminated)
            .collect();
        assert_eq!(term.len(), 1);
        assert_eq!(term[0].name, "Slack");
    }

    #[test]
    fn heavy_app_terminated_sets_flag() {
        let mut s = WindowSensor::new();
        s.tick(&make_procs(&[(1, "launchd"), (300, "Brave Browser")]), None);
        let d = s.tick(&make_procs(&[(1, "launchd")]), None);
        assert!(d.freed_heavy_app);
        assert_eq!(d.estimated_freed_mb, 200);
    }

    #[test]
    fn tab_delta_negative_on_tab_close() {
        let mut s = WindowSensor::new();
        s.tick(
            &make_procs(&[
                (1, "Brave Browser"),
                (101, "Brave Browser Helper (Renderer)"),
                (102, "Brave Browser Helper (Renderer)"),
                (103, "Brave Browser Helper (Renderer)"),
            ]),
            None,
        );
        let d = s.tick(
            &make_procs(&[
                (1, "Brave Browser"),
                (101, "Brave Browser Helper (Renderer)"),
                (102, "Brave Browser Helper (Renderer)"),
            ]),
            None,
        );
        assert_eq!(d.tab_delta, -1);
        assert_eq!(d.renderer_count, 2);
    }

    #[test]
    fn tab_delta_positive_on_tab_open() {
        let mut s = WindowSensor::new();
        s.tick(
            &make_procs(&[
                (1, "Brave Browser"),
                (101, "Brave Browser Helper (Renderer)"),
            ]),
            None,
        );
        let d = s.tick(
            &make_procs(&[
                (1, "Brave Browser"),
                (101, "Brave Browser Helper (Renderer)"),
                (102, "Brave Browser Helper (Renderer)"),
                (103, "Brave Browser Helper (Renderer)"),
            ]),
            None,
        );
        assert_eq!(d.tab_delta, 2);
    }

    #[test]
    fn renderer_terminations_not_emitted_as_events() {
        let mut s = WindowSensor::new();
        s.tick(
            &make_procs(&[
                (1, "Brave Browser"),
                (200, "Brave Browser Helper (Renderer)"),
                (201, "Brave Browser Helper (Renderer)"),
            ]),
            None,
        );
        let d = s.tick(&make_procs(&[(1, "Brave Browser")]), None);
        assert_eq!(d.tab_delta, -2);
        let r: Vec<_> = d
            .events
            .iter()
            .filter(|e| e.kind == AppEventKind::Terminated && e.name.contains("Renderer"))
            .collect();
        assert!(r.is_empty());
    }

    #[test]
    fn foreground_changed_detected() {
        let mut s = WindowSensor::new();
        let procs = make_procs(&[(1, "Brave Browser"), (2, "Warp")]);
        s.tick(&procs, Some("Brave Browser"));
        let d = s.tick(&procs, Some("Warp"));
        assert!(d.foreground_changed);
        assert_eq!(d.foreground_name, "Warp");
    }

    #[test]
    fn foreground_unchanged_not_flagged() {
        let mut s = WindowSensor::new();
        let procs = make_procs(&[(1, "Brave Browser")]);
        s.tick(&procs, Some("Brave Browser"));
        let d = s.tick(&procs, Some("Brave Browser"));
        assert!(!d.foreground_changed);
    }

    #[test]
    fn detects_new_app_launched() {
        let mut s = WindowSensor::new();
        s.tick(&make_procs(&[(1, "launchd")]), None);
        let d = s.tick(&make_procs(&[(1, "launchd"), (500, "Notion")]), None);
        let launched: Vec<_> = d
            .events
            .iter()
            .filter(|e| e.kind == AppEventKind::Launched)
            .collect();
        assert_eq!(launched.len(), 1);
        assert_eq!(launched[0].name, "Notion");
    }

    #[test]
    fn gpu_helper_counted_as_renderer() {
        let mut s = WindowSensor::new();
        s.tick(
            &make_procs(&[(1, "Brave Browser"), (10, "Brave Browser Helper (GPU)")]),
            None,
        );
        let d = s.tick(&make_procs(&[(1, "Brave Browser")]), None);
        assert_eq!(d.tab_delta, -1);
    }

    #[test]
    fn gc_removes_stale_pids() {
        let mut s = WindowSensor::new();
        s.tick(&make_procs(&[(1, "launchd"), (999, "Zombie")]), None);
        let alive: std::collections::HashSet<u32> = [1u32].into_iter().collect();
        s.gc(&alive);
        assert!(!s.prev_procs.contains_key(&999));
        assert!(s.prev_procs.contains_key(&1));
    }

    // ── Pressure floor correction [Denning 1968] ──────────────────────────────

    #[test]
    fn pressure_floor_correction_scales_with_renderers() {
        let mut s = WindowSensor::new();
        let procs: Vec<(u32, &str)> = (0..13u32)
            .map(|i| (i + 100, "Brave Browser Helper (Renderer)"))
            .collect();
        let d = s.tick(&procs, None);
        // 13 renderers × 0.012 = 0.156
        assert!((d.pressure_floor_correction - 0.156).abs() < 0.001);
    }

    #[test]
    fn pressure_floor_correction_capped_at_025() {
        let mut s = WindowSensor::new();
        // 30 renderers × 0.012 = 0.36 → capped at 0.25
        let procs: Vec<(u32, &str)> = (0..30u32)
            .map(|i| (i + 100, "Brave Browser Helper (Renderer)"))
            .collect();
        let d = s.tick(&procs, None);
        assert!((d.pressure_floor_correction - 0.25).abs() < 0.001);
    }

    // ── WorkloadIntent ────────────────────────────────────────────────────────

    #[test]
    fn workload_build_session_detected() {
        let procs = make_procs(&[(1, "cargo"), (2, "rustc"), (3, "Brave Browser")]);
        assert_eq!(classify_workload(&procs, 0), WorkloadIntent::BuildSession);
    }

    #[test]
    fn workload_ai_session_detected() {
        let procs = make_procs(&[(1, "ollama"), (2, "Brave Browser")]);
        assert_eq!(classify_workload(&procs, 0), WorkloadIntent::AISession);
    }

    #[test]
    fn workload_research_session_detected() {
        let procs = make_procs(&[(1, "Brave Browser"), (2, "launchd")]);
        // 10 renderers, no build/ai/media
        assert_eq!(
            classify_workload(&procs, 10),
            WorkloadIntent::ResearchSession
        );
    }

    #[test]
    fn workload_media_session_detected() {
        let procs = make_procs(&[(1, "Spotify"), (2, "launchd")]);
        assert_eq!(classify_workload(&procs, 0), WorkloadIntent::MediaSession);
    }

    #[test]
    fn workload_build_wins_over_ai() {
        // If both cargo and ollama are running, Build wins (higher priority)
        let procs = make_procs(&[(1, "cargo"), (2, "ollama")]);
        assert_eq!(classify_workload(&procs, 0), WorkloadIntent::BuildSession);
    }

    #[test]
    fn workload_general_when_no_signal() {
        let procs = make_procs(&[(1, "launchd"), (2, "Warp")]);
        assert_eq!(classify_workload(&procs, 3), WorkloadIntent::General);
    }

    // ── SessionPhase [Pirolli & Card 1999] ───────────────────────────────────

    #[test]
    fn session_phase_idle_when_no_renderers() {
        let mut s = WindowSensor::new();
        s.tick(&make_procs(&[(1, "launchd")]), None);
        // Advance past cold start
        for _ in 0..5 {
            s.tick(&make_procs(&[(1, "launchd")]), None);
        }
        let d = s.tick(&make_procs(&[(1, "launchd")]), None);
        assert_eq!(d.session_phase, SessionPhase::Idle);
    }

    #[test]
    fn session_phase_cold_start_initially() {
        let mut s = WindowSensor::new();
        let procs = make_procs(&[
            (1, "Brave Browser"),
            (10, "Brave Browser Helper (Renderer)"),
        ]);
        s.tick(&procs, None); // init
        let d = s.tick(&procs, None); // cycle 1
        assert_eq!(d.session_phase, SessionPhase::ColdStart);
    }

    #[test]
    fn session_phase_ramping_on_positive_velocity() {
        let mut s = WindowSensor::new();
        // Build up positive velocity: open 3 tabs per cycle
        let mut procs: Vec<(u32, &str)> = vec![(1, "Brave Browser")];
        s.tick(&procs, None);
        // Open tabs rapidly across multiple cycles to build up velocity EMA
        for i in 0..10u32 {
            procs.push((100 + i, "Brave Browser Helper (Renderer)"));
            let d = s.tick(&procs, None);
            if d.session_phase == SessionPhase::Ramping {
                return; // Test passed
            }
        }
        panic!("Never reached Ramping phase");
    }

    #[test]
    fn session_phase_winding_down_on_negative_velocity() {
        let mut s = WindowSensor::new();
        // Start with many tabs
        let mut procs: Vec<(u32, &str)> = vec![(1, "Brave Browser")];
        for i in 0..15u32 {
            procs.push((100 + i, "Brave Browser Helper (Renderer)"));
        }
        s.tick(&procs, None);
        // Advance past cold start with stable state
        for _ in 0..5 {
            s.tick(&procs, None);
        }
        // Now close tabs rapidly
        for _ in 0..10 {
            procs.pop();
            let d = s.tick(&procs, None);
            if d.session_phase == SessionPhase::WindingDown {
                return; // Test passed
            }
        }
        panic!("Never reached WindingDown phase");
    }

    // ── tab_velocity_ema ─────────────────────────────────────────────────────

    #[test]
    fn tab_velocity_ema_decays_to_zero() {
        let mut s = WindowSensor::new();
        let procs = make_procs(&[
            (1, "Brave Browser"),
            (10, "Brave Browser Helper (Renderer)"),
        ]);
        s.tick(&procs, None);
        // Stable — no delta — velocity should decay toward 0
        for _ in 0..20 {
            s.tick(&procs, None);
        }
        let d = s.tick(&procs, None);
        assert!(
            d.tab_velocity_ema.abs() < 0.01,
            "velocity should decay near 0, got {}",
            d.tab_velocity_ema
        );
    }

    #[test]
    fn foreground_switch_ema_increases_on_switches() {
        let mut s = WindowSensor::new();
        let procs = make_procs(&[(1, "Brave Browser"), (2, "Warp"), (3, "Notion")]);
        s.tick(&procs, Some("Brave Browser"));
        let apps = ["Warp", "Notion", "Brave Browser", "Warp", "Notion"];
        let mut last_ema = 0.0;
        for app in &apps {
            let d = s.tick(&procs, Some(app));
            last_ema = d.foreground_switch_ema;
        }
        assert!(
            last_ema > 0.3,
            "switch EMA should be elevated, got {}",
            last_ema
        );
    }
}
