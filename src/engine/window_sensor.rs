//! Window and app lifecycle event sensor for Apollo daemon.
//!
//! Tracks process-list deltas between cycles to infer app lifecycle events:
//! - App terminated (process disappeared) → likely freed RAM
//! - App launched (process appeared) → incoming memory pressure
//! - Browser tab closed/opened — browser renderer processes are one-per-tab
//!   on Chromium-based browsers. Tracking "Helper (Renderer)" process count
//!   gives a tab-level signal without Accessibility API access.
//!
//! # Design
//!
//! Polling-based diff approach (vs. push NSWorkspace notifications) because:
//! - Root daemons have no guaranteed window server session
//! - sysinfo process list is already collected every cycle
//! - Same 500ms granularity as rest of daemon — no latency penalty
//!
//! # Paper
//!
//! [GoF 1994] "Design Patterns" — Observer Pattern.
//! Applied as a cycle-to-cycle diff instead of callback registration,
//! which avoids thread synchronization requirements in a root daemon.
//!
//! [Mogul & Heidemann 1997] USENIX — "Eliminating Receive Livelock in an
//! Interrupt-Driven Kernel": polling at bounded frequency is preferable to
//! interrupt-driven callbacks when the interrupt rate may exceed processing
//! capacity. Same reasoning applies here: app termination events cluster
//! (e.g., closing a browser with 20 tabs) and batch processing is safer.

use std::collections::HashMap;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Process name substrings that identify browser renderer child processes.
/// Each of these corresponds to an open tab or rendering context.
/// Counted to derive `tab_delta` between cycles.
const RENDERER_PATTERNS: &[&str] = &[
    "Helper (Renderer)",
    "Helper (GPU)",
    "Helper (Plugin)",
];

/// Process name substrings for high-memory apps whose termination likely
/// frees significant RAM (≥100 MB). Used to compute `freed_heavy_app`.
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

// ── Types ─────────────────────────────────────────────────────────────────────

/// Kind of app lifecycle event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppEventKind {
    /// Process terminated (disappeared from process table).
    Terminated,
    /// New process appeared.
    Launched,
    /// Process name matches foreground app this cycle (focus gained).
    ForegroundGained,
}

/// A single app lifecycle event detected during a sensor tick.
#[derive(Debug, Clone)]
pub struct AppEvent {
    pub pid: u32,
    pub name: String,
    pub kind: AppEventKind,
}

/// Summary of all app lifecycle changes detected in one daemon cycle.
#[derive(Debug, Clone, Default)]
pub struct WindowDelta {
    /// All detected events this cycle.
    pub events: Vec<AppEvent>,
    /// Net change in browser renderer processes (tabs).
    /// Negative = tabs were closed; Positive = tabs were opened.
    pub tab_delta: i32,
    /// Current renderer process count (proxy for open tab count).
    pub renderer_count: u32,
    /// True if a high-memory (heavy) app terminated this cycle.
    /// Apollo can relax pressure thresholds for 2-3 cycles.
    pub freed_heavy_app: bool,
    /// Estimated MB freed from terminated heavy apps (rough estimate).
    pub estimated_freed_mb: u32,
    /// Name of the current foreground app (empty if unknown).
    pub foreground_name: String,
    /// True if the foreground app changed since last cycle.
    pub foreground_changed: bool,
}

// ── Sensor ────────────────────────────────────────────────────────────────────

/// Tracks process-list deltas between daemon cycles to infer app events.
///
/// # Usage
///
/// ```ignore
/// let mut sensor = WindowSensor::new();
/// loop {
///     let procs: Vec<(u32, &str)> = system.processes().iter()
///         .map(|(pid, p)| (pid.as_u32(), p.name())).collect();
///     let delta = sensor.tick(&procs, foreground_name.as_deref());
///     if delta.tab_delta < 0 {
///         // tabs closed — pressure will drop, relax thresholds
///     }
/// }
/// ```
pub struct WindowSensor {
    /// Previous cycle's process map: pid → name.
    prev_procs: HashMap<u32, String>,
    /// Previous cycle's renderer count.
    prev_renderer_count: u32,
    /// Previous foreground app name.
    prev_foreground: String,
    /// Whether we have seen at least one cycle (first tick has no delta).
    initialized: bool,
}

impl WindowSensor {
    pub fn new() -> Self {
        Self {
            prev_procs: HashMap::new(),
            prev_renderer_count: 0,
            prev_foreground: String::new(),
            initialized: false,
        }
    }

    /// Run one sensor tick. Call once per daemon cycle after process collection.
    ///
    /// `procs` — slice of (pid, name) pairs from the current cycle's process table.
    /// `foreground` — optional name of the currently focused app.
    pub fn tick(&mut self, procs: &[(u32, &str)], foreground: Option<&str>) -> WindowDelta {
        let mut delta = WindowDelta::default();

        // ── Foreground tracking ───────────────────────────────────────────────
        let fg_name = foreground.unwrap_or("").to_string();
        delta.foreground_name = fg_name.clone();
        delta.foreground_changed = fg_name != self.prev_foreground;
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

        // ── First tick: no diff possible, just initialize ─────────────────────
        if !self.initialized {
            self.prev_procs = current;
            self.prev_renderer_count = renderer_count;
            self.initialized = true;
            return delta;
        }

        // ── Detect terminations ───────────────────────────────────────────────
        for (pid, name) in &self.prev_procs {
            if !current.contains_key(pid) {
                // Process disappeared → terminated
                if is_heavy_app(name) {
                    delta.freed_heavy_app = true;
                    // Rough estimate: heavy apps typically hold 200-800MB.
                    // Conservative estimate: 200MB per heavy app terminated.
                    delta.estimated_freed_mb += 200;
                }
                // Only emit non-renderer terminations as explicit events
                // (renderer terminations are summarized via tab_delta).
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

        // ── Tab delta ─────────────────────────────────────────────────────────
        delta.tab_delta = renderer_count as i32 - self.prev_renderer_count as i32;

        // ── Foreground gained event ───────────────────────────────────────────
        if delta.foreground_changed && !fg_name.is_empty() {
            // Find PID of the foreground app in current procs
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

    /// Remove stale PIDs from internal state. Call periodically (e.g., every 100 cycles).
    /// In practice the diff already cleans up terminated PIDs, so this is a safety net.
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

fn is_renderer(name: &str) -> bool {
    RENDERER_PATTERNS.iter().any(|p| name.contains(p))
}

fn is_heavy_app(name: &str) -> bool {
    HEAVY_APP_PATTERNS.iter().any(|p| name.contains(p))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_procs<'a>(list: &'a [(u32, &'a str)]) -> Vec<(u32, &'a str)> {
        list.to_vec()
    }

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
        let procs1 = make_procs(&[(1, "launchd"), (200, "Slack")]);
        s.tick(&procs1, None);

        // Slack terminated
        let procs2 = make_procs(&[(1, "launchd")]);
        let d = s.tick(&procs2, None);
        let term: Vec<_> = d
            .events
            .iter()
            .filter(|e| e.kind == AppEventKind::Terminated)
            .collect();
        assert_eq!(term.len(), 1);
        assert_eq!(term[0].name, "Slack");
        assert_eq!(term[0].pid, 200);
    }

    #[test]
    fn heavy_app_terminated_sets_flag() {
        let mut s = WindowSensor::new();
        let procs1 = make_procs(&[(1, "launchd"), (300, "Brave Browser")]);
        s.tick(&procs1, None);

        let procs2 = make_procs(&[(1, "launchd")]);
        let d = s.tick(&procs2, None);
        assert!(d.freed_heavy_app);
        assert_eq!(d.estimated_freed_mb, 200);
    }

    #[test]
    fn tab_delta_negative_on_tab_close() {
        let mut s = WindowSensor::new();
        // 3 renderer processes (3 tabs)
        let procs1 = make_procs(&[
            (1, "Brave Browser"),
            (101, "Brave Browser Helper (Renderer)"),
            (102, "Brave Browser Helper (Renderer)"),
            (103, "Brave Browser Helper (Renderer)"),
        ]);
        s.tick(&procs1, None);

        // 2 renderer processes (one tab closed)
        let procs2 = make_procs(&[
            (1, "Brave Browser"),
            (101, "Brave Browser Helper (Renderer)"),
            (102, "Brave Browser Helper (Renderer)"),
        ]);
        let d = s.tick(&procs2, None);
        assert_eq!(d.tab_delta, -1);
        assert_eq!(d.renderer_count, 2);
    }

    #[test]
    fn tab_delta_positive_on_tab_open() {
        let mut s = WindowSensor::new();
        let procs1 = make_procs(&[
            (1, "Brave Browser"),
            (101, "Brave Browser Helper (Renderer)"),
        ]);
        s.tick(&procs1, None);

        let procs2 = make_procs(&[
            (1, "Brave Browser"),
            (101, "Brave Browser Helper (Renderer)"),
            (102, "Brave Browser Helper (Renderer)"),
            (103, "Brave Browser Helper (Renderer)"),
        ]);
        let d = s.tick(&procs2, None);
        assert_eq!(d.tab_delta, 2);
    }

    #[test]
    fn renderer_terminations_not_emitted_as_events() {
        let mut s = WindowSensor::new();
        let procs1 = make_procs(&[
            (1, "Brave Browser"),
            (200, "Brave Browser Helper (Renderer)"),
            (201, "Brave Browser Helper (Renderer)"),
        ]);
        s.tick(&procs1, None);

        // Both renderers close
        let procs2 = make_procs(&[(1, "Brave Browser")]);
        let d = s.tick(&procs2, None);
        // tab_delta = -2
        assert_eq!(d.tab_delta, -2);
        // No Terminated events for renderers (they're summarized as tab_delta)
        let renderer_events: Vec<_> = d
            .events
            .iter()
            .filter(|e| {
                e.kind == AppEventKind::Terminated && e.name.contains("Renderer")
            })
            .collect();
        assert!(renderer_events.is_empty());
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
        let procs1 = make_procs(&[(1, "launchd")]);
        s.tick(&procs1, None);

        let procs2 = make_procs(&[(1, "launchd"), (500, "Notion")]);
        let d = s.tick(&procs2, None);
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
        let procs1 = make_procs(&[
            (1, "Brave Browser"),
            (10, "Brave Browser Helper (GPU)"),
        ]);
        s.tick(&procs1, None);

        let procs2 = make_procs(&[(1, "Brave Browser")]);
        let d = s.tick(&procs2, None);
        assert_eq!(d.tab_delta, -1);
    }

    #[test]
    fn gc_removes_stale_pids() {
        let mut s = WindowSensor::new();
        let procs = make_procs(&[(1, "launchd"), (999, "Zombie")]);
        s.tick(&procs, None);

        let alive: std::collections::HashSet<u32> = [1u32].into_iter().collect();
        s.gc(&alive);
        assert!(!s.prev_procs.contains_key(&999));
        assert!(s.prev_procs.contains_key(&1));
    }
}
