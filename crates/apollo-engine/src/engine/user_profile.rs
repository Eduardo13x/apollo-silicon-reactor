//! User Profile — adaptive behavioural model built purely from observation
//!
//! No LLM required. Learns what the user actually does by tracking:
//!   - Which apps are in the foreground and for how long
//!   - Patterns by hour-of-day (morning email, afternoon code, evening video)
//!   - Which background processes correlate with "productive" sessions
//!
//! The resulting `UserProfile` is used by the adaptive governor to decide
//! what to freeze/unfreeze without ever asking the user.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

// ── Workload Detection ────────────────────────────────────────────────────────

/// High-level workload type inferred from foreground app + process mix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WorkloadType {
    /// Xcode, cargo, make, rustc, clang, gcc, cmake, git
    Coding,
    /// Zoom, Teams, Meet, Slack (video) + camera/mic active
    VideoCall,
    /// IINA, VLC, QuickTime, Netflix, YouTube, Safari/video tab
    MediaPlayback,
    /// Final Cut, DaVinci, Adobe Premiere, HandBrake
    VideoEdit,
    /// Mail, Calendar, Notes, Safari for browsing
    OfficeWork,
    /// Terminal-heavy workload with no GUI development
    CommandLine,
    /// System appears idle: no foreground interaction for >5 min
    Idle,
    /// Unrecognised mix
    General,
}

impl WorkloadType {
    /// Sprint patch (2026-06-05). Canonical kebab-case wire name for any
    /// variant. Returns the same strings the
    /// `#[serde(rename_all = "kebab-case")]` attribute emits — paired with
    /// [`workload_type_from_str`] this is the documented round-trip pair.
    ///
    /// Replaces 5 producer sites that previously emitted
    /// `format!("{:?}", v).to_lowercase()` — which collapses the
    /// underscore-free `Debug` ("VideoCall" → "videocall") and could not
    /// be parsed back by [`workload_type_from_str`] (which expects
    /// "video-call"). The legacy mismatch silently dropped serialized
    /// VideoCall/MediaPlayback/VideoEdit/OfficeWork/CommandLine state on
    /// restart.
    pub fn as_kebab(&self) -> &'static str {
        match self {
            WorkloadType::Coding => "coding",
            WorkloadType::VideoCall => "video-call",
            WorkloadType::MediaPlayback => "media-playback",
            WorkloadType::VideoEdit => "video-edit",
            WorkloadType::OfficeWork => "office-work",
            WorkloadType::CommandLine => "command-line",
            WorkloadType::Idle => "idle",
            WorkloadType::General => "general",
        }
    }
}

/// Heuristic signatures — list of substrings to look for in process names.
pub fn workload_signatures() -> Vec<(WorkloadType, Vec<&'static str>)> {
    vec![
        (
            WorkloadType::Coding,
            vec![
                "Xcode",
                "xcodebuild",
                "cargo",
                "rustc",
                "clang",
                "gcc",
                "make",
                "cmake",
                "git",
                "IntelliJ",
                "CLion",
                "VSCode",
                "code",
                "Cursor",
                "lldb",
                "gdb",
                "python3",
                "node",
            ],
        ),
        (
            WorkloadType::VideoCall,
            vec![
                "zoom.us",
                "Teams",
                "Google Meet",
                "Slack",
                "FaceTime",
                "webex",
                "Discord",
            ],
        ),
        (
            WorkloadType::MediaPlayback,
            vec![
                "IINA",
                "VLC",
                "QuickTime",
                "Infuse",
                "Plex",
                "Music",
                "Spotify",
                "Podcasts",
            ],
        ),
        (
            WorkloadType::VideoEdit,
            vec![
                "Final Cut",
                "DaVinci",
                "Premiere",
                "HandBrake",
                "Motion",
                "Compressor",
                "ffmpeg",
            ],
        ),
        (
            WorkloadType::OfficeWork,
            vec![
                "Mail",
                "Calendar",
                "Notes",
                "Pages",
                "Numbers",
                "Keynote",
                "Word",
                "Excel",
                "PowerPoint",
                "Notion",
            ],
        ),
        (
            WorkloadType::CommandLine,
            vec![
                "Terminal",
                "iTerm",
                "Warp",
                "Alacritty",
                "kitty",
                "ssh",
                "tmux",
                "zsh",
                "bash",
            ],
        ),
    ]
}

// ── Session Tracking ─────────────────────────────────────────────────────────

/// A single foreground application usage window.
#[derive(Debug, Clone)]
pub struct AppSession {
    pub app_name: String,
    pub started_at: Instant,
    pub duration_secs: u64,
    pub workload: WorkloadType,
    pub hour_of_day: u8, // 0-23
}

/// Per-app accumulated statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AppStats {
    /// Total seconds this app was in the foreground.
    pub total_foreground_secs: u64,
    /// How many times it was launched / brought to front.
    pub launch_count: u32,
    /// Average session length in seconds.
    pub avg_session_secs: u64,
    /// Which workload type this app most often correlates with.
    pub dominant_workload: Option<WorkloadType>,
    /// Seconds since last interaction.
    pub secs_since_last_use: u64,
}

// ── Hour-of-day model ─────────────────────────────────────────────────────────

/// Probability distribution over WorkloadType for a given hour.
pub type HourProfile = HashMap<WorkloadType, f32>;

// ── User Profile ──────────────────────────────────────────────────────────────

pub struct UserProfile {
    /// Per-app cumulative stats.
    app_stats: HashMap<String, AppStats>,
    /// Per-hour workload distribution (learned over time).
    hour_model: [HourProfile; 24],
    /// Currently detected workload.
    current_workload: WorkloadType,
    /// Foreground app right now.
    current_foreground: Option<String>,
    /// When the current foreground session started.
    session_start: Option<Instant>,
    /// Recent session history (last 200 sessions).
    session_history: VecDeque<AppSession>,
    /// Classifiers — retained for serde/persisted-shape compatibility even
    /// though hot-path queries now use [`Self::workload_sig_acs`].
    #[allow(dead_code)]
    workload_sigs: Vec<(WorkloadType, Vec<&'static str>)>,
    /// Sprint patch (2026-06-05) — S6. Pre-built AhoCorasick matchers,
    /// one per signature group. Replaces the O(P) per-name `contains` walk
    /// in `process_relevance` and `detect_workload` with one AC pass. Built
    /// once at construction so the hot path pays zero alloc.
    workload_sig_acs: Vec<(WorkloadType, aho_corasick::AhoCorasick)>,
    /// Timestamp of last observe() call, for computing secs_since_last_use.
    last_observe: Option<Instant>,
}

impl UserProfile {
    pub fn new() -> Self {
        // Initialise each hour with equal probability for each workload
        let hour_model = std::array::from_fn(|_| {
            let mut m = HourProfile::new();
            m.insert(WorkloadType::General, 1.0);
            m
        });

        let workload_sigs = workload_signatures();
        // Sprint patch (2026-06-05) — S6. Pre-build the AhoCorasick matcher
        // per signature group so `process_relevance` / `detect_workload`
        // become one O(N) walk per name rather than O(N×P) substring scan.
        let workload_sig_acs: Vec<(WorkloadType, aho_corasick::AhoCorasick)> = workload_sigs
            .iter()
            .map(|(wl, pats)| {
                let ac = aho_corasick::AhoCorasick::builder()
                    .match_kind(aho_corasick::MatchKind::Standard)
                    .build(pats)
                    .expect("workload_signatures contains valid literals");
                (*wl, ac)
            })
            .collect();
        Self {
            app_stats: HashMap::new(),
            hour_model,
            current_workload: WorkloadType::Idle,
            current_foreground: None,
            session_start: None,
            session_history: VecDeque::new(),
            workload_sigs,
            workload_sig_acs,
            last_observe: None,
        }
    }

    // ── Ingestion ─────────────────────────────────────────────────────────

    /// Call every cycle with the name of the currently active foreground app
    /// and the list of *all* running process names.
    pub fn observe(
        &mut self,
        foreground_app: Option<&str>,
        running_processes: &[&str],
        hour_of_day: u8,
    ) {
        // Track staleness: increment secs_since_last_use for all apps NOT in foreground.
        let elapsed_secs = self
            .last_observe
            .map(|t| t.elapsed().as_secs())
            .unwrap_or(0);
        self.last_observe = Some(Instant::now());
        for (name, stats) in &mut self.app_stats {
            if foreground_app == Some(name.as_str()) {
                stats.secs_since_last_use = 0;
            } else {
                stats.secs_since_last_use = stats.secs_since_last_use.saturating_add(elapsed_secs);
            }
        }

        let new_workload = self.detect_workload(foreground_app, running_processes);
        self.current_workload = new_workload;

        match (foreground_app, &self.current_foreground) {
            (Some(new), Some(old)) if new != old => {
                // App switched — close previous session
                self.close_session(old.clone(), hour_of_day);
                self.open_session(new.to_string());
            }
            (Some(new), None) => {
                self.open_session(new.to_string());
            }
            (None, Some(old)) => {
                self.close_session(old.clone(), hour_of_day);
                self.current_foreground = None;
                self.session_start = None;
            }
            _ => {}
        }

        self.current_foreground = foreground_app.map(|s| s.to_string());

        // Update hour model
        let entry = self.hour_model[hour_of_day as usize]
            .entry(new_workload)
            .or_insert(0.0);
        *entry += 1.0;
    }

    // ── Queries ───────────────────────────────────────────────────────────

    pub fn current_workload(&self) -> WorkloadType {
        self.current_workload
    }

    /// Most likely workload at a given hour based on history.
    pub fn likely_workload_at_hour(&self, hour: u8) -> WorkloadType {
        let profile = &self.hour_model[hour as usize];
        profile
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| *k)
            .unwrap_or(WorkloadType::General)
    }

    /// Read-only access to the hour-of-day model for ML classifier.
    pub fn hour_model_ref(&self) -> &[HourProfile; 24] {
        &self.hour_model
    }

    /// Read-only access to per-app stats for ML classifier.
    pub fn app_stats_ref(&self) -> &HashMap<String, AppStats> {
        &self.app_stats
    }

    /// True if the given app name has been used by this user in the past.
    pub fn user_ever_used(&self, app_name: &str) -> bool {
        self.app_stats.contains_key(app_name)
    }

    /// Return apps that the user has NOT used in `threshold_secs` seconds
    /// but that are currently running.
    pub fn stale_apps(&self, running: &[&str], threshold_secs: u64) -> Vec<String> {
        running
            .iter()
            .filter(|name| {
                match self.app_stats.get(**name) {
                    Some(stats) => stats.secs_since_last_use > threshold_secs,
                    None => true, // Never seen = assume stale
                }
            })
            .map(|s| s.to_string())
            .collect()
    }

    /// Hygiene: decay then evict app stats so the model reflects current
    /// behaviour rather than ancient history.
    ///
    /// Two phases:
    ///
    /// 1. **Decay** — apps idle longer than `decay_after_secs` (default 30 d)
    ///    have their `total_foreground_secs` multiplied by `decay_factor`
    ///    (default 0.5). Their imprint shrinks each pass without dropping
    ///    them, so an app reopened after a month doesn't have to rebuild
    ///    history from zero — it just weighs less than active apps.
    ///
    /// 2. **Evict** — apps idle longer than `evict_after_secs` (default 90 d)
    ///    are removed entirely UNLESS their lifetime usage is high
    ///    (`total_foreground_secs >= grace_secs`, default 50 h). High-usage
    ///    apps are kept indefinitely even if dormant — Eduardo might still
    ///    open Final Cut after a quarter.
    ///
    /// Returns the number of evicted entries (decay is silent).
    /// Caller should call this on a low-frequency tick (once per ~hour);
    /// running it every cycle is wasteful and stats only change on observe().
    pub fn prune_stale(
        &mut self,
        decay_after_secs: u64,
        evict_after_secs: u64,
        decay_factor: f32,
        grace_secs: u64,
    ) -> usize {
        let mut to_remove: Vec<String> = Vec::new();
        for (name, stats) in &mut self.app_stats {
            if stats.secs_since_last_use > evict_after_secs
                && stats.total_foreground_secs < grace_secs
            {
                to_remove.push(name.clone());
            } else if stats.secs_since_last_use > decay_after_secs {
                // Cosmetic for current relevance buckets, but future-proof:
                // any future weight that reads total_foreground_secs sees a
                // smaller imprint for dormant apps.
                stats.total_foreground_secs =
                    ((stats.total_foreground_secs as f32) * decay_factor) as u64;
            }
        }
        let evicted = to_remove.len();
        for name in to_remove {
            self.app_stats.remove(&name);
        }
        evicted
    }

    /// Confidence (0.0–1.0) that a process is needed for the current workload.
    pub fn process_relevance(&self, process_name: &str) -> f32 {
        // Sprint patch (2026-06-05) — S6. Pre-built AC replaces the
        // O(P) substring scan over `patterns`.
        let current = self.current_workload;
        for (workload, ac) in &self.workload_sig_acs {
            if *workload == current && ac.is_match(process_name) {
                return 1.0; // Directly relevant
            }
        }

        // Check usage history
        if let Some(stats) = self.app_stats.get(process_name) {
            let recency = match stats.secs_since_last_use {
                0..=300 => 0.8,
                301..=3600 => 0.5,
                3601..=86400 => 0.2,
                _ => 0.0,
            };
            return recency;
        }

        0.0 // Unknown process — no relevance established
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    fn detect_workload(&self, foreground: Option<&str>, all_procs: &[&str]) -> WorkloadType {
        let mut scores: HashMap<WorkloadType, u32> = HashMap::new();

        // Sprint patch (2026-06-05) — S6. Pre-built AC walk replaces the
        // O(N×P) substring scan that previously fired per name × pattern.
        let check_target = |name: &str| {
            for (workload, ac) in &self.workload_sig_acs {
                if ac.is_match(name) {
                    return Some(*workload);
                }
            }
            None
        };

        // Foreground app gets 3× weight
        if let Some(fg) = foreground {
            if let Some(wl) = check_target(fg) {
                *scores.entry(wl).or_insert(0) += 3;
            }
        }

        // Background processes get 1× weight
        for proc in all_procs {
            if let Some(wl) = check_target(proc) {
                *scores.entry(wl).or_insert(0) += 1;
            }
        }

        scores
            .iter()
            .max_by_key(|(_, v)| *v)
            .map(|(k, _)| *k)
            .unwrap_or(WorkloadType::General)
    }

    fn open_session(&mut self, app: String) {
        self.current_foreground = Some(app);
        self.session_start = Some(Instant::now());
    }

    fn close_session(&mut self, app: String, hour: u8) {
        let duration_secs = self
            .session_start
            .take()
            .map(|s| s.elapsed().as_secs())
            .unwrap_or(0);

        let workload = self.current_workload;

        let stats = self.app_stats.entry(app.clone()).or_default();
        stats.total_foreground_secs += duration_secs;
        stats.launch_count += 1;
        stats.avg_session_secs = stats.total_foreground_secs / stats.launch_count as u64;
        stats.secs_since_last_use = 0;
        // Only set dominant_workload if not yet determined
        if stats.dominant_workload.is_none() {
            stats.dominant_workload = Some(workload);
        }

        if self.session_history.len() >= 200 {
            self.session_history.pop_front();
        }
        self.session_history.push_back(AppSession {
            app_name: app,
            started_at: Instant::now(),
            duration_secs,
            workload,
            hour_of_day: hour,
        });
    }
}

// ── Persistence ───────────────────────────────────────────────────────────────

/// Serialisable snapshot of the parts of UserProfile that are worth persisting
/// across daemon restarts (app_stats + hour model). `session_history` is
/// intentionally excluded (contains `Instant` which is not serialisable and is
/// ephemeral by nature).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UserProfilePersisted {
    /// Per-app cumulative stats.
    pub app_stats: HashMap<String, AppStats>,
    /// Hour-of-day model: 24 entries (one per hour), each a map of
    /// WorkloadType variant name → observation count.
    pub hour_model: Vec<HashMap<String, f32>>,
}

fn workload_type_from_str(s: &str) -> Option<WorkloadType> {
    match s {
        "coding" => Some(WorkloadType::Coding),
        "video-call" => Some(WorkloadType::VideoCall),
        "media-playback" => Some(WorkloadType::MediaPlayback),
        "video-edit" => Some(WorkloadType::VideoEdit),
        "office-work" => Some(WorkloadType::OfficeWork),
        "command-line" => Some(WorkloadType::CommandLine),
        "idle" => Some(WorkloadType::Idle),
        "general" => Some(WorkloadType::General),
        _ => None,
    }
}

impl UserProfile {
    /// Serialize the learnable state to a persisted snapshot.
    pub fn to_persisted(&self) -> UserProfilePersisted {
        let hour_model = self
            .hour_model
            .iter()
            .map(|hp| {
                hp.iter()
                    .map(|(wl, count)| {
                        // Sprint patch (2026-06-05): use canonical kebab — round-trips
                        // with `workload_type_from_str`. The legacy
                        // `format!("{:?}", wl).to_lowercase()` fallback collapsed
                        // "VideoCall" → "videocall" which workload_type_from_str
                        // could not parse.
                        (wl.as_kebab().to_string(), *count)
                    })
                    .collect()
            })
            .collect();
        UserProfilePersisted {
            app_stats: self.app_stats.clone(),
            hour_model,
        }
    }

    /// Restore a UserProfile from a persisted snapshot.
    pub fn from_persisted(p: UserProfilePersisted) -> Self {
        let hour_model: [HourProfile; 24] = std::array::from_fn(|i| {
            if let Some(hour_data) = p.hour_model.get(i) {
                hour_data
                    .iter()
                    .filter_map(|(k, v)| workload_type_from_str(k).map(|wl| (wl, *v)))
                    .collect()
            } else {
                let mut m = HourProfile::new();
                m.insert(WorkloadType::General, 1.0);
                m
            }
        });
        let mut profile = Self::new();
        profile.app_stats = p.app_stats;
        profile.hour_model = hour_model;
        profile
    }
}

impl Default for UserProfile {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sprint patch (2026-06-05): `as_kebab` must round-trip with
    /// `workload_type_from_str`. The legacy `format!("{:?}", v).to_lowercase()`
    /// shape silently dropped multi-word variants because it collapsed
    /// "VideoCall" → "videocall" which the parser does not accept.
    #[test]
    fn as_kebab_round_trips_with_workload_type_from_str() {
        for wt in [
            WorkloadType::Coding,
            WorkloadType::VideoCall,
            WorkloadType::MediaPlayback,
            WorkloadType::VideoEdit,
            WorkloadType::OfficeWork,
            WorkloadType::CommandLine,
            WorkloadType::Idle,
            WorkloadType::General,
        ] {
            let s = wt.as_kebab();
            let parsed = workload_type_from_str(s)
                .unwrap_or_else(|| panic!("workload_type_from_str({s:?}) must accept kebab"));
            assert_eq!(parsed, wt, "as_kebab → from_str round trip for {wt:?}");
        }
    }

    #[test]
    fn workload_type_roundtrip_serde() {
        for wt in [
            WorkloadType::Coding,
            WorkloadType::VideoCall,
            WorkloadType::MediaPlayback,
            WorkloadType::VideoEdit,
            WorkloadType::OfficeWork,
            WorkloadType::CommandLine,
            WorkloadType::Idle,
            WorkloadType::General,
        ] {
            let json = serde_json::to_string(&wt).expect("serialize WorkloadType");
            let rt: WorkloadType = serde_json::from_str(&json).expect("deserialize WorkloadType");
            assert_eq!(rt, wt);
        }
    }

    #[test]
    fn workload_type_uses_kebab_case() {
        let json = serde_json::to_string(&WorkloadType::VideoCall)
            .expect("serialize WorkloadType::VideoCall");
        assert!(
            json.contains('-'),
            "expected kebab-case in JSON, got: {json}"
        );
    }

    #[test]
    fn app_stats_default_zero() {
        let stats = AppStats::default();
        assert_eq!(stats.total_foreground_secs, 0);
        assert_eq!(stats.launch_count, 0);
        assert!(stats.dominant_workload.is_none());
    }

    #[test]
    fn user_profile_new_starts_idle() {
        let up = UserProfile::new();
        assert_eq!(up.current_workload(), WorkloadType::Idle);
    }

    #[test]
    fn user_profile_observe_detects_coding_workload() {
        let mut up = UserProfile::new();
        up.observe(Some("Xcode"), &["rustc", "cargo"], 10);
        assert_eq!(up.current_workload(), WorkloadType::Coding);
    }

    #[test]
    fn user_profile_persisted_roundtrip() {
        let mut up = UserProfile::new();
        up.observe(Some("cargo"), &["rustc"], 9);
        up.observe(Some("Xcode"), &[], 14);

        let persisted = up.to_persisted();
        let json = serde_json::to_string(&persisted).expect("serialize UserProfilePersisted");
        let rt: UserProfilePersisted =
            serde_json::from_str(&json).expect("deserialize UserProfilePersisted");

        // hour_model always has 24 entries after to_persisted
        assert_eq!(rt.hour_model.len(), 24);
    }

    #[test]
    fn likely_workload_at_hour_returns_general_with_no_data() {
        let up = UserProfile::new();
        // With only the default "General: 1.0" seed, should return General
        assert_eq!(up.likely_workload_at_hour(0), WorkloadType::General);
    }

    #[test]
    fn process_relevance_unknown_returns_zero() {
        let up = UserProfile::new();
        assert!((up.process_relevance("unknown-process-xyz") - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn process_relevance_coding_process_returns_one_when_coding() {
        let mut up = UserProfile::new();
        // Put the profile in Coding workload
        up.observe(Some("cargo"), &["rustc", "cargo"], 10);
        // "cargo" should be fully relevant
        let rel = up.process_relevance("cargo");
        assert!(
            rel > 0.9,
            "expected relevance ~1.0 for coding process during coding workload, got {rel}"
        );
    }

    #[test]
    fn prune_stale_evicts_idle_low_usage_apps() {
        let mut up = UserProfile::new();
        up.app_stats.insert(
            "AncientLowUsage".to_string(),
            AppStats {
                total_foreground_secs: 60, // 1 min — low usage
                launch_count: 1,
                avg_session_secs: 60,
                dominant_workload: None,
                secs_since_last_use: 100 * 86_400, // 100 days idle
            },
        );
        let evicted = up.prune_stale(30 * 86_400, 90 * 86_400, 0.5, 50 * 3_600);
        assert_eq!(evicted, 1);
        assert!(!up.app_stats.contains_key("AncientLowUsage"));
    }

    #[test]
    fn prune_stale_keeps_high_usage_app_even_when_dormant() {
        let mut up = UserProfile::new();
        up.app_stats.insert(
            "FinalCutPro".to_string(),
            AppStats {
                total_foreground_secs: 200 * 3_600, // 200 h lifetime — past grace
                launch_count: 50,
                avg_session_secs: 4 * 3_600,
                dominant_workload: Some(WorkloadType::VideoEdit),
                secs_since_last_use: 100 * 86_400, // 100 days idle
            },
        );
        let evicted = up.prune_stale(30 * 86_400, 90 * 86_400, 0.5, 50 * 3_600);
        assert_eq!(evicted, 0);
        assert!(up.app_stats.contains_key("FinalCutPro"));
    }

    #[test]
    fn prune_stale_decays_idle_apps_without_eviction() {
        let mut up = UserProfile::new();
        up.app_stats.insert(
            "OldButRecent".to_string(),
            AppStats {
                total_foreground_secs: 1000,
                launch_count: 10,
                avg_session_secs: 100,
                dominant_workload: None,
                secs_since_last_use: 45 * 86_400, // 45 days — between decay and evict
            },
        );
        let evicted = up.prune_stale(30 * 86_400, 90 * 86_400, 0.5, 50 * 3_600);
        assert_eq!(evicted, 0);
        let stats = up.app_stats.get("OldButRecent").expect("not evicted");
        assert_eq!(stats.total_foreground_secs, 500); // halved
    }

    #[test]
    fn prune_stale_leaves_active_apps_untouched() {
        let mut up = UserProfile::new();
        up.app_stats.insert(
            "DailyDriver".to_string(),
            AppStats {
                total_foreground_secs: 10_000,
                launch_count: 100,
                avg_session_secs: 100,
                dominant_workload: None,
                secs_since_last_use: 60, // active 1 min ago
            },
        );
        let evicted = up.prune_stale(30 * 86_400, 90 * 86_400, 0.5, 50 * 3_600);
        assert_eq!(evicted, 0);
        let stats = up.app_stats.get("DailyDriver").expect("not evicted");
        assert_eq!(stats.total_foreground_secs, 10_000); // untouched
    }
}
