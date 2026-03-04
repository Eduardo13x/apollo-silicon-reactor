//! User Profile — adaptive behavioural model built purely from observation
//!
//! No LLM required. Learns what the user actually does by tracking:
//!   - Which apps are in the foreground and for how long
//!   - Patterns by hour-of-day (morning email, afternoon code, evening video)
//!   - Which background processes correlate with "productive" sessions
//!
//! The resulting `UserProfile` is used by the adaptive governor to decide
//! what to freeze/unfreeze without ever asking the user.

use std::collections::HashMap;
use std::time::Instant;
use serde::{Deserialize, Serialize};

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

/// Heuristic signatures — list of substrings to look for in process names.
pub fn workload_signatures() -> Vec<(WorkloadType, Vec<&'static str>)> {
    vec![
        (
            WorkloadType::Coding,
            vec![
                "Xcode", "xcodebuild", "cargo", "rustc", "clang", "gcc", "make",
                "cmake", "git", "IntelliJ", "CLion", "VSCode", "code", "Cursor",
                "lldb", "gdb", "python3", "node",
            ],
        ),
        (
            WorkloadType::VideoCall,
            vec![
                "zoom.us", "Teams", "Google Meet", "Slack", "FaceTime",
                "webex", "Discord",
            ],
        ),
        (
            WorkloadType::MediaPlayback,
            vec![
                "IINA", "VLC", "QuickTime", "Infuse", "Plex",
                "Music", "Spotify", "Podcasts",
            ],
        ),
        (
            WorkloadType::VideoEdit,
            vec![
                "Final Cut", "DaVinci", "Premiere", "HandBrake",
                "Motion", "Compressor", "ffmpeg",
            ],
        ),
        (
            WorkloadType::OfficeWork,
            vec![
                "Mail", "Calendar", "Notes", "Pages", "Numbers", "Keynote",
                "Word", "Excel", "PowerPoint", "Notion",
            ],
        ),
        (
            WorkloadType::CommandLine,
            vec![
                "Terminal", "iTerm", "Warp", "Alacritty", "kitty",
                "ssh", "tmux", "zsh", "bash",
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
    session_history: Vec<AppSession>,
    /// Classifiers
    workload_sigs: Vec<(WorkloadType, Vec<&'static str>)>,
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

        Self {
            app_stats: HashMap::new(),
            hour_model,
            current_workload: WorkloadType::Idle,
            current_foreground: None,
            session_start: None,
            session_history: Vec::new(),
            workload_sigs: workload_signatures(),
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
        let elapsed_secs = self.last_observe.map(|t| t.elapsed().as_secs()).unwrap_or(0);
        self.last_observe = Some(Instant::now());
        for (name, stats) in &mut self.app_stats {
            if foreground_app == Some(name.as_str()) {
                stats.secs_since_last_use = 0;
            } else {
                stats.secs_since_last_use =
                    stats.secs_since_last_use.saturating_add(elapsed_secs);
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

    /// Confidence (0.0–1.0) that a process is needed for the current workload.
    pub fn process_relevance(&self, process_name: &str) -> f32 {
        // Check if process directly matches current workload signature
        let current = self.current_workload;
        for (workload, patterns) in &self.workload_sigs {
            if *workload == current && patterns.iter().any(|p| process_name.contains(p)) {
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

    fn detect_workload(
        &self,
        foreground: Option<&str>,
        all_procs: &[&str],
    ) -> WorkloadType {
        let mut scores: HashMap<WorkloadType, u32> = HashMap::new();

        let check_target = |name: &str| {
            for (workload, patterns) in &self.workload_sigs {
                if patterns.iter().any(|p| name.contains(p)) {
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
        stats.dominant_workload = Some(workload);

        if self.session_history.len() >= 200 {
            self.session_history.remove(0);
        }
        self.session_history.push(AppSession {
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
                        // Use serde_json to get the kebab-case name, falling back to Debug.
                        let key = serde_json::to_value(wl)
                            .ok()
                            .and_then(|v| v.as_str().map(|s| s.to_string()))
                            .unwrap_or_else(|| format!("{:?}", wl).to_lowercase());
                        (key, *count)
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
