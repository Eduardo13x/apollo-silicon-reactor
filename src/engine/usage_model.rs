use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};

use crate::collector::{ProcessStats, SystemSnapshot};
use crate::engine::safety::protected_processes;

const PRESENCE_HALF_LIFE_DAYS: f64 = 5.0;
const JANK_HALF_LIFE_DAYS: f64 = 2.0;

// Conservative default caps.
const BOOTCAMP_DAYS: i64 = 5;
const PROMOTIONS_PER_DAY_BOOTCAMP: u32 = 3;
const PROMOTIONS_PER_DAY_STABLE: u32 = 1;

const MIN_AGE_HOURS_FOR_PROMOTION: i64 = 12;

// These thresholds assume EMA values in [0, 1].
const MIN_PRESENCE_FOR_INTERACTIVE: f64 = 0.18;
const MIN_INTERACTIVE_EMA: f64 = 0.14;
const MIN_USAGE_SCORE: f64 = 0.22;

const MIN_JANK_EMA: f64 = 0.10;
const MAX_INTERACTIVE_FOR_NOISE: f64 = 0.10;
const MIN_NOISE_SCORE: f64 = 0.18;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UsageModelPersisted {
    pub schema_version: u32,
    pub started_at: Option<DateTime<Utc>>,
    pub updated_at: Option<DateTime<Utc>>,
    pub entries: HashMap<String, UsageEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UsageEntry {
    pub raw_name: String,
    pub norm_name: String,
    pub first_seen_at: Option<DateTime<Utc>>,
    pub last_seen_at: Option<DateTime<Utc>>,

    pub presence_ema: f64,
    pub cpu_ema: f64,
    pub mem_ema: f64,
    pub interactive_ema: f64,
    pub jank_ema: f64,

    pub seen_count_total: u64,
    pub seen_interactive_total: u64,
    pub seen_jank_total: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageEntrySummary {
    pub name: String,
    pub usage_score: f64,
    pub noise_score: f64,
    pub presence_ema: f64,
    pub interactive_ema: f64,
    pub jank_ema: f64,
    pub cpu_ema: f64,
    pub mem_ema: f64,
    pub first_seen_at: Option<DateTime<Utc>>,
    pub last_seen_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageTopReport {
    pub interactive_candidates: Vec<UsageEntrySummary>,
    pub noise_candidates: Vec<UsageEntrySummary>,
    pub model_started_at: Option<DateTime<Utc>>,
    pub model_updated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Default)]
pub struct UsageModel {
    persisted: UsageModelPersisted,
}

impl UsageModel {
    pub fn load(path: &Path) -> Self {
        let data = std::fs::read_to_string(path).ok();
        if let Some(data) = data {
            if let Ok(mut p) = serde_json::from_str::<UsageModelPersisted>(&data) {
                if p.schema_version == 0 {
                    p.schema_version = 1;
                }
                return Self { persisted: p };
            }
        }
        Self {
            persisted: UsageModelPersisted {
                schema_version: 1,
                started_at: None,
                updated_at: None,
                entries: HashMap::new(),
            },
        }
    }

    pub fn persist(&self, path: &Path) {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(&self.persisted) {
            let _ = std::fs::write(path, json);
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
            }
        }
    }

    pub fn update_from_snapshot(
        &mut self,
        snapshot: &SystemSnapshot,
        now: DateTime<Utc>,
        interactive_proxy: bool,
        jank_proxy: bool,
        top_n: usize,
    ) {
        if self.persisted.started_at.is_none() {
            self.persisted.started_at = Some(now);
        }
        self.persisted.updated_at = Some(now);

        let procs: Vec<&ProcessStats> = snapshot.top_processes.iter().take(top_n).collect();
        for p in procs {
            let norm = normalize_name(&p.name);
            let entry = self
                .persisted
                .entries
                .entry(norm.clone())
                .or_insert_with(|| UsageEntry {
                    raw_name: p.name.clone(),
                    norm_name: norm.clone(),
                    first_seen_at: Some(now),
                    last_seen_at: Some(now),
                    ..UsageEntry::default()
                });

            let dt = entry
                .last_seen_at
                .map(|t| (now - t).to_std().ok())
                .flatten()
                .unwrap_or_default();
            let decay_presence = decay_factor(dt.as_secs_f64(), PRESENCE_HALF_LIFE_DAYS);
            let decay_jank = decay_factor(dt.as_secs_f64(), JANK_HALF_LIFE_DAYS);

            entry.last_seen_at = Some(now);
            entry.seen_count_total += 1;

            // Presence: binary.
            entry.presence_ema = ema_update(entry.presence_ema, 1.0, decay_presence);

            // CPU/memory normalized.
            let cpu_norm = (p.cpu_usage as f64 / 100.0).clamp(0.0, 1.0);
            let mem_norm =
                (p.memory_usage as f64 / (4.0 * 1024.0 * 1024.0 * 1024.0)).clamp(0.0, 1.0);
            entry.cpu_ema = ema_update(entry.cpu_ema, cpu_norm, decay_presence);
            entry.mem_ema = ema_update(entry.mem_ema, mem_norm, decay_presence);

            if interactive_proxy {
                entry.seen_interactive_total += 1;
                entry.interactive_ema = ema_update(entry.interactive_ema, 1.0, decay_presence);
            } else {
                entry.interactive_ema = ema_update(entry.interactive_ema, 0.0, decay_presence);
            }

            if jank_proxy {
                entry.seen_jank_total += 1;
                entry.jank_ema = ema_update(entry.jank_ema, 1.0, decay_jank);
            } else {
                entry.jank_ema = ema_update(entry.jank_ema, 0.0, decay_jank);
            }
        }
    }

    pub fn entry_summary(&self, name: &str) -> Option<UsageEntrySummary> {
        let norm = normalize_name(name);
        let e = self.persisted.entries.get(&norm)?;
        Some(summarize_entry(e))
    }

    pub fn top_report(&self, limit: usize) -> UsageTopReport {
        let mut entries: Vec<UsageEntrySummary> = self
            .persisted
            .entries
            .values()
            .map(summarize_entry)
            .collect();

        // Filter protected and obviously critical substrings.
        let protected = protected_processes();
        entries.retain(|e| !protected.iter().any(|p| e.name.contains(p)));

        let mut interactive = entries.clone();
        // BUG 20 fix: unwrap on partial_cmp could panic if NaN. Use unwrap_or(Equal).
        interactive.sort_by(|a, b| {
            b.usage_score
                .partial_cmp(&a.usage_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        interactive.truncate(limit);

        let mut noise = entries;
        // BUG 21 fix: same as BUG 20 for noise_score.
        noise.sort_by(|a, b| {
            b.noise_score
                .partial_cmp(&a.noise_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        noise.retain(|n| !interactive.iter().any(|i| i.name == n.name));
        noise.truncate(limit);

        UsageTopReport {
            interactive_candidates: interactive,
            noise_candidates: noise,
            model_started_at: self.persisted.started_at,
            model_updated_at: self.persisted.updated_at,
        }
    }

    pub fn maybe_promote_patterns(
        &self,
        now: DateTime<Utc>,
        existing_interactive: &[String],
        existing_noise: &[String],
        daily_promotions_used: u32,
        started_at: Option<DateTime<Utc>>,
    ) -> Vec<(String, String)> {
        // Returns (kind, pattern) where kind in {interactive, noise}.
        let bootcamp = started_at
            .map(|t| now - t < ChronoDuration::days(BOOTCAMP_DAYS))
            .unwrap_or(false);
        let cap = if bootcamp {
            PROMOTIONS_PER_DAY_BOOTCAMP
        } else {
            PROMOTIONS_PER_DAY_STABLE
        };
        if daily_promotions_used >= cap {
            return Vec::new();
        }

        let protected = protected_processes();
        let mut candidates: Vec<UsageEntrySummary> = self
            .persisted
            .entries
            .values()
            .map(summarize_entry)
            .collect();
        candidates.retain(|e| !protected.iter().any(|p| e.name.contains(p)));

        // Conservative: require age.
        candidates.retain(|e| {
            e.first_seen_at
                .map(|t| now - t > ChronoDuration::hours(MIN_AGE_HOURS_FOR_PROMOTION))
                .unwrap_or(false)
        });

        let mut promotions: Vec<(String, String)> = Vec::new();

        // Interactive promotions.
        let mut interactive = candidates.clone();
        // BUG 22 fix: unwrap on partial_cmp could panic if NaN.
        interactive.sort_by(|a, b| {
            b.usage_score
                .partial_cmp(&a.usage_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for e in interactive {
            if promotions.len() as u32 >= cap - daily_promotions_used {
                break;
            }
            if e.presence_ema < MIN_PRESENCE_FOR_INTERACTIVE {
                continue;
            }
            if e.interactive_ema < MIN_INTERACTIVE_EMA {
                continue;
            }
            if e.usage_score < MIN_USAGE_SCORE {
                continue;
            }
            if existing_interactive.iter().any(|p| e.name.contains(p)) {
                continue;
            }
            promotions.push(("interactive".to_string(), e.name));
        }

        // Noise promotions.
        let mut noise = candidates;
        // BUG 23 fix: unwrap on partial_cmp could panic if NaN.
        noise.sort_by(|a, b| {
            b.noise_score
                .partial_cmp(&a.noise_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for e in noise {
            if promotions.len() as u32 >= cap - daily_promotions_used {
                break;
            }
            if e.jank_ema < MIN_JANK_EMA {
                continue;
            }
            if e.interactive_ema > MAX_INTERACTIVE_FOR_NOISE {
                continue;
            }
            if e.noise_score < MIN_NOISE_SCORE {
                continue;
            }
            if existing_noise.iter().any(|p| e.name.contains(p)) {
                continue;
            }
            promotions.push(("noise".to_string(), e.name));
        }

        promotions
    }
}

pub fn usage_model_path_root(is_root: bool) -> PathBuf {
    if is_root {
        PathBuf::from("/var/lib/apollo/usage_model.json")
    } else {
        PathBuf::from("/tmp/apollo-usage_model.json")
    }
}

fn normalize_name(name: &str) -> String {
    name.trim().to_string()
}

fn decay_factor(dt_secs: f64, half_life_days: f64) -> f64 {
    let hl = half_life_days * 24.0 * 3600.0;
    if hl <= 0.0 {
        return 0.0;
    }
    // decay = 0.5^(dt/hl)
    (0.5_f64).powf(dt_secs / hl)
}

fn ema_update(prev: f64, value: f64, decay: f64) -> f64 {
    (prev * decay) + (value * (1.0 - decay))
}

fn summarize_entry(e: &UsageEntry) -> UsageEntrySummary {
    let impact = ((e.cpu_ema + e.mem_ema) * 0.5).clamp(0.0, 1.0);
    let usage_score = (0.50 * e.presence_ema) + (0.35 * e.interactive_ema) + (0.15 * impact);
    let noise_score = (0.60 * e.jank_ema) + (0.40 * impact);
    UsageEntrySummary {
        name: e.raw_name.clone(),
        usage_score: usage_score.clamp(0.0, 1.0),
        noise_score: noise_score.clamp(0.0, 1.0),
        presence_ema: e.presence_ema,
        interactive_ema: e.interactive_ema,
        jank_ema: e.jank_ema,
        cpu_ema: e.cpu_ema,
        mem_ema: e.mem_ema,
        first_seen_at: e.first_seen_at,
        last_seen_at: e.last_seen_at,
    }
}
