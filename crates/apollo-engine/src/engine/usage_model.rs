use crate::engine::types::HardPath;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};

use crate::collector::{ProcessStats, SystemSnapshot};

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

    /// EMA of cpu_wall_ratio from proc_pid_rusage deltas.
    /// Low (< 0.05) = I/O-bound (behavior-interactive), high (> 0.70) = CPU-bound.
    /// Initialized to 0.5 (neutral) for cold-start / schema migration.
    #[serde(default = "default_cpu_wall_ratio_ema")]
    pub cpu_wall_ratio_ema: f64,

    pub seen_count_total: u64,
    pub seen_interactive_total: u64,
    pub seen_jank_total: u64,
}

fn default_cpu_wall_ratio_ema() -> f64 {
    0.5
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
    pub cpu_wall_ratio_ema: f64,
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
        let data = HardPath::read_to_string_limited(path, 10 * 1024 * 1024).ok();
        if let Some(data) = data {
            if let Ok(mut p) = serde_json::from_str::<UsageModelPersisted>(&data) {
                if p.schema_version == 0 {
                    p.schema_version = 1;
                }
                // Schema v1 → v2 migration: initialize cpu_wall_ratio_ema to
                // neutral 0.5 for all existing entries (serde default handles
                // missing fields, but bump version to track the migration).
                if p.schema_version < 2 {
                    for entry in p.entries.values_mut() {
                        if entry.cpu_wall_ratio_ema == 0.0 {
                            entry.cpu_wall_ratio_ema = 0.5;
                        }
                    }
                    p.schema_version = 2;
                }
                return Self { persisted: p };
            }
        }
        Self {
            persisted: UsageModelPersisted {
                schema_version: 2,
                started_at: None,
                updated_at: None,
                entries: HashMap::new(),
            },
        }
    }

    pub fn persist(&self, path: &Path) {
        crate::engine::llm::write_json(path, &self.persisted, Some(0o600));
    }

    pub fn update_from_snapshot(
        &mut self,
        snapshot: &SystemSnapshot,
        now: DateTime<Utc>,
        interactive_proxy: bool,
        jank_proxy: bool,
        top_n: usize,
        cpu_wall_ratios: &HashMap<String, f32>,
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
                .and_then(|t| (now - t).to_std().ok())
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

            // EMA-update cpu_wall_ratio if we have a measurement for this process.
            // This is a PARALLEL signal to interactive_ema — does NOT replace it.
            if let Some(&ratio) = cpu_wall_ratios.get(&p.name) {
                let ratio_f64 = (ratio as f64).clamp(0.0, 1.0);
                entry.cpu_wall_ratio_ema =
                    ema_update(entry.cpu_wall_ratio_ema, ratio_f64, decay_presence);
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

        // Filter protected processes via the unified exact-match oracle.
        // Previously used `protected_processes().iter().any(|p| name.contains(p))`
        // which over-protected by substring (e.g., "WindowServer-helper" was
        // wrongly excluded because it contained "WindowServer"). is_protected_name()
        // is the single authoritative classifier (Tier 1 exact / Tier 2 infra /
        // Tier 3 dev runtime) — Saltzer & Kaashoek 2009 §3.3 Complete Mediation.
        entries.retain(|e| !crate::engine::safety::is_protected_name(&e.name));

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

    /// Access the raw entries map (e.g. to build behavior-interactive PID sets).
    pub fn entries(&self) -> &HashMap<String, UsageEntry> {
        &self.persisted.entries
    }

    pub fn maybe_promote_patterns(
        &self,
        now: DateTime<Utc>,
        existing_interactive: &[String],
        existing_noise: &[String],
        existing_protected: &[String],
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

        let never_interactive = never_promote_interactive();
        let mut candidates: Vec<UsageEntrySummary> = self
            .persisted
            .entries
            .values()
            .map(summarize_entry)
            .collect();
        // Use unified exact-match oracle for protected processes (closes
        // substring false-positive bug — see top_report() comment above).
        // never_promote_interactive() retains substring scan: it is a
        // separate, intentionally-permissive heuristic over name fragments
        // like "helper", not exact OS-daemon identities.
        candidates.retain(|e| !crate::engine::safety::is_protected_name(&e.name));
        candidates.retain(|e| !never_interactive.iter().any(|p| e.name.contains(p)));

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
        let protected_candidates = candidates.clone();
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

        // Protected promotions: apps the user switches to frequently and
        // consistently get a "protected" safety label.  Unlike interactive/noise
        // promotions these do NOT count against the daily cap — they are safety
        // annotations that prevent the optimizer from freezing an app the user
        // actively uses as a secondary window (e.g. Antigravity while Chrome is
        // in the foreground).
        //
        // Thresholds are intentionally high: interactive_ema > 0.55 means the
        // app was observed as the foreground app in the majority of recent cycles.
        for e in &protected_candidates {
            if e.interactive_ema > 0.55
                && e.presence_ema > 0.40
                && !existing_protected
                    .iter()
                    .any(|p| e.name.contains(p.as_str()))
                && !promotions.iter().any(|(_, n)| n == &e.name)
            {
                promotions.push(("protected".to_string(), e.name.clone()));
                break; // One per call — policy accumulates over cycles.
            }
        }

        promotions
    }
}

/// Processes that must never be promoted to interactive patterns.
/// These are background daemons, telemetry, and transient launchers that
/// appear frequently but do not benefit from priority boosting.
fn never_promote_interactive() -> Vec<&'static str> {
    vec![
        // The optimizer itself (circular dependency)
        "apollo-optimizerd",
        // Rust toolchain wrapper — rustup antepone "stable"/"nightly"/etc. como
        // proceso padre de rustc durante compilación. Consume RAM proporcional
        // al build y NO es una app interactiva del usuario.
        "stable",
        "nightly",
        "beta",
        // Telemetry / analytics
        "UsageTrackingAgent",
        "amsengagementd",
        "ecosystemanalyticsd",
        "PerfPowerServices",
        "triald",
        // Background asset / sync daemons
        "assetsubscriptiond",
        "mobileassetd",
        "searchpartyd",
        "cloudd",
        "fileproviderd",
        "photolibraryd",
        "softwareupdated",
        "accessoryupdaterd",
        // Background ML / analysis
        "photoanalysisd",
        "mediaanalysisd",
        "ModelCatalogAgent",
        "duetexpertd",
        // Spotlight / indexing
        "corespotlightd",
        "spotlightknowledged",
        "spindump",
        // System daemons (no user-visible latency impact)
        "dasd",
        "deleted",
        "ecosystemd",
        "fseventsd",
        "logd",
        "runningboardd",
        "airportd",
        "corebrightnessd",
        // Siri / assistant background
        "assistantd",
        "contextstored",
        "corespeechd",
        "com.apple.siri.embeddedspeech",
        "suggestd",
        // Preference / contacts sync
        "cfprefsd",
        "contactsd",
        // Updaters (not drivers)
        "logioptionsplus_updater",
        // Security scanners
        "XprotectService",
        // Decorative
        "WallpaperAerialsExtension",
        // Transient launchers
        "xpcproxy",
        "iconservicesagent",
        "linkd",
        "siriactionsd",
        "com.apple.Safari.SafeBrowsing.Service",
    ]
}

/// Behavior-based interactivity detection via cpu_wall_ratio EMA.
///
/// A process is "behavior-interactive" when:
/// 1. Its cpu_wall_ratio EMA is low (< 0.05) — I/O-bound, waiting on user input
/// 2. It has sustained presence (presence_ema >= 0.15) — not a transient process
/// 3. It has been seen enough times (>= 10) — cold-start protection
///
/// This is analogous to Linux CFS's interactivity estimator: processes that
/// spend most of their time sleeping (low CPU/wall ratio) are interactive.
pub fn is_behavior_interactive(entry: &UsageEntry) -> bool {
    entry.cpu_wall_ratio_ema < 0.05 && entry.presence_ema >= 0.15 && entry.seen_count_total >= 10
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

    // Memory footprint signal: user-facing apps hold more resident memory
    // (~50MB-2GB) than background daemons (~1-20MB).  mem_ema is normalised
    // to 4 GB, so 100 MB → 0.024; amplify 15× to make it meaningful.
    let mem_signal = (e.mem_ema * 15.0).clamp(0.0, 1.0);

    // Cap presence at 0.6 — beyond this, always-on daemons and heavy user
    // apps are indistinguishable.  The cap prevents 24/7 daemons from
    // dominating the ranking solely because of constant uptime.
    let capped_presence = e.presence_ema.min(0.6);

    // Old formula: 0.50 * presence + 0.35 * interactive + 0.15 * impact
    // Problem: daemons with presence ~0.91 dominated; real apps ranked below.
    let usage_score = (0.25 * capped_presence)
        + (0.35 * e.interactive_ema)
        + (0.15 * impact)
        + (0.25 * mem_signal);

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
        cpu_wall_ratio_ema: e.cpu_wall_ratio_ema,
        first_seen_at: e.first_seen_at,
        last_seen_at: e.last_seen_at,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(cpu_wall_ratio_ema: f64, presence_ema: f64, seen_count: u64) -> UsageEntry {
        UsageEntry {
            raw_name: "TestApp".to_string(),
            norm_name: "TestApp".to_string(),
            first_seen_at: None,
            last_seen_at: None,
            presence_ema,
            cpu_ema: 0.0,
            mem_ema: 0.0,
            interactive_ema: 0.0,
            jank_ema: 0.0,
            cpu_wall_ratio_ema,
            seen_count_total: seen_count,
            seen_interactive_total: 0,
            seen_jank_total: 0,
        }
    }

    fn make_snapshot(names: &[&str]) -> SystemSnapshot {
        use crate::collector::*;
        SystemSnapshot {
            timestamp: Utc::now(),
            cpu: CpuStats {
                global_usage: 10.0,
                core_count: 8,
            },
            memory: MemoryStats {
                total_ram: 8 * 1024 * 1024 * 1024,
                used_ram: 4 * 1024 * 1024 * 1024,
                free_ram: 4 * 1024 * 1024 * 1024,
                total_swap: 0,
                used_swap: 0,
            },
            pressure: PressureStats {
                memory_pressure: 0.3,
                swap_used_bytes: 0,
                swap_total_bytes: 0,
                swap_delta_bytes_per_sec: 0.0,
                thermal_level: "nominal".to_string(),
                compressor_pressure: 0.0,
                thrashing_score: 0.0,
            },
            disks: vec![],
            networks: vec![],
            top_processes: names
                .iter()
                .enumerate()
                .map(|(i, n)| ProcessStats {
                    pid: (i + 1) as u32,
                    name: n.to_string(),
                    cpu_usage: 5.0,
                    memory_usage: 100 * 1024 * 1024,
                    cpu_wall_ratio: None,
                })
                .collect(),
        }
    }

    #[test]
    fn test_cpu_wall_ratio_ema_update() {
        let mut model = UsageModel::default();
        let snapshot = make_snapshot(&["TestApp"]);
        let now = Utc::now();

        // Feed a constant low ratio (0.02) for many cycles.
        let mut ratios = HashMap::new();
        ratios.insert("TestApp".to_string(), 0.02_f32);

        for i in 0..50 {
            let t = now + ChronoDuration::seconds(i * 3);
            model.update_from_snapshot(&snapshot, t, false, false, 10, &ratios);
        }

        let entry = model.entries().get("TestApp").unwrap();
        // After 50 updates with ratio=0.02, the EMA should converge close to 0.02.
        assert!(
            entry.cpu_wall_ratio_ema < 0.06,
            "EMA should converge near 0.02, got {}",
            entry.cpu_wall_ratio_ema
        );
        assert!(
            entry.cpu_wall_ratio_ema > 0.0,
            "EMA should be positive, got {}",
            entry.cpu_wall_ratio_ema
        );
    }

    #[test]
    fn test_is_behavior_interactive_low_ratio() {
        // Low ratio, high presence, enough observations → interactive.
        let entry = make_entry(0.03, 0.30, 20);
        assert!(
            is_behavior_interactive(&entry),
            "low cpu_wall_ratio + high presence + enough seen → should be interactive"
        );
    }

    #[test]
    fn test_is_behavior_interactive_high_ratio() {
        // High ratio (CPU-bound) → NOT interactive.
        let entry = make_entry(0.80, 0.30, 20);
        assert!(
            !is_behavior_interactive(&entry),
            "high cpu_wall_ratio → should NOT be interactive"
        );

        // Borderline: ratio = 0.10 (above 0.05 threshold) → NOT interactive.
        let entry2 = make_entry(0.10, 0.30, 20);
        assert!(
            !is_behavior_interactive(&entry2),
            "ratio 0.10 (> 0.05 threshold) → should NOT be interactive"
        );
    }

    #[test]
    fn test_is_behavior_interactive_insufficient_presence() {
        // Low ratio but low presence → NOT interactive (transient process).
        let entry = make_entry(0.02, 0.10, 20);
        assert!(
            !is_behavior_interactive(&entry),
            "low presence_ema → should NOT be interactive"
        );
    }

    #[test]
    fn test_is_behavior_interactive_insufficient_seen_count() {
        // Low ratio, high presence, but too few observations → NOT interactive (cold start).
        let entry = make_entry(0.02, 0.30, 5);
        assert!(
            !is_behavior_interactive(&entry),
            "seen_count < 10 → should NOT be interactive (cold start)"
        );
    }

    #[test]
    fn test_cpu_wall_ratio_ema_no_ratio_preserves_default() {
        let mut model = UsageModel::default();
        let snapshot = make_snapshot(&["NoRatioApp"]);
        let now = Utc::now();

        // Update WITHOUT providing a ratio for this process.
        let empty_ratios: HashMap<String, f32> = HashMap::new();
        model.update_from_snapshot(&snapshot, now, false, false, 10, &empty_ratios);

        let entry = model.entries().get("NoRatioApp").unwrap();
        // Default cpu_wall_ratio_ema should be 0.0 (Default trait) since it's
        // a new entry and no ratio was provided. The serde default (0.5) only
        // applies when deserializing from JSON.
        assert!(
            (entry.cpu_wall_ratio_ema - 0.0).abs() < f64::EPSILON,
            "new entry without ratio should keep default 0.0, got {}",
            entry.cpu_wall_ratio_ema
        );
    }

    #[test]
    fn test_summarize_entry_includes_cpu_wall_ratio_ema() {
        let entry = make_entry(0.03, 0.30, 20);
        let summary = summarize_entry(&entry);
        assert!(
            (summary.cpu_wall_ratio_ema - 0.03).abs() < f64::EPSILON,
            "summary should include cpu_wall_ratio_ema"
        );
    }
}
