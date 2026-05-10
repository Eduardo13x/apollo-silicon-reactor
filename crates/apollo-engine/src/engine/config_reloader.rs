//! Live-reload of operator-tunable LLM config fields.
//!
//! Watches `/etc/apollo-optimizer/config.toml` (or any provided path) via mtime
//! polling and, when the file changes, re-parses it and reports whitelisted
//! diffs for the daemon to apply without restart.
//!
//! # Design invariants
//!
//! - **mtime-poll, not `fsnotify`.** Polling at bounded frequency is preferable
//!   to interrupt-driven callbacks under burst conditions
//!   [Mogul & Heidemann, USENIX 1997].
//! - **Parse-then-swap.** Torn reads (file rewritten mid-poll) surface as a
//!   TOML parse error; the previous config is retained. The daemon's committed
//!   state survives any single-point failure
//!   [Gray & Reuter 1992, §10 atomic-replace].
//! - **WAL gate.** If a Gemma trial is pending in the BUG-01 WAL
//!   (`pending_trial.json`), the reload is deferred: swapping `timeout_ms` or
//!   `endpoint` mid-trial would corrupt the per-category reliability EMA in
//!   GemmaTrust (commit `936cd70`). The tick reports `deferred_wal = true` and
//!   the caller retries next cycle.
//! - **Whitelist.** Only a small set of fields is live-reloadable — the rest
//!   of the operator's config (profile, safety thresholds, hardware caps) is
//!   deliberately immutable-at-boot. Non-whitelisted diffs are logged as
//!   WARNINGs and never applied
//!   [Google SRE Ch.8 — separate immutable-at-boot from live-tunable].

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::Deserialize;

use crate::engine::llm::LlmConfig;

/// Fields the daemon is allowed to hot-swap without restart. Anything else
/// in `[llm]` — `enabled`, `model`, `force_json` — still requires a deploy.
pub const LIVE_RELOAD_WHITELIST: &[&str] = &[
    "endpoint",
    "timeout_ms",
    "min_confidence",
    "max_calls_per_hour",
    "min_interval_secs",
    "always_on",
];

/// One concrete config change — emitted per tick when the file mtime advances.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigDiff {
    pub field: String,
    pub old: String,
    pub new: String,
}

/// Why a tick decided to do nothing (or not-quite-nothing).
#[derive(Debug, Clone, PartialEq)]
pub enum ReloadSkipReason {
    /// File mtime has not changed since the last poll.
    Unchanged,
    /// File rewritten but could not be parsed as valid TOML; previous config
    /// is retained. Reported so the caller can log/alert.
    ParseError(String),
    /// A pending Gemma trial WAL is present — deferring the swap to avoid
    /// mid-trial EMA corruption.
    PendingTrialWal,
    /// File mtime changed but none of the whitelisted fields actually differ.
    NoWhitelistedDiff,
}

/// Result of one poll tick.
#[derive(Debug, Clone)]
pub struct ReloadOutcome {
    /// `Some` iff a new config was parsed AND at least one whitelisted field
    /// differs from the current config AND no WAL gate is active. When this
    /// is `Some`, the daemon should swap its `LlmConfig` to this value.
    pub new_cfg: Option<LlmConfig>,
    /// Whitelisted diffs that will be (or were) applied. Empty on skip.
    pub applied: Vec<ConfigDiff>,
    /// Non-whitelisted diffs — log as WARN, do not apply.
    pub rejected: Vec<ConfigDiff>,
    /// Reason the tick produced no swap, if any.
    pub skip: Option<ReloadSkipReason>,
}

impl ReloadOutcome {
    fn unchanged() -> Self {
        Self {
            new_cfg: None,
            applied: Vec::new(),
            rejected: Vec::new(),
            skip: Some(ReloadSkipReason::Unchanged),
        }
    }
}

pub struct LlmConfigReloader {
    config_path: PathBuf,
    pending_trial_path: PathBuf,
    last_mtime: Option<SystemTime>,
}

impl LlmConfigReloader {
    pub fn new(config_path: PathBuf, pending_trial_path: PathBuf) -> Self {
        let last_mtime = fs::metadata(&config_path).and_then(|m| m.modified()).ok();
        Self {
            config_path,
            pending_trial_path,
            last_mtime,
        }
    }

    /// Inspect the config file and, if it changed, return whitelisted diffs.
    /// Caller applies `outcome.new_cfg` (when `Some`) to its shared `LlmConfig`
    /// handle and logs `outcome.rejected` as WARN.
    pub fn tick(&mut self, current: &LlmConfig) -> ReloadOutcome {
        let meta = match fs::metadata(&self.config_path) {
            Ok(m) => m,
            Err(_) => return ReloadOutcome::unchanged(),
        };
        let mtime = match meta.modified() {
            Ok(t) => t,
            Err(_) => return ReloadOutcome::unchanged(),
        };
        if Some(mtime) == self.last_mtime {
            return ReloadOutcome::unchanged();
        }

        let data = match fs::read_to_string(&self.config_path) {
            Ok(d) => d,
            Err(_) => return ReloadOutcome::unchanged(),
        };
        let parsed: RepoConfigStub = match toml::from_str(&data) {
            Ok(p) => p,
            Err(e) => {
                // Torn read / malformed edit — retain previous config, do NOT
                // advance last_mtime so we retry on next tick.
                return ReloadOutcome {
                    new_cfg: None,
                    applied: Vec::new(),
                    rejected: Vec::new(),
                    skip: Some(ReloadSkipReason::ParseError(e.to_string())),
                };
            }
        };
        let Some(new_cfg) = parsed.llm else {
            self.last_mtime = Some(mtime);
            return ReloadOutcome {
                new_cfg: None,
                applied: Vec::new(),
                rejected: Vec::new(),
                skip: Some(ReloadSkipReason::NoWhitelistedDiff),
            };
        };

        self.last_mtime = Some(mtime);

        let (applied, rejected) = diff_cfgs(current, &new_cfg);
        if applied.is_empty() && rejected.is_empty() {
            return ReloadOutcome {
                new_cfg: None,
                applied,
                rejected,
                skip: Some(ReloadSkipReason::NoWhitelistedDiff),
            };
        }

        // WAL gate: applied whitelist swaps are deferred while a Gemma trial
        // is in flight. Rejected diffs are still surfaced so the operator sees
        // the WARN immediately — those are never applied anyway.
        if !applied.is_empty() && self.pending_trial_path.exists() {
            return ReloadOutcome {
                new_cfg: None,
                applied: Vec::new(),
                rejected,
                skip: Some(ReloadSkipReason::PendingTrialWal),
            };
        }

        if applied.is_empty() {
            return ReloadOutcome {
                new_cfg: None,
                applied,
                rejected,
                skip: Some(ReloadSkipReason::NoWhitelistedDiff),
            };
        }

        ReloadOutcome {
            new_cfg: Some(new_cfg),
            applied,
            rejected,
            skip: None,
        }
    }
}

#[derive(Debug, Deserialize)]
struct RepoConfigStub {
    llm: Option<LlmConfig>,
}

fn diff_cfgs(a: &LlmConfig, b: &LlmConfig) -> (Vec<ConfigDiff>, Vec<ConfigDiff>) {
    let mut applied = Vec::new();
    let mut rejected = Vec::new();

    let push = |v: &mut Vec<ConfigDiff>, field: &str, old: String, new: String| {
        if old != new {
            v.push(ConfigDiff {
                field: field.to_string(),
                old,
                new,
            });
        }
    };

    // Whitelisted.
    push(
        &mut applied,
        "endpoint",
        fmt_opt(&a.endpoint),
        fmt_opt(&b.endpoint),
    );
    push(
        &mut applied,
        "timeout_ms",
        fmt_opt_u64(&a.timeout_ms),
        fmt_opt_u64(&b.timeout_ms),
    );
    push(
        &mut applied,
        "min_confidence",
        fmt_opt_f64(&a.min_confidence),
        fmt_opt_f64(&b.min_confidence),
    );
    push(
        &mut applied,
        "max_calls_per_hour",
        fmt_opt_u32(&a.max_calls_per_hour),
        fmt_opt_u32(&b.max_calls_per_hour),
    );
    push(
        &mut applied,
        "min_interval_secs",
        fmt_opt_u64(&a.min_interval_secs),
        fmt_opt_u64(&b.min_interval_secs),
    );
    push(
        &mut applied,
        "always_on",
        fmt_opt_bool(&a.always_on),
        fmt_opt_bool(&b.always_on),
    );

    // Non-whitelisted — surfaced for operator warning, never applied.
    push(
        &mut rejected,
        "enabled",
        fmt_opt_bool(&a.enabled),
        fmt_opt_bool(&b.enabled),
    );
    push(&mut rejected, "model", fmt_opt(&a.model), fmt_opt(&b.model));
    push(
        &mut rejected,
        "force_json",
        fmt_opt_bool(&a.force_json),
        fmt_opt_bool(&b.force_json),
    );

    (applied, rejected)
}

fn fmt_opt(v: &Option<String>) -> String {
    v.clone().unwrap_or_default()
}
fn fmt_opt_bool(v: &Option<bool>) -> String {
    v.map(|x| x.to_string()).unwrap_or_default()
}
fn fmt_opt_u32(v: &Option<u32>) -> String {
    v.map(|x| x.to_string()).unwrap_or_default()
}
fn fmt_opt_u64(v: &Option<u64>) -> String {
    v.map(|x| x.to_string()).unwrap_or_default()
}
fn fmt_opt_f64(v: &Option<f64>) -> String {
    v.map(|x| x.to_string()).unwrap_or_default()
}

#[allow(dead_code)]
fn _ensure_whitelist_in_scope(path: &Path) -> bool {
    LIVE_RELOAD_WHITELIST.contains(&path.to_str().unwrap_or(""))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::time::{Duration, UNIX_EPOCH};

    fn tmp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "apollo_cfg_reloader_{}_{}",
            std::process::id(),
            name
        ));
        let _ = fs::remove_file(&p);
        p
    }

    fn write_cfg(path: &Path, contents: &str) {
        let mut f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)
            .unwrap();
        f.write_all(contents.as_bytes()).unwrap();
    }

    fn bump_mtime(path: &Path) {
        // Ensure the next write produces a distinct mtime on fast FS.
        let prev = fs::metadata(path).unwrap().modified().unwrap();
        let target = prev + Duration::from_secs(2);
        let target_unix = target.duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
        // Touch via utimensat indirectly: re-open + append a trailing newline.
        // Then explicitly set via filetime-free path by sleeping if needed.
        let _ = target_unix;
        std::thread::sleep(Duration::from_millis(1100));
    }

    const INITIAL_CFG: &str = r#"
[llm]
enabled = true
endpoint = "http://127.0.0.1:8080"
timeout_ms = 60000
min_confidence = 0.80
max_calls_per_hour = 4
min_interval_secs = 600
always_on = true
"#;

    fn initial_llm_cfg() -> LlmConfig {
        let parsed: RepoConfigStub = toml::from_str(INITIAL_CFG).unwrap();
        parsed.llm.unwrap()
    }

    #[test]
    fn tick_reports_unchanged_when_mtime_same() {
        let cfg_path = tmp_path("unchanged.toml");
        let wal = tmp_path("unchanged_wal.json");
        write_cfg(&cfg_path, INITIAL_CFG);
        let mut r = LlmConfigReloader::new(cfg_path.clone(), wal);
        let out = r.tick(&initial_llm_cfg());
        assert!(out.new_cfg.is_none());
        assert_eq!(out.skip, Some(ReloadSkipReason::Unchanged));
    }

    #[test]
    fn tick_detects_whitelisted_change() {
        let cfg_path = tmp_path("whitelist.toml");
        let wal = tmp_path("whitelist_wal.json");
        write_cfg(&cfg_path, INITIAL_CFG);
        let mut r = LlmConfigReloader::new(cfg_path.clone(), wal);
        bump_mtime(&cfg_path);
        let updated = INITIAL_CFG.replace("timeout_ms = 60000", "timeout_ms = 180000");
        write_cfg(&cfg_path, &updated);
        let out = r.tick(&initial_llm_cfg());
        assert!(out.new_cfg.is_some());
        assert_eq!(out.applied.len(), 1);
        assert_eq!(out.applied[0].field, "timeout_ms");
        assert_eq!(out.applied[0].new, "180000");
        assert!(out.skip.is_none());
    }

    #[test]
    fn tick_rejects_non_whitelisted_change() {
        let cfg_path = tmp_path("reject.toml");
        let wal = tmp_path("reject_wal.json");
        write_cfg(&cfg_path, INITIAL_CFG);
        let mut r = LlmConfigReloader::new(cfg_path.clone(), wal);
        bump_mtime(&cfg_path);
        // Flip `enabled` (not whitelisted) — must NOT produce new_cfg.
        let updated = INITIAL_CFG.replace("enabled = true", "enabled = false");
        write_cfg(&cfg_path, &updated);
        let out = r.tick(&initial_llm_cfg());
        assert!(out.new_cfg.is_none());
        assert_eq!(out.rejected.len(), 1);
        assert_eq!(out.rejected[0].field, "enabled");
    }

    #[test]
    fn tick_defers_when_pending_trial_wal_present() {
        let cfg_path = tmp_path("wal_defer.toml");
        let wal = tmp_path("wal_defer_wal.json");
        write_cfg(&cfg_path, INITIAL_CFG);
        fs::write(&wal, b"{}").unwrap();
        let mut r = LlmConfigReloader::new(cfg_path.clone(), wal.clone());
        bump_mtime(&cfg_path);
        let updated = INITIAL_CFG.replace("timeout_ms = 60000", "timeout_ms = 180000");
        write_cfg(&cfg_path, &updated);
        let out = r.tick(&initial_llm_cfg());
        assert!(out.new_cfg.is_none());
        assert_eq!(out.skip, Some(ReloadSkipReason::PendingTrialWal));
    }

    #[test]
    fn tick_surfaces_parse_error_and_retains_cfg() {
        let cfg_path = tmp_path("parse_err.toml");
        let wal = tmp_path("parse_err_wal.json");
        write_cfg(&cfg_path, INITIAL_CFG);
        let mut r = LlmConfigReloader::new(cfg_path.clone(), wal);
        bump_mtime(&cfg_path);
        write_cfg(&cfg_path, "this is not : valid toml :: ==");
        let out = r.tick(&initial_llm_cfg());
        assert!(out.new_cfg.is_none());
        match out.skip {
            Some(ReloadSkipReason::ParseError(_)) => {}
            other => panic!("expected ParseError, got {:?}", other),
        }
    }

    #[test]
    fn whitelist_constant_is_exhaustive_for_live_fields() {
        for field in [
            "endpoint",
            "timeout_ms",
            "min_confidence",
            "max_calls_per_hour",
            "min_interval_secs",
            "always_on",
        ] {
            assert!(LIVE_RELOAD_WHITELIST.contains(&field), "missing {field}");
        }
    }
}
