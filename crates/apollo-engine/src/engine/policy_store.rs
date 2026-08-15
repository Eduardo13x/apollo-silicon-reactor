//! Local policy persistence and bounded configuration loading.
//!
//! This module intentionally contains no prompt construction, model client,
//! API key, or free-form advice parser. Runtime policy changes come from
//! Apollo's measured local learning pipeline.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::engine::types::HardPath;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RepoConfig {
    #[serde(default)]
    pub history: Option<crate::engine::daemon_metrics_history::HistoryConfig>,
    #[serde(default)]
    pub reflex: ReflexConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReflexConfig {
    #[serde(default = "default_reflex_enabled")]
    pub enabled: bool,
    #[serde(default = "default_reflex_shadow_cycles")]
    pub shadow_cycles: u64,
}

impl Default for ReflexConfig {
    fn default() -> Self {
        Self {
            enabled: default_reflex_enabled(),
            shadow_cycles: default_reflex_shadow_cycles(),
        }
    }
}

impl ReflexConfig {
    pub fn effective_shadow_cycles(&self) -> u64 {
        self.shadow_cycles.max(500)
    }
}

fn default_reflex_enabled() -> bool {
    true
}

fn default_reflex_shadow_cycles() -> u64 {
    500
}

pub fn load_repo_config(path: &Path) -> RepoConfig {
    let data = match HardPath::read_to_string_limited(path, 1024 * 1024) {
        Ok(data) => data,
        Err(_) => return RepoConfig::default(),
    };
    toml::from_str(&data).unwrap_or_default()
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LearnedPolicy {
    /// Arc-backed lists make reads O(1); mutation remains copy-on-write.
    pub interactive_patterns: std::sync::Arc<Vec<String>>,
    pub noise_patterns: std::sync::Arc<Vec<String>>,
    pub protected_patterns: std::sync::Arc<Vec<String>>,
    pub learned_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub pattern_weights:
        std::collections::HashMap<String, crate::engine::outcome_tracker::PatternWeight>,
}

pub fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let data = HardPath::read_to_string_limited(path, 1024 * 1024).ok()?;
    serde_json::from_str(&data).ok()
}

/// Best-effort atomic JSON write. The committed file is old-or-new, never a
/// partially written document. `fsync` is reserved for crash-critical state.
pub fn write_json_fsync(path: &Path, value: &impl Serialize, mode: Option<u32>, fsync: bool) {
    let _ = HardPath::verify_no_symlink(path);
    if let Some(parent) = path.parent() {
        let _ = HardPath::secure_create_dir_all(parent);
        #[cfg(unix)]
        if unsafe { libc::getuid() } == 0 {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(parent, fs::Permissions::from_mode(0o700));
        }
    }
    if let Ok(json) = serde_json::to_string_pretty(value) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            let tmp_path = path.with_extension("tmp");
            let file_mode = mode.unwrap_or(0o644);
            if let Ok(mut file) = fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(file_mode)
                .open(&tmp_path)
            {
                let wrote = file.write_all(json.as_bytes()).is_ok();
                let synced = !fsync || file.sync_all().is_ok();
                if wrote && synced && fs::rename(&tmp_path, path).is_ok() {
                    return;
                }
                let _ = fs::remove_file(&tmp_path);
            }
        }
        let _ = fs::write(path, json);
    }
}

pub fn write_json(path: &Path, value: &impl Serialize, mode: Option<u32>) {
    write_json_fsync(path, value, mode, false);
}

pub fn write_json_critical(path: &Path, value: &impl Serialize, mode: Option<u32>) {
    write_json_fsync(path, value, mode, true);
}

/// Strict transactional checkpoint for irreplaceable learned state.
pub fn write_json_transactional(
    path: &Path,
    value: &impl Serialize,
    mode: Option<u32>,
) -> std::io::Result<()> {
    static CHECKPOINT_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    let _checkpoint_guard = CHECKPOINT_LOCK
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    fn io_other(error: impl std::fmt::Display) -> std::io::Error {
        std::io::Error::other(error.to_string())
    }

    HardPath::verify_no_symlink(path).map_err(io_other)?;
    let parent = path
        .parent()
        .ok_or_else(|| std::io::Error::other("checkpoint path has no parent"))?;
    HardPath::secure_create_dir_all(parent).map_err(io_other)?;
    let json = serde_json::to_vec_pretty(value).map_err(io_other)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| std::io::Error::other("checkpoint path has no UTF-8 file name"))?;
    let tmp_path = parent.join(format!(".{file_name}.new"));
    let previous_path = parent.join(format!("{file_name}.previous"));
    let previous_tmp = parent.join(format!(".{file_name}.previous.new"));
    for candidate in [&tmp_path, &previous_path, &previous_tmp] {
        HardPath::verify_no_symlink(candidate).map_err(io_other)?;
    }

    #[cfg(unix)]
    fn write_synced(path: &Path, bytes: &[u8], mode: u32) -> std::io::Result<()> {
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(mode)
            .open(path)?;
        file.write_all(bytes)?;
        file.sync_all()
    }
    #[cfg(not(unix))]
    fn write_synced(path: &Path, bytes: &[u8], _mode: u32) -> std::io::Result<()> {
        fs::write(path, bytes)
    }

    let mode = mode.unwrap_or(0o600);
    if path.exists() {
        if previous_tmp.exists() {
            fs::remove_file(&previous_tmp)?;
        }
        if fs::hard_link(path, &previous_tmp).is_err() {
            let previous = fs::read(path)?;
            write_synced(&previous_tmp, &previous, mode)?;
        }
        fs::rename(&previous_tmp, &previous_path)?;
    }
    if let Err(error) = write_synced(&tmp_path, &json, mode)
        .and_then(|()| fs::rename(&tmp_path, path))
        .and_then(|()| fs::File::open(parent)?.sync_all())
    {
        let _ = fs::remove_file(&tmp_path);
        let _ = fs::remove_file(&previous_tmp);
        return Err(error);
    }
    Ok(())
}

pub fn delete_file_best_effort(path: &Path) {
    if HardPath::verify_no_symlink(path).is_ok() {
        let _ = fs::remove_file(path);
    }
}

pub fn policy_path_root(is_root: bool) -> PathBuf {
    if is_root {
        PathBuf::from("/var/lib/apollo/learned_policy.json")
    } else {
        PathBuf::from("/tmp/apollo-learned_policy.json")
    }
}

pub fn pending_trial_path(is_root: bool) -> PathBuf {
    if is_root {
        PathBuf::from("/var/lib/apollo/pending_trial.json")
    } else {
        PathBuf::from("/tmp/apollo-pending_trial.json")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackEntry {
    pub at: DateTime<Utc>,
    pub rating: String,
    pub note: Option<String>,
}

pub fn append_jsonl(path: &Path, value: &impl Serialize) {
    if HardPath::verify_no_symlink(path).is_err() {
        return;
    }
    if let Some(parent) = path.parent() {
        if HardPath::secure_create_dir_all(parent).is_err() {
            return;
        }
    }
    #[cfg(unix)]
    let open_result = {
        use std::os::unix::fs::OpenOptionsExt;
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .open(path)
    };
    #[cfg(not(unix))]
    let open_result = fs::OpenOptions::new().create(true).append(true).open(path);
    if let Ok(mut file) = open_result {
        if let Ok(line) = serde_json::to_string(value) {
            let _ = writeln!(file, "{line}");
        }
    }
}

pub fn feedback_path_root(is_root: bool) -> PathBuf {
    if is_root {
        PathBuf::from("/var/lib/apollo/learn/feedback.jsonl")
    } else {
        PathBuf::from("/tmp/apollo-feedback.jsonl")
    }
}

/// Paths retained only so an upgraded daemon can delete old Teacher secrets.
pub fn legacy_teacher_paths(is_root: bool) -> (PathBuf, PathBuf) {
    if is_root {
        (
            PathBuf::from("/var/lib/apollo/llm_state.json"),
            PathBuf::from("/var/lib/apollo/llm_api_key"),
        )
    } else {
        (
            PathBuf::from("/tmp/apollo-llm_state.json"),
            PathBuf::from("/tmp/apollo-llm_api_key"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_llm_config_is_ignored_while_history_still_loads() {
        let config: RepoConfig = toml::from_str(
            r#"
[llm]
enabled = true
endpoint = "http://127.0.0.1:8080"

[history]
enabled = false
"#,
        )
        .expect("unknown legacy table must remain forward-compatible");
        assert_eq!(config.history.expect("history").enabled, Some(false));
    }

    #[test]
    fn learned_policy_json_roundtrips_without_teacher_state() {
        let policy = LearnedPolicy {
            interactive_patterns: std::sync::Arc::new(vec!["Editor".to_string()]),
            noise_patterns: std::sync::Arc::new(vec!["Updater".to_string()]),
            ..LearnedPolicy::default()
        };
        let json = serde_json::to_string(&policy).expect("serialize");
        assert!(!json.contains("suggestion"));
        assert!(!json.contains("prompt"));
        let restored: LearnedPolicy = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.interactive_patterns.as_slice(), ["Editor"]);
    }

    #[test]
    fn reflex_config_defaults_on_and_bounds_shadow_window() {
        let legacy: RepoConfig = toml::from_str("").expect("legacy config");
        assert!(legacy.reflex.enabled);
        assert_eq!(legacy.reflex.shadow_cycles, 500);

        let configured: RepoConfig = toml::from_str(
            r#"
[reflex]
enabled = false
shadow_cycles = 3
"#,
        )
        .expect("reflex config");
        assert!(!configured.reflex.enabled);
        assert_eq!(configured.reflex.effective_shadow_cycles(), 500);
    }
}
