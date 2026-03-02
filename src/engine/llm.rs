use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::collector::SystemSnapshot;
use crate::engine::types::{LatencyTarget, LlmRunMode, OptimizationProfile};

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RepoConfig {
    pub llm: Option<LlmConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    /// Master switch (in addition to key TTL state).
    pub enabled: Option<bool>,
    /// OpenAI-compatible chat completions endpoint.
    pub endpoint: Option<String>,
    pub model: Option<String>,
    pub min_confidence: Option<f64>,
    pub max_calls_per_hour: Option<u32>,
    pub min_interval_secs: Option<u64>,
    pub timeout_ms: Option<u64>,
    /// If true (default), request JSON-only output where supported.
    pub force_json: Option<bool>,
}

impl LlmConfig {
    pub fn enabled(&self) -> bool {
        self.enabled.unwrap_or(false)
    }
    pub fn endpoint(&self) -> String {
        self.endpoint
            .clone()
            .unwrap_or_else(|| "https://api.openai.com/v1/chat/completions".to_string())
    }
    pub fn model(&self) -> String {
        self.model
            .clone()
            .unwrap_or_else(|| "gpt-4.1-mini".to_string())
    }
    pub fn min_confidence(&self) -> f64 {
        self.min_confidence.unwrap_or(0.80)
    }
    pub fn max_calls_per_hour(&self) -> u32 {
        self.max_calls_per_hour.unwrap_or(2)
    }
    pub fn min_interval_secs(&self) -> u64 {
        self.min_interval_secs.unwrap_or(15 * 60)
    }
    pub fn timeout(&self) -> Duration {
        // Cloud LLM calls routinely take a few seconds; default to a conservative timeout.
        Duration::from_millis(self.timeout_ms.unwrap_or(5000))
    }

    pub fn force_json(&self) -> bool {
        self.force_json.unwrap_or(true)
    }
}

pub fn load_repo_config(path: &Path) -> RepoConfig {
    let data = match fs::read_to_string(path) {
        Ok(v) => v,
        Err(_) => return RepoConfig::default(),
    };
    toml::from_str(&data).unwrap_or_default()
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct LlmState {
    pub enabled: bool,
    pub training_started_at: Option<DateTime<Utc>>,
    pub training_expires_at: Option<DateTime<Utc>>,
    pub last_call_at: Option<DateTime<Utc>>,
    pub last_attempt_at: Option<DateTime<Utc>>,
    pub last_http_status: Option<u16>,
    pub last_error: Option<String>,
    pub last_trigger_reason: Option<String>,
    pub consecutive_failures: u32,
    pub hour_window_started_at: Option<DateTime<Utc>>,
    pub calls_in_window: u32,

    pub calls_today_day: Option<String>,
    pub calls_today: u32,
    pub mode: LlmRunMode,
    pub last_trigger_at: Option<DateTime<Utc>>,
    pub trigger_events: Vec<DateTime<Utc>>,
    pub no_trigger_since: Option<DateTime<Utc>>,

    pub last_suggestion: Option<LlmSuggestion>,
    pub policy_updates_day: Option<DateTime<Utc>>,
    pub policy_updates_today: u32,
}

impl LlmState {
    pub fn training_active(&self) -> bool {
        if !self.enabled {
            return false;
        }
        match self.training_expires_at {
            Some(t) => t > Utc::now(),
            None => false,
        }
    }
}

// Note: quiet-hours and mode-governing are implemented in the daemon (binary)
// so they can use local time and runtime heuristics.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmSuggestion {
    pub suggested_profile: Option<OptimizationProfile>,
    pub suggested_latency_target: Option<LatencyTarget>,
    pub add_interactive_patterns: Vec<String>,
    pub add_noise_patterns: Vec<String>,
    pub add_protected_patterns: Vec<String>,
    pub confidence: f64,
    pub rationale: String,
}

impl Default for LlmSuggestion {
    fn default() -> Self {
        Self {
            suggested_profile: None,
            suggested_latency_target: None,
            add_interactive_patterns: Vec::new(),
            add_noise_patterns: Vec::new(),
            add_protected_patterns: Vec::new(),
            confidence: 0.0,
            rationale: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LearnedPolicy {
    pub interactive_patterns: Vec<String>,
    pub noise_patterns: Vec<String>,
    pub protected_patterns: Vec<String>,
    pub learned_at: Option<DateTime<Utc>>,
}

pub fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let data = fs::read_to_string(path).ok()?;
    serde_json::from_str(&data).ok()
}

pub fn write_json(path: &Path, value: &impl Serialize, mode: Option<u32>) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(value) {
        let _ = fs::write(path, json);
        if let Some(mode) = mode {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = fs::set_permissions(path, fs::Permissions::from_mode(mode));
            }
        }
    }
}

pub fn write_secret(path: &Path, value: &str) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, value)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

pub fn delete_file_best_effort(path: &Path) {
    let _ = fs::remove_file(path);
}

pub fn state_paths_root(is_root: bool) -> (PathBuf, PathBuf) {
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

pub fn policy_path_root(is_root: bool) -> PathBuf {
    if is_root {
        PathBuf::from("/var/lib/apollo/learned_policy.json")
    } else {
        PathBuf::from("/tmp/apollo-learned_policy.json")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackEntry {
    pub at: DateTime<Utc>,
    pub rating: String,
    pub note: Option<String>,
}

pub fn append_jsonl(path: &Path, value: &impl Serialize) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(path) {
        if let Ok(line) = serde_json::to_string(value) {
            let _ = writeln!(f, "{}", line);
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

pub fn suggestions_path_root(is_root: bool) -> PathBuf {
    if is_root {
        PathBuf::from("/var/lib/apollo/learn/suggestions.jsonl")
    } else {
        PathBuf::from("/tmp/apollo-suggestions.jsonl")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiChatRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<OpenAiResponseFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiResponseFormat {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiChatResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiChoice {
    message: OpenAiChoiceMessage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiChoiceMessage {
    content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LlmWireResponse {
    #[serde(default)]
    suggest_profile: Option<String>,
    #[serde(default)]
    suggest_latency_target: Option<String>,
    #[serde(default)]
    suggest_lists: Option<LlmWireLists>,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    rationale: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct LlmWireLists {
    #[serde(default)]
    add_interactive_patterns: Vec<String>,
    #[serde(default)]
    add_noise_patterns: Vec<String>,
    #[serde(default)]
    add_protected_patterns: Vec<String>,
}

fn parse_profile(s: &str) -> Option<OptimizationProfile> {
    match s {
        "balanced-root" => Some(OptimizationProfile::BalancedRoot),
        "aggressive-root" => Some(OptimizationProfile::AggressiveRoot),
        "safe-root" => Some(OptimizationProfile::SafeRoot),
        _ => None,
    }
}

fn parse_latency_target(s: &str) -> Option<LatencyTarget> {
    match s {
        "low" => Some(LatencyTarget::Low),
        "normal" => Some(LatencyTarget::Normal),
        "max" => Some(LatencyTarget::Max),
        _ => None,
    }
}

fn sanitize_pattern_list(mut v: Vec<String>, max: usize) -> Vec<String> {
    v.retain(|s| {
        let s = s.trim();
        !s.is_empty() && s.len() <= 80 && !s.contains('\n') && !s.contains('\r')
    });
    v.truncate(max);
    v
}

pub struct LlmAdvisor {
    cfg: LlmConfig,
    last_attempt: Option<Instant>,
}

impl LlmAdvisor {
    pub fn new(cfg: LlmConfig) -> Self {
        Self {
            cfg,
            last_attempt: None,
        }
    }

    pub fn update_cfg(&mut self, cfg: LlmConfig) {
        self.cfg = cfg;
    }

    pub fn call_raw(
        &mut self,
        snapshot: &SystemSnapshot,
        api_key: &str,
    ) -> Result<LlmSuggestion, LlmCallError> {
        // Extra guard: don't try too frequently on repeated failures.
        if let Some(last) = self.last_attempt {
            if last.elapsed() < Duration::from_secs(20) {
                return Err(LlmCallError::Cooldown);
            }
        }
        self.last_attempt = Some(Instant::now());

        let summary = build_summary(snapshot);
        let system_prompt = r#"You are an optimization advisor for a macOS system optimizer daemon.

Return ONLY valid JSON with this shape:
{
  "suggest_profile": "balanced-root"|"aggressive-root"|"safe-root"|null,
  "suggest_latency_target": "low"|"normal"|"max"|null,
  "suggest_lists": {
    "add_interactive_patterns": ["..."],
    "add_noise_patterns": ["..."],
    "add_protected_patterns": ["..."]
  },
  "confidence": 0.0,
  "rationale": "short reason"
}

Constraints:
- Do NOT suggest touching Spotlight stack (mds/mdworker/mds_stores/Spotlight).
- Keep pattern strings short substrings; no regex.
- If unsure, set suggestions to null and confidence low.
"#;

        let user_prompt = format!(
            "SystemSummary:\n{}\n\nGoal: maximize perceived responsiveness and stability.",
            summary
        );

        let req = OpenAiChatRequest {
            model: self.cfg.model(),
            messages: vec![
                OpenAiMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                },
                OpenAiMessage {
                    role: "user".to_string(),
                    content: user_prompt,
                },
            ],
            temperature: 0.1,
            response_format: if self.cfg.force_json() {
                Some(OpenAiResponseFormat {
                    kind: "json_object".to_string(),
                })
            } else {
                None
            },
        };

        let timeout = self.cfg.timeout();
        let payload = serde_json::to_value(&req)
            .map_err(|e| LlmCallError::Parse(format!("request serialize: {}", e)))?;
        let response = ureq::AgentBuilder::new()
            .timeout_connect(timeout)
            .timeout_read(timeout)
            .timeout_write(timeout)
            .build()
            .post(&self.cfg.endpoint())
            .set("Authorization", &format!("Bearer {}", api_key.trim()))
            .set("Content-Type", "application/json")
            .send_json(payload);

        let response = match response {
            Ok(r) => r,
            Err(ureq::Error::Status(code, r)) => {
                let body = r.into_string().ok();
                return Err(LlmCallError::HttpStatus {
                    code: code as u16,
                    body_excerpt: body.as_deref().map(excerpt_200),
                });
            }
            Err(e) => return Err(LlmCallError::Transport(e.to_string())),
        };

        let response_text = response
            .into_string()
            .map_err(|e| LlmCallError::Transport(e.to_string()))?;

        let parsed: OpenAiChatResponse = serde_json::from_str(&response_text)
            .map_err(|e| LlmCallError::Parse(format!("chat response parse: {}", e)))?;
        let content = parsed
            .choices
            .first()
            .ok_or_else(|| LlmCallError::Parse("no choices".to_string()))?
            .message
            .content
            .trim()
            .to_string();

        let json = extract_first_json_object(&content)
            .ok_or_else(|| LlmCallError::Parse("no json object in model content".to_string()))?;

        let wire: LlmWireResponse = serde_json::from_str(&json)
            .map_err(|e| LlmCallError::Parse(format!("suggestion parse: {}", e)))?;
        let lists = wire.suggest_lists.unwrap_or_default();
        let s = LlmSuggestion {
            suggested_profile: wire.suggest_profile.as_deref().and_then(parse_profile),
            suggested_latency_target: wire
                .suggest_latency_target
                .as_deref()
                .and_then(parse_latency_target),
            add_interactive_patterns: sanitize_pattern_list(lists.add_interactive_patterns, 6),
            add_noise_patterns: sanitize_pattern_list(lists.add_noise_patterns, 6),
            add_protected_patterns: sanitize_pattern_list(lists.add_protected_patterns, 6),
            confidence: wire.confidence.unwrap_or(0.0).clamp(0.0, 1.0),
            rationale: wire.rationale.unwrap_or_default(),
        };

        // Hard guard: never accept Spotlight stack patterns.
        let spotlight = ["mds", "mdworker", "mds_stores", "Spotlight"];
        if s.add_interactive_patterns
            .iter()
            .chain(s.add_noise_patterns.iter())
            .chain(s.add_protected_patterns.iter())
            .any(|p| spotlight.iter().any(|sp| p.contains(sp)))
        {
            return Err(LlmCallError::Rejected("forbidden pattern".to_string()));
        }

        Ok(s)
    }
}

#[derive(Debug, Clone)]
pub enum LlmCallError {
    Cooldown,
    HttpStatus {
        code: u16,
        body_excerpt: Option<String>,
    },
    Transport(String),
    Parse(String),
    Rejected(String),
}

fn excerpt_200(s: &str) -> String {
    let mut out = s.trim().to_string();
    if out.len() > 200 {
        out.truncate(200);
    }
    out
}

fn extract_first_json_object(s: &str) -> Option<String> {
    let t = s.trim();
    let mut start = None;
    let mut depth: i32 = 0;
    let mut in_str = false;
    let mut escape = false;
    for (i, c) in t.char_indices() {
        if in_str {
            if escape {
                escape = false;
                continue;
            }
            if c == '\\' {
                escape = true;
                continue;
            }
            if c == '"' {
                in_str = false;
            }
            continue;
        }

        if c == '"' {
            in_str = true;
            continue;
        }
        if c == '{' {
            if start.is_none() {
                start = Some(i);
            }
            depth += 1;
        } else if c == '}' {
            depth -= 1;
            if depth == 0 {
                if let Some(st) = start {
                    return Some(t[st..=i].to_string());
                }
            }
        }
    }
    None
}

fn build_summary(snapshot: &SystemSnapshot) -> String {
    #[derive(Serialize)]
    struct Summary<'a> {
        cpu_global: f32,
        mem_pressure: f64,
        swap_used_bytes: u64,
        swap_delta_bps: f64,
        thermal_level: &'a str,
        top_processes: &'a [crate::collector::ProcessStats],
    }
    let s = Summary {
        cpu_global: snapshot.cpu.global_usage,
        mem_pressure: snapshot.pressure.memory_pressure,
        swap_used_bytes: snapshot.pressure.swap_used_bytes,
        swap_delta_bps: snapshot.pressure.swap_delta_bytes_per_sec,
        thermal_level: &snapshot.pressure.thermal_level,
        top_processes: &snapshot.top_processes,
    };
    serde_json::to_string_pretty(&s).unwrap_or_default()
}
