use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::collector::SystemSnapshot;
use crate::engine::types::{HardPath, LatencyTarget, LlmRunMode, OptimizationProfile};

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RepoConfig {
    pub llm: Option<LlmConfig>,
}

#[derive(Debug, Clone, Deserialize, Default)]
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
    /// If true, bypass the training-TTL gate and auto-enable for local endpoints (Gemma 4, Ollama).
    pub always_on: Option<bool>,
}

impl LlmConfig {
    pub fn enabled(&self) -> bool {
        self.enabled.unwrap_or(false)
    }
    pub fn always_on(&self) -> bool {
        self.always_on.unwrap_or(false)
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
    let data = match HardPath::read_to_string_limited(path, 1024 * 1024) {
        Ok(s) => s,
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

    // ── Feedback loop: rastreo del resultado de sugerencias de Gemma ──────
    /// Presión en el momento en que se aplicó la última sugerencia (baseline).
    #[serde(default)]
    pub pending_outcome_pressure: Option<f64>,
    /// Timestamp cuando se aplicó la sugerencia (medir delta 30s después).
    #[serde(default)]
    pub pending_outcome_at: Option<DateTime<Utc>>,
    /// Snippet del rationale de la sugerencia pendiente.
    #[serde(default)]
    pub pending_outcome_rationale: Option<String>,
    /// Resultado medido de la última sugerencia — cerrado el loop.
    #[serde(default)]
    pub last_suggestion_outcome: Option<SuggestionOutcome>,
    /// Protected patterns added by the pending suggestion — used to revert on WORSENED outcome.
    #[serde(default)]
    pub pending_added_protected: Vec<String>,
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

/// Resultado medido de una sugerencia de Gemma — cierra el loop de feedback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestionOutcome {
    pub applied_at: DateTime<Utc>,
    pub pressure_before: f64,
    pub pressure_after: f64,
    /// Negativo = presión mejoró. Positivo = empeoró.
    pub pressure_delta: f64,
    /// Primeras 80 chars del rationale de Gemma para contexto.
    pub rationale_snippet: String,
}

/// Contexto rico que Apollo pasa al LLM teacher en cada llamada.
/// Contiene todo lo que Apollo ha aprendido sobre el sistema.
pub struct TeacherContext<'a> {
    /// Scores Bayesianos por proceso: (nombre, throttle_count, effectiveness 0–1).
    /// Solo incluye patrones con ≥3 throttles (señal estadística suficiente).
    pub pattern_scores: &'a [(String, u32, f64)],
    /// Resultado medido de la sugerencia anterior de Gemma (si existe).
    pub previous_outcome: Option<&'a SuggestionOutcome>,
    /// El OutcomeTracker detectó que el heurístico está fallando.
    pub heuristic_struggling: bool,
    /// Procesos actualmente congelados por Apollo.
    pub frozen_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LearnedPolicy {
    /// Arc-wrapped pattern Vecs eliminate O(N) deep clone in hot daemon read
    /// sites (main.rs:2742, daemon_dispatch_tick.rs:292). Reads = Arc refcount
    /// bump (O(1)). Mutations via Arc::make_mut (clone-on-write semantics).
    /// serde "rc" feature enables transparent Vec<String> round-trip on disk.
    pub interactive_patterns: std::sync::Arc<Vec<String>>,
    pub noise_patterns: std::sync::Arc<Vec<String>>,
    pub protected_patterns: std::sync::Arc<Vec<String>>,
    pub learned_at: Option<DateTime<Utc>>,
    /// Pesos Bayesianos por proceso: cuántas veces se throttleó y cuántas fue efectivo.
    /// Backward-compatible: campo opcional, deserializa a HashMap vacío si ausente.
    #[serde(default)]
    pub pattern_weights:
        std::collections::HashMap<String, crate::engine::outcome_tracker::PatternWeight>,
}

pub fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let data = HardPath::read_to_string_limited(path, 1024 * 1024).ok()?;
    serde_json::from_str(&data).ok()
}

/// Write JSON atomically (temp → rename). `fsync` controls whether to call
/// `sync_all()` before rename. Use `true` only for crash-critical files
/// (journal, learned_state) — it adds ~5-30ms per write via F_FULLFSYNC.
pub fn write_json_fsync(path: &Path, value: &impl Serialize, mode: Option<u32>, fsync: bool) {
    let _ = HardPath::verify_no_symlink(path);

    if let Some(parent) = path.parent() {
        let _ = HardPath::secure_create_dir_all(parent);
        // Restrict parent directory to root-only if we're root.
        #[cfg(unix)]
        if unsafe { libc::getuid() } == 0 {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(parent, fs::Permissions::from_mode(0o700));
        }
    }
    if let Ok(json) = serde_json::to_string_pretty(value) {
        // Atomic write: temp file → fsync → rename.
        // rename() on the same filesystem is atomic in POSIX, so a crash mid-write
        // leaves the old file intact rather than a truncated/empty file.
        #[cfg(unix)]
        {
            use std::io::Write as _;
            use std::os::unix::fs::OpenOptionsExt;
            let tmp_path = path.with_extension("tmp");
            let m = mode.unwrap_or(0o644);
            if let Ok(mut f) = fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(m)
                .open(&tmp_path)
            {
                let wrote = f.write_all(json.as_bytes()).is_ok();
                let synced = !fsync || f.sync_all().is_ok();
                if wrote && synced {
                    if fs::rename(&tmp_path, path).is_ok() {
                        return;
                    }
                }
                // Cleanup temp on failure.
                let _ = fs::remove_file(&tmp_path);
            }
        }
        // Fallback for non-unix or if atomic write failed.
        let _ = fs::write(path, json);
    }
}

/// Atomic write without fsync — for non-critical state files (wake_state,
/// governor_state, metrics, profile). Fast path: no F_FULLFSYNC syscall.
pub fn write_json(path: &Path, value: &impl Serialize, mode: Option<u32>) {
    write_json_fsync(path, value, mode, false);
}

/// Atomic write with fsync — for crash-critical files (learned_state, journal).
pub fn write_json_critical(path: &Path, value: &impl Serialize, mode: Option<u32>) {
    write_json_fsync(path, value, mode, true);
}

pub fn write_secret(path: &Path, value: &str) -> anyhow::Result<()> {
    HardPath::verify_no_symlink(path)?;

    if let Some(parent) = path.parent() {
        HardPath::secure_create_dir_all(parent)?;
    }
    #[cfg(unix)]
    {
        use std::fs::OpenOptions;
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)?;
        f.write_all(value.as_bytes())?;
    }
    #[cfg(not(unix))]
    {
        fs::write(path, value)?;
    }
    Ok(())
}

pub fn delete_file_best_effort(path: &Path) {
    if HardPath::verify_no_symlink(path).is_ok() {
        let _ = fs::remove_file(path);
    }
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

/// Write-ahead log for pending_trial_skill — survives daemon crash (BUG-01).
/// Written immediately when a skill trial is registered; deleted on resolution.
/// [Gray & Reuter 1992 §11 — crash recovery via write-ahead]
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
    if let Ok(mut f) = open_result {
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
        policy: Option<&LearnedPolicy>,
        teacher: Option<&TeacherContext<'_>>,
    ) -> Result<LlmSuggestion, LlmCallError> {
        // Extra guard: don't try too frequently on repeated failures.
        if let Some(last) = self.last_attempt {
            if last.elapsed() < Duration::from_secs(20) {
                return Err(LlmCallError::Cooldown);
            }
        }
        self.last_attempt = Some(Instant::now());

        let summary = build_summary(snapshot, teacher);
        let policy_context = policy.map(build_policy_context).unwrap_or_default();
        let system_prompt = r#"You are an optimization advisor for a macOS system optimizer daemon.

Return ONLY valid JSON (no markdown, no ```). Shape:
{
  "suggest_profile": "balanced-root"|"aggressive-root"|"safe-root"|null,
  "suggest_latency_target": "low"|"normal"|"max"|null,
  "suggest_lists": {
    "add_interactive_patterns": [],
    "add_noise_patterns": [],
    "add_protected_patterns": []
  },
  "confidence": 0.0,
  "rationale": "short reason"
}

HARD RULES (violation = rejected):
1. NEVER add a process to a category where it ALREADY belongs (e.g. don't add "Brave" to interactive if it's already there). You MAY move a process from noise→protected if PatternEffectiveness shows it is NOT causing pressure.
2. NEVER put the same process in both noise and protected in the same response. Pick one.
3. NEVER suggest Spotlight stack (mds/mdworker/mds_stores/Spotlight).
4. Keep pattern strings as short substrings, no regex.

GUIDANCE:
- If PatternEffectiveness shows effectiveness < 0.30 → that process does NOT cause pressure. Suggest it for protected (not noise).
- If PreviousGemmaSuggestion outcome was WORSENED or NO_EFFECT → revise your strategy for this call.
- If HEURISTIC_STRUGGLING=true → Apollo's rules are failing. Be decisive.
- If nothing new to suggest, return empty lists with confidence 0.80 and rationale explaining why current policy is sufficient.
- Confidence should reflect how sure you are. 0.70+ means you have clear evidence. Do not default to 0.65.
"#;

        let user_prompt = format!(
            "SystemSummary:\n{}\n\n{}\nGoal: maximize perceived responsiveness and stability.",
            summary, policy_context
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
        let endpoint = self.cfg.endpoint();
        // Allow HTTP only for loopback endpoints (Ollama, llama.cpp, LM Studio, etc.).
        // Remote endpoints must use HTTPS to protect the API key.
        let is_loopback = endpoint.starts_with("http://localhost")
            || endpoint.starts_with("http://127.0.0.1")
            || endpoint.starts_with("http://[::1]");
        if !endpoint.starts_with("https://") && !is_loopback {
            return Err(LlmCallError::Rejected(
                "LLM endpoint must use HTTPS to protect API key".to_string(),
            ));
        }
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
                    code,
                    body_excerpt: body.as_deref().map(excerpt_200),
                });
            }
            Err(e) => return Err(LlmCallError::Transport(e.to_string())),
        };

        let mut response_text = String::new();
        response
            .into_reader()
            .take(1024 * 1024) // 1MB limit for LLM response
            .read_to_string(&mut response_text)
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
            rationale: {
                let mut r = wire.rationale.unwrap_or_default();
                if r.len() > 1024 {
                    r.truncate(1024);
                }
                r
            },
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

fn build_policy_context(policy: &LearnedPolicy) -> String {
    // Truncate lists to keep prompt short for small local models.
    fn join_truncated(v: &[String], max: usize) -> String {
        let slice = if v.len() > max { &v[..max] } else { v };
        slice.join(", ")
    }
    let low_value: Vec<String> = policy
        .pattern_weights
        .iter()
        .filter(|(_, w)| w.is_low_value())
        .map(|(name, _)| name.clone())
        .collect();
    let mut out = String::from("CurrentPolicy (already classified — do NOT re-add these):\n");
    if !policy.interactive_patterns.is_empty() {
        out.push_str(&format!(
            "  interactive: {}\n",
            join_truncated(&policy.interactive_patterns, 20)
        ));
    }
    if !policy.noise_patterns.is_empty() {
        out.push_str(&format!(
            "  noise: {}\n",
            join_truncated(&policy.noise_patterns, 20)
        ));
    }
    if !policy.protected_patterns.is_empty() {
        out.push_str(&format!(
            "  protected: {}\n",
            join_truncated(&policy.protected_patterns, 20)
        ));
    }
    if !low_value.is_empty() {
        out.push_str(&format!(
            "  low-value (throttled ≥5×, zero effect — move to protected or leave alone): {}\n",
            join_truncated(&low_value, 15)
        ));
    }
    out
}

fn build_summary(snapshot: &SystemSnapshot, teacher: Option<&TeacherContext<'_>>) -> String {
    #[derive(Serialize)]
    struct Summary<'a> {
        cpu_global: f32,
        mem_pressure: f64,
        compressor_pressure: f64,
        swap_used_bytes: u64,
        swap_delta_bps: f64,
        thermal_level: &'a str,
        frozen_count: usize,
        top_processes: &'a [crate::collector::ProcessStats],
    }
    // Cap top_processes to 8 — keeps prompt under ~600 tokens for local models.
    let proc_slice = if snapshot.top_processes.len() > 8 {
        &snapshot.top_processes[..8]
    } else {
        &snapshot.top_processes
    };
    let frozen_count = teacher.map(|t| t.frozen_count).unwrap_or(0);
    let s = Summary {
        cpu_global: snapshot.cpu.global_usage,
        mem_pressure: snapshot.pressure.memory_pressure,
        compressor_pressure: snapshot.pressure.compressor_pressure,
        swap_used_bytes: snapshot.pressure.swap_used_bytes,
        swap_delta_bps: snapshot.pressure.swap_delta_bytes_per_sec,
        thermal_level: &snapshot.pressure.thermal_level,
        frozen_count,
        top_processes: proc_slice,
    };
    let mut out = serde_json::to_string_pretty(&s).unwrap_or_default();

    if let Some(ctx) = teacher {
        // Bayesian effectiveness scores por proceso
        let scored: Vec<_> = ctx
            .pattern_scores
            .iter()
            .filter(|(_, count, _)| *count >= 3)
            .collect();
        if !scored.is_empty() {
            out.push_str("\n\nPatternEffectiveness (Bayesian, throttle_count ≥ 3):\n");
            for (name, count, eff) in scored.iter().take(14) {
                let tag = if *eff < 0.30 {
                    "NOT_CAUSING_PRESSURE"
                } else if *eff > 0.75 {
                    "confirmed_noise"
                } else {
                    "uncertain"
                };
                out.push_str(&format!(
                    "  {} count={} effectiveness={:.2} [{}]\n",
                    name, count, eff, tag
                ));
            }
        }

        // Resultado de la sugerencia anterior de Gemma
        if let Some(prev) = ctx.previous_outcome {
            let verdict = if prev.pressure_delta < -0.05 {
                "IMPROVED"
            } else if prev.pressure_delta > 0.03 {
                "WORSENED"
            } else {
                "NO_EFFECT"
            };
            out.push_str(&format!(
                "\nPreviousGemmaSuggestion outcome: {} (pressure delta {:+.3}, before={:.3} after={:.3})\nYour rationale was: \"{}\"\n",
                verdict,
                prev.pressure_delta,
                prev.pressure_before,
                prev.pressure_after,
                prev.rationale_snippet
            ));
        }

        if ctx.heuristic_struggling {
            out.push_str(
                "\nHEURISTIC_STRUGGLING=true — Apollo's rule-based optimizer cannot reduce pressure with current patterns.\n",
            );
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(toml_src: &str) -> LlmConfig {
        let cfg: RepoConfig = toml::from_str(toml_src).expect("test fixture must be valid TOML");
        cfg.llm.expect("test fixture must set [llm]")
    }

    #[test]
    fn always_on_defaults_to_false() {
        let cfg = parse("[llm]\nenabled = true\n");
        assert!(
            !cfg.always_on(),
            "always_on must default false so cloud configs keep training-TTL gating"
        );
    }

    #[test]
    fn always_on_parses_true_from_toml() {
        let cfg = parse("[llm]\nenabled = true\nalways_on = true\n");
        assert!(cfg.always_on());
        assert!(cfg.enabled());
    }

    #[test]
    fn always_on_true_does_not_force_enabled_true() {
        // `always_on` is independent from the master `enabled` switch. The
        // daemon tick must still honor enabled=false (the user's kill
        // switch) — always_on only changes the TTL-bypass path, not the
        // initial gate. Regression guard: if these ever collapse into one
        // flag, the user loses the kill-switch mid-session.
        let cfg = parse("[llm]\nenabled = false\nalways_on = true\n");
        assert!(
            !cfg.enabled(),
            "enabled=false must disable even when always_on=true"
        );
        assert!(cfg.always_on());
    }

    #[test]
    fn timeout_uses_override_when_provided() {
        // Regression guard for the 180s Gemma-4-on-M1 config: local models
        // need longer than the 5s cloud default or every inference trips
        // the timeout before it completes.
        let cfg = parse("[llm]\nenabled = true\ntimeout_ms = 180000\n");
        assert_eq!(cfg.timeout(), Duration::from_millis(180_000));
    }

    #[test]
    fn timeout_falls_back_to_5s_default_when_unset() {
        let cfg = parse("[llm]\nenabled = true\n");
        assert_eq!(cfg.timeout(), Duration::from_millis(5_000));
    }

    #[test]
    fn min_interval_secs_override_respected() {
        // Production Gemma 4 config sets 1800s (30min). Tests were relying
        // on the 900s default — a silent regression here would flood the
        // local model and drain battery.
        let cfg = parse("[llm]\nenabled = true\nmin_interval_secs = 1800\n");
        assert_eq!(cfg.min_interval_secs(), 1800);
    }
}
