//! Local policy and usage learning for the daemon.
//!
//! This module is fully local. It observes process usage, promotes bounded
//! policy patterns, and applies them behind the existing safety gates. External
//! prompts, model calls, API keys, and free-form advice are not part of this
//! control path.

use std::collections::{HashMap, HashSet};

use chrono::{Duration as ChronoDuration, Local, Utc};

use apollo_engine::engine::audit_types::DecisionReason;
use apollo_engine::engine::daemon_helpers::pid_start_time;
use apollo_engine::engine::lock_ext::LockRecover;
use apollo_engine::engine::policy_store::{append_jsonl, write_json, LearnedPolicy};
use apollo_engine::engine::safety::pattern_conflicts_with_protected;
use apollo_engine::engine::types::RootAction;

use super::SharedState;

pub fn windowserver_cpu(snapshot: &apollo_engine::collector::SystemSnapshot) -> f32 {
    snapshot
        .top_processes
        .iter()
        .find(|p| p.name.contains("WindowServer"))
        .map(|p| p.cpu_usage)
        .unwrap_or(0.0)
}

// ── Usage Learning Tick ────────────────────────────────────────────────────

pub fn usage_learning_tick(
    state: &SharedState,
    snapshot: &apollo_engine::collector::SystemSnapshot,
    has_foreground: bool,
    cpu_wall_ratios: &HashMap<String, f32>,
) {
    let now = Utc::now();
    let ws_cpu = windowserver_cpu(snapshot);
    // Refine interactive_proxy: require both CPU activity signals AND an actual
    // foreground app (not idle/screensaver). This prevents background CPU spikes
    // from triggering interactive mode when the user isn't at the keyboard.
    let cpu_proxy = ws_cpu >= 10.0 || snapshot.cpu.global_usage >= 15.0;
    let interactive_proxy = cpu_proxy && has_foreground;
    let mem_pressure = snapshot.pressure.memory_pressure;
    let swap_delta = snapshot.pressure.swap_delta_bytes_per_sec;

    let jank_proxy = ws_cpu >= 35.0
        && (mem_pressure >= 0.75 || swap_delta >= 20.0 * 1024.0 * 1024.0)
        || matches!(
            snapshot.pressure.thermal_level.as_str(),
            "serious" | "critical"
        );

    {
        let mut usage = state.usage.lock_recover();
        usage.usage_model.update_from_snapshot(
            snapshot,
            now,
            interactive_proxy,
            jank_proxy,
            10,
            cpu_wall_ratios,
        );
    }

    // Persist usage model periodically (every ~2 minutes).
    {
        let mut usage = state.usage.lock_recover();
        let due = usage
            .usage_tracker
            .last_persist_at
            .map(|t| now - t > ChronoDuration::minutes(2))
            .unwrap_or(true);
        if due {
            let path = usage.usage_model_path.clone();
            usage.usage_model.persist(&path);
            usage.usage_tracker.last_persist_at = Some(now);
        }
    }

    // Daily promotion counters (conservative).
    let today = Local::now().date_naive().to_string();
    let promotions_used = {
        let mut usage = state.usage.lock_recover();
        if usage.usage_tracker.promotions_day.as_deref() != Some(&today) {
            usage.usage_tracker.promotions_day = Some(today.clone());
            usage.usage_tracker.promotions_today = 0;
        }
        usage.usage_tracker.promotions_today
    };
    // Propose promotions without holding locks across scoring.
    let (started_at, existing_interactive, existing_noise, existing_protected) = {
        let model = state.usage.lock_recover();
        let started_at = model.usage_model.top_report(1).model_started_at;
        drop(model);
        let policy = state.policy.lock_recover().learned_policy.clone();
        (
            started_at,
            policy.interactive_patterns,
            policy.noise_patterns,
            policy.protected_patterns,
        )
    };
    let promotions = {
        let model = state.usage.lock_recover();
        model.usage_model.maybe_promote_patterns(
            now,
            &existing_interactive,
            &existing_noise,
            &existing_protected,
            promotions_used,
            started_at,
        )
    };

    if promotions.is_empty() {
        return;
    }

    // Apply promotions to learned policy.
    let mut applied = 0u32;
    let learned_policy_path = state.policy.lock_recover().learned_policy_path.clone();
    let lp_snap = {
        let mut pg = state.policy.lock_recover();
        for (kind, pattern) in &promotions {
            match kind.as_str() {
                "interactive"
                    if !pg.learned_policy.interactive_patterns.contains(pattern)
                        && !pattern_conflicts_with_protected(pattern)
                    => {
                        std::sync::Arc::make_mut(&mut pg.learned_policy.interactive_patterns).push(pattern.clone());
                        applied += 1;
                    }
                "noise"
                    if !pg.learned_policy.noise_patterns.contains(pattern)
                        && !pattern_conflicts_with_protected(pattern)
                        && !noise_pattern_conflicts(
                            pattern,
                            &pg.learned_policy.interactive_patterns,
                        )
                    => {
                        std::sync::Arc::make_mut(&mut pg.learned_policy.noise_patterns).push(pattern.clone());
                        applied += 1;
                    }
                "protected"
                    // Protected patterns are safety labels — they bypass the daily
                    // cap and only require that the pattern isn't already present.
                    if !pg.learned_policy.protected_patterns.contains(pattern)
                        && !pattern_conflicts_with_protected(pattern)
                    => {
                        std::sync::Arc::make_mut(&mut pg.learned_policy.protected_patterns).push(pattern.clone());
                        applied += 1;
                    }
                _ => {}
            }
        }
        if applied > 0 {
            std::sync::Arc::make_mut(&mut pg.learned_policy.interactive_patterns).sort();
            std::sync::Arc::make_mut(&mut pg.learned_policy.noise_patterns).sort();
            std::sync::Arc::make_mut(&mut pg.learned_policy.protected_patterns).sort();
            pg.learned_policy.learned_at = Some(now);
        }
        let snap = pg.learned_policy.clone();
        if applied > 0 {
            pg.adaptive_governor.update_learned_policy(&snap);
        }
        snap
    };
    if applied > 0 {
        // Persist after releasing the policy lock.
        write_json(&learned_policy_path, &lp_snap, Some(0o600));
    }

    if applied > 0 {
        let events_path = {
            let mut usage = state.usage.lock_recover();
            usage.usage_tracker.promotions_today += applied;
            usage.usage_events_path.clone()
        };
        append_jsonl(
            &events_path,
            &serde_json::json!({"at": now, "promotions": promotions}),
        );
    }
}

// ── Apply Learned Policy Actions ───────────────────────────────────────────

/// 2026-05-16: per-PID TTL dedup for learned-policy boost/throttle emissions.
/// Without this, `apply_learned_policy_actions` re-emitted BoostProcess for
/// every interactive PID every cycle (464/500 journal entries = boosts).
/// Each emit ≡ mach_qos.set_tier syscall; mach_qos IS already sticky so
/// the kernel work was redundant — but each emit also walked the safety
/// stack + journal write, consuming 10% Apollo CPU and contributing to
/// the very thrashing Apollo was trying to mitigate. 30s TTL chosen to
/// match typical foreground app dwell time without re-firing on every
/// per-cycle snapshot. Stale entries pruned lazily on next access.
fn boost_dedup_cache() -> &'static std::sync::Mutex<HashMap<u32, std::time::Instant>> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<HashMap<u32, std::time::Instant>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

fn learned_policy_boost_ttl_secs() -> u64 {
    120
}

/// A boost may only target a process the user is actually using: the
/// foreground app or one with a visible window. Blocks headless background
/// processes (e.g. a `node` dev server matching the bare "node" interactive
/// pattern) from being boosted on a name match alone — boosting them steals
/// P-core scheduling from the foreground app / a live call. [2026-06-18]
///
/// 2026-06-21 — yield to frontmost media: when the FRONTMOST app is actively
/// playing media (`frontmost_media_active`), a visible-but-NOT-frontmost
/// candidate is refused. A boost marks the target TASK_FOREGROUND_APPLICATION +
/// QOS_TIER_0 + nice -10, lying to CLPC about P-core contention; doing that to a
/// background-but-visible terminal steals compositing timeshare from the
/// frontmost 4K app → occasional frame drop (node-54x class). The frontmost app
/// itself is ALWAYS boostable (never yields). Survival bypasses this entirely
/// (the caller gates on `!survival` before this). Subtractive — only ever adds a
/// skip; over-yielding merely leaves a non-frontmost app at its correct default
/// QoS.
fn boost_visibility_ok(
    pid: u32,
    fg_pid: Option<u32>,
    visible_pids: &HashSet<u32>,
    frontmost_media_active: bool,
) -> bool {
    if Some(pid) == fg_pid {
        return true; // the frontmost app is always boostable — never yields
    }
    // visible-but-not-frontmost: yield to the frontmost media app
    visible_pids.contains(&pid) && !frontmost_media_active
}

/// A locally promoted NOISE pattern conflicts (must be rejected) if it would
/// shadow an interactive/protected process. Historical imported policy once
/// moved `language_server` (the LSP) to noise.
/// The prior guard used exact `.contains()`, which missed `language_server` vs
/// the stored interactive `language_server_macos_arm`. Reject if the pattern is
/// a safety-protected name OR matches an existing interactive pattern
/// (truncation-aware, both directions).
fn noise_pattern_conflicts(pattern: &str, interactive: &[String]) -> bool {
    if apollo_engine::engine::safety::is_protected_name(pattern) {
        return true;
    }
    let pl = pattern.to_lowercase();
    let lpm = apollo_engine::engine::decide_actions::learned_pattern_matches;
    interactive.iter().any(|ip| {
        let il = ip.to_lowercase();
        lpm(&pl, &il) || lpm(&il, &pl)
    })
}

fn should_emit_boost(pid: u32, ttl_secs: u64) -> bool {
    let now = std::time::Instant::now();
    let ttl = std::time::Duration::from_secs(ttl_secs);
    let mut cache = boost_dedup_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // Prune stale entries opportunistically.
    cache.retain(|_, t| now.duration_since(*t) < ttl);
    match cache.get(&pid) {
        Some(t) if now.duration_since(*t) < ttl => false,
        _ => {
            cache.insert(pid, now);
            true
        }
    }
}

pub fn apply_learned_policy_actions(
    snapshot: &apollo_engine::collector::SystemSnapshot,
    policy: &LearnedPolicy,
    mut actions: Vec<RootAction>,
    // 2026-06-21: frontmost app NAME, for the boost yield-to-media gate. The
    // frontmost PID arrives via shadow_signals; the name is only known to the
    // caller (main.rs foreground_app). None = unknown → never yields (safe).
    foreground_app: Option<&str>,
) -> Vec<RootAction> {
    // Filter: never act on protected patterns (case-insensitive).
    if !policy.protected_patterns.is_empty() {
        actions.retain(|a| {
            let name = match a {
                RootAction::BoostProcess { name, .. }
                | RootAction::ThrottleProcess { name, .. }
                | RootAction::FreezeProcess { name, .. }
                | RootAction::UnfreezeProcess { name, .. } => name,
                _ => return true,
            };
            let name_lc = name.to_lowercase();
            !policy
                .protected_patterns
                .iter()
                .any(|p| name_lc.contains(&p.to_lowercase()))
        });
    }

    // Add targeted boost/throttle for top processes if policy matches.
    if policy.interactive_patterns.is_empty() && policy.noise_patterns.is_empty() {
        return actions;
    }
    let mut seen: HashSet<(u32, &'static str)> = HashSet::new();
    for a in &actions {
        match a {
            RootAction::BoostProcess { pid, .. } => {
                seen.insert((*pid, "boost"));
            }
            RootAction::ThrottleProcess { pid, .. } => {
                seen.insert((*pid, "throttle"));
            }
            _ => {}
        }
    }

    let survival = apollo_engine::engine::safety::survival_mode_active_total(
        snapshot.pressure.memory_pressure,
        snapshot.pressure.swap_used_bytes,
        snapshot.pressure.swap_total_bytes,
    );

    // ROOT FIX (2026-06-18 node boost-loop): a BOOST raises QoS to make the app
    // the user is actually using snappier. Firing it on a NAME match alone
    // boosts headless background processes — a `node` dev server matching the
    // bare "node" interactive pattern got boosted 54×, stealing P-core
    // scheduling from Meet's video (microstutter). Same class as Brave-0607.
    // Gate on real foreground / visible-window state: only the foreground app
    // or a process with a visible window may be boosted. The visible-pid
    // syscall (~1-3ms) is computed only when an interactive-name candidate
    // actually exists, so most cycles pay nothing.
    let fg_pid = apollo_engine::engine::shadow_signals::get_foreground_pid();
    let any_interactive_candidate = snapshot.top_processes.iter().any(|p| {
        !seen.contains(&(p.pid, "boost"))
            && policy
                .interactive_patterns
                .iter()
                .any(|pat| p.name.contains(pat))
    });
    let visible_pids = if any_interactive_candidate {
        apollo_engine::engine::cg_window::visible_pids()
    } else {
        HashSet::new()
    };
    // 2026-06-21 — yield-to-frontmost-media: true iff the frontmost app is a
    // media host (browser/player, NOT chat/call) AND audio is live now. Only
    // then do visible-but-not-frontmost boosts yield. Computed once/cycle,
    // gated on a candidate existing so quiet cycles pay nothing. is_audio*
    // is ~50µs and fail-false. None foreground name → false (never yields).
    let frontmost_media_active = any_interactive_candidate
        && foreground_app
            .map(apollo_engine::engine::window_sensor::is_media_host)
            .unwrap_or(false)
        && apollo_engine::engine::coreaudio_active::is_audio_running_somewhere();

    for p in &snapshot.top_processes {
        if policy
            .interactive_patterns
            .iter()
            .any(|pat| p.name.contains(pat))
            && !seen.contains(&(p.pid, "boost"))
            && !survival
            && boost_visibility_ok(p.pid, fg_pid, &visible_pids, frontmost_media_active)
            && should_emit_boost(p.pid, learned_policy_boost_ttl_secs())
        {
            // Round-4 hotfix (2026-06-07): the FIX-1 guard at
            // decide_actions.rs:597/644/871 protected the rule-based
            // boost paths but missed this legacy policy emit site, which
            // applies the LEARNED policy classification (the very
            // signal corrupted by the Brave loop). Without this guard,
            // 4 Brave boosts/cycle slipped through post-deploy despite
            // 36 hard_protected_boost_skipped_total bumps elsewhere.
            // Apply complete-mediation here too — `is_boost_forbidden`
            // returns true for hard-protected names + Chromium
            // family-roots.
            if apollo_engine::engine::safety::is_boost_forbidden(&p.name) {
                apollo_engine::engine::lse_counters::LSE_COUNTERS
                    .inc_hard_protected_boost_skipped();
                continue;
            }
            let (ss, su) = pid_start_time(p.pid);
            actions.push(RootAction::BoostProcess {
                pid: p.pid,
                name: p.name.clone(),
                reason: "learned-policy interactive".to_string(),
                decision_reason: DecisionReason::PressureContext,
                start_sec: ss,
                start_usec: su,
            });
            seen.insert((p.pid, "boost"));
        }
        if policy.noise_patterns.iter().any(|pat| p.name.contains(pat))
            && !seen.contains(&(p.pid, "throttle"))
            // Complete mediation (2026-06-18 bug-class sweep): a noise pattern
            // can match a protected/Apple/dev process or a Chromium helper. The
            // survival-mode upgrade makes this an AGGRESSIVE throttle (SIGSTOP
            // pulses) — on Chromium that breaks Brave's IPC contract. Never
            // throttle a protected name or the Chromium family on a name match.
            && !apollo_engine::engine::safety::is_protected_name(&p.name)
            && !apollo_engine::engine::safety::is_chromium_family(&p.name)
            // 2026-06-20: a legacy policy noise-classified `language_server` (the
            // LSP) even though it is interactive — a name-match conflict the
            // exact-contains add-guard missed. NEVER throttle a process that is
            // also interactive-classified, regardless of how it got into noise.
            // Truncation-aware (language_server vs language_server_macos_arm).
            && !policy.interactive_patterns.iter().any(|ip| {
                let nl = p.name.to_lowercase();
                let il = ip.to_lowercase();
                apollo_engine::engine::decide_actions::learned_pattern_matches(&nl, &il)
                    || apollo_engine::engine::decide_actions::learned_pattern_matches(&il, &nl)
            })
        {
            let (ss, su) = pid_start_time(p.pid);
            // Under survival mode, upgrade to aggressive throttle. Non-aggressive
            // (background QoS demotion) is too weak when swap ≥4GB — the process
            // still pages in/out at the same rate. Aggressive adds SIGSTOP pulses.
            // [Nygard 2018 §5] — under load, shed harder on processes already
            // classified as noise.
            actions.push(RootAction::ThrottleProcess {
                pid: p.pid,
                name: p.name.clone(),
                aggressive: survival,
                reason: if survival {
                    "learned-policy noise (survival)".to_string()
                } else {
                    "learned-policy noise".to_string()
                },
                start_sec: ss,
                start_usec: su,
                decision_reason: DecisionReason::PressureContext,
            });
            seen.insert((p.pid, "throttle"));
        }
    }

    actions
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_engine::collector::{
        CpuStats, MemoryStats, PressureStats, ProcessStats, SystemSnapshot,
    };

    fn foreground_signal_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    struct ForegroundSignalReset;

    impl Drop for ForegroundSignalReset {
        fn drop(&mut self) {
            apollo_engine::engine::shadow_signals::set_foreground_pid(None);
        }
    }

    #[test]
    fn noise_add_rejects_interactive_and_protected() {
        // A legacy policy noise-classified the LSP. A noise pattern that shadows an
        // interactive entry (truncation-aware) or a protected name must be
        // rejected — exact .contains() missed language_server vs the stored
        // language_server_macos_arm.
        let interactive = vec!["language_server_macos_arm".to_string(), "Brave".to_string()];
        assert!(
            noise_pattern_conflicts("language_server", &interactive),
            "truncated LSP name must conflict with the full interactive pattern"
        );
        assert!(
            noise_pattern_conflicts("WindowServer", &interactive),
            "a safety-protected name must conflict"
        );
        assert!(
            !noise_pattern_conflicts("GoogleUpdater", &interactive),
            "a genuine noise process must NOT conflict"
        );
    }

    #[test]
    fn boost_only_foreground_or_visible() {
        let mut visible = HashSet::new();
        visible.insert(42u32); // a process with a visible window
        let no_media = false; // frontmost is not an active media host
                              // Foreground app → boostable.
        assert!(boost_visibility_ok(7, Some(7), &visible, no_media));
        // Visible window → boostable.
        assert!(boost_visibility_ok(42, Some(7), &visible, no_media));
        // Headless background (e.g. a node dev server): not foreground, no
        // window → MUST NOT be boosted on a name match alone.
        assert!(!boost_visibility_ok(999, Some(7), &visible, no_media));
        assert!(!boost_visibility_ok(999, None, &HashSet::new(), no_media));
    }

    #[test]
    fn boost_yields_to_frontmost_media() {
        // 2026-06-21: a visible-but-NOT-frontmost interactive process (e.g. a
        // terminal at pid 42) must YIELD its boost while the FRONTMOST app is an
        // active media host (Brave playing 4K) — the boost would steal P-core
        // compositing timeshare → occasional frame drop (node-54x class).
        let mut visible = HashSet::new();
        visible.insert(42u32); // the visible-but-background terminal
        let fg = Some(7u32); // frontmost app is pid 7 (the browser)

        // media active in the frontmost app:
        // (a) the FRONTMOST app itself is ALWAYS boostable — never yields.
        assert!(
            boost_visibility_ok(7, fg, &visible, true),
            "frontmost app must always be boostable, even during its own media"
        );
        // (b) the visible-but-not-frontmost terminal YIELDS.
        assert!(
            !boost_visibility_ok(42, fg, &visible, true),
            "visible non-frontmost boost must yield to the frontmost media app"
        );

        // media NOT active (or frontmost is not a media host) → unchanged:
        // (c) the same terminal is boosted as before.
        assert!(
            boost_visibility_ok(42, fg, &visible, false),
            "without frontmost media, a visible process is boosted as before"
        );
    }

    #[test]
    fn is_media_host_matches_players_not_chat() {
        use apollo_engine::engine::window_sensor::is_media_host;
        // Browsers + dedicated players → media hosts (yield).
        assert!(is_media_host("Brave Browser"));
        assert!(is_media_host("Google Chrome"));
        assert!(is_media_host("Safari"));
        assert!(is_media_host("Spotify"));
        assert!(is_media_host("VLC"));
        assert!(is_media_host("IINA"));
        // Chat/call apps → NOT media hosts (a Slack ping must not starve the
        // terminal; calls are handled by is_realtime_call_active).
        assert!(!is_media_host("Slack"));
        assert!(!is_media_host("Discord"));
        assert!(!is_media_host("zoom.us"));
        assert!(!is_media_host("Microsoft Teams"));
        // A terminal/editor → not a media host.
        assert!(!is_media_host("alacritty"));
        assert!(!is_media_host("Code"));
    }

    fn snapshot_with(processes: Vec<ProcessStats>) -> SystemSnapshot {
        SystemSnapshot {
            timestamp: chrono::Utc::now(),
            cpu: CpuStats {
                global_usage: 0.0,
                core_count: 1,
            },
            memory: MemoryStats {
                total_ram: 0,
                used_ram: 0,
                free_ram: 0,
                total_swap: 0,
                used_swap: 0,
            },
            pressure: PressureStats {
                memory_pressure: 0.0,
                swap_used_bytes: 0,
                swap_total_bytes: 0,
                swap_delta_bytes_per_sec: 0.0,
                thermal_level: "nominal".into(),
                compressor_pressure: 0.0,
                thrashing_score: 0.0,
                memory_pressure_raw: 0.0,
                refault_delta_per_sec: 0.0,
            },
            disks: vec![],
            networks: vec![],
            top_processes: processes,
        }
    }

    fn proc(pid: u32, name: &str, cpu: f32) -> ProcessStats {
        ProcessStats {
            pid,
            name: name.into(),
            cpu_usage: cpu,
            memory_usage: 0,
            cpu_wall_ratio: None,
        }
    }

    fn policy(interactive: &[&str], noise: &[&str], protected: &[&str]) -> LearnedPolicy {
        LearnedPolicy {
            interactive_patterns: std::sync::Arc::new(
                interactive.iter().map(|s| s.to_string()).collect(),
            ),
            noise_patterns: std::sync::Arc::new(noise.iter().map(|s| s.to_string()).collect()),
            protected_patterns: std::sync::Arc::new(
                protected.iter().map(|s| s.to_string()).collect(),
            ),
            learned_at: None,
            pattern_weights: HashMap::new(),
        }
    }

    // ── windowserver_cpu ─────────────────────────────────────────────────────

    #[test]
    fn windowserver_cpu_empty_snapshot_returns_zero() {
        assert_eq!(windowserver_cpu(&snapshot_with(vec![])), 0.0);
    }

    #[test]
    fn windowserver_cpu_finds_exact_name() {
        let snap = snapshot_with(vec![proc(1, "WindowServer", 42.5)]);
        assert_eq!(windowserver_cpu(&snap), 42.5);
    }

    #[test]
    fn windowserver_cpu_matches_substring() {
        let snap = snapshot_with(vec![proc(1, "com.apple.WindowServer", 10.0)]);
        assert_eq!(windowserver_cpu(&snap), 10.0);
    }

    #[test]
    fn windowserver_cpu_case_sensitive_miss() {
        let snap = snapshot_with(vec![proc(1, "windowserver", 99.0)]);
        assert_eq!(windowserver_cpu(&snap), 0.0, "lookup is case-sensitive");
    }

    // ── apply_learned_policy_actions ─────────────────────────────────────────

    #[test]
    fn apply_empty_policy_passthrough() {
        let snap = snapshot_with(vec![]);
        let actions = vec![RootAction::BoostProcess {
            pid: 1,
            name: "app".into(),
            reason: "r".into(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        }];
        let result = apply_learned_policy_actions(&snap, &policy(&[], &[], &[]), actions, None);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn apply_protected_pattern_removes_freeze() {
        let snap = snapshot_with(vec![]);
        let actions = vec![RootAction::FreezeProcess {
            pid: 1,
            name: "claude".into(),
            reason: "r".into(),
            start_sec: 0,
            start_usec: 0,
            decision_reason: DecisionReason::PressureContext,
        }];
        let result =
            apply_learned_policy_actions(&snap, &policy(&[], &[], &["claude"]), actions, None);
        assert!(result.is_empty(), "claude must be protected");
    }

    #[test]
    fn apply_protected_pattern_keeps_non_matching() {
        let snap = snapshot_with(vec![]);
        let actions = vec![RootAction::FreezeProcess {
            pid: 2,
            name: "slack".into(),
            reason: "r".into(),
            start_sec: 0,
            start_usec: 0,
            decision_reason: DecisionReason::PressureContext,
        }];
        let result =
            apply_learned_policy_actions(&snap, &policy(&[], &[], &["claude"]), actions, None);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn apply_interactive_pattern_adds_boost() {
        let _serial = foreground_signal_test_lock();
        let _reset = ForegroundSignalReset;
        boost_dedup_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        apollo_engine::engine::shadow_signals::set_foreground_pid(Some(42));
        let snap = snapshot_with(vec![proc(42, "Xcode", 20.0)]);
        let result =
            apply_learned_policy_actions(&snap, &policy(&["Xcode"], &[], &[]), vec![], None);
        assert_eq!(result.len(), 1);
        match &result[0] {
            RootAction::BoostProcess { pid, name, .. } => {
                assert_eq!(*pid, 42);
                assert_eq!(name, "Xcode");
            }
            _ => panic!("expected BoostProcess"),
        }
    }

    #[test]
    fn apply_no_duplicate_boost_when_already_present() {
        let _serial = foreground_signal_test_lock();
        let _reset = ForegroundSignalReset;
        apollo_engine::engine::shadow_signals::set_foreground_pid(Some(42));
        let snap = snapshot_with(vec![proc(42, "Xcode", 20.0)]);
        let existing = vec![RootAction::BoostProcess {
            pid: 42,
            name: "Xcode".into(),
            reason: "existing".into(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        }];
        let result =
            apply_learned_policy_actions(&snap, &policy(&["Xcode"], &[], &[]), existing, None);
        let boosts = result
            .iter()
            .filter(|a| matches!(a, RootAction::BoostProcess { .. }))
            .count();
        assert_eq!(boosts, 1, "must not duplicate existing boost");
    }

    #[test]
    fn learned_policy_boost_ttl_is_longer_than_thirty_seconds() {
        assert!(
            learned_policy_boost_ttl_secs() >= 120,
            "mach QoS boosts are sticky; re-emitting every 30s creates journal noise"
        );
    }
}
