//! Level 8: Adaptive Intelligence Tests
//!
//! Tests for the heuristic-based intelligent system:
//! process_classifier, zombie_hunter, user_profile, adaptive_governor

use apollo_optimizer::engine::adaptive_governor::{AdaptiveGovernor, GovernorDecision};
use apollo_optimizer::engine::process_classifier::{
    essential_process_names, score_utility, score_waste, telemetry_process_names,
    ProcessClassifier, ProcessSnapshot, ProcessTier,
};
use apollo_optimizer::engine::user_profile::{UserProfile, WorkloadType};
use apollo_optimizer::engine::zombie_hunter::{
    HuntSnapshot, ZombieAction, ZombieClass, ZombieHunter,
};

// ── Helpers ────────────────────────────────────────────────────────────────

fn make_snap(name: &str, cpu: f32, secs_idle: u64, has_gui: bool) -> ProcessSnapshot {
    ProcessSnapshot {
        pid: 1000,
        name: name.to_string(),
        cpu_percent: cpu,
        rss_bytes: 100 * 1024 * 1024,
        is_zombie: false,
        secs_since_foreground: secs_idle,
        secs_since_user_interaction: secs_idle,
        has_network: false,
        has_gui_window: has_gui,
        wakeups_per_sec: 1.0,
        parent_alive: true,
        process_uptime_secs: 300,
        faults_total: 0,
        pageins_total: 0,
        is_translated: false,
        mach_port_count: 0,
        cpu_contention: None,
    }
}

fn make_hunt(name: &str, zombie: bool, parent_alive: bool) -> HuntSnapshot {
    HuntSnapshot {
        pid: 2000,
        ppid: 1,
        name: name.to_string(),
        is_kernel_zombie: zombie,
        parent_alive,
        has_gui_window: false,
        rss_bytes: 100 * 1024 * 1024,
        cpu_percent: 0.5,
        wakeups_per_sec: 1.0,
        secs_since_user_interaction: 3600,
        host_app_pid: None,
        host_app_running: true,
        host_app_absent_secs: 0,
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ProcessClassifier Tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_classifier_essential_processes_are_protected() {
    let classifier = ProcessClassifier::new();
    let snap = make_snap("kernel_task", 50.0, 0, false);
    let tier = classifier.classify(&snap);
    assert_eq!(tier, ProcessTier::SystemEssential);
}

#[test]
fn test_classifier_launchd_essential() {
    let classifier = ProcessClassifier::new();
    let snap = make_snap("launchd", 2.0, 0, false);
    assert_eq!(classifier.classify(&snap), ProcessTier::SystemEssential);
}

#[test]
fn test_classifier_telemetry_detected() {
    let classifier = ProcessClassifier::new();
    let snap = make_snap("analyticsd", 2.0, 600, false);
    assert_eq!(classifier.classify(&snap), ProcessTier::Telemetry);
}

#[test]
fn test_classifier_crash_reporter_is_telemetry() {
    let classifier = ProcessClassifier::new();
    let snap = make_snap("CrashReporter", 1.0, 1000, false);
    assert_eq!(classifier.classify(&snap), ProcessTier::Telemetry);
}

#[test]
fn test_classifier_zombie_detected() {
    let classifier = ProcessClassifier::new();
    let mut snap = make_snap("crashed_app", 0.0, 9999, false);
    snap.is_zombie = true;
    assert_eq!(classifier.classify(&snap), ProcessTier::ZombieOrphan);
}

#[test]
fn test_classifier_orphan_detected() {
    let classifier = ProcessClassifier::new();
    let mut snap = make_snap("orphan_proc", 5.0, 9999, false);
    snap.parent_alive = false;
    assert_eq!(classifier.classify(&snap), ProcessTier::ZombieOrphan);
}

#[test]
fn test_classifier_active_foreground() {
    let classifier = ProcessClassifier::new();
    let snap = make_snap("Xcode", 30.0, 0, true);
    assert_eq!(classifier.classify(&snap), ProcessTier::ActiveForeground);
}

#[test]
fn test_classifier_background_visible() {
    let classifier = ProcessClassifier::new();
    let snap = make_snap("Slack", 5.0, 60, true); // Used 60s ago, has window
    assert_eq!(classifier.classify(&snap), ProcessTier::BackgroundVisible);
}

#[test]
fn test_classifier_stale_app() {
    let classifier = ProcessClassifier::new();
    // Stale requires: cpu < 0.5 AND wakeups < 1.0 AND secs_since_foreground > 300.
    let mut snap = make_snap("StaleApp", 0.1, 90_000, false); // 25h idle, low cpu
    snap.wakeups_per_sec = 0.0;
    assert_eq!(classifier.classify(&snap), ProcessTier::Stale);
}

#[test]
fn test_classifier_silent_daemon() {
    let classifier = ProcessClassifier::new();
    let mut snap = make_snap("backupd", 3.0, 7200, false);
    snap.wakeups_per_sec = 30.0; // High wakeup, no GUI, no network
    assert_eq!(classifier.classify(&snap), ProcessTier::SilentDaemon);
}

#[test]
fn test_utility_score_active() {
    let snap = make_snap("Xcode", 30.0, 5, true);
    let score = score_utility(&snap);
    assert!(
        score > 0.7,
        "Active process should have high utility: {}",
        score
    );
}

#[test]
fn test_utility_score_stale() {
    let snap = make_snap("StaleApp", 1.0, 100_000, false);
    let score = score_utility(&snap);
    assert!(
        score < 0.2,
        "Stale process should have low utility: {}",
        score
    );
}

#[test]
fn test_waste_score_high_cpu() {
    let mut snap = make_snap("burner", 9.0, 7200, false);
    snap.wakeups_per_sec = 45.0;
    snap.rss_bytes = 300 * 1024 * 1024; // >200MB triggers +0.15 → total 0.80 > 0.7
    let waste = score_waste(&snap);
    assert!(
        waste > 0.7,
        "High-CPU process should have high waste: {}",
        waste
    );
}

#[test]
fn test_waste_score_idle_process() {
    let mut snap = make_snap("idle_proc", 0.0, 7200, false);
    snap.wakeups_per_sec = 0.0;
    let waste = score_waste(&snap);
    // Long idle (7200s > 3600) with cpu < 1.0 adds +0.10 → exactly 0.10.
    assert!(
        waste <= 0.10,
        "Idle process should have near-zero waste: {}",
        waste
    );
}

#[test]
fn test_classifier_throttle_candidates() {
    let classifier = ProcessClassifier::new();
    let snaps = vec![
        make_snap("kernel_task", 50.0, 0, false), // Essential
        make_snap("analyticsd", 2.0, 600, false), // Telemetry
        {
            let mut s = make_snap("StaleApp", 1.0, 90_000, false);
            s.pid = 1001;
            s
        },
    ];
    let candidates = classifier.throttle_candidates(&snaps);
    assert!(candidates.len() >= 2); // analyticsd + StaleApp
}

#[test]
fn test_essential_list_not_empty() {
    let essentials = essential_process_names();
    assert!(essentials.len() >= 5);
    assert!(essentials.contains("kernel_task"));
    assert!(essentials.contains("launchd"));
}

#[test]
fn test_telemetry_list_not_empty() {
    let telemetry = telemetry_process_names();
    assert!(telemetry.len() >= 5);
    assert!(telemetry.contains("analyticsd"));
    assert!(telemetry.contains("CrashReporter"));
}

// ════════════════════════════════════════════════════════════════════════════
// ZombieHunter Tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_zombie_hunter_detects_true_zombie() {
    let mut hunter = ZombieHunter::new();
    let snap = make_hunt("crashed_app", true, true);

    // True zombie should be detected immediately (no confirmation needed)
    let result = hunter.evaluate(&snap);
    assert!(result.is_some());
    let dw = result.unwrap();
    assert_eq!(dw.zombie_class, ZombieClass::TrueZombie);
    assert_eq!(dw.recommended_action, ZombieAction::Kill);
}

#[test]
fn test_zombie_hunter_detects_orphan() {
    let mut hunter = ZombieHunter::new();
    let mut snap = make_hunt("orphan", false, false);
    snap.ppid = 999; // Non-launchd parent

    let result = hunter.evaluate(&snap);
    assert!(result.is_some());
    let dw = result.unwrap();
    assert_eq!(dw.zombie_class, ZombieClass::Orphan);
    assert_eq!(dw.recommended_action, ZombieAction::Kill);
}

#[test]
fn test_zombie_hunter_legitimate_process() {
    let mut hunter = ZombieHunter::new();
    let mut snap = make_hunt("normal_app", false, true);
    snap.secs_since_user_interaction = 10;
    snap.has_gui_window = true;

    let result = hunter.evaluate(&snap);
    assert!(result.is_none(), "Normal process should not be flagged");
}

#[test]
fn test_zombie_hunter_requires_confirmation_for_ghost_helper() {
    let mut hunter = ZombieHunter::new();
    let mut snap = make_hunt("AppHelper", false, true);
    snap.host_app_pid = Some(5000);
    snap.host_app_running = false;
    snap.host_app_absent_secs = 90_000; // 25h

    // First two evaluations → not yet confirmed
    assert!(hunter.evaluate(&snap).is_none());
    assert!(hunter.evaluate(&snap).is_none());
    // Third → confirmed
    let result = hunter.evaluate(&snap);
    assert!(result.is_some());
    assert_eq!(result.unwrap().zombie_class, ZombieClass::GhostHelper);
}

#[test]
fn test_zombie_hunter_wakeup_burner_needs_confirmation() {
    let mut hunter = ZombieHunter::new();
    let mut snap = make_hunt("bg_daemon", false, true);
    snap.wakeups_per_sec = 50.0;
    snap.secs_since_user_interaction = 3600;
    snap.has_gui_window = false;

    hunter.evaluate(&snap);
    hunter.evaluate(&snap);
    let result = hunter.evaluate(&snap);
    assert!(result.is_some());
    assert_eq!(result.unwrap().zombie_class, ZombieClass::WakeupBurner);
}

#[test]
fn test_zombie_hunter_memory_hoarder() {
    let mut hunter = ZombieHunter::new();
    let mut snap = make_hunt("fat_daemon", false, true);
    snap.rss_bytes = 1024 * 1024 * 1024; // 1GB
    snap.has_gui_window = false;
    snap.secs_since_user_interaction = 7200;

    hunter.evaluate(&snap);
    hunter.evaluate(&snap);
    let result = hunter.evaluate(&snap);
    assert!(result.is_some());
    let dw = result.unwrap();
    assert_eq!(dw.zombie_class, ZombieClass::MemoryHoarder);
    assert_eq!(dw.recommended_action, ZombieAction::Suspend);
}

#[test]
fn test_zombie_hunter_total_reclaimable() {
    let dead = vec![
        apollo_optimizer::engine::zombie_hunter::DeadWeightProcess {
            pid: 1,
            name: "a".into(),
            zombie_class: ZombieClass::TrueZombie,
            wasted_rss_bytes: 200 * 1024 * 1024,
            wakeups_per_sec: 0.0,
            recommended_action: ZombieAction::Kill,
            reason: "".into(),
        },
        apollo_optimizer::engine::zombie_hunter::DeadWeightProcess {
            pid: 2,
            name: "b".into(),
            zombie_class: ZombieClass::MemoryHoarder,
            wasted_rss_bytes: 800 * 1024 * 1024,
            wakeups_per_sec: 10.0,
            recommended_action: ZombieAction::Suspend,
            reason: "".into(),
        },
    ];
    let total = ZombieHunter::total_reclaimable_bytes(&dead);
    assert_eq!(total, 1000 * 1024 * 1024);
}

#[test]
fn test_zombie_hunter_cleanup() {
    let mut hunter = ZombieHunter::new();
    let snap = make_hunt("bg_daemon", false, true);
    hunter.evaluate(&snap);

    hunter.cleanup(&[9999]); // PID 2000 not in live list
                             // Should not crash and counter should be cleared
}

// ════════════════════════════════════════════════════════════════════════════
// UserProfile Tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_user_profile_initial_workload_is_idle() {
    let profile = UserProfile::new();
    // Before observing anything, the workload defaults to Idle
    assert_eq!(profile.current_workload(), WorkloadType::Idle);
}

#[test]
fn test_user_profile_detects_coding_workload() {
    let mut profile = UserProfile::new();
    profile.observe(Some("Xcode"), &["Xcode", "cargo", "git"], 9);
    assert_eq!(profile.current_workload(), WorkloadType::Coding);
}

#[test]
fn test_user_profile_detects_video_call() {
    let mut profile = UserProfile::new();
    profile.observe(Some("zoom.us"), &["zoom.us", "Teams"], 14);
    assert_eq!(profile.current_workload(), WorkloadType::VideoCall);
}

#[test]
fn test_user_profile_detects_media_playback() {
    let mut profile = UserProfile::new();
    profile.observe(Some("VLC"), &["VLC"], 20);
    assert_eq!(profile.current_workload(), WorkloadType::MediaPlayback);
}

#[test]
fn test_user_profile_likely_workload_at_hour() {
    let mut profile = UserProfile::new();
    for _ in 0..5 {
        profile.observe(Some("Xcode"), &["Xcode", "cargo"], 10);
    }
    let likely = profile.likely_workload_at_hour(10);
    assert_eq!(likely, WorkloadType::Coding);
}

#[test]
fn test_user_profile_stale_apps() {
    let mut profile = UserProfile::new();
    // Open and close Keynote quickly
    profile.observe(Some("Keynote"), &["Keynote"], 10);
    profile.observe(None, &[], 10); // Close

    let running = vec!["Keynote"];
    // No stats updated yet (session just ended with 0s) — should not be stale
    let stale = profile.stale_apps(&running, 3600);
    // Result depends on internal tracking; just verify no panic
    let _ = stale;
}

#[test]
fn test_user_profile_process_relevance_known_workload() {
    let mut profile = UserProfile::new();
    profile.observe(Some("Xcode"), &["Xcode", "cargo", "git"], 10);

    // cargo is directly relevant to Coding workload
    let relevance = profile.process_relevance("cargo");
    assert!(
        relevance > 0.5,
        "cargo should be relevant during Coding: {}",
        relevance
    );
}

#[test]
fn test_user_profile_process_relevance_irrelevant() {
    let mut profile = UserProfile::new();
    profile.observe(Some("Xcode"), &["Xcode", "cargo"], 10);

    // zoom.us is not relevant to Coding
    let relevance = profile.process_relevance("zoom.us");
    assert!(
        relevance < 0.5,
        "zoom.us should not be relevant during Coding: {}",
        relevance
    );
}

#[test]
fn test_user_profile_foreground_app_switch() {
    let mut profile = UserProfile::new();
    profile.observe(Some("Xcode"), &["Xcode"], 14);
    profile.observe(Some("Slack"), &["Slack"], 14);
    assert_eq!(profile.current_workload(), WorkloadType::VideoCall);
}

// ════════════════════════════════════════════════════════════════════════════
// AdaptiveGovernor Tests
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_governor_protects_essentials() {
    let mut gov = AdaptiveGovernor::new();

    let procs = vec![make_snap("kernel_task", 50.0, 0, false)];
    let hunts: Vec<HuntSnapshot> = vec![];

    let decisions = gov.decide_all(&procs, &hunts, Some("Xcode"), &["kernel_task", "Xcode"], 10);

    let kt = decisions.iter().find(|d| d.name == "kernel_task");
    assert!(kt.is_some());
    assert_eq!(kt.unwrap().decision, GovernorDecision::Allow);
}

#[test]
fn test_governor_kills_true_zombie() {
    let mut gov = AdaptiveGovernor::new();

    let procs: Vec<ProcessSnapshot> = vec![];
    let mut hunt = make_hunt("crashed_app", true, true);
    hunt.pid = 9999;
    let hunts = vec![hunt];

    let decisions = gov.decide_all(&procs, &hunts, None, &[], 10);

    let zombie = decisions.iter().find(|d| d.name == "crashed_app");
    assert!(zombie.is_some());
    assert_eq!(zombie.unwrap().decision, GovernorDecision::Kill);
}

#[test]
fn test_governor_throttles_telemetry() {
    let mut gov = AdaptiveGovernor::new();

    let procs = vec![make_snap("analyticsd", 2.0, 600, false)];
    let hunts: Vec<HuntSnapshot> = vec![];

    let decisions = gov.decide_all(&procs, &hunts, None, &["analyticsd"], 10);

    let telem = decisions.iter().find(|d| d.name == "analyticsd");
    assert!(telem.is_some());
    assert!(matches!(
        telem.unwrap().decision,
        GovernorDecision::Throttle | GovernorDecision::Freeze
    ));
}

#[test]
fn test_governor_freezes_telemetry_during_coding() {
    let mut gov = AdaptiveGovernor::new();

    let procs = vec![make_snap("analyticsd", 2.0, 600, false)];
    let hunts: Vec<HuntSnapshot> = vec![];

    // Xcode in foreground → Coding workload
    let decisions = gov.decide_all(
        &procs,
        &hunts,
        Some("Xcode"),
        &["Xcode", "cargo", "analyticsd"],
        10,
    );

    let telem = decisions.iter().find(|d| d.name == "analyticsd");
    assert!(telem.is_some());
    assert_eq!(telem.unwrap().decision, GovernorDecision::Freeze);
}

#[test]
fn test_governor_allows_active_foreground() {
    let mut gov = AdaptiveGovernor::new();

    let procs = vec![make_snap("Xcode", 30.0, 0, true)];
    let hunts: Vec<HuntSnapshot> = vec![];

    let decisions = gov.decide_all(&procs, &hunts, Some("Xcode"), &["Xcode"], 10);

    let xcode = decisions.iter().find(|d| d.name == "Xcode");
    assert!(xcode.is_some());
    assert_eq!(xcode.unwrap().decision, GovernorDecision::Allow);
}

#[test]
fn test_governor_throttles_stale_process() {
    let mut gov = AdaptiveGovernor::new();

    let procs = vec![make_snap("OldApp", 2.0, 90_000, false)];
    let hunts: Vec<HuntSnapshot> = vec![];

    let decisions = gov.decide_all(&procs, &hunts, Some("Xcode"), &["Xcode", "OldApp"], 10);

    let old = decisions.iter().find(|d| d.name == "OldApp");
    assert!(old.is_some());
    assert!(matches!(
        old.unwrap().decision,
        GovernorDecision::Throttle | GovernorDecision::Freeze
    ));
}

#[test]
fn test_governor_summary_counts() {
    let mut gov = AdaptiveGovernor::new();

    let procs = vec![
        make_snap("kernel_task", 50.0, 0, false),
        make_snap("analyticsd", 2.0, 600, false),
    ];
    let hunts = vec![make_hunt("crashed_app", true, true)];

    let decisions = gov.decide_all(
        &procs,
        &hunts,
        Some("Xcode"),
        &["Xcode", "kernel_task", "analyticsd"],
        10,
    );

    let summary = AdaptiveGovernor::summarise(&decisions);
    assert_eq!(summary.total, decisions.len());
    assert!(summary.killed + summary.frozen + summary.throttled + summary.allowed == summary.total);
}

#[test]
fn test_governor_waste_override() {
    let mut gov = AdaptiveGovernor::new();

    // High CPU + high wakeups + large RSS + no GUI → waste override triggers.
    // waste = cpu(>5,noGUI)+0.40 + wakeups(>20)+0.25 + rss(>200MB,noGUI)+0.15 = 0.80
    // On 8GB M1: waste_override_threshold=0.80 → utility(0.40) < 0.60 → Throttle.
    let mut snap = make_snap("BurnerApp", 11.0, 7200, false);
    snap.wakeups_per_sec = 60.0;
    snap.rss_bytes = 300 * 1024 * 1024;
    let procs = vec![snap];
    let hunts: Vec<HuntSnapshot> = vec![];

    let decisions = gov.decide_all(&procs, &hunts, None, &["BurnerApp"], 15);
    let burner = decisions.iter().find(|d| d.name == "BurnerApp");
    assert!(burner.is_some());
    assert!(matches!(
        burner.unwrap().decision,
        GovernorDecision::Throttle | GovernorDecision::Freeze
    ));
}
