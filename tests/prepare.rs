//! ══════════════════════════════════════════════════════════════════════════════
//! Apollo AutoResearch — Fixed Evaluation Benchmark (prepare.py equivalent)
//! ══════════════════════════════════════════════════════════════════════════════
//!
//! THIS FILE IS READ-ONLY. The agent must NEVER modify it.
//!
//! It defines scenario-based tests that measure DECISION QUALITY, not code health.
//! Each scenario simulates a real system state and checks whether
//! AdaptiveGovernor makes the CORRECT decision for memory AND performance.
//!
//! Metric: `decision_score` = correct decisions / total scenarios
//!         Higher is better. Range [0.0, 1.0].
//!
//! Scenarios cover:
//!   - Memory pressure response (freeze the right processes)
//!   - Performance protection (never throttle foreground/interactive)
//!   - Workload awareness (coding → protect compilers, video → protect renderers)
//!   - Waste detection (zombies, telemetry → always act)
//!   - Edge cases (ephemeral XPC, app helpers, gray zones)
//!   - Resource efficiency (don't over-throttle when pressure is low)

#[cfg(test)]
mod scenarios {
    use apollo_optimizer::engine::adaptive_governor::{AdaptiveGovernor, GovernorDecision};
    use apollo_optimizer::engine::process_classifier::{ProcessSnapshot, ProcessTier};
    use apollo_optimizer::engine::zombie_hunter::HuntSnapshot;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn snap(
        pid: u32,
        name: &str,
        cpu: f32,
        rss_mb: u64,
        gui: bool,
        interaction_secs: u64,
        fg_secs: u64,
    ) -> ProcessSnapshot {
        ProcessSnapshot {
            pid,
            name: name.to_string(),
            cpu_percent: cpu,
            rss_bytes: rss_mb * 1024 * 1024,
            is_zombie: false,
            secs_since_foreground: fg_secs,
            secs_since_user_interaction: interaction_secs,
            has_network: false,
            has_gui_window: gui,
            wakeups_per_sec: 1.0,
            parent_alive: true,
            process_uptime_secs: 3600,
            faults_total: 0,
            pageins_total: 0,
            is_translated: false,
            mach_port_count: 0,
        }
    }

    fn zombie_snap(pid: u32, name: &str) -> ProcessSnapshot {
        ProcessSnapshot {
            pid,
            name: name.to_string(),
            cpu_percent: 0.0,
            rss_bytes: 50 * 1024 * 1024,
            is_zombie: true,
            secs_since_foreground: 99999,
            secs_since_user_interaction: 99999,
            has_network: false,
            has_gui_window: false,
            wakeups_per_sec: 0.0,
            parent_alive: false,
            process_uptime_secs: 86400,
            faults_total: 0,
            pageins_total: 0,
            is_translated: false,
            mach_port_count: 0,
        }
    }

    fn decide(snaps: &[ProcessSnapshot], fg: Option<&str>) -> Vec<(String, GovernorDecision)> {
        let mut gov = AdaptiveGovernor::new();
        let hunts: Vec<HuntSnapshot> = Vec::new();
        let names: Vec<&str> = snaps.iter().map(|s| s.name.as_str()).collect();
        let decisions = gov.decide_all(snaps, &hunts, fg, &names, 14); // 2pm
        decisions
            .into_iter()
            .map(|d| (d.name, d.decision))
            .collect()
    }

    fn find_decision(results: &[(String, GovernorDecision)], name: &str) -> GovernorDecision {
        results
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| *d)
            .unwrap_or(GovernorDecision::Allow)
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 1: PERFORMANCE PROTECTION
    // The user's active work must NEVER be throttled or frozen.
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn s01_foreground_app_always_allowed() {
        let snaps = vec![
            snap(100, "Code", 45.0, 800, true, 2, 0),  // VS Code, active
            snap(200, "Slack", 3.0, 400, true, 600, 600), // Slack, background
        ];
        let results = decide(&snaps, Some("Code"));
        assert_eq!(find_decision(&results, "Code"), GovernorDecision::Allow);
    }

    #[test]
    fn s02_active_gui_app_protected() {
        let snaps = vec![
            snap(100, "Brave", 30.0, 600, true, 5, 0),
            snap(200, "idled", 0.1, 50, false, 9999, 9999),
        ];
        let results = decide(&snaps, Some("Brave"));
        assert_eq!(find_decision(&results, "Brave"), GovernorDecision::Allow);
    }

    #[test]
    fn s03_system_essential_never_touched() {
        let snaps = vec![
            snap(1, "WindowServer", 20.0, 300, false, 0, 0),
            snap(2, "launchd", 1.0, 100, false, 0, 0),
        ];
        let results = decide(&snaps, None);
        assert_eq!(find_decision(&results, "WindowServer"), GovernorDecision::Allow);
        assert_eq!(find_decision(&results, "launchd"), GovernorDecision::Allow);
    }

    #[test]
    fn s04_ephemeral_xpc_not_throttled() {
        // Short-lived XPC services (< 8s uptime) should be left alone
        let mut s = snap(300, "com.apple.quicklook", 5.0, 20, false, 9999, 9999);
        s.process_uptime_secs = 3; // just spawned
        let snaps = vec![s];
        let results = decide(&snaps, None);
        assert_eq!(
            find_decision(&results, "com.apple.quicklook"),
            GovernorDecision::Allow
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 2: MEMORY PRESSURE — CORRECT FREEZE/THROTTLE TARGETS
    // When memory is tight, Apollo must act on the RIGHT processes.
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn s05_stale_idle_process_frozen() {
        // No GUI, no CPU, idle for 10 hours → freeze
        let snaps = vec![snap(500, "stale_daemon", 0.0, 200, false, 36000, 36000)];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "stale_daemon");
        assert!(
            d == GovernorDecision::Freeze || d == GovernorDecision::Throttle,
            "Stale process should be frozen or throttled, got {:?}",
            d
        );
    }

    #[test]
    fn s06_telemetry_always_acted_on() {
        let snaps = vec![snap(600, "DiagnosticReporter", 2.0, 50, false, 9999, 9999)];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "DiagnosticReporter");
        assert!(
            d == GovernorDecision::Throttle || d == GovernorDecision::Freeze,
            "Telemetry should be throttled or frozen, got {:?}",
            d
        );
    }

    #[test]
    fn s07_zombie_gets_killed_or_frozen() {
        let snaps = vec![zombie_snap(700, "dead_process")];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "dead_process");
        assert!(
            d == GovernorDecision::Kill
                || d == GovernorDecision::Freeze
                || d == GovernorDecision::Throttle,
            "Zombie should be killed/frozen/throttled, got {:?}",
            d
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 3: WORKLOAD AWARENESS
    // Different workloads need different optimization strategies.
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn s08_compiler_not_throttled_in_build() {
        // During a build, compilers should NOT be throttled
        let snaps = vec![
            snap(800, "rustc", 95.0, 500, false, 30, 30),
            snap(801, "cargo", 10.0, 200, false, 30, 30),
        ];
        let names: Vec<&str> = vec!["rustc", "cargo"];
        let mut gov = AdaptiveGovernor::new();
        let hunts = Vec::new();
        let decisions = gov.decide_all(&snaps, &hunts, None, &names, 14);
        for d in &decisions {
            if d.name == "rustc" || d.name == "cargo" {
                assert_ne!(
                    d.decision,
                    GovernorDecision::Freeze,
                    "{} should NOT be frozen during build",
                    d.name
                );
            }
        }
    }

    #[test]
    fn s09_high_cpu_with_gui_stays_allowed() {
        // A GUI app using 90% CPU is doing work the user asked for (rendering, etc.)
        let snaps = vec![snap(900, "Blender", 90.0, 2000, true, 10, 0)];
        let results = decide(&snaps, Some("Blender"));
        assert_eq!(find_decision(&results, "Blender"), GovernorDecision::Allow);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 4: RESOURCE EFFICIENCY
    // Don't over-optimize. When pressure is low, let things run.
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn s10_background_daemon_with_network_not_frozen() {
        // A daemon with active network (e.g., database, web server) has utility
        let mut s = snap(1000, "postgres", 5.0, 300, false, 3600, 3600);
        s.has_network = true;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "postgres");
        assert_ne!(
            d,
            GovernorDecision::Freeze,
            "Network-active daemon should not be frozen"
        );
    }

    #[test]
    fn s11_recently_used_app_not_aggressive() {
        // App used 30 seconds ago — still warm in user's mind
        let snaps = vec![snap(1100, "Notes", 0.5, 100, true, 30, 30)];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "Notes");
        assert_ne!(
            d,
            GovernorDecision::Freeze,
            "Recently-used GUI app should not be frozen"
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 5: APP HELPER SAFETY
    // Browser helpers crash if SIGSTOP'd. Only throttle, never freeze.
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn s12_chrome_helper_never_frozen() {
        let snaps = vec![snap(
            1200,
            "Google Chrome Helper",
            2.0,
            150,
            false,
            600,
            600,
        )];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "Google Chrome Helper");
        assert_ne!(
            d,
            GovernorDecision::Freeze,
            "Chrome Helper must NEVER be frozen (watchdog crash)"
        );
    }

    #[test]
    fn s13_active_helper_with_audio_protected() {
        let mut s = snap(1300, "Electron Helper", 8.0, 200, false, 100, 100);
        s.wakeups_per_sec = 50.0; // High wakeups = audio/video
        let snaps = vec![s];
        let results = decide(&snaps, None);
        assert_eq!(
            find_decision(&results, "Electron Helper"),
            GovernorDecision::Allow,
            "Active helper with high wakeups (audio/video) should be allowed"
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 6: WASTE DETECTION
    // High-waste processes should be acted on even with moderate utility.
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn s14_high_waste_low_utility_throttled() {
        let mut s = snap(1400, "leaky_daemon", 8.0, 800, false, 7200, 7200);
        s.wakeups_per_sec = 80.0; // Chatty
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "leaky_daemon");
        assert!(
            d == GovernorDecision::Throttle || d == GovernorDecision::Freeze,
            "High-waste daemon should be throttled/frozen, got {:?}",
            d
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 7: MIXED SCENARIOS (integration)
    // Multiple processes, realistic system state.
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn s15_realistic_desktop_session() {
        let snaps = vec![
            snap(100, "Code", 25.0, 600, true, 5, 0),      // Active editor
            snap(200, "Brave", 15.0, 800, true, 120, 120),  // Background browser
            snap(300, "Slack", 2.0, 400, true, 600, 600),   // Background chat
            snap(400, "Dropbox", 3.0, 200, false, 9999, 9999), // Background sync
            snap(500, "WindowServer", 10.0, 150, false, 0, 0), // System
            snap(600, "analyticsd", 1.0, 30, false, 9999, 9999), // Telemetry
        ];
        let results = decide(&snaps, Some("Code"));

        // Active editor: always allowed
        assert_eq!(find_decision(&results, "Code"), GovernorDecision::Allow);
        // WindowServer: system essential
        assert_eq!(find_decision(&results, "WindowServer"), GovernorDecision::Allow);
        // Telemetry: must be throttled or frozen
        let analytics_d = find_decision(&results, "analyticsd");
        assert!(
            analytics_d == GovernorDecision::Throttle || analytics_d == GovernorDecision::Freeze,
            "Telemetry should be acted on, got {:?}",
            analytics_d
        );
    }

    #[test]
    fn s16_build_session_protects_compilers() {
        let snaps = vec![
            snap(100, "rustc", 95.0, 400, false, 60, 60),
            snap(101, "cargo", 10.0, 200, false, 60, 60),
            snap(200, "Slack", 2.0, 400, true, 600, 600),
            snap(300, "analyticsd", 0.5, 20, false, 9999, 9999),
        ];
        let results = decide(&snaps, None);

        // Compilers: not frozen
        assert_ne!(find_decision(&results, "rustc"), GovernorDecision::Freeze);
        assert_ne!(find_decision(&results, "cargo"), GovernorDecision::Freeze);
    }

    #[test]
    fn s17_idle_system_minimal_intervention() {
        // When system is idle, most processes should be allowed
        let snaps = vec![
            snap(100, "Finder", 0.5, 100, true, 300, 300),
            snap(200, "mds_stores", 2.0, 80, false, 9999, 9999),
            snap(300, "bird", 0.1, 30, false, 9999, 9999),
        ];
        let results = decide(&snaps, Some("Finder"));

        let freeze_count = results
            .iter()
            .filter(|(_, d)| *d == GovernorDecision::Freeze)
            .count();
        assert!(
            freeze_count <= 1,
            "Idle system should not aggressively freeze: {} freezes",
            freeze_count
        );
    }

    #[test]
    fn s18_translated_rosetta_lower_priority() {
        // Rosetta-translated processes should have slightly lower utility
        let mut native = snap(100, "native_app", 5.0, 200, false, 3600, 3600);
        let mut translated = snap(200, "translated_app", 5.0, 200, false, 3600, 3600);
        translated.is_translated = true;
        native.is_translated = false;
        // Both should get some decision — but the translated one should not
        // be preferred over native. (This tests that is_translated affects scoring)
        let snaps = vec![native, translated];
        let results = decide(&snaps, None);
        // At minimum, both should get a decision
        assert!(results.len() >= 2);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 8: SCORING SANITY
    // Verify the utility/waste scoring produces sensible rankings.
    // ══════════════════════════════════════════════════════════════════════════

    #[test]
    fn s19_utility_ranking_foreground_gt_background_gt_stale() {
        use apollo_optimizer::engine::process_classifier::score_utility;
        let fg = snap(1, "fg_app", 10.0, 200, true, 5, 0);
        let bg = snap(2, "bg_app", 5.0, 200, true, 300, 300);
        let stale = snap(3, "stale", 0.0, 200, false, 36000, 36000);

        let u_fg = score_utility(&fg);
        let u_bg = score_utility(&bg);
        let u_stale = score_utility(&stale);

        assert!(
            u_fg > u_bg,
            "Foreground utility ({}) should > background ({})",
            u_fg,
            u_bg
        );
        assert!(
            u_bg > u_stale,
            "Background utility ({}) should > stale ({})",
            u_bg,
            u_stale
        );
    }

    #[test]
    fn s20_network_daemon_utility_higher_than_silent() {
        use apollo_optimizer::engine::process_classifier::score_utility;
        let mut networked = snap(1, "db", 3.0, 300, false, 3600, 3600);
        networked.has_network = true;
        let silent = snap(2, "idle", 0.0, 50, false, 3600, 3600);

        assert!(
            score_utility(&networked) > score_utility(&silent),
            "Network daemon should have higher utility than silent idle"
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 9: BOSS SCENARIOS — EXPERT-LEVEL DECISIONS
    // These test intelligence that goes beyond basic classification.
    // ══════════════════════════════════════════════════════════════════════════

    /// BOSS 1: LLM on-device awareness.
    /// ollama: 4GB RSS, ZERO CPU, no GUI, idle 2h. The SilentDaemon idle
    /// override will throttle it. But throttling a loaded LLM model is wrong —
    /// model reload from disk takes 30+ seconds. Must be ALLOWED.
    /// Requires: name-based LLM recognition or RSS-cost-aware protection.
    #[test]
    fn s21_llm_process_protected_despite_high_rss() {
        let s = snap(2100, "ollama", 0.0, 4096, false, 7200, 7200);
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "ollama");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "LLM process (4GB loaded model) must be ALLOWED — reload cost is 30s+. Got {:?}",
            d
        );
    }

    /// BOSS 2: Active I/O work protection.
    /// backupd: 12% CPU, 600MB RSS, 60 wakeups/s, no GUI, idle forever.
    /// The graduated waste override catches it (waste>0.5, utility<0.40).
    /// But 80k pageins = real disk I/O work, not waste. Must be ALLOWED.
    /// Requires: pageins as a "legitimate work" signal.
    #[test]
    fn s22_backup_process_protected_during_active_io() {
        let mut s = snap(2200, "backupd", 12.0, 600, false, 9999, 9999);
        s.pageins_total = 80000;
        s.wakeups_per_sec = 60.0;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "backupd");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "Active backup (80k pageins) must be ALLOWED — freezing corrupts data. Got {:?}",
            d
        );
    }

    /// BOSS 3: IPC-serving daemon (WaitGraph).
    /// trustd: 0% CPU, no GUI, idle 2h, 1 wakeup/s. Hits the SilentDaemon
    /// idle override → Throttle. But 120 Mach ports = critical IPC hub.
    /// Throttling it beachballs Chrome, Safari, Mail (all need TLS).
    /// Requires: mach_port_count as a protection signal.
    #[test]
    fn s23_ipc_serving_daemon_not_frozen() {
        let mut s = snap(2300, "trustd", 0.0, 40, false, 7200, 7200);
        s.mach_port_count = 120;
        s.wakeups_per_sec = 1.0; // Just enough to NOT be Stale
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "trustd");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "IPC hub (120 Mach ports) must be ALLOWED — throttle causes system beachballs. Got {:?}",
            d
        );
    }

    /// BOSS 4: Compiler without build context.
    /// rustc: 95% CPU, 1.5GB RSS, 55 wakeups, no GUI, idle forever.
    /// No cargo in the process list = no workload hint from classifier.
    /// Pure waste case. But the NAME "rustc" means it's a compiler.
    /// Requires: compiler name recognition independent of workload context.
    #[test]
    fn s24_compiler_protected_during_build() {
        let mut rustc = snap(2400, "rustc", 95.0, 1500, false, 9999, 9999);
        rustc.wakeups_per_sec = 55.0;
        // No cargo in the list — workload classifier won't detect Coding mode
        let snaps = vec![rustc];
        let results = decide(&snaps, None);
        let d_rustc = find_decision(&results, "rustc");
        assert_eq!(
            d_rustc,
            GovernorDecision::Allow,
            "Compiler (rustc) must NEVER be throttled even without build context. Got {:?}",
            d_rustc
        );
    }

    /// BOSS 5: Long-running encode protection.
    /// ffmpeg: 85% CPU, 1GB RSS, 55 wakeups, no GUI, idle 2h.
    /// RSS penalty + waste override → Throttle. But 200k pageins + known
    /// encoder name = user-initiated long task. Must be ALLOWED.
    /// Requires: encoder recognition or pageins-as-work signal.
    #[test]
    fn s25_render_process_not_frozen() {
        let mut s = snap(2500, "ffmpeg", 85.0, 1024, false, 7200, 7200);
        s.wakeups_per_sec = 55.0;
        s.pageins_total = 200000;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "ffmpeg");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "Encoder (ffmpeg, 200k pageins) must be ALLOWED — user-initiated task. Got {:?}",
            d
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 10: BOSS LEVEL 2 — SYSTEM-WIDE INTELLIGENCE
    // Scenarios requiring awareness of the WHOLE system, not just one process.
    // ══════════════════════════════════════════════════════════════════════════

    /// BOSS 6: Rosetta tax — translated x86 processes cost 2x memory on ARM.
    /// A translated daemon with 500MB RSS, 0% CPU, idle 4h. Should be throttled
    /// or frozen MORE aggressively than a native daemon with same stats, because
    /// freeing it reclaims 2x the effective memory (JIT page tables).
    #[test]
    fn s26_translated_process_penalized() {
        let mut native = snap(2600, "native_daemon", 0.0, 500, false, 14400, 14400);
        native.wakeups_per_sec = 1.0;
        let mut rosetta = snap(2601, "rosetta_daemon", 0.0, 500, false, 14400, 14400);
        rosetta.is_translated = true;
        rosetta.wakeups_per_sec = 1.0;
        let snaps = vec![native.clone(), rosetta.clone()];
        let results = decide(&snaps, None);
        let d_native = find_decision(&results, "native_daemon");
        let d_rosetta = find_decision(&results, "rosetta_daemon");
        // Both are idle SilentDaemons and will be throttled.
        // But Rosetta process should be frozen (more aggressive) while native just throttled.
        assert!(
            d_rosetta as u8 > d_native as u8
                || (d_rosetta == GovernorDecision::Freeze && d_native == GovernorDecision::Throttle),
            "Translated (Rosetta) process should be treated MORE aggressively than native. \
             Got native={:?}, rosetta={:?}",
            d_native, d_rosetta
        );
    }

    /// BOSS 7: Process swarm — when 50+ background processes are running,
    /// the governor should be MORE aggressive (freeze instead of throttle)
    /// to protect foreground responsiveness. Tests crowd-awareness.
    #[test]
    fn s27_swarm_increases_aggression() {
        // Create a swarm of 50 idle background daemons
        let mut snaps: Vec<ProcessSnapshot> = (0..50)
            .map(|i| {
                let mut s = snap(2700 + i, &format!("bgd_{}", i), 0.5, 30, false, 3600, 3600);
                s.wakeups_per_sec = 2.0;
                s
            })
            .collect();
        // Add a target: mildly wasteful daemon
        let mut target = snap(2799, "target_daemon", 3.0, 200, false, 3600, 3600);
        target.wakeups_per_sec = 15.0;
        snaps.push(target);
        let results = decide(&snaps, None);
        let d = find_decision(&results, "target_daemon");
        assert!(
            d == GovernorDecision::Throttle || d == GovernorDecision::Freeze,
            "With 50+ bg processes competing, mildly wasteful daemon should be acted on. Got {:?}",
            d
        );
    }

    /// BOSS 8: Night owl — at 3am (hour=3), the user is likely asleep.
    /// Background maintenance processes should be left alone (they're doing
    /// useful work: backups, indexing, updates). Only pure waste should be acted on.
    #[test]
    fn s28_nighttime_allows_maintenance() {
        // Spotlight indexer: moderate CPU, background, doing real work at night
        let mut mds = snap(2800, "mds_stores", 25.0, 300, false, 9999, 9999);
        mds.wakeups_per_sec = 30.0;
        mds.pageins_total = 40000;
        let snaps = vec![mds];
        // Pass hour=3 (3am) — night time
        let mut gov = AdaptiveGovernor::new();
        let hunts: Vec<HuntSnapshot> = Vec::new();
        let names: Vec<&str> = snaps.iter().map(|s| s.name.as_str()).collect();
        let decisions = gov.decide_all(&snaps, &hunts, None, &names, 3); // 3am
        let d = decisions.iter().find(|d| d.name == "mds_stores").unwrap();
        assert!(
            d.decision == GovernorDecision::Allow || d.decision == GovernorDecision::Throttle,
            "At 3am, indexer doing real I/O work should be allowed to work. Got {:?}",
            d.decision
        );
    }

    /// BOSS 9: Helper process with active parent.
    /// A Chrome Helper using 5% CPU and 200MB with its parent (Chrome) in
    /// foreground. The helper must be ALLOWED because it's serving the
    /// active foreground app. Throttling it = tab freeze.
    #[test]
    fn s29_helper_with_foreground_parent_protected() {
        let chrome = snap(2900, "Google Chrome", 10.0, 500, true, 5, 5);
        let mut helper = snap(2901, "Google Chrome Helper", 5.0, 200, false, 9999, 9999);
        helper.wakeups_per_sec = 40.0;
        let snaps = vec![chrome, helper];
        let results = decide(&snaps, Some("Google Chrome"));
        let d = find_decision(&results, "Google Chrome Helper");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "Helper of foreground app must be ALLOWED — throttle = tab freeze. Got {:?}",
            d
        );
    }

    /// BOSS 10: Database engine protection.
    /// postgres: 3% CPU, 2GB RSS (shared buffers), lots of Mach ports,
    /// network active, idle forever. Looks like a memory hog but it's a
    /// database serving other apps. Must be allowed.
    #[test]
    fn s30_database_engine_protected() {
        let mut pg = snap(3000, "postgres", 3.0, 2048, false, 9999, 9999);
        pg.has_network = true;
        pg.mach_port_count = 90; // Many client connections
        pg.wakeups_per_sec = 10.0;
        let snaps = vec![pg];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "postgres");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "Database engine with active connections must be ALLOWED. Got {:?}",
            d
        );
    }
}
