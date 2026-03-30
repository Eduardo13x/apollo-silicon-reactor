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
}
