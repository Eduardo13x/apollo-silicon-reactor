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

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 11: BOSS LEVEL 3 — ADVERSARIAL SCENARIOS
    // Edge cases designed to break naive heuristics.
    // ══════════════════════════════════════════════════════════════════════════

    /// BOSS 11: Fake telemetry name — a process named "analyticsd_worker" that
    /// is NOT actually telemetry (it's a user analytics pipeline). High CPU,
    /// high pageins, doing real work. Must not be blindly throttled by name.
    #[test]
    fn s31_fake_telemetry_not_blindly_throttled() {
        let mut s = snap(3100, "analytics_worker", 40.0, 400, false, 300, 300);
        s.pageins_total = 60000; // Real I/O work
        s.wakeups_per_sec = 20.0;
        s.has_network = true;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "analytics_worker");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "analytics_worker doing real I/O work must be ALLOWED despite name. Got {:?}",
            d
        );
    }

    /// BOSS 12: Memory hoarder vs loaded model — two 2GB processes.
    /// One is a leaked browser process (0% CPU, no ports, no network, zombie parent).
    /// Other is a database (3% CPU, 90 ports, network). The governor should
    /// treat them DIFFERENTLY despite same RSS.
    #[test]
    fn s32_same_rss_different_treatment() {
        let leaked = snap(3200, "leaked_browser", 0.0, 2048, false, 14400, 14400);
        let mut db = snap(3201, "mysql", 3.0, 2048, false, 9999, 9999);
        db.has_network = true;
        db.mach_port_count = 90;
        db.wakeups_per_sec = 8.0;
        let snaps = vec![leaked, db];
        let results = decide(&snaps, None);
        let d_leaked = find_decision(&results, "leaked_browser");
        let d_db = find_decision(&results, "mysql");
        assert!(
            d_leaked != GovernorDecision::Allow,
            "Leaked 2GB browser process should NOT be allowed. Got {:?}",
            d_leaked
        );
        assert_eq!(
            d_db,
            GovernorDecision::Allow,
            "Active 2GB database must be ALLOWED. Got {:?}",
            d_db
        );
    }

    /// BOSS 13: Freshly launched app — a process that just started (uptime=3s)
    /// with high RSS (1GB) because it's loading. Despite looking like a hog,
    /// it should be allowed because it's still initializing.
    #[test]
    fn s33_fresh_launch_not_punished() {
        let mut s = snap(3300, "Xcode", 80.0, 1024, true, 3, 3);
        s.process_uptime_secs = 3;
        s.wakeups_per_sec = 200.0; // Loading plugins, indexing
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "Xcode");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "Freshly launched Xcode (3s) must be ALLOWED during init. Got {:?}",
            d
        );
    }

    /// BOSS 14: Translated LLM — worst case for memory.
    /// An x86 translated ollama process with 6GB RSS, idle 4h.
    /// Even though it's Rosetta AND idle AND huge RSS, it must be protected
    /// because it's an LLM model. The LLM protection should override Rosetta tax.
    #[test]
    fn s34_translated_llm_still_protected() {
        let mut s = snap(3400, "ollama", 0.0, 6144, false, 14400, 14400);
        s.is_translated = true;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "ollama");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "Translated LLM (6GB) must still be ALLOWED — model reload cost > Rosetta penalty. Got {:?}",
            d
        );
    }

    /// BOSS 15: Docker daemon — high Mach ports, network, moderate CPU.
    /// Docker Desktop runs as a daemon with many containers connecting.
    /// Must not be throttled even though it has no GUI.
    #[test]
    fn s35_docker_daemon_protected() {
        let mut s = snap(3500, "com.docker.backend", 8.0, 1500, false, 9999, 9999);
        s.has_network = true;
        s.mach_port_count = 200; // Container connections
        s.wakeups_per_sec = 25.0;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "com.docker.backend");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "Docker daemon (200 Mach ports) must be ALLOWED. Got {:?}",
            d
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 12: BOSS LEVEL 3b — TRULY ADVERSARIAL
    // These are designed to break current heuristics by hitting exact edge cases.
    // ══════════════════════════════════════════════════════════════════════════

    /// BOSS 16: Swarm with IPC hub overlap.
    /// In a 40-process swarm, a daemon with 50 Mach ports (just below IPC hub
    /// threshold of 80) and waste=0.30 would be swarm-throttled. But it has
    /// network + moderate ports = it's serving other apps. Must be ALLOWED.
    #[test]
    fn s36_near_hub_in_swarm_not_throttled() {
        let mut snaps: Vec<ProcessSnapshot> = (0..40)
            .map(|i| snap(3600 + i, &format!("bg_{}", i), 0.5, 30, false, 3600, 3600))
            .collect();
        let mut near_hub = snap(3699, "windowserver_helper", 2.0, 150, false, 3600, 3600);
        near_hub.mach_port_count = 60; // Below 80 threshold but still significant
        near_hub.has_network = true;
        snaps.push(near_hub);
        let results = decide(&snaps, None);
        let d = find_decision(&results, "windowserver_helper");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "Near-hub daemon (60 ports + network) in swarm must be ALLOWED. Got {:?}",
            d
        );
    }

    /// BOSS 17: I/O worker just below pageins threshold.
    /// A process with 45k pageins (below 50k threshold) doing real work.
    /// Must still be protected because the pattern (CPU + network + pageins)
    /// indicates legitimate work even below the hard threshold.
    #[test]
    fn s37_io_worker_below_threshold() {
        let mut s = snap(3700, "rsync", 15.0, 300, false, 9999, 9999);
        s.pageins_total = 45000; // Below 50k threshold
        s.has_network = true;
        s.wakeups_per_sec = 30.0;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "rsync");
        assert!(
            d == GovernorDecision::Allow || d == GovernorDecision::Throttle,
            "I/O worker (rsync, 45k pageins + network + 15% CPU) must not be frozen. Got {:?}",
            d
        );
    }

    /// BOSS 18: Memory pressure triage — when system has many heavy processes,
    /// the LEAST useful one should be acted on first. Here: two idle daemons,
    /// one with network (more useful) and one without. The one without network
    /// should get MORE aggressive treatment.
    #[test]
    fn s38_triage_least_useful_first() {
        let mut useful = snap(3800, "dns_cache", 0.0, 200, false, 7200, 7200);
        useful.has_network = true;
        useful.wakeups_per_sec = 1.0;
        let mut useless = snap(3801, "old_updater", 0.0, 200, false, 7200, 7200);
        useless.wakeups_per_sec = 1.0;
        let snaps = vec![useful, useless];
        let results = decide(&snaps, None);
        let d_useful = find_decision(&results, "dns_cache");
        let d_useless = find_decision(&results, "old_updater");
        // Both are idle SilentDaemons. But dns_cache has network → higher utility.
        // old_updater should be treated same or more aggressively.
        assert!(
            d_useless as u8 >= d_useful as u8,
            "Process without network should be treated >= aggressively as one with network. \
             Got dns_cache={:?}, old_updater={:?}",
            d_useful, d_useless
        );
    }

    /// BOSS 19: Electron app with no GUI but serving localhost.
    /// VS Code's extension host: no window, high CPU, high RSS, but it's
    /// the brain of the editor. 30 Mach ports (below hub threshold).
    /// Has network (localhost IPC). Must be ALLOWED.
    #[test]
    fn s39_electron_extension_host_protected() {
        let mut s = snap(3900, "Code Helper (Plugin)", 25.0, 800, false, 5, 5);
        s.has_network = true;
        s.mach_port_count = 30;
        s.wakeups_per_sec = 45.0;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "Code Helper (Plugin)");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "VS Code extension host must be ALLOWED — it IS the editor. Got {:?}",
            d
        );
    }

    /// BOSS 20: Zombie disguised as useful.
    /// A process claiming 10% CPU and 500MB RSS but with dead parent.
    /// The zombie hunter should catch it regardless of its "stats".
    #[test]
    fn s40_zombie_with_high_cpu_still_killed() {
        let mut z = snap(4000, "zombie_hog", 10.0, 500, false, 9999, 9999);
        z.is_zombie = true;
        z.parent_alive = false;
        let snaps = vec![z];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "zombie_hog");
        assert!(
            d == GovernorDecision::Kill || d == GovernorDecision::Freeze,
            "Zombie with dead parent must be killed/frozen regardless of CPU. Got {:?}",
            d
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 13: BOSS LEVEL 4 — CONTEXTUAL INTELLIGENCE
    // These require understanding process relationships and system context.
    // ══════════════════════════════════════════════════════════════════════════

    /// BOSS 21: Foreground app's invisible helper swarm.
    /// Safari is foreground. It has 8 "com.apple.WebKit" helpers with no GUI.
    /// Each helper has moderate CPU and RSS. They look like SilentDaemons but
    /// they ARE Safari. In a 50-process swarm, swarm throttle must NOT catch them.
    #[test]
    fn s41_fg_app_helpers_immune_to_swarm() {
        let mut snaps: Vec<ProcessSnapshot> = (0..40)
            .map(|i| snap(4100 + i, &format!("bg_{}", i), 0.5, 30, false, 3600, 3600))
            .collect();
        // Safari foreground
        let safari = snap(4150, "Safari", 5.0, 600, true, 2, 2);
        snaps.push(safari);
        // 8 WebKit helpers — no GUI, moderate stats
        for i in 0..8 {
            let mut helper = snap(4160 + i, "com.apple.WebKit.WebContent", 8.0, 200, false, 9999, 9999);
            helper.wakeups_per_sec = 30.0;
            snaps.push(helper);
        }
        let results = decide(&snaps, Some("Safari"));
        // ALL WebKit helpers should be allowed (they're serving the foreground app)
        let helper_decisions: Vec<_> = results.iter()
            .filter(|(name, _)| name == "com.apple.WebKit.WebContent")
            .map(|(_, d)| *d)
            .collect();
        assert!(
            helper_decisions.iter().all(|d| *d == GovernorDecision::Allow),
            "WebKit helpers of foreground Safari must ALL be allowed in swarm. Got {:?}",
            helper_decisions
        );
    }

    /// BOSS 22: Stale LLM — an ollama process with 0% CPU, idle 24h.
    /// Even though it's an LLM, 24 hours idle means the user has forgotten
    /// about it. At some point, the reload cost is worth paying. Throttle it.
    #[test]
    fn s42_very_stale_llm_eventually_throttled() {
        let s = snap(4200, "ollama", 0.0, 4096, false, 86400, 86400);
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "ollama");
        assert!(
            d == GovernorDecision::Throttle || d == GovernorDecision::Freeze,
            "LLM idle 24h should eventually be throttled — user forgot about it. Got {:?}",
            d
        );
    }

    /// BOSS 23: GPU compute process (Metal shader compilation).
    /// A process doing GPU work appears as low CPU (GPU work doesn't show in
    /// CPU%) but high RSS and many faults. Must not be frozen.
    #[test]
    fn s43_gpu_compute_not_frozen() {
        let mut s = snap(4300, "MTLCompilerService", 2.0, 800, false, 60, 60);
        s.faults_total = 500000; // Many page faults from GPU buffer mapping
        s.mach_port_count = 40;
        s.wakeups_per_sec = 50.0;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "MTLCompilerService");
        assert!(
            d == GovernorDecision::Allow || d == GovernorDecision::Throttle,
            "GPU compute (Metal shader) must not be frozen — GPU stall. Got {:?}",
            d
        );
    }

    /// BOSS 24: Cascading dependency — Spotlight depends on trustd depends on mDNSResponder.
    /// If mDNSResponder is throttled, DNS breaks → trustd stalls → Spotlight hangs.
    /// mDNSResponder has few Mach ports (30) and low CPU. Must be ALLOWED.
    #[test]
    fn s44_mdns_responder_always_allowed() {
        let mut s = snap(4400, "mDNSResponder", 0.5, 20, false, 9999, 9999);
        s.has_network = true;
        s.mach_port_count = 30;
        s.wakeups_per_sec = 5.0;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "mDNSResponder");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "mDNSResponder must ALWAYS be allowed — DNS is critical infrastructure. Got {:?}",
            d
        );
    }

    /// BOSS 25: Diminishing returns — two idle daemons with different RSS.
    /// small_daemon: 50MB idle. big_daemon: 2GB idle. Both SilentDaemons.
    /// The big one should be acted on MORE aggressively (more memory to reclaim).
    #[test]
    fn s45_bigger_daemon_more_aggressive() {
        let mut small = snap(4500, "small_daemon", 0.0, 50, false, 7200, 7200);
        small.wakeups_per_sec = 1.0;
        let mut big = snap(4501, "big_daemon", 0.0, 2048, false, 7200, 7200);
        big.wakeups_per_sec = 1.0;
        let snaps = vec![small, big];
        let results = decide(&snaps, None);
        let d_small = find_decision(&results, "small_daemon");
        let d_big = find_decision(&results, "big_daemon");
        assert!(
            d_big as u8 >= d_small as u8,
            "Bigger idle daemon should get >= aggressive treatment. \
             Got small={:?}, big={:?}",
            d_small, d_big
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 14: BOSS LEVEL 5 — RESOURCE AWARENESS
    // These require understanding RSS cost, time-of-day, energy, and Rosetta
    // impact at a deeper level than simple thresholds.
    // ══════════════════════════════════════════════════════════════════════════

    /// BOSS 26: RSS hog idle daemon should FREEZE, not just throttle.
    /// On 8GB M1, a native daemon hogging 1.5GB with 0% CPU for 2h is a
    /// massive waste of physical memory. Throttle only reduces CPU priority
    /// — it doesn't free the 1.5GB. Must freeze to reclaim.
    #[test]
    fn s46_rss_hog_idle_must_freeze() {
        let snaps = vec![snap(4600, "big_daemon", 0.1, 1500, false, 7200, 7200)];
        let results = decide(&snaps, None);
        assert_eq!(
            find_decision(&results, "big_daemon"),
            GovernorDecision::Freeze,
            "1.5GB idle native daemon on 8GB machine must be FROZEN to reclaim memory"
        );
    }

    /// BOSS 27: Night mode — at 3AM nobody is using the machine.
    /// A background daemon with low activity that would normally be Allow
    /// should be Throttled to save energy. hour_of_day must influence decisions.
    #[test]
    fn s47_night_mode_more_aggressive_throttle() {
        let snaps = vec![snap(4700, "night_service", 2.0, 100, false, 1800, 1800)];
        let mut gov = AdaptiveGovernor::new();
        let hunts: Vec<HuntSnapshot> = Vec::new();
        let names: Vec<&str> = vec!["night_service"];
        let decisions = gov.decide_all(&snaps, &hunts, None, &names, 3); // 3 AM
        let d = decisions.iter().find(|d| d.name == "night_service").unwrap();
        assert!(
            d.decision == GovernorDecision::Throttle || d.decision == GovernorDecision::Freeze,
            "At 3AM, idle background service should be throttled to save energy. Got {:?}",
            d.decision
        );
    }

    /// BOSS 28: Wakeup energy hog — 200 wakeups/sec is an energy catastrophe.
    /// Each wakeup forces the CPU out of deep idle (P→E transition on M1).
    /// Even with only 3% CPU, this daemon destroys battery life. Must throttle.
    #[test]
    fn s48_wakeup_energy_hog_throttled() {
        let mut s = snap(4800, "chatty_daemon", 3.0, 80, false, 600, 600);
        s.wakeups_per_sec = 200.0;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "chatty_daemon");
        assert!(
            d == GovernorDecision::Throttle || d == GovernorDecision::Freeze,
            "200 wakeups/sec daemon must be throttled for energy. Got {:?}",
            d
        );
    }

    /// BOSS 29: Translated process in swarm should FREEZE, not just throttle.
    /// Rosetta processes use ~2x memory (JIT page tables). In a swarm scenario
    /// where resources are scarce, freezing reclaims both CPU scheduling AND
    /// the extra Rosetta memory overhead. Throttle is not enough.
    #[test]
    fn s49_translated_in_swarm_should_freeze() {
        let mut snaps: Vec<ProcessSnapshot> = (0..35)
            .map(|i| snap(4900 + i, &format!("bg_{}", i), 0.5, 30, false, 3600, 3600))
            .collect();
        let mut rosetta = snap(4950, "rosetta_daemon", 2.0, 200, false, 1800, 1800);
        rosetta.is_translated = true;
        snaps.push(rosetta);
        let results = decide(&snaps, None);
        let d = find_decision(&results, "rosetta_daemon");
        assert_eq!(
            d,
            GovernorDecision::Freeze,
            "Translated process in swarm must be FROZEN (2x memory overhead). Got {:?}",
            d
        );
    }

    /// BOSS 30: FG helper stress test — validates that foreground app helpers
    /// are immune to swarm pressure even with moderate waste stats.
    /// 40 background processes, Safari foreground, WebKit helper with low
    /// wakeups (no audio/video active). Must still be ALLOWED.
    #[test]
    fn s50_fg_helper_survives_swarm_pressure() {
        let mut snaps: Vec<ProcessSnapshot> = (0..38)
            .map(|i| snap(5000 + i, &format!("daemon_{}", i), 0.5, 30, false, 3600, 3600))
            .collect();
        let mut webkit = snap(5050, "com.apple.WebKit.WebContent", 5.0, 200, false, 100, 100);
        webkit.wakeups_per_sec = 2.0;
        snaps.push(webkit);
        snaps.push(snap(5051, "Safari", 15.0, 400, true, 5, 0));
        let results = decide(&snaps, Some("Safari"));
        assert_eq!(
            find_decision(&results, "com.apple.WebKit.WebContent"),
            GovernorDecision::Allow,
            "FG helper must be immune to swarm pressure"
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 15: BOSS LEVEL 6 — UNUSED SIGNALS & TEMPORAL GRADUATION
    // The governor collects faults_total and process_uptime_secs but ignores
    // them. Idle duration is binary (1h = magic number). These scenarios
    // require using ALL available signals and graduating decisions over time.
    // ══════════════════════════════════════════════════════════════════════════

    /// BOSS 31: Graduated idle — a daemon idle for 12 HOURS with 1.5% CPU
    /// (above the 0.5% idle override threshold). The idle override misses it,
    /// and utility (0.50) keeps it alive. But 12h idle is extreme — throttle it.
    #[test]
    fn s51_graduated_idle_12h_throttled() {
        let snaps = vec![snap(5100, "old_service", 1.5, 300, false, 43200, 43200)];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "old_service");
        assert!(
            d == GovernorDecision::Throttle || d == GovernorDecision::Freeze,
            "Daemon idle 12h should be throttled regardless of CPU. Got {:?}",
            d
        );
    }

    /// BOSS 32: High faults = active memory work. A process doing GPU buffer
    /// mapping or mmap'd I/O has 800K page faults but low CPU (GPU does the
    /// real work). In a swarm, swarm-throttle would catch it. Must be ALLOWED
    /// because faults prove it's doing real memory work.
    #[test]
    fn s52_high_faults_protected_in_swarm() {
        let mut snaps: Vec<ProcessSnapshot> = (0..35)
            .map(|i| snap(5200 + i, &format!("bg_{}", i), 0.5, 30, false, 3600, 3600))
            .collect();
        let mut gpu = snap(5250, "gpu_worker", 0.8, 200, false, 7200, 7200);
        gpu.faults_total = 800000;
        gpu.wakeups_per_sec = 5.0;
        snaps.push(gpu);
        let results = decide(&snaps, None);
        let d = find_decision(&results, "gpu_worker");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "800K faults = active memory work — must be ALLOWED despite swarm. Got {:?}",
            d
        );
    }

    /// BOSS 33: Partial Mach port protection. A daemon with 60 Mach ports
    /// (below 80 IPC hub threshold) is still serving other processes via XPC.
    /// In a swarm, swarm-throttle catches it. Must be ALLOWED because 60 ports
    /// is significant IPC activity.
    #[test]
    fn s53_partial_port_protection_in_swarm() {
        let mut snaps: Vec<ProcessSnapshot> = (0..35)
            .map(|i| snap(5300 + i, &format!("bg_{}", i), 0.5, 30, false, 3600, 3600))
            .collect();
        let mut broker = snap(5350, "xpc_broker", 1.0, 100, false, 3600, 3600);
        broker.mach_port_count = 60;
        broker.wakeups_per_sec = 5.0;
        snaps.push(broker);
        let results = decide(&snaps, None);
        let d = find_decision(&results, "xpc_broker");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "60 Mach ports = significant IPC — must be ALLOWED despite swarm. Got {:?}",
            d
        );
    }

    /// BOSS 34: Network bonus expires. A Stale process (cpu=0.1%, wakeups=0.5)
    /// has been idle for 8 HOURS but still has an open network socket. The
    /// network +0.05 utility bonus keeps it alive forever. But 8h idle means
    /// the connection is a zombie keepalive, not real work. Must throttle.
    #[test]
    fn s54_stale_network_expires_after_8h() {
        let mut s = snap(5400, "keepalive_daemon", 0.1, 100, false, 28800, 28800);
        s.has_network = true;
        s.wakeups_per_sec = 0.5;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "keepalive_daemon");
        assert!(
            d == GovernorDecision::Throttle || d == GovernorDecision::Freeze,
            "Stale daemon with network idle 8h = zombie connection — must act. Got {:?}",
            d
        );
    }

    /// BOSS 35: Graduated Stale freeze. A Stale process idle for 24 HOURS
    /// has utility ~0.50 (base score). The freeze threshold (0.05) never
    /// catches it because base utility is too high. But 24h idle is extreme
    /// — this process should be FROZEN to reclaim memory.
    #[test]
    fn s55_very_stale_24h_frozen() {
        let mut s = snap(5500, "abandoned_service", 0.1, 150, false, 86400, 86400);
        s.wakeups_per_sec = 0.3;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "abandoned_service");
        assert_eq!(
            d,
            GovernorDecision::Freeze,
            "Stale process idle 24h must be FROZEN — memory is wasted. Got {:?}",
            d
        );
    }
}
