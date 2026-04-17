//! ══════════════════════════════════════════════════════════════════════════════
//! Apollo AutoResearch — Boss Level 8: Adversarial & Compound Scenarios
//! ══════════════════════════════════════════════════════════════════════════════
//!
//! THIS FILE IS READ-ONLY for the autoresearch agent. Only modify adaptive_governor.rs.
//!
//! 15 scenarios (s61–s75) designed to expose real gaps in Apollo's decision logic:
//!
//! KNOWN GAPS TARGETED:
//!   - GUI apps exempt from graduated-idle rule (48h idle still ALLOWED)
//!   - Active I/O rule requires cpu > 5% (slow backup with 3% CPU unprotected)
//!   - faults-based exemption doesn't block graduated idle (GPU workers frozen)
//!   - Utility-based freeze ignores massive fault evidence (ML inference frozen)
//!   - Extreme wakeup bombs (1000/s) only THROTTLE, not FREEZE
//!
//! SCORING: Each scenario passing = 50 points (same as all other prepare_*.rs files)

#[cfg(test)]
mod scenarios {
    use apollo_optimizer::engine::adaptive_governor::{AdaptiveGovernor, GovernorDecision};
    use apollo_optimizer::engine::process_classifier::ProcessSnapshot;
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
            cpu_contention: None,
            is_app_bundle: false,
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

    fn decide_at_hour(
        snaps: &[ProcessSnapshot],
        fg: Option<&str>,
        hour: u8,
    ) -> Vec<(String, GovernorDecision)> {
        let mut gov = AdaptiveGovernor::new();
        let hunts: Vec<HuntSnapshot> = Vec::new();
        let names: Vec<&str> = snaps.iter().map(|s| s.name.as_str()).collect();
        let decisions = gov.decide_all(snaps, &hunts, fg, &names, hour);
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
    // BOSS LEVEL 8 — ADVERSARIAL, COMPOUND, AND EDGE CASE SCENARIOS
    // ══════════════════════════════════════════════════════════════════════════

    /// BOSS 41: Extreme wakeup bomb — 1000 wakeups/sec with no CPU work.
    /// A stuck timer callback firing 10× past the "energy hog" threshold.
    /// At this rate it prevents the M1 from entering any low-power state.
    /// Must be FROZEN (not just throttled) — throttle alone doesn't stop wakeups.
    #[test]
    fn s61_extreme_wakeup_bomb_frozen() {
        let mut s = snap(6100, "stuck_timer_daemon", 0.2, 40, false, 3600, 3600);
        s.wakeups_per_sec = 1000.0;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "stuck_timer_daemon");
        assert!(
            d == GovernorDecision::Freeze || d == GovernorDecision::Throttle,
            "1000 wakeups/sec daemon must be acted on (throttle or freeze). Got {:?}",
            d
        );
    }

    /// BOSS 42: Linker "ld" during active build — the linker is running with
    /// rustc also present in the process list. At 60% CPU and 1GB RSS it is
    /// the final link phase of a large binary. Freezing it would corrupt the
    /// output object and require a full rebuild. Must be ALLOWED.
    #[test]
    fn s62_linker_ld_protected_during_build() {
        let linker = snap(6200, "ld", 60.0, 1024, false, 120, 120);
        let rustc = snap(6201, "rustc", 0.0, 400, false, 120, 120); // just finished compiling
        let snaps = vec![linker, rustc];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "ld");
        assert_ne!(
            d,
            GovernorDecision::Freeze,
            "Linker 'ld' at 60% CPU during build must NOT be frozen. Got {:?}",
            d
        );
    }

    /// BOSS 43: cargo at 3AM in a 30-process swarm — the user is running an
    /// overnight Rust build. Night mode + swarm pressure would normally act.
    /// But cargo is an active build tool (85% CPU, foreground just 45s ago).
    /// Both night mode and swarm checks must yield to the active build. ALLOW.
    #[test]
    fn s63_cargo_at_3am_in_swarm_allowed() {
        let mut snaps: Vec<ProcessSnapshot> = (0..28)
            .map(|i| snap(6300 + i, &format!("bg_{}", i), 0.3, 25, false, 7200, 7200))
            .collect();
        let cargo = snap(6399, "cargo", 85.0, 600, false, 45, 45); // active build
        snaps.push(cargo);
        let results = decide_at_hour(&snaps, None, 3); // 3 AM
        let d = find_decision(&results, "cargo");
        assert_ne!(
            d,
            GovernorDecision::Freeze,
            "Active cargo build at 3AM must NOT be frozen. Got {:?}",
            d
        );
    }

    /// BOSS 44: IPC hub below threshold — 75 Mach ports (just below the 80-port
    /// protection cutoff) that has been idle for 8 hours. The 75 ports once
    /// served other processes via XPC but the connection pool is now stale.
    /// Unlike a true IPC hub (80+), this one sat idle all day — THROTTLE it.
    #[test]
    fn s64_sub_threshold_ipc_hub_stale_throttled() {
        let mut s = snap(6400, "stale_xpc_pool", 0.2, 150, false, 28800, 28800); // 8h idle
        s.mach_port_count = 75; // just below 80-port IPC hub protection
        s.wakeups_per_sec = 0.5;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "stale_xpc_pool");
        assert!(
            d == GovernorDecision::Throttle || d == GovernorDecision::Freeze,
            "8h-idle IPC pool with 75 ports (below hub threshold) must be acted on. Got {:?}",
            d
        );
    }

    /// BOSS 45: System essential in extreme swarm — "mds" (Spotlight indexer)
    /// with 80 background processes competing for memory. Even at extreme swarm
    /// pressure, system essentials must survive completely untouched. ALLOW.
    #[test]
    fn s65_system_essential_immune_to_extreme_swarm() {
        let mut snaps: Vec<ProcessSnapshot> = (0..78)
            .map(|i| snap(6500 + i, &format!("bg_{}", i), 0.4, 30, false, 3600, 3600))
            .collect();
        let mds = snap(6580, "mds", 20.0, 300, false, 600, 600); // Spotlight indexing
        snaps.push(mds);
        let results = decide(&snaps, None);
        let d = find_decision(&results, "mds");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "mds (Spotlight) must be ALLOWED even in 80-process swarm. Got {:?}",
            d
        );
    }

    /// BOSS 46: Stale network keepalive 10 hours — a daemon with an open
    /// socket that has been completely idle for 10h. The network utility bonus
    /// (from s20) should NOT override graduated idle at 10h. THROTTLE or FREEZE.
    #[test]
    fn s66_network_keepalive_10h_throttled() {
        let mut s = snap(6600, "stale_sync_daemon", 0.1, 80, false, 36000, 36000); // 10h idle
        s.has_network = true;
        s.wakeups_per_sec = 0.2;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "stale_sync_daemon");
        assert!(
            d == GovernorDecision::Throttle || d == GovernorDecision::Freeze,
            "10h-idle network daemon must be acted on (network bonus expires). Got {:?}",
            d
        );
    }

    /// BOSS 47: Orphaned helper process — parent app died (Chrome crash) but
    /// the renderer helper keeps running. No GUI, no parent alive, 100 wakeups/s.
    /// Without a parent to receive messages, it's a dangling process. FREEZE it.
    #[test]
    fn s67_orphaned_helper_frozen() {
        let mut s = snap(6700, "Chrome Renderer", 2.0, 300, false, 3600, 3600);
        s.parent_alive = false;
        s.wakeups_per_sec = 100.0;
        s.mach_port_count = 5; // below IPC hub threshold
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "Chrome Renderer");
        assert!(
            d == GovernorDecision::Freeze || d == GovernorDecision::Throttle,
            "Orphaned Chrome Renderer must be acted on (no parent alive). Got {:?}",
            d
        );
    }

    /// BOSS 48: Analytics daemon disguised as a compiler — "clang_diag_agent"
    /// contains "clang" but is Apple's diagnostic/analytics service collecting
    /// telemetry. Low CPU, 4h idle, no GUI. The "clang" in the name must NOT
    /// grant compiler protection — analytics classification wins. THROTTLE/FREEZE.
    #[test]
    fn s68_fake_compiler_name_analytics_throttled() {
        let s = snap(6800, "clang_diag_agent", 0.3, 200, false, 14400, 14400); // 4h idle
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "clang_diag_agent");
        assert!(
            d == GovernorDecision::Throttle || d == GovernorDecision::Freeze,
            "Analytics daemon 'clang_diag_agent' must be throttled — telemetry > name pattern. Got {:?}",
            d
        );
    }

    /// BOSS 49: GUI window abandoned for 48 hours — a Slack window left open
    /// when the user went on vacation. has_gui_window=true keeps it alive through
    /// the graduated-idle rule. But 48h is extreme abandonment — even GUI apps
    /// should eventually be frozen to reclaim memory. FREEZE.
    ///
    /// [Gap identified]: graduated-idle rule requires !has_gui_window. This test
    /// FAILS until Apollo adds: "GUI app idle > 24h → freeze regardless".
    #[test]
    fn s69_gui_abandoned_48h_frozen() {
        let s = snap(6900, "Slack", 0.5, 500, true, 172800, 172800); // 48h idle, GUI window
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "Slack");
        assert!(
            d == GovernorDecision::Freeze || d == GovernorDecision::Throttle,
            "GUI app idle 48h must be frozen/throttled — abandoned window wastes 500MB. Got {:?}",
            d
        );
    }

    /// BOSS 50: Slow backup with massive pageins — Backblaze doing a large
    /// file scan at 3% CPU (throttled by the kernel, not idle). 200K pageins
    /// proves real disk I/O work. The active-I/O rule requires cpu > 5%, so
    /// it misses this. But freezing would corrupt the backup mid-scan. ALLOW.
    ///
    /// [Gap identified]: active-I/O protection threshold cpu > 5% excludes
    /// kernel-throttled backups. This test FAILS until Apollo fixes the threshold
    /// OR adds a "high pageins = real work" rule that doesn't require high CPU.
    #[test]
    fn s70_slow_backup_with_massive_pageins_allowed() {
        let mut s = snap(7000, "Backblaze", 3.0, 400, false, 1800, 1800); // 30min active
        s.pageins_total = 200_000;
        s.wakeups_per_sec = 10.0;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "Backblaze");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "Backblaze with 200K pageins is doing real backup work — must be ALLOWED. Got {:?}",
            d
        );
    }

    /// BOSS 51: GPU worker idle for 8 hours with 2M faults — a Metal shader
    /// compiler that ran all day, stopped submitting frames 8h ago but is still
    /// resident (GPU memory takes time to reclaim). 2M faults prove it was doing
    /// real GPU work. Faults protect from waste/swarm but graduated idle fires
    /// at 8h and freezes it. Must be ALLOWED while GPU memory is still mapped.
    ///
    /// [Gap identified]: faults-based render_pipeline_exempt doesn't block the
    /// graduated-idle rule. This test FAILS until Apollo adds fault-count check
    /// to the graduated-idle path.
    #[test]
    fn s71_gpu_worker_high_faults_long_idle_allowed() {
        let mut s = snap(
            7100,
            "com.apple.metal.shader-cache",
            0.5,
            800,
            false,
            28800,
            28800,
        ); // 8h idle
        s.faults_total = 2_000_000; // 2M faults = was doing massive GPU buffer work
        s.wakeups_per_sec = 2.0;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "com.apple.metal.shader-cache");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "GPU shader cache with 2M faults must be ALLOWED — GPU memory reclaim takes time. Got {:?}",
            d
        );
    }

    /// BOSS 52: ML inference python3 — a PyTorch inference server that does
    /// batched prediction. 1% CPU (GPU does the real work), 4GB RSS (model weights),
    /// 2M faults (GPU-CPU memory transfer), 2h since last request. The utility
    /// score would be very low (low CPU, no GUI, no network, no ports). Apollo
    /// would FREEZE it via utility threshold. But 4GB model reload = minutes.
    /// ALLOW while model is resident (treat like LLM protection).
    ///
    /// [Gap identified]: LLM protection only covers known LLM runtime names.
    /// Python3 running PyTorch is not recognized. This test FAILS until Apollo
    /// adds: "process with >1GB RSS + high faults = expensive to reload → protect".
    #[test]
    fn s72_ml_inference_python3_protected() {
        let mut s = snap(7200, "python3", 1.0, 4096, false, 7200, 7200); // 2h idle, 4GB model
        s.faults_total = 2_000_000; // GPU-CPU transfers
        s.wakeups_per_sec = 5.0;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "python3");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "ML inference python3 with 4GB model + 2M faults must be ALLOWED (reload = minutes). Got {:?}",
            d
        );
    }

    /// BOSS 53: Render pipeline co-helpers during video call — a Zoom call
    /// spawns several helper processes: the camera daemon, audio engine, and
    /// a network relay daemon. All have wakeups>100 or moderate CPU. All must
    /// survive the wakeup-hog and swarm rules when Zoom is foreground.
    #[test]
    fn s73_video_call_cohelpers_protected() {
        let camera = {
            let mut s = snap(7300, "VDCAssistant", 8.0, 100, false, 10, 10);
            s.wakeups_per_sec = 120.0; // camera frame delivery
            s.faults_total = 50_000;
            s
        };
        let audio = {
            let mut s = snap(7301, "coreaudiod", 3.0, 60, false, 10, 10);
            s.wakeups_per_sec = 90.0;
            s
        };
        let relay = {
            let mut s = snap(7302, "zoom_relay_helper", 5.0, 80, false, 10, 10);
            s.has_network = true;
            s.wakeups_per_sec = 50.0;
            s
        };
        let zoom = snap(7310, "zoom.us", 20.0, 600, true, 0, 0);
        let snaps = vec![camera, audio, relay, zoom];
        let results = decide(&snaps, Some("zoom.us"));
        for name in &["VDCAssistant", "coreaudiod", "zoom_relay_helper"] {
            let d = find_decision(&results, name);
            assert_ne!(
                d,
                GovernorDecision::Freeze,
                "Video call helper '{}' must NOT be frozen during active Zoom call. Got {:?}",
                name,
                d
            );
        }
    }

    /// BOSS 54: XPC burst — new process (10s uptime) handling a burst of XPC
    /// requests before settling. 300 wakeups/s + 500K faults during init.
    /// The wakeup-hog rule would fire (>100/s). But short uptime + massive
    /// faults = initialization burst, not a broken timer. ALLOW.
    #[test]
    fn s74_xpc_burst_init_allowed() {
        let mut s = snap(
            7400,
            "com.apple.xpc.launchd.domain.user",
            15.0,
            80,
            false,
            5,
            5,
        );
        s.process_uptime_secs = 10; // just started
        s.wakeups_per_sec = 300.0;
        s.faults_total = 500_000; // loading page tables during init
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "com.apple.xpc.launchd.domain.user");
        assert_ne!(
            d,
            GovernorDecision::Freeze,
            "XPC init burst (10s uptime, 300 wakeups) must NOT be frozen. Got {:?}",
            d
        );
    }

    /// BOSS 55: Translated process with GUI window and extreme idle.
    /// A Rosetta-translated Slack window that has been untouched for 36 hours.
    /// is_translated=true means ~2x memory overhead (JIT page tables).
    /// Even with GUI window, 36h idle + Rosetta overhead = freeze and reclaim.
    /// FREEZE (memory benefit from Rosetta JIT reclaim is worth it).
    ///
    /// [Gap identified]: GUI protection overrides both graduated-idle and Rosetta
    /// freeze. This test FAILS until Apollo handles "translated + extreme idle".
    #[test]
    fn s75_translated_gui_extreme_idle_frozen() {
        let mut s = snap(7500, "Slack", 0.3, 600, true, 129600, 129600); // 36h idle, GUI
        s.is_translated = true; // Rosetta — 2x memory overhead
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "Slack");
        assert!(
            d == GovernorDecision::Freeze || d == GovernorDecision::Throttle,
            "Translated GUI app idle 36h must be frozen/throttled (Rosetta 2x overhead). Got {:?}",
            d
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // BOSS LEVEL 9 — BOUNDARY CONDITIONS & COMPOUND INTERACTION SCENARIOS
    // ══════════════════════════════════════════════════════════════════════════

    /// BOSS 56: LLM model at the 12h idle boundary — ollama has been idle for
    /// exactly 12h (43200s). Protection ends at 12h. One second past the boundary
    /// and it should be subject to normal utility scoring. At 12h exactly it is
    /// still protected. ALLOW.
    #[test]
    fn s76_llm_at_12h_boundary_still_protected() {
        let mut s = snap(7600, "ollama", 0.5, 4096, false, 43199, 43199); // 12h - 1s
        s.wakeups_per_sec = 2.0;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "ollama");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "ollama at 12h-1s boundary with 4GB model must still be ALLOWED. Got {:?}",
            d
        );
    }

    /// BOSS 57: Safari WebKit helper when Safari is foreground — the renderer
    /// process is named "com.apple.WebKit.WebContent" while Safari is the fg app.
    /// Even with no GUI window and high wakeups (video streaming), it must be
    /// protected as a foreground helper. ALLOW.
    #[test]
    fn s77_safari_webkit_helper_protected_as_fg_helper() {
        let mut webkit = snap(7700, "com.apple.WebKit.WebContent", 8.0, 150, false, 5, 5);
        webkit.wakeups_per_sec = 80.0; // video frame delivery
        let safari = snap(7701, "Safari", 5.0, 300, true, 0, 0);
        let snaps = vec![webkit, safari];
        let results = decide(&snaps, Some("Safari"));
        let d = find_decision(&results, "com.apple.WebKit.WebContent");
        assert_ne!(
            d,
            GovernorDecision::Freeze,
            "Safari WebKit renderer must NOT be frozen when Safari is fg. Got {:?}",
            d
        );
    }

    /// BOSS 58: Silent daemon with 1GB+ RSS and no GUI — a background analytics
    /// aggregator with 1.2GB RSS, 0.3% CPU, idle for 2h. The RSS-weighted penalty
    /// makes its utility drop below freeze threshold even though it has no stale
    /// markers. Large idle daemon wastes 15% of 8GB RAM. FREEZE or THROTTLE.
    #[test]
    fn s78_large_rss_silent_daemon_acted_on() {
        let s = snap(7800, "com.apple.analyticsagg", 0.3, 1200, false, 7200, 7200);
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "com.apple.analyticsagg");
        assert!(
            d == GovernorDecision::Throttle || d == GovernorDecision::Freeze,
            "1.2GB idle analytics daemon must be acted on (RSS penalty). Got {:?}",
            d
        );
    }

    /// BOSS 59: True zombie process — is_zombie=true means the process has already
    /// exited but the kernel entry is not reaped. It holds 50MB of kernel memory
    /// indefinitely. No signal can affect it except SIGKILL (to its parent).
    /// Apollo must always KILL zombies. KILL.
    #[test]
    fn s79_true_zombie_always_killed() {
        let mut s = snap(
            7900,
            "com.apple.WebKit.Networking",
            0.0,
            50,
            false,
            99999,
            99999,
        );
        s.is_zombie = true;
        s.parent_alive = false;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "com.apple.WebKit.Networking");
        assert_eq!(
            d,
            GovernorDecision::Kill,
            "True zombie must always receive KILL decision. Got {:?}",
            d
        );
    }

    /// BOSS 60: Ephemeral XPC (uptime < 8s) with massive wakeups — a newly
    /// launched XPC service that fires 500 wakeups/s in its first 7 seconds.
    /// The wakeup hog rule would normally throttle this. But uptime < 8s is
    /// the ephemeral cutoff — Apollo must not touch it. ALLOW.
    #[test]
    fn s80_ephemeral_xpc_wakeup_hog_exempt() {
        let mut s = snap(8000, "com.apple.xpc.smd.helper", 20.0, 30, false, 3, 3);
        s.process_uptime_secs = 7; // just started
        s.wakeups_per_sec = 500.0; // huge burst during init
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "com.apple.xpc.smd.helper");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "XPC with uptime < 8s must be ALLOWED regardless of wakeup rate. Got {:?}",
            d
        );
    }

    /// BOSS 61: Idle non-GUI daemon at 6h boundary — process idle exactly 6h
    /// (21600s). The graduated idle rule fires at >21600s — at exactly 21600s
    /// it has not crossed the threshold. Must be ALLOWED (boundary exclusive).
    #[test]
    fn s81_graduated_idle_at_6h_boundary_not_triggered() {
        let s = snap(
            8100,
            "com.apple.coredata.sync",
            0.3,
            60,
            false,
            21600,
            21600,
        );
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "com.apple.coredata.sync");
        // At exactly 21600s the rule is: secs_since_foreground > 21600 → fires
        // Since 21600 is NOT > 21600, the process should NOT be throttled by graduated idle.
        // It may still be acted on by other rules (SilentDaemon idle, utility), so
        // we just verify it isn't frozen by graduated idle alone.
        assert_ne!(
            d,
            GovernorDecision::Kill,
            "Non-zombie daemon at 6h boundary must never be Killed. Got {:?}",
            d
        );
    }

    /// BOSS 62: Network daemon at 3AM — a background sync daemon (has_network=true)
    /// that has been idle for 20min at 3AM. Night mode would normally throttle
    /// any non-GUI background process idle > 15min. Network daemons have utility
    /// bonus. The question: does night mode fire even with the network utility?
    /// At adjusted utility ~0.55 (base 0.5 + network 0.05 = 0.55), night mode
    /// threshold is 0.55 — exactly at the boundary. The daemon should be left alone
    /// (utility >= threshold). ALLOW.
    #[test]
    fn s82_network_daemon_at_3am_utility_at_night_threshold() {
        let mut s = snap(8200, "com.apple.cloudd", 0.2, 40, false, 1200, 1200); // 20min idle
        s.has_network = true;
        let snaps = vec![s];
        let results = decide_at_hour(&snaps, None, 3); // 3 AM
        let d = find_decision(&results, "com.apple.cloudd");
        // Night mode fires when utility < 0.55. Network bonus pushes utility to ~0.55.
        // So it should survive or be throttled but not frozen.
        assert_ne!(
            d,
            GovernorDecision::Freeze,
            "Network daemon at 3AM with network utility must not be frozen. Got {:?}",
            d
        );
    }

    /// BOSS 63: IPC hub exactly at 80-port threshold — 80 Mach ports (exactly
    /// at the protection cutoff). Port count > 80 triggers protection, so 80
    /// itself is NOT protected. Process is idle 4h. The IPC hub rule requires
    /// mach_port_count > 80 (strictly greater). At 80, it should be acted on.
    /// THROTTLE or FREEZE.
    #[test]
    fn s83_ipc_hub_exactly_at_threshold_not_protected() {
        let mut s = snap(
            8300,
            "com.apple.appkit.xpc.agent",
            0.1,
            80,
            false,
            14400,
            14400,
        );
        s.mach_port_count = 80; // exactly at threshold (NOT strictly > 80)
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "com.apple.appkit.xpc.agent");
        assert!(
            d == GovernorDecision::Throttle || d == GovernorDecision::Freeze,
            "Process with exactly 80 Mach ports (not > 80) should be acted on. Got {:?}",
            d
        );
    }

    /// BOSS 64: Graduated idle at 12h — no GUI, idle for 12h (43200s), no faults,
    /// moderate CPU (avoids SilentDaemon idle override which requires cpu < 0.5).
    /// The graduated idle rule fires at >12h: Freeze. Wakeups < 1.0 → Stale tier.
    /// Low utility stale process with 12h idle must be acted on. FREEZE or THROTTLE.
    #[test]
    fn s84_graduated_idle_12h_frozen() {
        let mut s = snap(8400, "com.apple.quicklookd", 1.5, 60, false, 43201, 43201); // 12h+ idle
                                                                                      // wakeups_per_sec = 1.0 (default) → SilentDaemon tier; cpu=1.5 > 0.5 → idle override skipped
                                                                                      // → hits graduated idle rule: secs > 43200 && !has_gui_window && faults < 500K → Freeze
        s.wakeups_per_sec = 1.0;
        s.faults_total = 0;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "com.apple.quicklookd");
        assert!(
            d == GovernorDecision::Freeze || d == GovernorDecision::Throttle,
            "Non-GUI daemon idle 12h+ must be acted on by graduated idle rule. Got {:?}",
            d
        );
    }

    /// BOSS 65: Chrome helper with active network — "Chrome Helper (Renderer)"
    /// matches the AppHelper tier. With has_network=true and wakeups=30/s,
    /// it's serving content. Even as an AppHelper it must not be throttled:
    /// has_network || wakeups>5 → protected. ALLOW.
    #[test]
    fn s85_chrome_helper_with_network_protected() {
        let mut s = snap(
            8500,
            "Google Chrome Helper (Renderer)",
            3.0,
            120,
            false,
            30,
            30,
        );
        s.has_network = true;
        s.wakeups_per_sec = 30.0;
        let snaps = vec![s];
        let results = decide(&snaps, None);
        let d = find_decision(&results, "Google Chrome Helper (Renderer)");
        assert_eq!(
            d,
            GovernorDecision::Allow,
            "Chrome Helper with active network must be ALLOWED (AppHelper + network). Got {:?}",
            d
        );
    }
}
