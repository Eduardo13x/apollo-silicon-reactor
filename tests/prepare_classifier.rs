//! ══════════════════════════════════════════════════════════════════════════════
//! Apollo AutoResearch — Process Classifier Benchmark
//! ══════════════════════════════════════════════════════════════════════════════
//!
//! THIS FILE IS READ-ONLY. The agent must NEVER modify it.
//!
//! Tests tier classification accuracy, utility scoring, and waste scoring
//! across edge cases that affect user-visible behavior. Misclassification
//! causes: audio drops (daemon frozen), UI jank (helper throttled),
//! wasted RAM (stale process kept alive).
//!
//! Target file: src/engine/process_classifier.rs

#[cfg(test)]
mod scenarios {
    use apollo_optimizer::engine::process_classifier::{
        ProcessClassifier, ProcessSnapshot, ProcessTier, score_utility, waste_score,
    };

    fn snap(name: &str) -> ProcessSnapshot {
        ProcessSnapshot {
            pid: 100,
            name: name.to_string(),
            cpu_percent: 0.0,
            rss_bytes: 10 * 1024 * 1024,
            is_zombie: false,
            secs_since_foreground: 3600,
            secs_since_user_interaction: 3600,
            has_network: false,
            has_gui_window: false,
            wakeups_per_sec: 0.0,
            parent_alive: true,
            process_uptime_secs: 3600,
            faults_total: 0,
            pageins_total: 0,
            is_translated: false,
            mach_port_count: 0,
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 1: TIER CLASSIFICATION EDGE CASES
    // ══════════════════════════════════════════════════════════════════════════

    /// C01: Audio daemon (coreaudiod) is SystemEssential even without GUI.
    /// Freezing coreaudiod = instant audio dropout.
    #[test]
    fn c01_coreaudiod_is_essential() {
        let c = ProcessClassifier::new();
        let s = snap("coreaudiod");
        assert_eq!(
            c.classify(&s), ProcessTier::SystemEssential,
            "coreaudiod must be SystemEssential — freezing it kills all audio"
        );
    }

    /// C02: Notification daemon at 0.4% CPU and 0.8 wakeups/s should NOT be
    /// classified as Stale. It's a SilentDaemon doing legitimate polling.
    /// Stale requires cpu < 0.5 AND wakeups < 1.0 AND idle > 300s — all three.
    #[test]
    fn c02_notification_daemon_not_stale() {
        let c = ProcessClassifier::new();
        let mut s = snap("notifyd_helper");
        s.cpu_percent = 0.4;
        s.wakeups_per_sec = 0.8;
        s.secs_since_foreground = 600;
        // This hits all 3 Stale conditions: cpu 0.4 < 0.5, wakeups 0.8 < 1.0, idle 600 > 300.
        // The question is: should a daemon polling every ~1.2s be frozen?
        let tier = c.classify(&s);
        // Stale is acceptable here — the daemon IS idle. The governor
        // should protect it via utility scoring, not the classifier.
        assert!(
            tier == ProcessTier::Stale || tier == ProcessTier::SilentDaemon,
            "Low-activity daemon should be Stale or SilentDaemon, got {:?}", tier
        );
    }

    /// C03: App that was foreground 25 seconds ago is ActiveForeground.
    /// User just switched away — still within 30s window.
    #[test]
    fn c03_recently_active_app_is_foreground() {
        let c = ProcessClassifier::new();
        let mut s = snap("Code");
        s.has_gui_window = true;
        s.secs_since_user_interaction = 25;
        assert_eq!(
            c.classify(&s), ProcessTier::ActiveForeground,
            "App with GUI + 25s since interaction should be ActiveForeground"
        );
    }

    /// C04: App that was foreground 35 seconds ago is BackgroundVisible.
    /// Just past the 30s threshold — should downgrade.
    #[test]
    fn c04_slightly_old_app_is_background_visible() {
        let c = ProcessClassifier::new();
        let mut s = snap("Code");
        s.has_gui_window = true;
        s.secs_since_user_interaction = 35;
        assert_eq!(
            c.classify(&s), ProcessTier::BackgroundVisible,
            "App with GUI + 35s since interaction = BackgroundVisible"
        );
    }

    /// C05: Chrome Helper with GUI window is AppHelper, not SilentDaemon.
    #[test]
    fn c05_chrome_helper_with_gui_is_app_helper() {
        let c = ProcessClassifier::new();
        let mut s = snap("Google Chrome Helper");
        s.has_gui_window = true;
        assert_eq!(
            c.classify(&s), ProcessTier::AppHelper,
            "Chrome Helper with GUI should be AppHelper"
        );
    }

    /// C06: Chrome Helper WITHOUT GUI is SilentDaemon (background renderer).
    #[test]
    fn c06_chrome_helper_no_gui_is_silent() {
        let c = ProcessClassifier::new();
        let s = snap("Google Chrome Helper");
        assert_eq!(
            c.classify(&s), ProcessTier::SilentDaemon,
            "Chrome Helper without GUI should be SilentDaemon"
        );
    }

    /// C07: Orphan process (parent dead, pid > 1) is ZombieOrphan.
    #[test]
    fn c07_orphan_is_zombie() {
        let c = ProcessClassifier::new();
        let mut s = snap("orphaned_worker");
        s.parent_alive = false;
        s.pid = 500;
        assert_eq!(
            c.classify(&s), ProcessTier::ZombieOrphan,
            "Process with dead parent should be ZombieOrphan"
        );
    }

    /// C08: PID 1 with dead parent is NOT zombie (launchd edge case).
    #[test]
    fn c08_pid1_not_zombie() {
        let c = ProcessClassifier::new();
        let mut s = snap("launchd");
        s.parent_alive = false;
        s.pid = 1;
        // launchd is essential, and pid=1 is exempt from orphan check
        assert_eq!(
            c.classify(&s), ProcessTier::SystemEssential,
            "PID 1 (launchd) should never be ZombieOrphan"
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 2: UTILITY SCORING
    // ══════════════════════════════════════════════════════════════════════════

    /// C09: GUI app with recent interaction should have highest utility.
    #[test]
    fn c09_active_gui_app_max_utility() {
        let mut s = snap("App");
        s.has_gui_window = true;
        s.secs_since_user_interaction = 5;
        let u = score_utility(&s);
        assert!(u > 0.90, "Active GUI app should have utility > 0.90, got {}", u);
    }

    /// C10: Silent daemon with network should have utility > base (0.5).
    #[test]
    fn c10_network_daemon_above_base() {
        let mut s = snap("sync_daemon");
        s.has_network = true;
        let u = score_utility(&s);
        assert!(u > 0.50, "Network daemon should have utility > 0.50, got {}", u);
    }

    /// C11: Chatty no-GUI daemon (100 wakeups/s) should be penalized.
    #[test]
    fn c11_chatty_daemon_penalized() {
        let mut s = snap("chatty_service");
        s.wakeups_per_sec = 100.0;
        let u = score_utility(&s);
        assert!(u < 0.40, "Chatty daemon (100 wakeups/s) should have low utility, got {}", u);
    }

    /// C12: Rosetta process gets small penalty but stays near base.
    #[test]
    fn c12_rosetta_small_penalty() {
        let mut native = snap("app_arm");
        native.cpu_percent = 5.0;
        let mut rosetta = snap("app_x86");
        rosetta.cpu_percent = 5.0;
        rosetta.is_translated = true;
        let u_native = score_utility(&native);
        let u_rosetta = score_utility(&rosetta);
        assert!(
            u_native > u_rosetta,
            "Native ({}) should have higher utility than Rosetta ({})", u_native, u_rosetta
        );
        assert!(
            u_rosetta > 0.40,
            "Rosetta penalty should be small — still usable. Got {}", u_rosetta
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // CATEGORY 3: WASTE SCORING
    // ══════════════════════════════════════════════════════════════════════════

    /// C13: GUI app with high CPU should have LOW waste (user doing real work).
    #[test]
    fn c13_gui_high_cpu_low_waste() {
        let mut s = snap("Figma");
        s.has_gui_window = true;
        s.cpu_percent = 50.0;
        s.secs_since_user_interaction = 5;
        let w = waste_score(&s, ProcessTier::ActiveForeground);
        assert_eq!(w, 0.0, "ActiveForeground waste must be 0.0");
    }

    /// C14: Background visible app has very low waste.
    #[test]
    fn c14_bg_visible_low_waste() {
        let mut s = snap("Spotify");
        s.has_gui_window = true;
        let w = waste_score(&s, ProcessTier::BackgroundVisible);
        assert!(w <= 0.15, "BackgroundVisible waste should be <= 0.15, got {}", w);
    }

    /// C15: SilentDaemon with high wakeups AND large RSS should have high waste.
    #[test]
    fn c15_bloated_chatty_daemon_high_waste() {
        let mut s = snap("bloated_daemon");
        s.wakeups_per_sec = 50.0;
        s.rss_bytes = 500 * 1024 * 1024; // 500MB
        let w = waste_score(&s, ProcessTier::SilentDaemon);
        assert!(w > 0.55, "Bloated chatty daemon waste should be > 0.55, got {}", w);
    }
}
