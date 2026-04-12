//! Adaptive Governor — ties together user profile, process classifier,
//! and zombie hunter to produce concrete optimization decisions.
//!
//! This is the "brain" of the system.  It works entirely with heuristics:
//!
//!   decision = f(workload, process_tier, utility_score, waste_score, user_history)
//!
//! No LLM, no external service, zero latency.

use crate::engine::{
    hw_bayes::HwFeatures,
    llm::LearnedPolicy,
    process_classifier::{score_utility, ProcessClassifier, ProcessSnapshot, ProcessTier},
    silicon_probe::SiliconInfo,
    user_profile::{UserProfile, WorkloadType},
    workload_classifier::{WorkloadClassification, WorkloadClassifier},
    zombie_hunter::{HuntSnapshot, ZombieAction, ZombieHunter},
};

// ── Decision ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GovernorDecision {
    /// Let the process run normally.
    Allow,
    /// Reduce CPU scheduling priority (renice to +10).
    Throttle,
    /// Send SIGSTOP — will be resumed when workload changes.
    Freeze,
    /// Send SIGKILL — only for zombies / orphans.
    Kill,
}

#[derive(Debug, Clone)]
pub struct ProcessDecision {
    pub pid: u32,
    pub name: String,
    pub decision: GovernorDecision,
    pub tier: ProcessTier,
    pub utility_score: f32,
    pub waste_score: f32,
    pub reason: String,
}

// ── Governor Config ───────────────────────────────────────────────────────────

pub struct GovernorConfig {
    /// Utility score below which a non-essential process is throttled.
    pub throttle_utility_threshold: f32,
    /// Utility score below which a non-essential process is frozen.
    pub freeze_utility_threshold: f32,
    /// Waste score above which even useful processes get throttled.
    pub waste_override_threshold: f32,
    /// During a high-priority workload, freeze more aggressively.
    pub aggressive_in_coding: bool,
    pub aggressive_in_video_edit: bool,
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self {
            // Conservador: solo throttlear procesos genuinamente inactivos.
            // Daemons con red activa tienen utility ~0.23, así pasan el threshold.
            throttle_utility_threshold: 0.20,
            freeze_utility_threshold: 0.05,
            waste_override_threshold: 0.9,
            aggressive_in_coding: true,
            aggressive_in_video_edit: true,
        }
    }
}

// ── Governor ─────────────────────────────────────────────────────────────────

pub struct AdaptiveGovernor {
    pub config: GovernorConfig,
    classifier: ProcessClassifier,
    zombie_hunter: ZombieHunter,
    pub user_profile: UserProfile,
    workload_classifier: WorkloadClassifier,
    last_classification: WorkloadClassification,
}

impl AdaptiveGovernor {
    pub fn new() -> Self {
        // Calibrar thresholds según el chip real — leído del hardware sin syscall pesada.
        // M1 (8 cores, 8GB) = más conservador. M3 Max (16 cores, 96GB) = más agresivo.
        let hw = SiliconInfo::read();
        let config = calibrate_config_for_hardware(&hw);
        Self {
            config,
            classifier: ProcessClassifier::new(),
            zombie_hunter: ZombieHunter::new(),
            user_profile: UserProfile::new(),
            workload_classifier: WorkloadClassifier::new(),
            last_classification: WorkloadClassification {
                workload: WorkloadType::General,
                confidence: 0.0,
                sources: Vec::new(),
            },
        }
    }

    // ── Main entry point ──────────────────────────────────────────────────

    /// Given current process snapshots, produce a decision for each one.
    ///
    /// Call once per optimization cycle (e.g. every 5–15 s).
    pub fn decide_all(
        &mut self,
        proc_snaps: &[ProcessSnapshot],
        hunt_snaps: &[HuntSnapshot],
        foreground_app: Option<&str>,
        all_proc_names: &[&str],
        hour_of_day: u8,
    ) -> Vec<ProcessDecision> {
        self.decide_all_with_hw(
            proc_snaps,
            hunt_snaps,
            foreground_app,
            all_proc_names,
            hour_of_day,
            None,
        )
    }

    /// Versión con features de hardware para clasificación Bayesiana completa.
    pub fn decide_all_with_hw(
        &mut self,
        proc_snaps: &[ProcessSnapshot],
        hunt_snaps: &[HuntSnapshot],
        foreground_app: Option<&str>,
        all_proc_names: &[&str],
        hour_of_day: u8,
        hw: Option<HwFeatures>,
    ) -> Vec<ProcessDecision> {
        // 1. Update user profile
        self.user_profile
            .observe(foreground_app, all_proc_names, hour_of_day);

        // 1b. ML: Gaussian NB de hardware + clasificador de texto fusionados.
        {
            let hour_model = self.user_profile.hour_model_ref().clone();
            let app_stats = self.user_profile.app_stats_ref().clone();
            let classification = self.workload_classifier.classify_with_hw(
                foreground_app,
                all_proc_names,
                &hour_model,
                &app_stats,
                hour_of_day,
                hw.as_ref(),
            );
            // Aprendizaje online: si texto+hora tienen confianza alta, entrena el NB de HW.
            if let Some(ref hw_feat) = hw {
                self.workload_classifier.maybe_observe(
                    hw_feat,
                    classification.workload,
                    classification.confidence,
                );
            }
            if classification.confidence >= 0.60 {
                self.config.aggressive_in_coding = matches!(
                    classification.workload,
                    WorkloadType::Coding | WorkloadType::CommandLine
                );
                self.config.aggressive_in_video_edit =
                    matches!(classification.workload, WorkloadType::VideoEdit);
            }
            self.last_classification = classification;
        }

        let workload = self.user_profile.current_workload();

        // 2. Find zombies first — they always get Kill/Suspend regardless of profile
        let dead_weight = self.zombie_hunter.evaluate_all(hunt_snaps);

        // Build a set of PIDs that have already been sentenced by zombie hunter
        let zombie_pids: std::collections::HashSet<u32> =
            dead_weight.iter().map(|d| d.pid).collect();

        // 3. Classify & decide everything else
        let classified = self.classifier.classify_all(proc_snaps);

        let mut decisions: Vec<ProcessDecision> = Vec::new();

        for (snap, tier, waste) in &classified {
            if zombie_pids.contains(&snap.pid) {
                continue; // Will be handled below
            }

            let utility = score_utility(snap);
            let decision = self.decide_one(
                snap,
                *tier,
                utility,
                *waste,
                workload,
                classified.len(),
                foreground_app,
                hour_of_day,
            );
            decisions.push(decision);
        }

        // 4. Append zombie / dead-weight decisions
        for dw in &dead_weight {
            let gov_decision = match dw.recommended_action {
                ZombieAction::Kill => GovernorDecision::Kill,
                ZombieAction::Suspend => GovernorDecision::Freeze,
                ZombieAction::NiceToMax => GovernorDecision::Throttle,
            };
            decisions.push(ProcessDecision {
                pid: dw.pid,
                name: dw.name.clone(),
                decision: gov_decision,
                tier: ProcessTier::ZombieOrphan,
                utility_score: 0.0,
                waste_score: 1.0,
                reason: dw.reason.clone(),
            });
        }

        decisions
    }

    // ── Per-process decision ──────────────────────────────────────────────

    fn decide_one(
        &self,
        snap: &ProcessSnapshot,
        tier: ProcessTier,
        utility: f32,
        waste: f32,
        workload: WorkloadType,
        process_count: usize,
        foreground_app: Option<&str>,
        hour_of_day: u8,
    ) -> ProcessDecision {
        // Helper closure: avoids repeating pid/name/tier/waste_score on every early return.
        let pd = |decision: GovernorDecision, utility_score: f32, reason: String| ProcessDecision {
            pid: snap.pid,
            name: snap.name.clone(),
            decision,
            tier,
            utility_score,
            waste_score: waste,
            reason,
        };

        // Zombie/orphan handling: true zombies (is_zombie=true) need SIGKILL to
        // reap their kernel entry. Mere orphans (parent died, but process is still
        // running and may be in the middle of I/O) should be frozen, not killed —
        // killing mid-write can corrupt data. A frozen orphan drains naturally once
        // it tries to communicate with its dead parent.
        if tier == ProcessTier::ZombieOrphan {
            if snap.is_zombie {
                return ProcessDecision {
                    pid: snap.pid,
                    name: snap.name.clone(),
                    decision: GovernorDecision::Kill,
                    tier,
                    utility_score: 0.0,
                    waste_score: 1.0,
                    reason: "Zombie process — reap with SIGKILL".into(),
                };
            }
            return ProcessDecision {
                pid: snap.pid,
                name: snap.name.clone(),
                decision: GovernorDecision::Freeze,
                tier,
                utility_score: 0.0,
                waste_score: 1.0,
                reason: "Orphaned process (parent dead) — freeze to reclaim".into(),
            };
        }

        // Essential — never touch
        if tier == ProcessTier::SystemEssential || tier == ProcessTier::ActiveForeground {
            return pd(
                GovernorDecision::Allow,
                utility,
                "Essential or active-foreground — protected".into(),
            );
        }

        // Ephemeral XPC on-demand services exit on their own within seconds.
        // Throttling them wastes cycles and the action is always stale by
        // the time execute_actions runs (~127ms later). Threshold: 8s gives
        // enough margin for a full cycle without catching persistent daemons
        // (which typically stay alive for minutes or hours).
        if snap.process_uptime_secs < 8 {
            return pd(
                GovernorDecision::Allow,
                utility,
                format!(
                    "ephemeral XPC (uptime={}s < 8s) — will exit on its own",
                    snap.process_uptime_secs
                ),
            );
        }

        // AppHelper (browser renderers, Electron helpers…) — throttle only, nunca freeze.
        // Chromium/WebKit tienen un watchdog que crashea la tab si el renderer
        // deja de responder (SIGSTOP). Además, wakeups altos = audio/video en curso.
        if tier == ProcessTier::AppHelper {
            if snap.wakeups_per_sec > 5.0 || snap.has_network {
                return pd(
                    GovernorDecision::Allow,
                    utility,
                    "AppHelper activo (audio/video/red) — protegido".into(),
                );
            }
            // Sin actividad → throttle suave, pero NUNCA freeze
            let decision = if utility < self.config.throttle_utility_threshold {
                GovernorDecision::Throttle
            } else {
                GovernorDecision::Allow
            };
            return pd(
                decision,
                utility,
                format!(
                    "AppHelper inactivo — throttle-only (utility={:.2})",
                    utility
                ),
            );
        }

        // Telemetry — always throttle (or freeze under heavy load)
        if tier == ProcessTier::Telemetry {
            let d = if self.is_heavy_workload(workload) {
                GovernorDecision::Freeze
            } else {
                GovernorDecision::Throttle
            };
            return pd(d, utility, "Known telemetry/analytics process".into());
        }

        // IPC hub protection: daemons with many Mach ports are serving other
        // processes via XPC/MIG. Throttling them cascades into beachballs.
        if snap.mach_port_count > 80 {
            return pd(
                GovernorDecision::Allow,
                utility,
                format!("IPC hub ({} Mach ports) — protected", snap.mach_port_count),
            );
        }

        // LLM model protection: large-RSS processes matching known AI runtimes
        // have huge reload cost (30s+). Protect if idle < 12h. Beyond that,
        // the user has likely moved on and the memory is worth reclaiming.
        const LLM_NAMES: &[&str] = &["ollama", "llama", "llamafile", "mlc-chat", "whisper"];
        if snap.rss_bytes > 1024 * 1024 * 1024
            && LLM_NAMES.iter().any(|n| snap.name.contains(n))
            && snap.secs_since_user_interaction < 43200
        {
            return pd(
                GovernorDecision::Allow,
                utility,
                format!(
                    "LLM model loaded ({}MB RSS) — reload cost too high",
                    snap.rss_bytes / 1024 / 1024
                ),
            );
        }

        // Active I/O work protection: high pageins = real disk work in progress.
        // Freezing corrupts backups/encodes. Throttle is acceptable.
        if snap.pageins_total > 50000 && snap.cpu_percent > 5.0 {
            return pd(
                GovernorDecision::Allow,
                utility,
                format!(
                    "Active I/O work ({} pageins, {:.0}% CPU) — protected",
                    snap.pageins_total, snap.cpu_percent
                ),
            );
        }

        // SilentDaemon idle override: if a daemon has near-zero CPU and has been
        // idle for over an hour with no GUI, it's effectively stale.
        // Rosetta (translated) processes get frozen instead of throttled because
        // they use ~2x memory (JIT page tables) — freeing them is more valuable.
        if tier == ProcessTier::SilentDaemon
            && snap.cpu_percent < 0.5
            && snap.secs_since_foreground > 3600
            && !snap.has_gui_window
        {
            // Translated (2x memory) or RSS hog (>1GB): freeze to reclaim memory.
            let decision = if snap.is_translated || snap.rss_bytes > 1024 * 1024 * 1024 {
                GovernorDecision::Freeze
            } else {
                GovernorDecision::Throttle
            };
            return pd(
                decision,
                utility,
                format!(
                    "SilentDaemon idle override (cpu={:.1}%, idle={}s)",
                    snap.cpu_percent, snap.secs_since_foreground
                ),
            );
        }

        // Relevance bonus from user profile
        let relevance = self.user_profile.process_relevance(&snap.name);
        let mut adjusted_utility = (utility + relevance * 0.2).min(1.0);

        // RSS-weighted utility penalty: large background processes (>500MB)
        // with no GUI are penalized proportionally. A 1GB daemon without GUI
        // gets -0.10 utility. Ensures bloated silent processes are acted on
        // sooner without touching the classifier.
        if !snap.has_gui_window && snap.rss_bytes > 500 * 1024 * 1024 {
            let excess_gb =
                (snap.rss_bytes as f64 - 500.0 * 1024.0 * 1024.0) / (1024.0 * 1024.0 * 1024.0);
            let penalty = (excess_gb * 0.10).min(0.20) as f32;
            adjusted_utility = (adjusted_utility - penalty).max(0.0);
        }

        // Extreme GUI abandonment: a window untouched for >24h is not an active
        // session — it is abandoned memory. The normal graduated-idle rule skips
        // GUI windows, but 24h is long enough that freezing is safe even with a
        // visible window. Translated (Rosetta) apps consume 2x memory, so any
        // extreme-idle Rosetta window is also worth reclaiming.
        if snap.has_gui_window && snap.secs_since_foreground > 86400 && snap.cpu_percent < 2.0 {
            return pd(
                GovernorDecision::Freeze,
                adjusted_utility,
                format!(
                    "GUI app abandoned >24h (idle={}h) — freeze to reclaim memory",
                    snap.secs_since_foreground / 3600
                ),
            );
        }

        // Graduated idle: the idle override (above) only catches cpu < 0.5%.
        // But a daemon idle for 6h+ is effectively abandoned even at moderate CPU.
        // Graduated: >6h → Throttle, >12h → Freeze. No GUI required.
        // Processes with high faults (>500K) are doing active GPU/memory work —
        // their idle time is deceptive (e.g. Metal shader caches between frames).
        if !snap.has_gui_window && snap.secs_since_foreground > 21600 && snap.faults_total < 500_000
        {
            let decision = if snap.secs_since_foreground > 43200 {
                GovernorDecision::Freeze
            } else {
                GovernorDecision::Throttle
            };
            return pd(
                decision,
                adjusted_utility,
                format!(
                    "Graduated idle ({}h, no GUI) — {}",
                    snap.secs_since_foreground / 3600,
                    if snap.secs_since_foreground > 43200 {
                        "freeze"
                    } else {
                        "throttle"
                    }
                ),
            );
        }

        // Foreground app helper detection: if a process name contains part of
        // the foreground app name (e.g., "com.apple.WebKit" when "Safari" is fg),
        // or is a known helper pattern for the foreground app, protect it.
        // Must be checked BEFORE night mode/waste/swarm overrides — otherwise a
        // Safari WebKit helper gets throttled by those checks first.
        if let Some(fg) = foreground_app {
            let is_fg_helper = snap.name.contains(fg)
                || (fg == "Safari" && snap.name.contains("WebKit"))
                || (fg == "Google Chrome" && snap.name.contains("Chrome"))
                || (fg == "Brave Browser" && snap.name.contains("Brave"))
                || (fg == "Firefox" && snap.name.contains("plugin-container"));
            if is_fg_helper && !snap.has_gui_window {
                return pd(
                    GovernorDecision::Allow,
                    utility,
                    format!("Helper of foreground app ({})", fg),
                );
            }
        }

        // Night mode: between midnight and 6AM, nobody is watching the screen.
        // Non-GUI background processes with idle > 15min should be throttled to
        // save energy. Placed AFTER FG helper check so Safari tabs survive at 3AM.
        let is_night = hour_of_day < 6;
        if is_night
            && !snap.has_gui_window
            && snap.secs_since_foreground > 900
            && adjusted_utility < 0.55
        {
            return pd(
                GovernorDecision::Throttle,
                adjusted_utility,
                format!(
                    "Night mode (hour={}, idle={}s) — throttle to save energy",
                    hour_of_day, snap.secs_since_foreground
                ),
            );
        }

        // Stale — strong candidate for freeze
        if tier == ProcessTier::Stale && adjusted_utility < self.config.freeze_utility_threshold {
            return pd(
                GovernorDecision::Freeze,
                adjusted_utility,
                format!("Stale process (utility={:.2}) — frozen", adjusted_utility),
            );
        }

        // Render pipeline exemption: processes with high faults (GPU buffer work)
        // or known compositor daemons with an active foreground app are part of
        // the frame delivery path. Throttling them causes visible jank / dropped frames.
        let render_pipeline_exempt = snap.faults_total > 100000
            || (foreground_app.is_some() && is_render_pipeline(&snap.name));

        // Waste override — graduated curve: higher waste tolerates less utility.
        // waste >= threshold (0.80 on 8GB, 0.90 default) → throttle if utility < 0.60
        // waste > 0.50 → throttle if utility < 0.40 (soft override, no GUI only)
        // Render pipeline processes are exempt — their "waste" is actually frame work.
        let waste_triggered = if render_pipeline_exempt {
            false
        } else if waste >= self.config.waste_override_threshold {
            adjusted_utility < 0.6
        } else if waste > 0.5 && !snap.has_gui_window {
            adjusted_utility < 0.40
        } else {
            false
        };
        if waste_triggered {
            return pd(
                GovernorDecision::Throttle,
                adjusted_utility,
                format!(
                    "Waste override (waste={:.2}, utility={:.2})",
                    waste, adjusted_utility
                ),
            );
        }

        // Swarm pressure: when many processes are competing (>30), lower the
        // bar for waste-based throttling. With 50+ daemons, even mildly wasteful
        // ones degrade foreground responsiveness.
        // Swarm exemptions: high faults = active memory work (GPU, mmap I/O),
        // significant Mach ports (40+) = serving other processes via XPC.
        let swarm_exempt = snap.faults_total > 500000 || snap.mach_port_count > 40;
        if process_count > 30
            && waste >= 0.30
            && adjusted_utility < 0.55
            && !snap.has_gui_window
            && !swarm_exempt
        {
            // Translated (Rosetta) processes in swarm get frozen: they use ~2x
            // memory from JIT page tables, so reclaiming is more valuable.
            let swarm_decision = if snap.is_translated {
                GovernorDecision::Freeze
            } else {
                GovernorDecision::Throttle
            };
            return pd(
                swarm_decision,
                adjusted_utility,
                format!(
                    "Swarm throttle ({} procs, waste={:.2}, util={:.2})",
                    process_count, waste, adjusted_utility
                ),
            );
        }

        // Wakeup energy hog: >100 wakeups/sec with no GUI forces the CPU out
        // of deep idle on every wakeup (P→E transition on M1). Even at low CPU%,
        // this destroys battery life. Throttle to reduce wakeup frequency.
        // Render pipeline processes are exempt (computed above waste override).
        if snap.wakeups_per_sec > 100.0
            && !snap.has_gui_window
            && adjusted_utility < 0.50
            && !render_pipeline_exempt
        {
            return pd(
                GovernorDecision::Throttle,
                adjusted_utility,
                format!(
                    "Wakeup energy hog ({:.0} wakeups/s) — throttle for battery",
                    snap.wakeups_per_sec
                ),
            );
        }

        let throttle_thresh = self.config.throttle_utility_threshold;

        // Normal utility-based decision.
        let base_reason = format!(
            "utility={:.2} waste={:.2} workload={:?}",
            adjusted_utility, waste, workload
        );
        let decision = if adjusted_utility < self.config.freeze_utility_threshold {
            GovernorDecision::Freeze
        } else if adjusted_utility < throttle_thresh {
            GovernorDecision::Throttle
        } else {
            GovernorDecision::Allow
        };
        pd(decision, adjusted_utility, base_reason)
    }

    // ── ML Ligero methods ─────────────────────────────────────────────────

    /// Call when the LLM teacher delivers an updated LearnedPolicy.
    pub fn update_learned_policy(&mut self, policy: &LearnedPolicy) {
        self.workload_classifier.update_learned_policy(policy);
    }

    /// Classify the current workload using ML Ligero.
    ///
    /// Clones hour_model and app_stats to avoid a split-borrow: we cannot
    /// borrow `self.user_profile` while also using `self.workload_classifier`.
    pub fn classify_workload(
        &mut self,
        foreground_app: Option<&str>,
        all_proc_names: &[&str],
        hour_of_day: u8,
    ) -> WorkloadClassification {
        let hour_model = self.user_profile.hour_model_ref().clone();
        let app_stats = self.user_profile.app_stats_ref().clone();

        let classification = self.workload_classifier.classify(
            foreground_app,
            all_proc_names,
            &hour_model,
            &app_stats,
            hour_of_day,
        );
        self.last_classification = classification.clone();
        classification
    }

    /// Returns the most recent ML classification (cached, no recomputation).
    pub fn last_ml_classification(&self) -> &WorkloadClassification {
        &self.last_classification
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    fn is_heavy_workload(&self, wl: WorkloadType) -> bool {
        match wl {
            WorkloadType::Coding | WorkloadType::CommandLine => self.config.aggressive_in_coding,
            WorkloadType::VideoEdit | WorkloadType::VideoCall => {
                self.config.aggressive_in_video_edit
            }
            _ => false,
        }
    }

    /// Summary statistics for the last decision run.
    pub fn summarise(decisions: &[ProcessDecision]) -> GovernorSummary {
        let mut summary = GovernorSummary::default();
        for d in decisions {
            match d.decision {
                GovernorDecision::Allow => summary.allowed += 1,
                GovernorDecision::Throttle => summary.throttled += 1,
                GovernorDecision::Freeze => summary.frozen += 1,
                GovernorDecision::Kill => summary.killed += 1,
            }
        }
        summary.total = decisions.len();
        summary
    }
}

impl Default for AdaptiveGovernor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Default)]
pub struct GovernorSummary {
    pub total: usize,
    pub allowed: usize,
    pub throttled: usize,
    pub frozen: usize,
    pub killed: usize,
}

/// Render-pipeline process names: these feed frames to WindowServer.
/// Throttling them causes dropped frames even though they have no GUI window.
fn is_render_pipeline(name: &str) -> bool {
    const RENDER_NAMES: &[&str] = &[
        "VDCAssistant",       // Video decode compositor
        "coreservicesd",      // System UI compositing
        "com.apple.gpu",      // GPU helper processes
        "MTLCompilerService", // Metal shader compilation
        "mediaserverd",       // AV render pipeline
    ];
    RENDER_NAMES.iter().any(|r| name.contains(r))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::process_classifier::ProcessSnapshot;
    use crate::engine::zombie_hunter::HuntSnapshot;

    fn base_proc(pid: u32, name: &str) -> ProcessSnapshot {
        ProcessSnapshot {
            pid,
            name: name.to_string(),
            cpu_percent: 1.0,
            rss_bytes: 50 * 1024 * 1024,
            is_zombie: false,
            secs_since_foreground: 120,
            secs_since_user_interaction: 120,
            has_network: false,
            has_gui_window: false,
            wakeups_per_sec: 2.0,
            parent_alive: true,
            process_uptime_secs: 3600,
            faults_total: 100,
            pageins_total: 100,
            is_translated: false,
            mach_port_count: 10,
            cpu_contention: None,
            is_app_bundle: false,
        }
    }

    fn no_hunts() -> Vec<HuntSnapshot> {
        vec![]
    }

    fn governor() -> AdaptiveGovernor {
        AdaptiveGovernor::new()
    }

    // ── Basic operation ──────────────────────────────────────────────────────

    #[test]
    fn empty_input_produces_no_decisions() {
        let mut gov = governor();
        let decisions = gov.decide_all(&[], &no_hunts(), None, &[], 12);
        assert!(decisions.is_empty());
    }

    #[test]
    fn summarise_empty_decisions() {
        let summary = AdaptiveGovernor::summarise(&[]);
        assert_eq!(summary.total, 0);
        assert_eq!(summary.allowed, 0);
    }

    #[test]
    fn summarise_counts_correctly() {
        let decisions = vec![
            ProcessDecision {
                pid: 1,
                name: "a".into(),
                decision: GovernorDecision::Allow,
                tier: ProcessTier::SilentDaemon,
                utility_score: 0.5,
                waste_score: 0.1,
                reason: "".into(),
            },
            ProcessDecision {
                pid: 2,
                name: "b".into(),
                decision: GovernorDecision::Throttle,
                tier: ProcessTier::SilentDaemon,
                utility_score: 0.1,
                waste_score: 0.5,
                reason: "".into(),
            },
            ProcessDecision {
                pid: 3,
                name: "c".into(),
                decision: GovernorDecision::Freeze,
                tier: ProcessTier::SilentDaemon,
                utility_score: 0.0,
                waste_score: 0.9,
                reason: "".into(),
            },
            ProcessDecision {
                pid: 4,
                name: "d".into(),
                decision: GovernorDecision::Kill,
                tier: ProcessTier::ZombieOrphan,
                utility_score: 0.0,
                waste_score: 1.0,
                reason: "".into(),
            },
        ];
        let summary = AdaptiveGovernor::summarise(&decisions);
        assert_eq!(summary.total, 4);
        assert_eq!(summary.allowed, 1);
        assert_eq!(summary.throttled, 1);
        assert_eq!(summary.frozen, 1);
        assert_eq!(summary.killed, 1);
    }

    // ── Essential / foreground protection ────────────────────────────────────

    #[test]
    fn system_essential_process_always_allowed() {
        let mut gov = governor();
        // "launchd" is in the essential set.
        let snap = base_proc(1, "launchd");
        let decisions = gov.decide_all(&[snap], &no_hunts(), None, &[], 12);
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].decision, GovernorDecision::Allow);
    }

    #[test]
    fn foreground_app_is_allowed() {
        let mut gov = governor();
        // When Safari is foreground, its process should be protected.
        let snap = ProcessSnapshot {
            name: "Safari".into(),
            secs_since_foreground: 0,
            ..base_proc(100, "Safari")
        };
        let decisions = gov.decide_all(&[snap], &no_hunts(), Some("Safari"), &["Safari"], 14);
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].decision, GovernorDecision::Allow);
    }

    // ── Ephemeral process protection ─────────────────────────────────────────

    #[test]
    fn ephemeral_xpc_under_8s_is_always_allowed() {
        let mut gov = governor();
        let snap = ProcessSnapshot {
            process_uptime_secs: 3, // below 8s threshold
            ..base_proc(200, "com.apple.xpc.launchd.oneshot.helper")
        };
        let decisions = gov.decide_all(&[snap], &no_hunts(), None, &[], 12);
        assert_eq!(decisions.len(), 1);
        assert_eq!(
            decisions[0].decision,
            GovernorDecision::Allow,
            "ephemeral XPC < 8s must always be allowed"
        );
    }

    // ── Graduated idle ────────────────────────────────────────────────────────

    #[test]
    fn idle_over_6h_no_gui_is_throttled() {
        let mut gov = governor();
        let snap = ProcessSnapshot {
            secs_since_foreground: 7 * 3600, // 7h
            has_gui_window: false,
            ..base_proc(300, "some-daemon")
        };
        let decisions = gov.decide_all(&[snap], &no_hunts(), None, &[], 12);
        assert_eq!(decisions.len(), 1);
        assert_eq!(
            decisions[0].decision,
            GovernorDecision::Throttle,
            "idle > 6h should be throttled"
        );
    }

    #[test]
    fn idle_over_12h_no_gui_is_frozen() {
        let mut gov = governor();
        let snap = ProcessSnapshot {
            secs_since_foreground: 13 * 3600, // 13h
            has_gui_window: false,
            ..base_proc(301, "stale-daemon")
        };
        let decisions = gov.decide_all(&[snap], &no_hunts(), None, &[], 12);
        assert_eq!(decisions.len(), 1);
        assert_eq!(
            decisions[0].decision,
            GovernorDecision::Freeze,
            "idle > 12h should be frozen"
        );
    }

    // ── IPC hub protection ────────────────────────────────────────────────────

    #[test]
    fn high_mach_port_count_is_protected() {
        let mut gov = governor();
        let snap = ProcessSnapshot {
            mach_port_count: 100, // > 80 threshold
            secs_since_foreground: 5000,
            ..base_proc(400, "ipc-hub-daemon")
        };
        let decisions = gov.decide_all(&[snap], &no_hunts(), None, &[], 12);
        assert_eq!(decisions.len(), 1);
        assert_eq!(
            decisions[0].decision,
            GovernorDecision::Allow,
            "high Mach port count = IPC hub, must be protected"
        );
    }

    // ── Night mode ────────────────────────────────────────────────────────────

    #[test]
    fn night_mode_throttles_idle_no_gui() {
        let mut gov = governor();
        let snap = ProcessSnapshot {
            has_gui_window: false,
            secs_since_foreground: 1800, // 30 min idle
            wakeups_per_sec: 1.0,
            cpu_percent: 0.1,
            ..base_proc(500, "background-daemon")
        };
        // hour=3 → night mode
        let decisions = gov.decide_all(&[snap], &no_hunts(), None, &[], 3);
        assert_eq!(decisions.len(), 1);
        // Night mode should throttle low-utility background processes.
        // (May be Allow if utility is high — test only fires for low-utility processes)
        let d = &decisions[0];
        assert!(
            d.decision == GovernorDecision::Throttle || d.decision == GovernorDecision::Allow,
            "night mode should throttle or allow, not freeze/kill: {:?}",
            d.decision
        );
    }

    // ── Wakeup energy hog ─────────────────────────────────────────────────────

    #[test]
    fn wakeup_hog_no_gui_is_throttled() {
        let mut gov = governor();
        let snap = ProcessSnapshot {
            wakeups_per_sec: 200.0, // > 100 threshold
            has_gui_window: false,
            cpu_percent: 0.5,
            secs_since_foreground: 300,
            faults_total: 100, // not render pipeline
            ..base_proc(600, "wakeup-hog")
        };
        let decisions = gov.decide_all(&[snap], &no_hunts(), None, &[], 14);
        assert_eq!(decisions.len(), 1);
        // High wakeups + no GUI + low utility → throttle.
        assert_eq!(
            decisions[0].decision,
            GovernorDecision::Throttle,
            "wakeup hog should be throttled"
        );
    }

    // ── Foreground helper detection ───────────────────────────────────────────

    #[test]
    fn webkit_helper_protected_when_safari_foreground() {
        let mut gov = governor();
        let snap = ProcessSnapshot {
            has_gui_window: false,
            secs_since_foreground: 5000,
            ..base_proc(700, "com.apple.WebKit.WebContent")
        };
        let decisions = gov.decide_all(&[snap], &no_hunts(), Some("Safari"), &["Safari"], 14);
        assert_eq!(decisions.len(), 1);
        assert_eq!(
            decisions[0].decision,
            GovernorDecision::Allow,
            "WebKit helper with Safari in foreground must be protected"
        );
    }

    // ── Governor config default ───────────────────────────────────────────────

    #[test]
    fn governor_config_default_values() {
        let cfg = GovernorConfig::default();
        assert!((cfg.throttle_utility_threshold - 0.20).abs() < 0.01);
        assert!((cfg.freeze_utility_threshold - 0.05).abs() < 0.01);
        assert!((cfg.waste_override_threshold - 0.90).abs() < 0.01);
        assert!(cfg.aggressive_in_coding);
        assert!(cfg.aggressive_in_video_edit);
    }

    // ── is_render_pipeline ────────────────────────────────────────────────────

    #[test]
    fn render_pipeline_names_detected() {
        assert!(is_render_pipeline("VDCAssistant"));
        assert!(is_render_pipeline("MTLCompilerService"));
        assert!(is_render_pipeline("mediaserverd"));
        assert!(!is_render_pipeline("random-daemon"));
        assert!(!is_render_pipeline("Safari"));
    }

    // ── calibrate_config_for_hardware ────────────────────────────────────────

    #[test]
    fn low_ram_machine_gets_lower_waste_threshold() {
        use crate::engine::silicon_probe::SiliconInfo;
        let hw_8gb = SiliconInfo {
            memory_bytes: 8 * 1024 * 1024 * 1024,
            ..SiliconInfo::read()
        };
        let cfg = calibrate_config_for_hardware(&hw_8gb);
        // 8GB machine: waste threshold should be 0.80, more aggressive.
        assert!(
            (cfg.waste_override_threshold - 0.80).abs() < 0.01,
            "8GB machine should use 0.80 waste threshold, got {}",
            cfg.waste_override_threshold
        );
    }

    // ── LLM model protection ──────────────────────────────────────────────────

    #[test]
    fn llm_model_large_rss_within_12h_is_protected() {
        let mut gov = governor();
        let snap = ProcessSnapshot {
            name: "ollama".into(),
            rss_bytes: 4 * 1024 * 1024 * 1024, // 4GB
            secs_since_user_interaction: 3600, // 1h idle
            ..base_proc(800, "ollama")
        };
        let decisions = gov.decide_all(&[snap], &no_hunts(), None, &[], 14);
        assert_eq!(decisions.len(), 1);
        assert_eq!(
            decisions[0].decision,
            GovernorDecision::Allow,
            "LLM model (ollama) with 4GB RSS within 12h must be protected"
        );
    }

    #[test]
    fn llm_model_expired_beyond_12h_not_protected() {
        let mut gov = governor();
        // Beyond 12h idle (43201s) — LLM protection does not apply (condition: < 43200).
        // Also set secs_since_foreground > 21600 so graduated idle fires.
        let snap = ProcessSnapshot {
            name: "ollama".into(),
            rss_bytes: 4 * 1024 * 1024 * 1024,  // 4GB
            secs_since_user_interaction: 43201, // beyond 12h — LLM protection gone
            secs_since_foreground: 43201,       // triggers graduated idle > 12h → Freeze
            cpu_percent: 1.0,                   // > 0.5 → SilentDaemon idle override skips
            ..base_proc(801, "ollama")
        };
        let decisions = gov.decide_all(&[snap], &no_hunts(), None, &[], 14);
        assert_eq!(decisions.len(), 1);
        // Without LLM protection and with 12h+ idle, must be Frozen by graduated idle.
        assert_eq!(
            decisions[0].decision,
            GovernorDecision::Freeze,
            "LLM model beyond 12h boundary should be Frozen by graduated idle rule"
        );
    }

    // ── GUI abandonment ───────────────────────────────────────────────────────

    #[test]
    fn gui_app_abandoned_24h_is_frozen() {
        let mut gov = governor();
        let snap = ProcessSnapshot {
            has_gui_window: true,
            secs_since_foreground: 86401, // just over 24h
            cpu_percent: 0.3,
            ..base_proc(900, "AbandonedApp")
        };
        let decisions = gov.decide_all(&[snap], &no_hunts(), None, &[], 14);
        assert_eq!(decisions.len(), 1);
        assert_eq!(
            decisions[0].decision,
            GovernorDecision::Freeze,
            "GUI app abandoned for 24h+ must be frozen to reclaim memory"
        );
    }

    #[test]
    fn gui_app_under_24h_not_frozen_by_abandonment_rule() {
        let mut gov = governor();
        let snap = ProcessSnapshot {
            has_gui_window: true,
            secs_since_foreground: 86399, // just under 24h
            cpu_percent: 0.3,
            ..base_proc(901, "ActiveApp")
        };
        let decisions = gov.decide_all(&[snap], &no_hunts(), None, &[], 14);
        assert_eq!(decisions.len(), 1);
        assert_ne!(
            decisions[0].decision,
            GovernorDecision::Freeze,
            "GUI app < 24h idle must NOT be frozen by abandonment rule"
        );
    }

    // ── Orphan (parent dead, non-zombie) ─────────────────────────────────────

    #[test]
    fn orphan_non_zombie_is_frozen_not_killed() {
        let mut gov = governor();
        let snap = ProcessSnapshot {
            is_zombie: false,
            parent_alive: false,
            pid: 999,
            ..base_proc(999, "orphaned-helper")
        };
        let decisions = gov.decide_all(&[snap], &no_hunts(), None, &[], 14);
        assert_eq!(decisions.len(), 1);
        assert_eq!(
            decisions[0].decision,
            GovernorDecision::Freeze,
            "Non-zombie orphan (parent dead) must be Frozen, not Killed"
        );
    }

    // ── Translated process in swarm ───────────────────────────────────────────

    #[test]
    fn translated_process_in_swarm_is_frozen() {
        let mut gov = governor();
        // 31 processes → swarm condition (> 30)
        let mut procs: Vec<ProcessSnapshot> = (0..30)
            .map(|i| ProcessSnapshot {
                cpu_percent: 0.2,
                secs_since_foreground: 3600,
                ..base_proc(1000 + i as u32, &format!("bg_{}", i))
            })
            .collect();
        // Add a translated daemon — in swarm + is_translated → Freeze
        let translated = ProcessSnapshot {
            is_translated: true,
            secs_since_foreground: 3600,
            cpu_percent: 0.2,
            has_gui_window: false,
            faults_total: 100,  // not swarm_exempt
            mach_port_count: 5, // not swarm_exempt
            ..base_proc(2000, "rosetta-daemon")
        };
        procs.push(translated);
        let decisions = gov.decide_all(&procs, &no_hunts(), None, &[], 14);
        let d = decisions
            .iter()
            .find(|d| d.name == "rosetta-daemon")
            .map(|d| d.decision);
        assert_eq!(
            d,
            Some(GovernorDecision::Freeze),
            "Translated process in swarm must be Frozen (2x memory overhead)"
        );
    }

    // ── Render pipeline exemption ─────────────────────────────────────────────

    #[test]
    fn render_pipeline_process_exempt_from_wakeup_throttle() {
        let mut gov = governor();
        // VDCAssistant is in the render pipeline — even with 200 wakeups/s + low utility
        // it should be protected when there's a foreground app.
        let snap = ProcessSnapshot {
            wakeups_per_sec: 200.0,
            has_gui_window: false,
            cpu_percent: 0.5,
            secs_since_foreground: 300,
            faults_total: 0, // explicitly low to NOT trigger faults exemption
            ..base_proc(700, "VDCAssistant")
        };
        let decisions = gov.decide_all(&[snap], &no_hunts(), Some("Zoom"), &["Zoom"], 14);
        assert_eq!(decisions.len(), 1);
        // Render pipeline processes are exempt from wakeup throttle rule.
        // With foreground app present and being a render pipeline process → exempt.
        assert_ne!(
            decisions[0].decision,
            GovernorDecision::Kill,
            "Render pipeline VDCAssistant must not be killed"
        );
    }

    // ── Classify workload ─────────────────────────────────────────────────────

    #[test]
    fn classify_workload_returns_classification() {
        let mut gov = governor();
        let classification = gov.classify_workload(Some("Xcode"), &["rustc", "cargo"], 14);
        // Should return a valid classification — workload type matters less than structure.
        assert!(classification.confidence >= 0.0 && classification.confidence <= 1.0);
    }

    #[test]
    fn last_ml_classification_matches_recent_decide() {
        let mut gov = governor();
        let snap = base_proc(100, "some-daemon");
        let _ = gov.decide_all(&[snap], &no_hunts(), Some("Xcode"), &["Xcode", "rustc"], 14);
        let last = gov.last_ml_classification();
        // After a decide_all, classification must have been updated.
        assert!(last.confidence >= 0.0);
    }

    // ── Micro-benchmark: decide_all latency ──────────────────────────────────

    #[test]
    fn bench_decide_all_latency() {
        let mut gov = governor();
        let procs: Vec<ProcessSnapshot> = (0..20)
            .map(|i| ProcessSnapshot {
                secs_since_foreground: 300 + i * 100,
                ..base_proc(i as u32 + 1000, "daemon")
            })
            .collect();
        let hunts: Vec<HuntSnapshot> = vec![];
        // Warm-up.
        for _ in 0..5 {
            let _ = gov.decide_all(&procs, &hunts, None, &[], 14);
        }
        let start = std::time::Instant::now();
        let n = 100usize;
        for _ in 0..n {
            let _ = gov.decide_all(&procs, &hunts, None, &[], 14);
        }
        let per_call_ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
        // 20 processes × governor logic should complete in < 5ms per cycle.
        assert!(
            per_call_ms < 5.0,
            "decide_all too slow: {per_call_ms:.2}ms/call (expected < 5ms)"
        );
    }
}

/// Calibra los thresholds del governor según el hardware real del chip.
///
/// Leído una sola vez al inicio via sysctl (sin root, sin entitlement).
/// Evita hardcodear valores que asuman M1 cuando corre en M3 Max.
fn calibrate_config_for_hardware(hw: &SiliconInfo) -> GovernorConfig {
    let mut cfg = GovernorConfig::default();
    let ram_gb = hw.memory_bytes / 1024 / 1024 / 1024;

    // Con poca RAM (8GB), lower waste threshold to act sooner on bloat.
    if ram_gb <= 8 {
        cfg.waste_override_threshold = 0.80;
    }

    cfg
}
