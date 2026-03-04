//! Adaptive Governor — ties together user profile, process classifier,
//! and zombie hunter to produce concrete optimization decisions.
//!
//! This is the "brain" of the system.  It works entirely with heuristics:
//!
//!   decision = f(workload, process_tier, utility_score, waste_score, user_history)
//!
//! No LLM, no external service, zero latency.

use crate::engine::{
    llm::LearnedPolicy,
    process_classifier::{ProcessClassifier, ProcessSnapshot, ProcessTier, score_utility},
    user_profile::{UserProfile, WorkloadType},
    workload_classifier::{WorkloadClassification, WorkloadClassifier},
    zombie_hunter::{HuntSnapshot, ZombieHunter, ZombieAction},
};

// ── Decision ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GovernerDecision {
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
    pub decision: GovernerDecision,
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
            throttle_utility_threshold: 0.4,
            freeze_utility_threshold: 0.1,
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
        Self {
            config: GovernorConfig::default(),
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
        // 1. Update user profile
        self.user_profile
            .observe(foreground_app, all_proc_names, hour_of_day);

        // 1b. ML Ligero: classify workload BEFORE heuristic decisions so F1 uses ML result.
        // Clone to avoid split-borrow (user_profile + workload_classifier on self simultaneously).
        {
            let hour_model = self.user_profile.hour_model_ref().clone();
            let app_stats = self.user_profile.app_stats_ref().clone();
            let classification = self.workload_classifier.classify(
                foreground_app,
                all_proc_names,
                &hour_model,
                &app_stats,
                hour_of_day,
            );
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
            let decision = self.decide_one(snap, *tier, utility, *waste, workload);
            decisions.push(decision);
        }

        // 4. Append zombie / dead-weight decisions
        for dw in &dead_weight {
            let gov_decision = match dw.recommended_action {
                ZombieAction::Kill => GovernerDecision::Kill,
                ZombieAction::Suspend => GovernerDecision::Freeze,
                ZombieAction::NiceToMax => GovernerDecision::Throttle,
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
    ) -> ProcessDecision {
        // Essential — never touch
        if tier == ProcessTier::SystemEssential || tier == ProcessTier::ActiveForeground {
            return ProcessDecision {
                pid: snap.pid,
                name: snap.name.clone(),
                decision: GovernerDecision::Allow,
                tier,
                utility_score: utility,
                waste_score: waste,
                reason: "Essential or active-foreground — protected".into(),
            };
        }

        // Telemetry — always throttle (or freeze under heavy load)
        if tier == ProcessTier::Telemetry {
            let d = if self.is_heavy_workload(workload) {
                GovernerDecision::Freeze
            } else {
                GovernerDecision::Throttle
            };
            return ProcessDecision {
                pid: snap.pid,
                name: snap.name.clone(),
                decision: d,
                tier,
                utility_score: utility,
                waste_score: waste,
                reason: "Known telemetry/analytics process".into(),
            };
        }

        // Relevance bonus from user profile
        let relevance = self.user_profile.process_relevance(&snap.name);
        let adjusted_utility = (utility + relevance * 0.2).min(1.0);

        // Stale — strong candidate for freeze
        if tier == ProcessTier::Stale && adjusted_utility < self.config.freeze_utility_threshold {
            return ProcessDecision {
                pid: snap.pid,
                name: snap.name.clone(),
                decision: GovernerDecision::Freeze,
                tier,
                utility_score: adjusted_utility,
                waste_score: waste,
                reason: format!("Stale process (utility={:.2}) — frozen", adjusted_utility),
            };
        }

        // Waste override — even "useful" processes get throttled if they're
        // burning too many resources while the user isn't using them.
        if waste >= self.config.waste_override_threshold
            && adjusted_utility < 0.6
        {
            return ProcessDecision {
                pid: snap.pid,
                name: snap.name.clone(),
                decision: GovernerDecision::Throttle,
                tier,
                utility_score: adjusted_utility,
                waste_score: waste,
                reason: format!("High waste ({:.2}) with low utility ({:.2})", waste, adjusted_utility),
            };
        }

        // Normal utility-based decision
        let decision = if adjusted_utility < self.config.freeze_utility_threshold
            && self.is_heavy_workload(workload)
        {
            GovernerDecision::Freeze
        } else if adjusted_utility < self.config.throttle_utility_threshold {
            GovernerDecision::Throttle
        } else {
            GovernerDecision::Allow
        };

        ProcessDecision {
            pid: snap.pid,
            name: snap.name.clone(),
            decision,
            tier,
            utility_score: adjusted_utility,
            waste_score: waste,
            reason: format!(
                "utility={:.2} waste={:.2} workload={:?}",
                adjusted_utility, waste, workload
            ),
        }
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
        matches!(wl, WorkloadType::Coding | WorkloadType::VideoEdit | WorkloadType::VideoCall)
            && (self.config.aggressive_in_coding || self.config.aggressive_in_video_edit)
    }

    /// Summary statistics for the last decision run.
    pub fn summarise(decisions: &[ProcessDecision]) -> GovernorSummary {
        let mut summary = GovernorSummary::default();
        for d in decisions {
            match d.decision {
                GovernerDecision::Allow => summary.allowed += 1,
                GovernerDecision::Throttle => summary.throttled += 1,
                GovernerDecision::Freeze => summary.frozen += 1,
                GovernerDecision::Kill => summary.killed += 1,
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
