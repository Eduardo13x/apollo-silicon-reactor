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
    silicon_probe::{fast_entropy, SiliconInfo},
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
            let decision = self.decide_one(snap, *tier, utility, *waste, workload);
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
    ) -> ProcessDecision {
        // Essential — never touch
        if tier == ProcessTier::SystemEssential || tier == ProcessTier::ActiveForeground {
            return ProcessDecision {
                pid: snap.pid,
                name: snap.name.clone(),
                decision: GovernorDecision::Allow,
                tier,
                utility_score: utility,
                waste_score: waste,
                reason: "Essential or active-foreground — protected".into(),
            };
        }

        // AppHelper (browser renderers, Electron helpers…) — throttle only, nunca freeze.
        // Chromium/WebKit tienen un watchdog que crashea la tab si el renderer
        // deja de responder (SIGSTOP). Además, wakeups altos = audio/video en curso.
        if tier == ProcessTier::AppHelper {
            if snap.wakeups_per_sec > 5.0 || snap.has_network {
                // Audio, video, o cargando contenido → no tocar
                return ProcessDecision {
                    pid: snap.pid,
                    name: snap.name.clone(),
                    decision: GovernorDecision::Allow,
                    tier,
                    utility_score: utility,
                    waste_score: waste,
                    reason: "AppHelper activo (audio/video/red) — protegido".into(),
                };
            }
            // Sin actividad → throttle suave, pero NUNCA freeze
            let decision = if utility < self.config.throttle_utility_threshold {
                GovernorDecision::Throttle
            } else {
                GovernorDecision::Allow
            };
            return ProcessDecision {
                pid: snap.pid,
                name: snap.name.clone(),
                decision,
                tier,
                utility_score: utility,
                waste_score: waste,
                reason: format!(
                    "AppHelper inactivo — throttle-only (utility={:.2})",
                    utility
                ),
            };
        }

        // Telemetry — always throttle (or freeze under heavy load)
        if tier == ProcessTier::Telemetry {
            let d = if self.is_heavy_workload(workload) {
                GovernorDecision::Freeze
            } else {
                GovernorDecision::Throttle
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
                decision: GovernorDecision::Freeze,
                tier,
                utility_score: adjusted_utility,
                waste_score: waste,
                reason: format!("Stale process (utility={:.2}) — frozen", adjusted_utility),
            };
        }

        // Waste override — even "useful" processes get throttled if they're
        // burning too many resources while the user isn't using them.
        if waste >= self.config.waste_override_threshold && adjusted_utility < 0.6 {
            return ProcessDecision {
                pid: snap.pid,
                name: snap.name.clone(),
                decision: GovernorDecision::Throttle,
                tier,
                utility_score: adjusted_utility,
                waste_score: waste,
                reason: format!(
                    "High waste ({:.2}) with low utility ({:.2})",
                    waste, adjusted_utility
                ),
            };
        }

        // Normal utility-based decision with stochastic tie-breaking.
        //
        // Cuando varios procesos caen exactamente en la zona gris alrededor de un
        // threshold (utility ≈ threshold ± GRAY_ZONE), Apollo sin entropía los actuaría
        // a TODOS en el mismo ciclo → stutter visible o thundering herd de wakeups.
        //
        // fast_entropy(pid) usa cntvct_el0 + xorshift64 (~3 ns, sin syscall) para
        // distribuir esas decisiones en ciclos sucesivos sin introducir ningún estado.
        const GRAY_ZONE: f32 = 0.02;

        let near_freeze =
            (adjusted_utility - self.config.freeze_utility_threshold).abs() < GRAY_ZONE;
        let near_throttle =
            (adjusted_utility - self.config.throttle_utility_threshold).abs() < GRAY_ZONE;

        let (decision, reason) = if adjusted_utility < self.config.freeze_utility_threshold
            && self.is_heavy_workload(workload)
        {
            if near_freeze {
                // Gray zone: stagger freeze/throttle con entropía para evitar herd
                let d = if fast_entropy(snap.pid as u64) % 2 == 0 {
                    GovernorDecision::Freeze
                } else {
                    GovernorDecision::Throttle
                };
                let r = format!(
                    "utility={:.2} waste={:.2} workload={:?} (gray-zone entropy)",
                    adjusted_utility, waste, workload
                );
                (d, r)
            } else {
                let r = format!(
                    "utility={:.2} waste={:.2} workload={:?}",
                    adjusted_utility, waste, workload
                );
                (GovernorDecision::Freeze, r)
            }
        } else if adjusted_utility < self.config.throttle_utility_threshold {
            if near_throttle {
                // Gray zone: stagger throttle/allow
                let d = if fast_entropy(snap.pid as u64) % 2 == 0 {
                    GovernorDecision::Throttle
                } else {
                    GovernorDecision::Allow
                };
                let r = format!(
                    "utility={:.2} waste={:.2} workload={:?} (gray-zone entropy)",
                    adjusted_utility, waste, workload
                );
                (d, r)
            } else {
                let r = format!(
                    "utility={:.2} waste={:.2} workload={:?}",
                    adjusted_utility, waste, workload
                );
                (GovernorDecision::Throttle, r)
            }
        } else {
            let r = format!(
                "utility={:.2} waste={:.2} workload={:?}",
                adjusted_utility, waste, workload
            );
            (GovernorDecision::Allow, r)
        };

        ProcessDecision {
            pid: snap.pid,
            name: snap.name.clone(),
            decision,
            tier,
            utility_score: adjusted_utility,
            waste_score: waste,
            reason,
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

/// Calibra los thresholds del governor según el hardware real del chip.
///
/// Leído una sola vez al inicio via sysctl (sin root, sin entitlement).
/// Evita hardcodear valores que asuman M1 cuando corre en M3 Max.
fn calibrate_config_for_hardware(hw: &SiliconInfo) -> GovernorConfig {
    let mut cfg = GovernorConfig::default();
    let cores = hw.physical_cores;
    let ram_gb = hw.memory_bytes / 1024 / 1024 / 1024;

    // Más cores = más margen para congelar procesos background agresivamente,
    // porque el workload activo tiene más cores propios disponibles.
    if cores >= 12 {
        // M3 Pro/Max, M4 Pro/Max — más cores, algo más agresivo pero sin pasarse
        cfg.freeze_utility_threshold = 0.08;
        cfg.throttle_utility_threshold = 0.25;
    } else if cores >= 10 {
        // M2 Pro, M3 — tier intermedio
        cfg.freeze_utility_threshold = 0.06;
        cfg.throttle_utility_threshold = 0.22;
    }
    // M1/M2 base (8 cores) — usa defaults (0.20/0.05)

    // Con poca RAM (8GB), congelar un poco más agresivo para evitar swap.
    if ram_gb <= 8 {
        cfg.freeze_utility_threshold = cfg.freeze_utility_threshold.max(0.07);
        cfg.waste_override_threshold = 0.85;
    }

    cfg
}
