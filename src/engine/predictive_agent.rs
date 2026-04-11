//! Predictive Agent — LinUCB contextual bandit for proactive memory management.
//!
//! Apollo's existing pipeline is **reactive**: it detects pressure then acts.
//! On an M1 with 8 GB the margin between "fine" and "swap storm" is ~2-3 GB,
//! and reactive interventions often arrive late.
//!
//! This module predicts pressure episodes 5-30s ahead using hardware signals
//! already collected, and executes **soft interventions** (never freeze/SIGSTOP)
//! that prepare the system before impact.
//!
//! ## LinUCB
//! A contextual bandit with 5 arms and 12-dimensional context.
//! Each arm maintains a 12×12 matrix A and a 12-vector b.
//! Selection: argmax_a (θ_a · x + α √(x' A_a⁻¹ x))
//! Update: A_a += x x', b_a += r x
//!
//! No external dependencies — pure f64 arithmetic.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::engine::nars_belief::DriftDetector;
use crate::engine::neon_ema;
use crate::engine::overflow_guard::OverflowThresholds;
use crate::engine::swap_predictor::SwapTrend;
use crate::engine::user_profile::WorkloadType;

// ── Constants ────────────────────────────────────────────────────────────────

const D: usize = 12; // feature dimensions
const K: usize = 5; // number of arms
const WARMUP_CYCLES: u32 = 200;
const SEEDED_WARMUP_CYCLES: u32 = 50;
const PERSIST_INTERVAL: u32 = 100;
const TIGHTEN_OFFSET: f64 = -0.03; // 3pp tighter thresholds

// ── Intervention (arm) ───────────────────────────────────────────────────────

/// The five soft interventions the agent can choose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Intervention {
    /// Do nothing — observe baseline.
    Observe,
    /// Temporarily tighten overflow thresholds by 3pp for this cycle.
    TightenThresholds,
    /// Suggest aggressive profile to the governor for 5 minutes.
    SuggestAggressive,
    /// Renice top 3 noise processes (+5 niceness, no SIGSTOP).
    PreThrottleNoise,
    /// Hint the kernel to purge background caches.
    ProactivePurge,
}

impl Intervention {
    fn index(self) -> usize {
        match self {
            Self::Observe => 0,
            Self::TightenThresholds => 1,
            Self::SuggestAggressive => 2,
            Self::PreThrottleNoise => 3,
            Self::ProactivePurge => 4,
        }
    }
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Self::Observe,
            1 => Self::TightenThresholds,
            2 => Self::SuggestAggressive,
            3 => Self::PreThrottleNoise,
            4 => Self::ProactivePurge,
            _ => Self::Observe,
        }
    }
}

// ── Specialist voting ────────────────────────────────────────────────────────

/// A specialist's vote: proposed intervention + confidence (0–1).
#[derive(Debug, Clone)]
pub struct SpecialistVote {
    /// Who is voting.
    pub name: &'static str,
    /// Proposed intervention.
    pub intervention: Intervention,
    /// Confidence in this proposal (0–1). Higher = more weight.
    pub confidence: f64,
}

/// Result of tallying specialist votes.
#[derive(Debug)]
pub struct VotingResult {
    /// The winning intervention.
    pub intervention: Intervention,
    /// Whether specialists disagreed (at least 2 different non-Observe proposals).
    pub had_disagreement: bool,
    /// Total weighted score for the winner.
    pub winning_score: f64,
}

/// Tally specialist votes using weighted scoring.
/// Each intervention accumulates confidence from its voters.
/// Highest total confidence wins. Ties go to the safer option (lower index).
pub fn tally_votes(votes: &[SpecialistVote]) -> VotingResult {
    let mut scores = [0.0_f64; K];
    for v in votes {
        scores[v.intervention.index()] += v.confidence;
    }

    // Find winner (highest score; ties favor lower index = safer).
    let mut best_idx = 0;
    let mut best_score = scores[0];
    for (i, &s) in scores.iter().enumerate().skip(1) {
        if s > best_score {
            best_score = s;
            best_idx = i;
        }
    }

    // Detect disagreement: ≥2 different non-Observe proposals.
    let non_observe_proposals: Vec<usize> = votes
        .iter()
        .filter(|v| v.intervention != Intervention::Observe)
        .map(|v| v.intervention.index())
        .collect();
    let unique_proposals: std::collections::HashSet<usize> =
        non_observe_proposals.iter().copied().collect();
    let had_disagreement = unique_proposals.len() >= 2;

    VotingResult {
        intervention: Intervention::from_index(best_idx),
        had_disagreement,
        winning_score: best_score,
    }
}

// ── Specialist accuracy tracker ─────────────────────────────────────────────

/// Specialist indices used by [`SpecialistAccuracyTracker`].
pub mod specialist {
    pub const LINUCB: usize = 0;
    pub const HAZARD: usize = 1;
    pub const MONOPOLY: usize = 2;
    pub const KALMAN: usize = 3;
    pub const COUNT: usize = 4;

    /// Human-readable names, indexed by the constants above.
    pub const NAMES: [&str; COUNT] = ["linucb", "hazard", "monopoly", "kalman"];

    /// Map a specialist name string to its index, or return `None`.
    pub fn index_of(name: &str) -> Option<usize> {
        NAMES.iter().position(|&n| n == name)
    }
}

/// Per-specialist EMA accuracy tracker (Super Learner–style adaptive weights).
///
/// Each specialist starts at 0.70 accuracy (matching the legacy hardcoded
/// multipliers). After each cycle outcome is known the tracker receives a
/// binary correct/incorrect signal and the EMA decays toward it:
///
/// ```text
/// accuracy[i] = 0.97 * accuracy[i] + 0.03 * (1 if correct else 0)
/// ```
///
/// The resulting accuracy value is used as the confidence multiplier when
/// building [`SpecialistVote`]s instead of fixed constants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialistAccuracyTracker {
    /// EMA accuracy per specialist, range [0, 1]. Init: 0.70.
    accuracy: [f64; specialist::COUNT],
    /// Total updates applied (for diagnostics / test assertions).
    updates: u64,
    /// NARS drift detector for specialist effectiveness beliefs.
    /// When a specialist's prediction accuracy shifts by >=20pp (sustained),
    /// drift is detected and weights are reset toward 0.70 (neutral prior).
    /// [Kuncheva 2004] "Changing Classifiers in Non-Stationary Environments"
    #[serde(default)]
    pub drift_detector: DriftDetector,
}

impl Default for SpecialistAccuracyTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl SpecialistAccuracyTracker {
    const EMA_ALPHA: f64 = 0.03;
    const INIT_ACCURACY: f64 = 0.70;

    /// Create a new tracker with all specialists at 0.70.
    pub fn new() -> Self {
        Self {
            accuracy: [Self::INIT_ACCURACY; specialist::COUNT],
            updates: 0,
            drift_detector: DriftDetector::new(),
        }
    }

    /// Update accuracy for a single specialist.
    ///
    /// `correct` = true when the specialist's prediction was confirmed by the
    /// observed outcome (pressure spiked when predicted high, or stayed calm
    /// when predicted low).
    pub fn update(&mut self, specialist_idx: usize, correct: bool) {
        if specialist_idx < specialist::COUNT {
            let signal = if correct { 1.0 } else { 0.0 };
            self.accuracy[specialist_idx] =
                (1.0 - Self::EMA_ALPHA) * self.accuracy[specialist_idx] + Self::EMA_ALPHA * signal;
            self.updates += 1;
            // NARS: track specialist drift belief.
            // When a specialist's accuracy regime changes (was accurate, now not),
            // the belief frequency will shift and trigger recalibration.
            let name = specialist::NAMES
                .get(specialist_idx)
                .copied()
                .unwrap_or("unknown");
            self.drift_detector.observe(name, correct);
            // Auto-recalibrate: if drift detected, reset drifted specialists toward neutral.
            if self.drift_detector.needs_recalibration() {
                // Build mask: 1.0 if this specialist needs recalibration, 0.0 if stable.
                // Drifted specialists get pulled toward INIT_ACCURACY (0.70); stable ones unchanged.
                // NEON version: process all 4 specialists in one vfmaq_f32 instruction.
                // [ARM NEON Guide §4] batch EMA: alpha * prior + (1-alpha) * current.
                // Mask encodes selective update: alpha[i] = 0.1 if drifted, 0.0 if stable.
                let mut ema_f32 = [0.0f32; 4];
                let mut alphas = [0.0f32; 4];
                let prior_f32 = Self::INIT_ACCURACY as f32;

                for i in 0..specialist::COUNT {
                    ema_f32[i] = self.accuracy[i] as f32;
                    let n = specialist::NAMES[i];
                    alphas[i] = if let Some(tv) = self.drift_detector.belief(n) {
                        if tv.confidence < 0.3
                            || (tv.frequency as f64 - self.accuracy[i]).abs() > 0.20
                        {
                            0.1_f32 // drifted: pull 10% toward prior
                        } else {
                            0.0_f32 // stable: no change
                        }
                    } else {
                        0.0_f32
                    };
                }

                // NEON batch recalibration: pull ALL 4 specialists toward prior simultaneously.
                // Specialists with alphas[i]=0.0 are unchanged by the EMA (0*prior + 1*ema = ema).
                // Specialists with alphas[i]=0.1 move 10% toward prior.
                // We process in two steps: for simplicity use uniform alpha=0.1 on all 4 then
                // restore the stable ones from saved ema_f32.
                // [ARM NEON Guide] ema_update_4 = 1 vfmaq_f32 instruction for all 4 specialists.
                let saved = ema_f32; // save for restore of stable specialists
                let prior4 = [prior_f32; 4];
                neon_ema::ema_update_4(
                    &mut ema_f32,
                    &prior4,
                    0.1_f32, // pull-toward-prior alpha
                );
                for i in 0..specialist::COUNT {
                    if alphas[i] > 0.0 {
                        self.accuracy[i] = ema_f32[i] as f64; // NEON result for drifted specialist
                    } else {
                        let _ = saved[i]; // stable: keep original (no-op, written for clarity)
                    }
                }
                self.drift_detector.acknowledge_recalibration();
            }
        }
    }

    /// Update accuracy by specialist name. No-op for unknown names.
    pub fn update_by_name(&mut self, name: &str, correct: bool) {
        if let Some(idx) = specialist::index_of(name) {
            self.update(idx, correct);
        }
    }

    /// Return the learned accuracy weight for a specialist (range [0, 1]).
    /// Used as the confidence multiplier when building `SpecialistVote`s.
    pub fn weight(&self, specialist_idx: usize) -> f64 {
        if specialist_idx < specialist::COUNT {
            self.accuracy[specialist_idx]
        } else {
            Self::INIT_ACCURACY
        }
    }

    /// Return all accuracy weights as a slice.
    pub fn weights(&self) -> &[f64; specialist::COUNT] {
        &self.accuracy
    }

    /// Mutable access to all accuracy weights (for validation/clamping on restore).
    pub fn weights_mut(&mut self) -> &mut [f64; specialist::COUNT] {
        &mut self.accuracy
    }

    /// Total update calls (diagnostic).
    pub fn total_updates(&self) -> u64 {
        self.updates
    }

    /// Record disagreement outcome: given the votes and whether the chosen
    /// intervention was effective, boost correct specialists and penalize wrong ones.
    ///
    /// When specialists disagree, one faction was right and the other wrong.
    /// After observing the outcome (pressure dropped = effective), we can
    /// attribute correctness:
    /// - Specialists who voted for the winning intervention: correct = effective
    /// - Specialists who voted differently: correct = !effective
    ///
    /// This closes the feedback loop where `had_disagreement` was logged but
    /// never used to improve specialist weights.
    pub fn record_disagreement_outcome(
        &mut self,
        votes: &[SpecialistVote],
        chosen: Intervention,
        was_effective: bool,
    ) {
        for vote in votes {
            let voted_for_winner = vote.intervention == chosen;
            // If the chosen action was effective, those who voted for it were right.
            // If it was ineffective, those who voted against it were right.
            let correct = if voted_for_winner {
                was_effective
            } else {
                !was_effective
            };
            if let Some(idx) = specialist::index_of(vote.name) {
                self.update(idx, correct);
            }
        }
    }
}

// ── Context vector ───────────────────────────────────────────────────────────

/// 12-dimensional context built from already-collected signals.
#[derive(Debug, Clone)]
pub struct AgentContext {
    pub features: [f64; D],
}

impl AgentContext {
    /// Build the context vector from existing daemon state.
    ///
    /// All inputs are already collected each cycle — no new syscalls.
    ///
    /// `outcome_effectiveness`: overall [0,1] from OutcomeTracker.
    /// `low_value_ratio`: fraction of tracked processes that are low-value [0,1].
    ///   When high, interventions are wasting effort — LinUCB learns to prefer
    ///   Observe or switch strategy.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        memory_pressure: f64,
        swap_trend: SwapTrend,
        time_to_critical_secs: Option<i32>,
        throughput_mips: f64,
        jitter_us: f64,
        cache_latency_us: f64,
        workload: WorkloadType,
        hour: u8,
        reactor_weight: f64,
        overflow_threshold_offset: f64,
        outcome_effectiveness: f64,
        low_value_ratio: f64,
    ) -> Self {
        let swap_ord = match swap_trend {
            SwapTrend::Decreasing => 0.0,
            SwapTrend::Stable => 0.25,
            SwapTrend::Increasing => 0.50,
            SwapTrend::Critical => 0.75,
        };
        let swap_urgency = match time_to_critical_secs {
            None => 0.0,
            Some(secs) => 1.0 / (1.0 + secs as f64),
        };
        let hour_f = hour as f64;
        let hour_sin = (2.0 * std::f64::consts::PI * hour_f / 24.0).sin();
        let hour_cos = (2.0 * std::f64::consts::PI * hour_f / 24.0).cos();
        let wl_ord = workload_ordinal(workload) as f64 / 7.0;

        // Slot 11: combined feedback signal.
        // effectiveness [0,1] penalized by low_value_ratio [0,1].
        // When low_value_ratio is high, the effective signal drops,
        // telling LinUCB that current interventions aren't working.
        let feedback_signal = (outcome_effectiveness * (1.0 - low_value_ratio)).clamp(0.0, 1.0);

        Self {
            features: [
                memory_pressure.clamp(0.0, 1.0),              // 0
                swap_ord,                                     // 1
                swap_urgency.clamp(0.0, 1.0),                 // 2
                (throughput_mips / 1200.0).clamp(0.0, 2.0),   // 3
                (jitter_us / 5000.0).clamp(0.0, 2.0),         // 4
                (cache_latency_us / 30000.0).clamp(0.0, 2.0), // 5
                wl_ord,                                       // 6
                hour_sin,                                     // 7
                hour_cos,                                     // 8
                reactor_weight.clamp(0.0, 1.0),               // 9
                overflow_threshold_offset.clamp(-0.20, 0.0),  // 10
                feedback_signal,                              // 11
            ],
        }
    }
}

fn workload_ordinal(wl: WorkloadType) -> u8 {
    match wl {
        WorkloadType::Coding => 0,
        WorkloadType::VideoCall => 1,
        WorkloadType::MediaPlayback => 2,
        WorkloadType::VideoEdit => 3,
        WorkloadType::OfficeWork => 4,
        WorkloadType::CommandLine => 5,
        WorkloadType::Idle => 6,
        WorkloadType::General => 7,
    }
}

// ── 12×12 matrix (row-major) ─────────────────────────────────────────────────

/// Fixed-size 12×12 matrix for LinUCB. Stored row-major as [f64; 144].
#[derive(Clone, Serialize, Deserialize)]
struct Mat12 {
    data: Vec<f64>, // length D*D = 144
}

impl Mat12 {
    fn identity() -> Self {
        let mut data = vec![0.0; D * D];
        for i in 0..D {
            data[i * D + i] = 1.0;
        }
        Self { data }
    }

    /// Compute A⁻¹ via Gauss-Jordan elimination on a 12×12 matrix.
    /// Returns None if singular (shouldn't happen with identity init + regularization).
    fn inverse(&self) -> Option<Self> {
        let mut aug = vec![0.0; D * 2 * D]; // D × 2D augmented matrix
        for i in 0..D {
            for j in 0..D {
                aug[i * 2 * D + j] = self.data[i * D + j];
            }
            aug[i * 2 * D + D + i] = 1.0;
        }
        for col in 0..D {
            // Partial pivoting
            let mut max_row = col;
            let mut max_val = aug[col * 2 * D + col].abs();
            for row in (col + 1)..D {
                let v = aug[row * 2 * D + col].abs();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }
            if max_val < 1e-15 {
                return None;
            }
            if max_row != col {
                for j in 0..(2 * D) {
                    let a = col * 2 * D + j;
                    let b = max_row * 2 * D + j;
                    aug.swap(a, b);
                }
            }
            let pivot = aug[col * 2 * D + col];
            for j in 0..(2 * D) {
                aug[col * 2 * D + j] /= pivot;
            }
            for row in 0..D {
                if row == col {
                    continue;
                }
                let factor = aug[row * 2 * D + col];
                for j in 0..(2 * D) {
                    aug[row * 2 * D + j] -= factor * aug[col * 2 * D + j];
                }
            }
        }
        let mut data = vec![0.0; D * D];
        for i in 0..D {
            for j in 0..D {
                data[i * D + j] = aug[i * 2 * D + D + j];
            }
        }
        Some(Self { data })
    }

    /// self += x * x' (outer product rank-1 update).
    fn add_outer(&mut self, x: &[f64; D]) {
        for i in 0..D {
            for j in 0..D {
                self.data[i * D + j] += x[i] * x[j];
            }
        }
    }

    /// Compute x' A x (quadratic form).
    fn quad_form(&self, x: &[f64; D]) -> f64 {
        let mut result = 0.0;
        for i in 0..D {
            for j in 0..D {
                result += x[i] * self.data[i * D + j] * x[j];
            }
        }
        result
    }
}

impl std::fmt::Debug for Mat12 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mat12[{}]", self.data.len())
    }
}

// ── LinUCB arm ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LinUCBArm {
    /// A matrix (D×D) — starts as identity.
    a: Mat12,
    /// b vector (D) — starts as zeros.
    b: Vec<f64>,
    /// Total pulls of this arm.
    pull_count: u64,
    /// Sum of rewards received.
    reward_sum: f64,
}

impl LinUCBArm {
    fn new() -> Self {
        Self {
            a: Mat12::identity(),
            b: vec![0.0; D],
            pull_count: 0,
            reward_sum: 0.0,
        }
    }

    /// UCB score for this arm given context x and exploration parameter alpha.
    fn score(&self, x: &[f64; D], alpha: f64) -> f64 {
        let a_inv = match self.a.inverse() {
            Some(inv) => inv,
            None => return 0.0, // degenerate — don't pick this arm
        };
        // θ = A⁻¹ b
        let mut theta = [0.0; D];
        for (i, th) in theta.iter_mut().enumerate() {
            for j in 0..D {
                *th += a_inv.data[i * D + j] * self.b[j];
            }
        }
        // exploitation = θ · x
        let exploit: f64 = theta.iter().zip(x.iter()).map(|(t, xi)| t * xi).sum();
        // exploration = α √(x' A⁻¹ x)
        let explore = alpha * a_inv.quad_form(x).max(0.0).sqrt();
        exploit + explore
    }

    /// Update arm after observing reward r for context x.
    fn update(&mut self, x: &[f64; D], reward: f64) {
        self.a.add_outer(x);
        for (bi, xi) in self.b.iter_mut().zip(x.iter()) {
            *bi += reward * xi;
        }
        self.pull_count += 1;
        self.reward_sum += reward;
    }
}

// ── Persisted state ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PredictiveAgentState {
    version: u32,
    arms: Vec<LinUCBArm>,
    alpha: f64,
    total_cycles: u64,
    warmup_remaining: u32,
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Predictive agent using LinUCB for proactive memory management.
pub struct PredictiveAgent {
    arms: [LinUCBArm; K],
    alpha: f64,
    total_cycles: u64,
    warmup_remaining: u32,
    /// Last chosen intervention and context (for delayed reward).
    last_action: Option<(Intervention, [f64; D], f64)>, // (arm, context, pressure_at_action)
    /// Cycles since last persist.
    cycles_since_persist: u32,
    path: PathBuf,
}

impl PredictiveAgent {
    /// Load from disk or create a fresh agent with warmup.
    pub fn load_or_default(path: &Path) -> Self {
        let default = || Self {
            arms: std::array::from_fn(|_| LinUCBArm::new()),
            alpha: 1.5,
            total_cycles: 0,
            warmup_remaining: WARMUP_CYCLES,
            last_action: None,
            cycles_since_persist: 0,
            path: path.to_path_buf(),
        };

        let loaded = std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str::<PredictiveAgentState>(&s).ok());

        match loaded {
            Some(state) if state.version == 1 && state.arms.len() == K => {
                let mut arms: [LinUCBArm; K] = std::array::from_fn(|_| LinUCBArm::new());
                for (i, arm) in state.arms.into_iter().enumerate() {
                    if arm.b.len() == D && arm.a.data.len() == D * D {
                        arms[i] = arm;
                    }
                }
                eprintln!(
                    "predictive-agent: loaded — {} cycles trained, warmup={}",
                    state.total_cycles, state.warmup_remaining
                );
                Self {
                    arms,
                    alpha: state.alpha,
                    total_cycles: state.total_cycles,
                    warmup_remaining: state.warmup_remaining,
                    last_action: None,
                    cycles_since_persist: 0,
                    path: path.to_path_buf(),
                }
            }
            _ => {
                eprintln!(
                    "predictive-agent: cold start — {} warmup cycles",
                    WARMUP_CYCLES
                );
                default()
            }
        }
    }

    /// Select the best intervention for the current context.
    pub fn select_action(&mut self, ctx: &AgentContext) -> Intervention {
        self.select_action_with_confidence(ctx).0
    }

    /// Like `select_action` but also returns the normalized UCB confidence [0, 1].
    ///
    /// The UCB score (exploit + explore term) is normalized by dividing by the
    /// score of the second-best arm so that margin of victory maps to confidence.
    /// When all arms are tied the confidence is 0.5; a dominant winner → near 1.0.
    /// During warmup returns (Observe, 0.0) — agent has no opinion yet.
    pub fn select_action_with_confidence(&mut self, ctx: &AgentContext) -> (Intervention, f64) {
        if self.warmup_remaining > 0 {
            self.warmup_remaining -= 1;
            self.last_action = Some((Intervention::Observe, ctx.features, ctx.features[0]));
            return (Intervention::Observe, 0.0);
        }

        let mut scores = [(0usize, f64::NEG_INFINITY); K];
        for i in 0..K {
            scores[i] = (i, self.arms[i].score(&ctx.features, self.alpha));
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best_arm = scores[0].0;
        let best_score = scores[0].1;
        let second_score = scores[1].1;

        // Confidence: how far ahead is the winner vs runner-up?
        // Normalize to [0.5, 1.0]: tie → 0.5, large margin → 1.0.
        let confidence = if best_score <= 0.0 || best_score == second_score {
            0.5
        } else {
            let margin = (best_score - second_score) / best_score.abs().max(1e-9);
            (0.5 + margin * 0.5).clamp(0.5, 1.0)
        };

        let intervention = Intervention::from_index(best_arm);
        self.last_action = Some((intervention, ctx.features, ctx.features[0]));
        (intervention, confidence)
    }

    /// Observe the outcome: current pressure after the intervention had time to act.
    /// Call this after execute_actions + outcome_tracker.tick().
    pub fn observe_outcome(&mut self, current_pressure: f64) {
        self.total_cycles += 1;

        let (intervention, features, pressure_at_action) = match self.last_action.take() {
            Some(v) => v,
            None => return,
        };

        let delta = pressure_at_action - current_pressure; // positive = pressure dropped

        let reward = if intervention == Intervention::Observe {
            // Observe: penalize only if pressure spiked while we did nothing.
            if delta < -0.05 {
                -0.3
            } else {
                0.0
            }
        } else {
            // Active intervention
            if delta > 0.05 {
                (delta * 5.0).clamp(0.0, 1.0)
            } else if delta < -0.03 {
                -0.5
            } else {
                -0.1 // cost of unnecessary action
            }
        };

        self.arms[intervention.index()].update(&features, reward);
    }

    /// Returns the threshold adjustment if TightenThresholds was chosen, else 0.
    pub fn threshold_adjustment(&self) -> f64 {
        match &self.last_action {
            Some((Intervention::TightenThresholds, _, _)) => TIGHTEN_OFFSET,
            _ => 0.0,
        }
    }

    /// Returns the last chosen intervention (for external logic like SuggestAggressive).
    pub fn last_intervention(&self) -> Option<Intervention> {
        self.last_action.as_ref().map(|(i, _, _)| *i)
    }

    /// Persist to disk every PERSIST_INTERVAL cycles.
    pub fn maybe_persist(&mut self) {
        self.cycles_since_persist += 1;
        if self.cycles_since_persist < PERSIST_INTERVAL {
            return;
        }
        self.cycles_since_persist = 0;
        self.persist();
    }

    fn persist(&self) {
        let state = PredictiveAgentState {
            version: 1,
            arms: self.arms.to_vec(),
            alpha: self.alpha,
            total_cycles: self.total_cycles,
            warmup_remaining: self.warmup_remaining,
        };
        if let Ok(json) = serde_json::to_string(&state) {
            let _ = std::fs::write(&self.path, json);
        }
    }

    /// Whether the agent is active (past warmup).
    pub fn is_active(&self) -> bool {
        self.warmup_remaining == 0
    }

    /// Total training cycles completed.
    pub fn total_cycles(&self) -> u64 {
        self.total_cycles
    }

    /// Pull counts per arm (for observability).
    pub fn arm_pulls(&self) -> [u64; K] {
        std::array::from_fn(|i| self.arms[i].pull_count)
    }

    /// Average reward per arm (for observability).
    pub fn arm_avg_rewards(&self) -> [f64; K] {
        std::array::from_fn(|i| {
            if self.arms[i].pull_count == 0 {
                0.0
            } else {
                self.arms[i].reward_sum / self.arms[i].pull_count as f64
            }
        })
    }

    /// Apply the chosen intervention's threshold adjustments to existing thresholds.
    /// Returns adjusted thresholds (only modifies if TightenThresholds was selected).
    pub fn adjust_thresholds(&self, mut thresholds: OverflowThresholds) -> OverflowThresholds {
        let adj = self.threshold_adjustment();
        if adj != 0.0 {
            thresholds.bg_pressure = (thresholds.bg_pressure + adj).max(0.50);
            thresholds.critical_pressure = (thresholds.critical_pressure + adj).max(0.60);
            thresholds.extreme_pressure = (thresholds.extreme_pressure + adj).max(0.65);
        }
        thresholds
    }

    /// ZeroTune: seed LinUCB arms with system-aware priors to reduce cold-start.
    ///
    /// Instead of 200 blind Observe cycles, inject synthetic observations based
    /// on hardware meta-features. This encodes domain knowledge:
    /// - Low RAM (≤8 GB): TightenThresholds and ProactivePurge are more valuable
    /// - High RAM (>16 GB): Observe is often sufficient
    /// - More cores: PreThrottleNoise is cheaper (more scheduling headroom)
    ///
    /// Call once at initialization when no persisted state exists.
    /// Reduces warmup from 200 → 50 cycles.
    pub fn meta_seed(&mut self, ram_gb: f64, cores: usize) {
        if self.total_cycles > 0 {
            return; // already trained, don't overwrite
        }

        // Synthetic context: moderate pressure scenario (the interesting regime).
        let mut ctx = [0.0; D];
        ctx[0] = 0.55; // memory_pressure (moderate-high)
        ctx[1] = 0.3; // swap_trend (rising)
        ctx[2] = 0.5; // time_to_critical (medium)

        // Prior rewards by arm, scaled by hardware.
        // Low RAM → proactive interventions are more valuable.
        let ram_factor = (16.0 / ram_gb).clamp(0.5, 2.0); // 8GB→2.0, 16GB→1.0, 32GB→0.5
        let core_factor = (cores as f64 / 8.0).clamp(0.5, 1.5); // 4→0.5, 8→1.0, 12→1.5

        let priors = [
            0.0,                // Observe: neutral
            0.3 * ram_factor,   // TightenThresholds: better with low RAM
            0.1,                // SuggestAggressive: mild prior
            0.15 * core_factor, // PreThrottleNoise: better with more cores
            0.25 * ram_factor,  // ProactivePurge: better with low RAM
        ];

        // Inject N synthetic pulls per arm (like pseudo-observations).
        const SEED_PULLS: usize = 5;
        for (arm_idx, &reward) in priors.iter().enumerate() {
            for _ in 0..SEED_PULLS {
                self.arms[arm_idx].update(&ctx, reward);
            }
        }

        self.warmup_remaining = SEEDED_WARMUP_CYCLES;
        eprintln!(
            "predictive-agent: ZeroTune seeded (ram={:.0}GB, cores={}) — warmup reduced to {}",
            ram_gb, cores, SEEDED_WARMUP_CYCLES
        );
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "apollo_test_predictive_{name}_{}.json",
            std::process::id()
        ));
        p
    }

    fn dummy_context(pressure: f64) -> AgentContext {
        AgentContext::build(
            pressure,
            SwapTrend::Stable,
            None,
            800.0,
            50.0,
            5000.0,
            WorkloadType::General,
            14,
            0.0,
            0.0,
            0.5,
            0.0, // low_value_ratio
        )
    }

    #[test]
    fn test_linucb_select_returns_observe_during_warmup() {
        let path = test_path("warmup");
        let mut agent = PredictiveAgent::load_or_default(&path);
        assert!(agent.warmup_remaining > 0);

        let ctx = dummy_context(0.5);
        for _ in 0..10 {
            let action = agent.select_action(&ctx);
            assert_eq!(action, Intervention::Observe);
            agent.observe_outcome(0.5);
        }
    }

    #[test]
    fn test_linucb_exploration_all_arms() {
        let path = test_path("explore");
        let mut agent = PredictiveAgent::load_or_default(&path);
        // Skip warmup
        agent.warmup_remaining = 0;

        let mut seen = [false; K];
        // With alpha=1.5 and identity matrices, exploration should try all arms.
        for pressure in 0..50 {
            let ctx = dummy_context(pressure as f64 / 100.0);
            let action = agent.select_action(&ctx);
            seen[action.index()] = true;
            agent.observe_outcome(ctx.features[0] - 0.01);
        }
        // At least Observe and one other arm should have been tried.
        assert!(seen[0], "Observe should be tried");
        let non_observe_tried = seen[1..].iter().any(|&s| s);
        assert!(
            non_observe_tried,
            "At least one non-observe arm should be explored"
        );
    }

    #[test]
    fn test_reward_pressure_drop() {
        let path = test_path("reward");
        let mut agent = PredictiveAgent::load_or_default(&path);
        agent.warmup_remaining = 0;

        // Simulate: agent chooses TightenThresholds, pressure drops.
        let ctx = dummy_context(0.8);
        let _action = agent.select_action(&ctx);
        // Force the last_action to TightenThresholds for testing.
        agent.last_action = Some((Intervention::TightenThresholds, ctx.features, 0.8));
        agent.observe_outcome(0.7); // delta = 0.1 > 0.05 → positive reward

        let pulls = agent.arm_pulls();
        assert_eq!(pulls[1], 1, "TightenThresholds should have 1 pull");
        let avg = agent.arm_avg_rewards();
        assert!(
            avg[1] > 0.0,
            "TightenThresholds avg reward should be positive"
        );
    }

    #[test]
    fn test_threshold_adjustment_only_when_tighten() {
        let path = test_path("thresh");
        let mut agent = PredictiveAgent::load_or_default(&path);
        agent.warmup_remaining = 0;

        // No action chosen yet → no adjustment.
        assert_eq!(agent.threshold_adjustment(), 0.0);

        // Force Observe
        let ctx = dummy_context(0.5);
        agent.last_action = Some((Intervention::Observe, ctx.features, 0.5));
        assert_eq!(agent.threshold_adjustment(), 0.0);

        // Force TightenThresholds
        agent.last_action = Some((Intervention::TightenThresholds, ctx.features, 0.5));
        assert!((agent.threshold_adjustment() - TIGHTEN_OFFSET).abs() < 1e-10);
    }

    #[test]
    fn test_persistence_roundtrip() {
        let path = test_path("persist");

        // Train a bit and persist.
        {
            let mut agent = PredictiveAgent::load_or_default(&path);
            agent.warmup_remaining = 0;
            for i in 0..20 {
                let ctx = dummy_context(0.5 + (i as f64) * 0.01);
                agent.select_action(&ctx);
                agent.observe_outcome(0.5);
            }
            agent.persist();
        }

        // Reload and verify state was preserved.
        {
            let agent = PredictiveAgent::load_or_default(&path);
            assert_eq!(agent.warmup_remaining, 0);
            assert_eq!(agent.total_cycles, 20);
            let pulls: u64 = agent.arm_pulls().iter().sum();
            assert_eq!(pulls, 20);
        }
    }

    #[test]
    fn test_adjust_thresholds() {
        let path = test_path("adjust");
        let mut agent = PredictiveAgent::load_or_default(&path);
        agent.warmup_remaining = 0;

        let base = OverflowThresholds::default();

        // Without TightenThresholds, thresholds unchanged.
        let ctx = dummy_context(0.5);
        agent.last_action = Some((Intervention::Observe, ctx.features, 0.5));
        let adj = agent.adjust_thresholds(base);
        assert!((adj.bg_pressure - base.bg_pressure).abs() < 1e-10);

        // With TightenThresholds, thresholds lowered.
        agent.last_action = Some((Intervention::TightenThresholds, ctx.features, 0.5));
        let adj = agent.adjust_thresholds(base);
        assert!(adj.bg_pressure < base.bg_pressure);
        assert!(adj.critical_pressure < base.critical_pressure);
        assert!(adj.extreme_pressure < base.extreme_pressure);
    }

    #[test]
    fn test_mat12_inverse_identity() {
        let id = Mat12::identity();
        let inv = id.inverse().unwrap();
        // Inverse of identity is identity.
        for i in 0..D {
            for j in 0..D {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (inv.data[i * D + j] - expected).abs() < 1e-10,
                    "inv[{}][{}] = {} (expected {})",
                    i,
                    j,
                    inv.data[i * D + j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_feedback_signal_combines_effectiveness_and_low_value() {
        // High effectiveness + no low-value → high feedback signal.
        let ctx_good = AgentContext::build(
            0.5,
            SwapTrend::Stable,
            None,
            800.0,
            50.0,
            5000.0,
            WorkloadType::General,
            14,
            0.0,
            0.0,
            0.80, // outcome_effectiveness
            0.0,  // low_value_ratio
        );
        assert!(
            (ctx_good.features[11] - 0.80).abs() < 1e-6,
            "no low-value: feedback should equal effectiveness"
        );

        // High effectiveness + high low-value → penalized feedback.
        let ctx_bad = AgentContext::build(
            0.5,
            SwapTrend::Stable,
            None,
            800.0,
            50.0,
            5000.0,
            WorkloadType::General,
            14,
            0.0,
            0.0,
            0.80, // outcome_effectiveness
            0.50, // 50% low-value
        );
        assert!(
            (ctx_bad.features[11] - 0.40).abs() < 1e-6,
            "50% low-value: feedback should be 0.80 * 0.50 = 0.40, got {}",
            ctx_bad.features[11]
        );

        // Darwinian: bad context should signal lower than good context.
        assert!(ctx_bad.features[11] < ctx_good.features[11]);
    }

    #[test]
    fn test_context_build_ranges() {
        let ctx = AgentContext::build(
            1.5, // will be clamped to 1.0
            SwapTrend::Critical,
            Some(5),
            2000.0,
            10000.0,
            60000.0,
            WorkloadType::Coding,
            23,
            1.0,
            -0.15,
            0.8,
            0.3, // low_value_ratio
        );
        // memory_pressure clamped
        assert!((ctx.features[0] - 1.0).abs() < 1e-10);
        // swap_trend critical = 0.75
        assert!((ctx.features[1] - 0.75).abs() < 1e-10);
        // swap_urgency = 1/(1+5) = 0.1667
        assert!((ctx.features[2] - 1.0 / 6.0).abs() < 1e-3);
    }

    // ── ZeroTune cold start tests ────────────────────────────────────────────

    #[test]
    fn test_meta_seed_reduces_warmup() {
        let path = test_path("zerotune_warmup");
        let mut agent = PredictiveAgent::load_or_default(&path);
        assert_eq!(agent.warmup_remaining, WARMUP_CYCLES);

        agent.meta_seed(8.0, 8);
        assert_eq!(agent.warmup_remaining, SEEDED_WARMUP_CYCLES);
        assert!(SEEDED_WARMUP_CYCLES < WARMUP_CYCLES);
    }

    #[test]
    fn test_meta_seed_injects_priors_into_arms() {
        let path = test_path("zerotune_priors");
        let mut agent = PredictiveAgent::load_or_default(&path);
        agent.meta_seed(8.0, 8);

        // After seeding, arms should have pull_count > 0.
        let pulls = agent.arm_pulls();
        assert!(
            pulls.iter().all(|&p| p > 0),
            "all arms should have pulls: {:?}",
            pulls
        );

        // TightenThresholds (arm 1) should have higher avg reward than Observe (arm 0)
        // on 8GB RAM (ram_factor=2.0 → 0.3*2.0=0.6 vs 0.0).
        let avg = agent.arm_avg_rewards();
        assert!(
            avg[1] > avg[0],
            "TightenThresholds should have higher prior than Observe on 8GB: {:?}",
            avg
        );
    }

    #[test]
    fn test_meta_seed_noop_after_training() {
        let path = test_path("zerotune_noop");
        let mut agent = PredictiveAgent::load_or_default(&path);
        // Simulate some training
        agent.warmup_remaining = 0;
        let ctx = dummy_context(0.5);
        agent.select_action(&ctx);
        agent.observe_outcome(0.5);
        assert!(agent.total_cycles() > 0);

        let pulls_before = agent.arm_pulls();
        agent.meta_seed(8.0, 8); // should be no-op
        let pulls_after = agent.arm_pulls();
        assert_eq!(
            pulls_before, pulls_after,
            "meta_seed should be no-op after training"
        );
    }

    #[test]
    fn test_meta_seed_low_vs_high_ram() {
        let path_low = test_path("zerotune_low_ram");
        let path_high = test_path("zerotune_high_ram");
        let mut agent_low = PredictiveAgent::load_or_default(&path_low);
        let mut agent_high = PredictiveAgent::load_or_default(&path_high);

        agent_low.meta_seed(8.0, 8);
        agent_high.meta_seed(32.0, 8);

        // On 8GB, TightenThresholds (arm 1) should have higher reward than on 32GB.
        let avg_low = agent_low.arm_avg_rewards();
        let avg_high = agent_high.arm_avg_rewards();
        assert!(
            avg_low[1] > avg_high[1],
            "TightenThresholds should be more valued on 8GB ({}) than 32GB ({})",
            avg_low[1],
            avg_high[1]
        );
    }

    // ── Specialist voting tests ──────────────────────────────────────────────

    #[test]
    fn test_voting_single_specialist_wins() {
        let votes = vec![SpecialistVote {
            name: "linucb",
            intervention: Intervention::TightenThresholds,
            confidence: 0.8,
        }];
        let result = tally_votes(&votes);
        assert_eq!(result.intervention, Intervention::TightenThresholds);
        assert!(!result.had_disagreement);
    }

    #[test]
    fn test_voting_highest_confidence_wins() {
        let votes = vec![
            SpecialistVote {
                name: "linucb",
                intervention: Intervention::Observe,
                confidence: 0.3,
            },
            SpecialistVote {
                name: "hazard",
                intervention: Intervention::SuggestAggressive,
                confidence: 0.9,
            },
        ];
        let result = tally_votes(&votes);
        assert_eq!(result.intervention, Intervention::SuggestAggressive);
    }

    #[test]
    fn test_voting_detects_disagreement() {
        let votes = vec![
            SpecialistVote {
                name: "hazard",
                intervention: Intervention::SuggestAggressive,
                confidence: 0.5,
            },
            SpecialistVote {
                name: "monopoly",
                intervention: Intervention::PreThrottleNoise,
                confidence: 0.5,
            },
        ];
        let result = tally_votes(&votes);
        assert!(
            result.had_disagreement,
            "two different non-Observe proposals = disagreement"
        );
    }

    #[test]
    fn test_voting_same_intervention_accumulates() {
        let votes = vec![
            SpecialistVote {
                name: "hazard",
                intervention: Intervention::TightenThresholds,
                confidence: 0.4,
            },
            SpecialistVote {
                name: "kalman",
                intervention: Intervention::TightenThresholds,
                confidence: 0.5,
            },
            SpecialistVote {
                name: "linucb",
                intervention: Intervention::Observe,
                confidence: 0.6,
            },
        ];
        let result = tally_votes(&votes);
        // TightenThresholds: 0.4+0.5 = 0.9 > Observe: 0.6
        assert_eq!(result.intervention, Intervention::TightenThresholds);
        assert!((result.winning_score - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_voting_tie_favors_safer_option() {
        let votes = vec![
            SpecialistVote {
                name: "a",
                intervention: Intervention::Observe,
                confidence: 0.5,
            },
            SpecialistVote {
                name: "b",
                intervention: Intervention::SuggestAggressive,
                confidence: 0.5,
            },
        ];
        let result = tally_votes(&votes);
        // Equal scores → lower index (Observe=0) wins.
        assert_eq!(result.intervention, Intervention::Observe);
    }

    // ── SpecialistAccuracyTracker tests ──────────────────────────────────────

    #[test]
    fn test_specialist_accuracy_tracker_init_is_0_70() {
        let tracker = SpecialistAccuracyTracker::new();
        for i in 0..specialist::COUNT {
            assert!(
                (tracker.weight(i) - 0.70).abs() < 1e-12,
                "specialist {} should init to 0.70, got {}",
                specialist::NAMES[i],
                tracker.weight(i)
            );
        }
        assert_eq!(tracker.total_updates(), 0);
    }

    #[test]
    fn test_specialist_accuracy_increases_on_correct_predictions() {
        let mut tracker = SpecialistAccuracyTracker::new();
        let initial = tracker.weight(specialist::HAZARD);
        // Many correct predictions should push accuracy above initial.
        for _ in 0..50 {
            tracker.update(specialist::HAZARD, true);
        }
        assert!(
            tracker.weight(specialist::HAZARD) > initial,
            "accuracy should rise after many correct predictions: {} > {}",
            tracker.weight(specialist::HAZARD),
            initial
        );
        assert_eq!(tracker.total_updates(), 50);
    }

    #[test]
    fn test_specialist_accuracy_decreases_on_false_alarms() {
        let mut tracker = SpecialistAccuracyTracker::new();
        let initial = tracker.weight(specialist::MONOPOLY);
        // Many incorrect predictions should push accuracy below initial.
        for _ in 0..50 {
            tracker.update(specialist::MONOPOLY, false);
        }
        assert!(
            tracker.weight(specialist::MONOPOLY) < initial,
            "accuracy should drop after many false alarms: {} < {}",
            tracker.weight(specialist::MONOPOLY),
            initial
        );
    }

    #[test]
    fn test_specialist_accuracy_update_by_name() {
        let mut tracker = SpecialistAccuracyTracker::new();
        let before = tracker.weight(specialist::KALMAN);
        tracker.update_by_name("kalman", true);
        let after = tracker.weight(specialist::KALMAN);
        // One correct update: 0.97 * 0.70 + 0.03 * 1.0 = 0.709
        let expected = 0.97 * 0.70 + 0.03 * 1.0;
        assert!(
            (after - expected).abs() < 1e-12,
            "after one correct update: expected {expected}, got {after}"
        );
        assert!(after > before);
    }

    #[test]
    fn test_specialist_accuracy_unknown_name_is_noop() {
        let mut tracker = SpecialistAccuracyTracker::new();
        let before: Vec<f64> = (0..specialist::COUNT).map(|i| tracker.weight(i)).collect();
        tracker.update_by_name("nonexistent", true);
        let after: Vec<f64> = (0..specialist::COUNT).map(|i| tracker.weight(i)).collect();
        assert_eq!(before, after, "unknown specialist name should be a no-op");
        assert_eq!(tracker.total_updates(), 0);
    }

    // ── Intervention from_index tests ────────────────────────────────────────

    #[test]
    fn test_from_index_all_valid_variants() {
        assert_eq!(Intervention::from_index(0), Intervention::Observe);
        assert_eq!(Intervention::from_index(1), Intervention::TightenThresholds);
        assert_eq!(Intervention::from_index(2), Intervention::SuggestAggressive);
        assert_eq!(Intervention::from_index(3), Intervention::PreThrottleNoise);
        assert_eq!(Intervention::from_index(4), Intervention::ProactivePurge);
    }

    #[test]
    fn test_from_index_out_of_range_falls_back_to_observe() {
        assert_eq!(Intervention::from_index(5), Intervention::Observe);
        assert_eq!(Intervention::from_index(99), Intervention::Observe);
        assert_eq!(Intervention::from_index(usize::MAX), Intervention::Observe);
    }

    #[test]
    fn test_intervention_index_roundtrip() {
        let variants = [
            Intervention::Observe,
            Intervention::TightenThresholds,
            Intervention::SuggestAggressive,
            Intervention::PreThrottleNoise,
            Intervention::ProactivePurge,
        ];
        for v in variants {
            assert_eq!(
                Intervention::from_index(v.index()),
                v,
                "from_index(index()) should be identity for {:?}",
                v
            );
        }
    }

    #[test]
    fn test_intervention_serde_roundtrip() {
        let variants = [
            Intervention::Observe,
            Intervention::TightenThresholds,
            Intervention::SuggestAggressive,
            Intervention::PreThrottleNoise,
            Intervention::ProactivePurge,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).expect("serialize");
            let back: Intervention = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, v, "serde roundtrip failed for {:?}", v);
        }
    }

    // ── SpecialistAccuracyTracker — weight convergence & clamping ────────────

    #[test]
    fn test_specialist_accuracy_weight_never_exceeds_one() {
        let mut tracker = SpecialistAccuracyTracker::new();
        for _ in 0..1000 {
            tracker.update(specialist::LINUCB, true);
        }
        let w = tracker.weight(specialist::LINUCB);
        assert!(
            w <= 1.0,
            "weight must not exceed 1.0 after 1000 correct: {w}"
        );
    }

    #[test]
    fn test_specialist_accuracy_weight_never_below_zero() {
        let mut tracker = SpecialistAccuracyTracker::new();
        for _ in 0..1000 {
            tracker.update(specialist::LINUCB, false);
        }
        let w = tracker.weight(specialist::LINUCB);
        assert!(
            w >= 0.0,
            "weight must not go below 0.0 after 1000 incorrect: {w}"
        );
    }

    #[test]
    fn test_specialist_accuracy_convergence_to_one_after_many_correct() {
        let mut tracker = SpecialistAccuracyTracker::new();
        // EMA with alpha=0.03 converges to 1.0 from 0.70: after ~300 steps it should be > 0.97.
        for _ in 0..300 {
            tracker.update(specialist::KALMAN, true);
        }
        let w = tracker.weight(specialist::KALMAN);
        assert!(
            w > 0.97,
            "after 300 correct, weight should approach 1.0, got {w}"
        );
    }

    #[test]
    fn test_specialist_accuracy_convergence_to_zero_after_many_incorrect() {
        let mut tracker = SpecialistAccuracyTracker::new();
        // EMA with alpha=0.03 converges to 0.0 from 0.70: after ~300 steps it should be < 0.03.
        for _ in 0..300 {
            tracker.update(specialist::HAZARD, false);
        }
        let w = tracker.weight(specialist::HAZARD);
        assert!(
            w < 0.03,
            "after 300 incorrect, weight should approach 0.0, got {w}"
        );
    }

    #[test]
    fn test_specialist_accuracy_out_of_range_index_returns_default() {
        let tracker = SpecialistAccuracyTracker::new();
        // Out-of-range index should return the default INIT_ACCURACY (0.70).
        let w = tracker.weight(specialist::COUNT + 10);
        assert!(
            (w - 0.70).abs() < 1e-12,
            "out-of-range index should return default 0.70, got {w}"
        );
    }

    #[test]
    fn test_specialist_accuracy_update_out_of_range_is_noop() {
        let mut tracker = SpecialistAccuracyTracker::new();
        let before = tracker.total_updates();
        tracker.update(specialist::COUNT + 5, true);
        assert_eq!(
            tracker.total_updates(),
            before,
            "out-of-range update should be no-op"
        );
    }

    #[test]
    fn test_specialist_accuracy_tracker_serde_roundtrip() {
        let mut tracker = SpecialistAccuracyTracker::new();
        for _ in 0..20 {
            tracker.update(specialist::LINUCB, true);
            tracker.update(specialist::HAZARD, false);
        }
        let json = serde_json::to_string(&tracker).expect("serialize");
        let back: SpecialistAccuracyTracker = serde_json::from_str(&json).expect("deserialize");
        for i in 0..specialist::COUNT {
            assert!(
                (back.weight(i) - tracker.weight(i)).abs() < 1e-12,
                "weight[{i}] mismatch after serde roundtrip"
            );
        }
        assert_eq!(back.total_updates(), tracker.total_updates());
    }

    // ── record_disagreement_outcome tests ─────────────────────────────────

    #[test]
    fn test_disagreement_outcome_boosts_correct_specialist() {
        let mut tracker = SpecialistAccuracyTracker::new();
        let votes = vec![
            SpecialistVote {
                name: "linucb",
                intervention: Intervention::TightenThresholds,
                confidence: 0.8,
            },
            SpecialistVote {
                name: "hazard",
                intervention: Intervention::ProactivePurge,
                confidence: 0.6,
            },
        ];
        let before_linucb = tracker.weight(specialist::LINUCB);
        let before_hazard = tracker.weight(specialist::HAZARD);
        // TightenThresholds won and was effective → linucb was right, hazard wrong
        tracker.record_disagreement_outcome(&votes, Intervention::TightenThresholds, true);
        assert!(
            tracker.weight(specialist::LINUCB) > before_linucb,
            "correct specialist should be boosted"
        );
        assert!(
            tracker.weight(specialist::HAZARD) < before_hazard,
            "wrong specialist should be penalized"
        );
    }

    #[test]
    fn test_disagreement_outcome_ineffective_penalizes_winner() {
        let mut tracker = SpecialistAccuracyTracker::new();
        let votes = vec![
            SpecialistVote {
                name: "linucb",
                intervention: Intervention::TightenThresholds,
                confidence: 0.8,
            },
            SpecialistVote {
                name: "hazard",
                intervention: Intervention::ProactivePurge,
                confidence: 0.6,
            },
        ];
        let before_linucb = tracker.weight(specialist::LINUCB);
        let before_hazard = tracker.weight(specialist::HAZARD);
        // TightenThresholds won but was NOT effective → linucb wrong, hazard right
        tracker.record_disagreement_outcome(&votes, Intervention::TightenThresholds, false);
        assert!(
            tracker.weight(specialist::LINUCB) < before_linucb,
            "winner of ineffective intervention should be penalized"
        );
        assert!(
            tracker.weight(specialist::HAZARD) > before_hazard,
            "dissenter should be boosted when winner was wrong"
        );
    }

    #[test]
    fn test_disagreement_outcome_unknown_specialist_is_noop() {
        let mut tracker = SpecialistAccuracyTracker::new();
        let votes = vec![SpecialistVote {
            name: "nonexistent",
            intervention: Intervention::Observe,
            confidence: 0.5,
        }];
        let updates_before = tracker.total_updates();
        tracker.record_disagreement_outcome(&votes, Intervention::Observe, true);
        // Unknown specialist name → no update
        assert_eq!(tracker.total_updates(), updates_before);
    }

    // ── AgentContext edge cases ──────────────────────────────────────────────

    #[test]
    fn test_context_extreme_zero_pressure_no_panic() {
        let ctx = AgentContext::build(
            0.0,
            SwapTrend::Decreasing,
            None,
            0.0,
            0.0,
            0.0,
            WorkloadType::Idle,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
        );
        assert!((ctx.features[0] - 0.0).abs() < 1e-10);
        assert!(
            ctx.features.iter().all(|f| f.is_finite()),
            "all features must be finite"
        );
    }

    #[test]
    fn test_context_extreme_max_pressure_no_panic() {
        let ctx = AgentContext::build(
            1.0,
            SwapTrend::Critical,
            Some(0),
            9999.0,
            99999.0,
            999999.0,
            WorkloadType::VideoEdit,
            23,
            1.0,
            -0.20,
            1.0,
            1.0,
        );
        assert!((ctx.features[0] - 1.0).abs() < 1e-10);
        assert!(
            ctx.features.iter().all(|f| f.is_finite()),
            "all features must be finite"
        );
    }

    // ── LinUCB post-warmup behavior & edge cases ─────────────────────────────

    #[test]
    fn test_linucb_post_warmup_returns_valid_intervention() {
        let path = test_path("post_warmup_valid");
        let mut agent = PredictiveAgent::load_or_default(&path);
        agent.warmup_remaining = 0;

        let ctx = dummy_context(0.6);
        let action = agent.select_action(&ctx);
        // Result must be one of the 5 valid variants (from_index ensures this).
        let valid = [
            Intervention::Observe,
            Intervention::TightenThresholds,
            Intervention::SuggestAggressive,
            Intervention::PreThrottleNoise,
            Intervention::ProactivePurge,
        ];
        assert!(
            valid.contains(&action),
            "select_action must return a valid Intervention"
        );
    }

    #[test]
    fn test_linucb_observe_outcome_without_prior_select_is_noop() {
        // observe_outcome with no last_action set should not panic.
        let path = test_path("outcome_no_action");
        let mut agent = PredictiveAgent::load_or_default(&path);
        agent.observe_outcome(0.5); // no last_action — should be silent no-op
        assert_eq!(agent.total_cycles(), 1); // total_cycles still increments
        let pulls = agent.arm_pulls();
        assert!(
            pulls.iter().all(|&p| p == 0),
            "no arms should have pulls: {:?}",
            pulls
        );
    }

    #[test]
    fn test_linucb_select_with_confidence_warmup_returns_zero_confidence() {
        let path = test_path("confidence_warmup");
        let mut agent = PredictiveAgent::load_or_default(&path);
        assert!(agent.warmup_remaining > 0);

        let ctx = dummy_context(0.5);
        let (action, confidence) = agent.select_action_with_confidence(&ctx);
        assert_eq!(action, Intervention::Observe);
        assert!(
            (confidence - 0.0).abs() < 1e-10,
            "during warmup confidence should be 0.0, got {confidence}"
        );
    }

    #[test]
    fn test_linucb_select_with_confidence_post_warmup_in_range() {
        let path = test_path("confidence_post");
        let mut agent = PredictiveAgent::load_or_default(&path);
        agent.warmup_remaining = 0;

        let ctx = dummy_context(0.5);
        let (_, confidence) = agent.select_action_with_confidence(&ctx);
        assert!(
            confidence >= 0.0 && confidence <= 1.0,
            "confidence must be in [0, 1], got {confidence}"
        );
    }

    #[test]
    fn test_linucb_is_active_only_after_warmup() {
        let path = test_path("is_active");
        let mut agent = PredictiveAgent::load_or_default(&path);
        assert!(!agent.is_active(), "should not be active during warmup");
        agent.warmup_remaining = 0;
        assert!(agent.is_active(), "should be active after warmup");
    }

    #[test]
    fn test_linucb_observe_negative_reward_on_pressure_spike() {
        // Observe intervention + pressure increased → negative reward for Observe arm.
        let path = test_path("negative_reward");
        let mut agent = PredictiveAgent::load_or_default(&path);
        agent.warmup_remaining = 0;

        let ctx = dummy_context(0.5);
        agent.last_action = Some((Intervention::Observe, ctx.features, 0.5));
        // pressure rose from 0.5 to 0.6: delta = 0.5 - 0.6 = -0.1 < -0.05 → penalty -0.3
        agent.observe_outcome(0.6);

        let pulls = agent.arm_pulls();
        assert_eq!(pulls[0], 1, "Observe arm should have 1 pull");
        let avg = agent.arm_avg_rewards();
        assert!(
            avg[0] < 0.0,
            "Observe should receive negative reward when pressure spikes: {}",
            avg[0]
        );
    }

    #[test]
    fn test_voting_empty_votes_returns_observe() {
        // Empty vote list: all scores = 0.0, lower index (Observe) wins.
        let result = tally_votes(&[]);
        assert_eq!(result.intervention, Intervention::Observe);
        assert!(!result.had_disagreement);
        assert!((result.winning_score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_voting_dominant_specialist_wins() {
        // One specialist has weight 0.99, all others 0.01.
        let votes = vec![
            SpecialistVote {
                name: "linucb",
                intervention: Intervention::ProactivePurge,
                confidence: 0.99,
            },
            SpecialistVote {
                name: "hazard",
                intervention: Intervention::Observe,
                confidence: 0.01,
            },
            SpecialistVote {
                name: "monopoly",
                intervention: Intervention::Observe,
                confidence: 0.01,
            },
            SpecialistVote {
                name: "kalman",
                intervention: Intervention::Observe,
                confidence: 0.01,
            },
        ];
        let result = tally_votes(&votes);
        assert_eq!(
            result.intervention,
            Intervention::ProactivePurge,
            "dominant specialist (0.99 vs 0.03) should win"
        );
    }

    #[test]
    fn test_specialist_index_of_all_known_names() {
        assert_eq!(specialist::index_of("linucb"), Some(specialist::LINUCB));
        assert_eq!(specialist::index_of("hazard"), Some(specialist::HAZARD));
        assert_eq!(specialist::index_of("monopoly"), Some(specialist::MONOPOLY));
        assert_eq!(specialist::index_of("kalman"), Some(specialist::KALMAN));
        assert_eq!(specialist::index_of("unknown"), None);
    }
}
