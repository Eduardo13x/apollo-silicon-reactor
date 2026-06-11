//! ActionPolicyScorer — composable decision surface replacing the gate tower.
//!
//! Every candidate `RootAction` is scored by a vector of `PolicyFeature`
//! implementations. Each feature contributes {benefit, cost, uncertainty,
//! hard_veto?}. The scorer aggregates and emits a `PolicyScore` with an
//! accept/reject decision and a reason string.
//!
//! **Shadow mode:** in the scaffold commit, the scorer is NOT wired into the
//! daemon main loop. Follow-up commits will run it alongside the existing
//! gate tower (logging disagreements to `BlockedActionJournal`), then cut
//! over once disagreement rate is bounded.
//!
//! # Motivation
//! The "Fix-N-minus-1" meta-pattern: every new failure mode spawned a new
//! bypass conditional. Gate C's thrashing-bypass (commit 69b0b8b) was the
//! 12th such fix in April. The root cause was that each gate was an
//! independent `if` branch with ad-hoc bypass chains — there was no
//! composable decision surface. Treating each gate as a *feature
//! contributing to a composite score* solves this structurally.
//!
//! # Theory
//! - [Puterman 1994] MDP — expected-utility decision frameworks: actions
//!   chosen by maximizing expected utility over composed value signals.
//! - [Ng 1999] potential-based reward shaping — safely inject new signals
//!   without changing the optimal policy.
//! - [Lakshminarayanan 2017] epistemic uncertainty estimation — weighted
//!   penalty for under-confident contributions.
//!
//! # Extension
//! Downstream features (F5 Deep Scan hot-page cost, F6 predictive
//! p_oom_30s, F7 sensor freshness) implement `PolicyFeature` and register
//! via `PolicyScorer::builder().feature(...)`. No core API change
//! required.

// Do NOT import from user_context, decide_actions, execute_actions — scaffold
// must be dependency-free of the current gate tower to keep migration clean.
use crate::engine::safety::ProtectionLevel;
use crate::engine::types::RootAction;

/// Context packaged for scoring one candidate action. All fields the scorer
/// needs for its decision. Fields here map 1-1 to current gate inputs.
#[derive(Debug, Clone)]
pub struct ActionContext {
    pub pressure: f64,
    pub swap_gb: f64,
    pub thrashing_score: f64,
    /// Predicted probability of OOM kill within the next 30s. F6 will wire.
    pub p_oom_30s: Option<f64>,
    /// Predicted probability of jank within the next 60s. F6 will wire.
    pub p_jank_60s: Option<f64>,
    pub has_sleep_assertion: bool,
    pub call_in_progress: bool,
    pub idle_secs: f64,
    pub foreground_pid: Option<u32>,
    pub is_foreground_family: bool,
    pub is_recently_active: bool,
    pub thermal_emergency: bool,
    /// 0..=3 — higher means deeper in an interrupt-driven grace period.
    pub interrupt_phase: u8,
    pub protection_level: ProtectionLevel,
    /// Fraction of process RSS that is "hot" (recently touched). F5 will wire.
    pub hot_page_fraction: Option<f64>,
    /// Working-set size in MiB from Deep Scan. F5 will wire.
    pub wss_mb: Option<f64>,
    /// Age of the freshest sensor sample feeding this decision, ms. F7 will wire.
    pub sensor_age_ms: Option<u64>,
    /// Global epistemic uncertainty ≥ 0; features may use or add to it.
    pub epistemic_uncertainty: f64,

    // ── Phase 5.2 wiring inputs (Sprint 10, 2026-05-16) ─────────────────────
    /// True when the machine is running on battery power (no AC charger
    /// detected via SMC). Sourced from `SmcSnapshot.charger_watts`: `Some(>0)`
    /// → on AC; `Some(0.0)` or `None` → on battery. Optional because not
    /// every code path constructs ActionContext from a hardware snapshot
    /// (tests, dry-run paths). When `None`, the battery-aware penalty
    /// returns 0.0 (no UX cost imposed without evidence).
    pub is_on_battery: Option<bool>,
    /// Recent wake-ups per second (sysinfo cumulative wakeups / cycle_dt).
    /// Used by Phase 5.2 to scale the "micro-wakeup tax" when running on
    /// battery. Optional same rationale as `is_on_battery`.
    pub wakeups_per_sec: Option<f64>,
    /// Recent context switches per second. Same wiring/optionality contract.
    pub ctx_switches_per_sec: Option<f64>,
}

/// A single feature's contribution to the composite score.
///
/// The first four fields drive the legacy RSS aggregation path
/// (`PolicyScorer::score`). The three `m_*` mass-assignment fields
/// drive the parallel Dempster-Shafer evidential path. RSS is the
/// default; DS activates only when
/// [`LearnedState::policy_aggregator_mode`] is set to `"ds"`.
///
/// # Dempster-Shafer mass semantics
///
/// Each feature reports a basic probability assignment (BPA) over a
/// two-element frame of discernment Ω = {accept, reject}:
/// - `m_belief`   → mass assigned to {accept} — confident the action is correct.
/// - `m_disbelief`→ mass assigned to {reject} — confident the action is wrong.
/// - `m_uncertain`→ mass assigned to Ω = {accept, reject} — feature is ignorant.
///
/// The three must sum to 1.0 (the BPA axiom in Shafer 1976 §2.2).
/// Features that do not opt into the DS layer leave these at the
/// default `(0.0, 0.0, 1.0)` — full ignorance, no influence on the
/// combined belief. The scorer accepts any normalisation issues as
/// `policy_scorer_ds_high_conflict_fallback_total` bumps and degrades
/// to RSS.
#[derive(Debug, Clone, Copy)]
// Default impl is manual (below) because `m_uncertain` must default
// to 1.0 (full ignorance) rather than the derived 0.0 — a Contribution
// from a feature that does not opt into DS must contribute vacuous
// evidence, not "zero ignorance" which Dempster's rule would
// misinterpret.
pub struct Contribution {
    /// Expected reclaim / risk-reduction (≥0).
    pub benefit: f64,
    /// Expected user disruption (≥0).
    pub cost: f64,
    /// Feature's confidence penalty (≥0).
    pub uncertainty: f64,
    /// If true, the scorer MUST reject regardless of aggregate score.
    pub hard_veto: bool,
    /// Group C Dempster-Shafer (2026-06-06). Mass assigned to {accept}.
    /// Default 0.0 — feature does not opt into DS aggregation.
    pub m_belief: f64,
    /// Group C Dempster-Shafer (2026-06-06). Mass assigned to {reject}.
    /// Default 0.0 — feature does not opt into DS aggregation.
    pub m_disbelief: f64,
    /// Group C Dempster-Shafer (2026-06-06). Mass assigned to Ω
    /// (ignorance / "could be either"). Default 1.0 — full ignorance,
    /// no influence on combined belief.
    pub m_uncertain: f64,
}

impl Default for Contribution {
    fn default() -> Self {
        Self::zero()
    }
}

impl Contribution {
    pub fn zero() -> Self {
        Self {
            benefit: 0.0,
            cost: 0.0,
            uncertainty: 0.0,
            hard_veto: false,
            // Default mass assignment is full ignorance — a feature
            // that does not opt into DS contributes nothing to
            // combined belief, in line with Shafer 1976 §2.2 vacuous
            // BPA semantics.
            m_belief: 0.0,
            m_disbelief: 0.0,
            m_uncertain: 1.0,
        }
    }

    /// Group C (2026-06-06) — build a DS-aware contribution. Mass
    /// arguments are renormalised if they do not sum to 1.0 within
    /// `EPS`; legacy {benefit, cost, uncertainty, hard_veto} fields are
    /// preserved so the same `Contribution` value drives both the RSS
    /// and DS aggregation paths.
    pub fn with_mass(
        benefit: f64,
        cost: f64,
        uncertainty: f64,
        hard_veto: bool,
        m_belief: f64,
        m_disbelief: f64,
        m_uncertain: f64,
    ) -> Self {
        // Defensive renormalisation. A feature that ships `(0.6, 0.4,
        // 0.2)` (sum 1.2) gets divided down to a valid BPA rather than
        // breaking Dempster's rule downstream. Negative masses are
        // clamped to 0 — they are not meaningful in the DS frame.
        let b = m_belief.max(0.0);
        let d = m_disbelief.max(0.0);
        let u = m_uncertain.max(0.0);
        let sum = b + d + u;
        let (mb, md, mu) = if sum > f64::EPSILON {
            (b / sum, d / sum, u / sum)
        } else {
            // Degenerate input → fall back to full ignorance.
            (0.0, 0.0, 1.0)
        };
        Self {
            benefit,
            cost,
            uncertainty,
            hard_veto,
            m_belief: mb,
            m_disbelief: md,
            m_uncertain: mu,
        }
    }
}

/// Implement this trait to contribute to the action policy score.
///
/// Features are invoked in registration order. Their contributions are
/// summed (benefits, costs, uncertainties). Any feature may veto.
pub trait PolicyFeature: Send + Sync {
    /// Stable short name for logging (e.g., "protection", "pressure_benefit").
    fn name(&self) -> &'static str;
    fn contribute(&self, action: &RootAction, ctx: &ActionContext) -> Contribution;
}

/// Result of running the scorer on one action.
#[derive(Debug, Clone)]
pub struct PolicyScore {
    pub action_kind: &'static str,
    pub total_benefit: f64,
    pub total_cost: f64,
    pub total_uncertainty: f64,
    /// Feature name that fired `hard_veto`, if any.
    pub vetoed_by: Option<String>,
    pub accept: bool,
    /// Human-readable reason, for journal/dashboard.
    pub reason: String,
    pub per_feature: Vec<(&'static str, Contribution)>,
    /// Net composite signal: `total_benefit − λ_cost·total_cost −
    /// λ_unc·total_uncertainty`. This is the scalar that `accept` thresholds
    /// against, exposed here so callers (Phase C scorer-override gate in
    /// `decide_actions`) can apply *strength* thresholds without re-deriving
    /// the math. A hard veto leaves `composite` at its raw arithmetic value;
    /// callers must consult `vetoed_by` separately for the absolute reject
    /// signal. Range is unbounded in principle (depends on registered
    /// features), but in the Sprint 11 feature set composite is empirically
    /// confined to roughly [−3.0, +3.5].
    pub composite: f64,
    /// Sprint patch (2026-06-05) — S11. Raw (pre-clamp) RSS uncertainty.
    /// Equals `total_uncertainty` when no clamp fired; strictly greater
    /// otherwise. Operators reading this alongside the new LSE counter
    /// `policy_scorer_uncertainty_saturated_total` can quantify the gap
    /// each scorer call introduced (was the clamp barely engaging or did
    /// it absorb a 3× compositional bomb?). Defaults to 0.0 so any future
    /// serde load of an old `PolicyScore` won't break.
    pub raw_uncertainty: f64,
    /// Group C Dempster-Shafer (2026-06-06). Aggregated belief mass on
    /// {accept} after Dempster's rule of combination. Populated only
    /// when [`PolicyScorer::aggregator_mode`] is `Dempster`; remains
    /// 0.0 under the default RSS path so legacy consumers see no
    /// behaviour change.
    pub ds_belief: f64,
    /// Group C Dempster-Shafer (2026-06-06). Aggregated mass on
    /// {reject}. Same activation contract as `ds_belief`.
    pub ds_disbelief: f64,
    /// Group C Dempster-Shafer (2026-06-06). Residual mass on Ω
    /// (combined ignorance). `ds_belief + ds_disbelief + ds_uncertain
    /// = 1.0` within float tolerance whenever DS aggregation actually
    /// ran (i.e. `ds_fallback_used == false`).
    pub ds_uncertain: f64,
    /// Group C Dempster-Shafer (2026-06-06). Yager conflict mass K
    /// from Dempster's rule. `K > ds_conflict_fallback_threshold`
    /// triggers RSS fallback (and bumps
    /// `policy_scorer_ds_high_conflict_fallback_total`).
    pub ds_conflict: f64,
    /// Group C Dempster-Shafer (2026-06-06). True iff DS aggregation
    /// produced K above the fallback threshold and the scorer used the
    /// RSS path instead. Operators can split metrics by this flag to
    /// see how often DS would have changed the decision.
    pub ds_fallback_used: bool,
}

/// Composable decision surface. Construct via [`PolicyScorer::builder`].
///
/// # Uncertainty composition — Root-Sum-Square (RSS) with saturation
///
/// Per-feature uncertainties are aggregated via RSS (`sqrt(Σ uᵢ²)`), bounded
/// at [`PolicyScorer::uncertainty_saturation`] (default 1.5). Linear sum
/// would assume each feature reports *independent additive* information
/// loss — that is wrong for overlapping feature scopes (e.g. cold-start
/// F5+F6+F7 all reporting `uncertainty=1.0` over the same missing-data
/// event would yield 3.0 and overwhelm any benefit, locking the scorer in
/// permanent OBSERVE). RSS treats contributions as independent
/// Gaussian-like variance sources, so growth is sublinear and the natural
/// ceiling (after saturation) lets high-benefit decisions still clear the
/// threshold.
///
/// Theory:
/// - [Shafer 1976] A Mathematical Theory of Evidence — belief-function
///   composition rules for multi-source evidence.
/// - [Gelman et al. 2013] Bayesian Data Analysis §3 — variance of a sum
///   vs. sum of variances for independent sources.
/// - [Lakshminarayanan 2017] Simple and Scalable Predictive Uncertainty
///   Estimation using Deep Ensembles — RSS aggregation of ensemble
///   uncertainty.
/// Group C Dempster-Shafer (2026-06-06). Aggregation strategy for
/// composing per-feature contributions into a single decision signal.
///
/// `Rss` (the shipped default) sums benefit/cost linearly and combines
/// uncertainty via root-sum-square with a saturation clamp — the
/// Sprint 11 design documented at the top of this module. `Dempster`
/// uses Dempster's rule of combination (Shafer 1976 §3) on the
/// per-feature mass assignments to compute belief / disbelief /
/// ignorance, then falls back to RSS when Yager conflict mass K
/// exceeds [`PolicyScorer::ds_conflict_fallback_threshold`] (default
/// 0.7 — see Yager 1987 §3 on incompatible evidence).
///
/// The mode is selected per-scorer at build time and is intended to
/// be controlled by [`crate::engine::learned_state::LearnedState::policy_aggregator_mode`]
/// so operators can flip strategies via the persisted config without
/// recompiling. The default keeps behaviour byte-equivalent to the
/// pre-Group-C scorer until the operator explicitly opts in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum AggregatorMode {
    /// Root-Sum-Square uncertainty composition (Sprint 11 default).
    #[default]
    Rss,
    /// Dempster's rule of combination over per-feature BPAs, with RSS
    /// fallback on high conflict.
    Dempster,
}


impl AggregatorMode {
    /// Parse the persisted [`LearnedState::policy_aggregator_mode`]
    /// string. Unknown strings default to RSS so a typo in the config
    /// file cannot silently switch on the experimental aggregator.
    pub fn from_persisted(s: &str) -> Self {
        match s {
            "ds" | "dempster" | "dempster_shafer" => Self::Dempster,
            _ => Self::Rss,
        }
    }
}

pub struct PolicyScorer {
    features: Vec<Box<dyn PolicyFeature>>,
    /// Accept threshold: accept iff
    /// `benefit - λ_cost*cost - λ_unc*uncertainty ≥ threshold` AND no veto.
    threshold: f64,
    lambda_cost: f64,
    lambda_unc: f64,
    /// Ceiling for aggregated uncertainty (post-RSS). Default 1.5.
    uncertainty_saturation: f64,
    /// Group C (2026-06-06). Aggregation strategy. Default `Rss`.
    aggregator_mode: AggregatorMode,
    /// Group C (2026-06-06). Yager conflict mass K above which the DS
    /// aggregator falls back to RSS. Default 0.7 — high enough to let
    /// disagreement between two features pass through (K can reach 0.5
    /// with two opposed singleton BPAs), low enough to catch three or
    /// more features in mutual conflict.
    ds_conflict_fallback_threshold: f64,
}

impl PolicyScorer {
    pub fn builder() -> PolicyScorerBuilder {
        PolicyScorerBuilder::default()
    }

    /// Score an action against all registered features.
    pub fn score(&self, action: &RootAction, ctx: &ActionContext) -> PolicyScore {
        let action_kind = action_kind_name(action);
        let mut per_feature: Vec<(&'static str, Contribution)> =
            Vec::with_capacity(self.features.len());
        let mut total_benefit = 0.0f64;
        let mut total_cost = 0.0f64;
        // Accumulate sum of squares; compose via RSS after the loop so N
        // features with uncertainty≈1.0 each can't linearly stack into a
        // compositional bomb. See [Shafer 1976], [Gelman 2013 BDA §3].
        let mut unc_ss = 0.0f64;
        let mut vetoed_by: Option<String> = None;

        for f in &self.features {
            let c = f.contribute(action, ctx);
            total_benefit += c.benefit.max(0.0);
            total_cost += c.cost.max(0.0);
            let u = c.uncertainty.max(0.0);
            unc_ss += u * u;
            if c.hard_veto && vetoed_by.is_none() {
                vetoed_by = Some(f.name().to_string());
            }
            per_feature.push((f.name(), c));
        }

        // Sprint patch (2026-06-05) — S11 telemetry. Compute the raw RSS
        // BEFORE clamping so the Apr-22 NotebookLM concern (3-feature × 1.0
        // bomb invisible) is observable. The LSE counter only bumps when
        // the clamp actually fired — non-zero rate means at least one
        // feature is reporting outsized uncertainty for some action class.
        let raw_uncertainty = unc_ss.sqrt();
        let total_uncertainty = raw_uncertainty.min(self.uncertainty_saturation);
        if raw_uncertainty > self.uncertainty_saturation {
            crate::engine::lse_counters::LSE_COUNTERS.inc_policy_scorer_uncertainty_saturated();
        }

        // Group C (2026-06-06) — Dempster-Shafer evidential aggregation,
        // shadow-mode by default. The classical RSS path above is always
        // computed (cheap, deterministic) so the DS path can fall back
        // without recomputing. DS activation is per-call: any feature
        // that opted into DS (m_belief + m_disbelief > 0) participates;
        // pure-RSS features contribute vacuous BPAs (full ignorance) so
        // the mixed regime is well-defined.
        let (ds_belief, ds_disbelief, ds_uncertain, ds_conflict, ds_fallback_used) =
            if self.aggregator_mode == AggregatorMode::Dempster {
                aggregate_dempster_shafer(&per_feature, self.ds_conflict_fallback_threshold)
            } else {
                // RSS mode — leave DS fields at neutral defaults so
                // legacy consumers see no change.
                (0.0, 0.0, 1.0, 0.0, false)
            };

        // Net composite: when DS aggregation succeeded (no fallback) we
        // use the DS belief-minus-disbelief signal scaled into the same
        // unit range as the legacy benefit-cost score, then add an
        // ignorance penalty proportional to `lambda_unc`. When DS fell
        // back to RSS — or RSS mode is selected — use the legacy net.
        // This keeps the threshold semantics stable across modes:
        // `accept iff composite >= threshold AND no veto`.
        let rss_net =
            total_benefit - self.lambda_cost * total_cost - self.lambda_unc * total_uncertainty;
        let net = if self.aggregator_mode == AggregatorMode::Dempster && !ds_fallback_used {
            // Map DS singleton masses to the same scale as RSS so the
            // shipped threshold (0.0) keeps its calibration. The
            // belief / disbelief delta plays the role of `benefit -
            // cost`; we scale by the RSS magnitude so a DS-confident
            // accept of an action with high RSS benefit stays high,
            // while a DS-confident reject of the same action goes
            // negative. The ignorance term inherits `lambda_unc`.
            let ds_signal = (ds_belief - ds_disbelief)
                * (total_benefit + self.lambda_cost * total_cost).max(1.0);
            ds_signal - self.lambda_unc * ds_uncertain
        } else {
            rss_net
        };
        let accept = vetoed_by.is_none() && net >= self.threshold;

        let reason = build_reason(
            action_kind,
            accept,
            net,
            self.threshold,
            total_benefit,
            total_cost,
            total_uncertainty,
            &vetoed_by,
            &per_feature,
        );

        PolicyScore {
            action_kind,
            total_benefit,
            total_cost,
            total_uncertainty,
            vetoed_by,
            accept,
            reason,
            per_feature,
            composite: net,
            raw_uncertainty,
            ds_belief,
            ds_disbelief,
            ds_uncertain,
            ds_conflict,
            ds_fallback_used,
        }
    }
}

/// Group C (2026-06-06) — Dempster's rule of combination for the
/// two-element frame Ω = {accept, reject}.
///
/// Returns `(belief, disbelief, uncertain, conflict_K, fell_back)`.
/// `fell_back == true` means K exceeded `conflict_fallback_threshold`
/// — the caller treats this scorer call as RSS-mode for the net
/// computation. The mass triple is still returned so audits can
/// observe what DS *would* have said had it been trusted.
///
/// # Algorithm
///
/// For each pair of features the orthogonal sum is:
///
/// ```text
/// m₁₂({A}) = (m₁(A)·m₂(A) + m₁(A)·m₂(Ω) + m₁(Ω)·m₂(A)) / (1 - K)
/// m₁₂({R}) = (m₁(R)·m₂(R) + m₁(R)·m₂(Ω) + m₁(Ω)·m₂(R)) / (1 - K)
/// m₁₂(Ω)   = m₁(Ω)·m₂(Ω) / (1 - K)
/// K        = m₁(A)·m₂(R) + m₁(R)·m₂(A)
/// ```
///
/// We accumulate over all features. K accumulates additively as the
/// running conflict mass *of the combination so far*; whenever a step
/// would divide by < `EPS` (total conflict, undefined Dempster's
/// rule), or the cumulative K exceeds the threshold, we abort and
/// return the fallback signal.
///
/// # References
/// - [Shafer 1976] A Mathematical Theory of Evidence §3 — orthogonal sum.
/// - [Yager 1987] On the Dempster-Shafer framework and new combination
///   rules — conflict mass interpretation.
/// - [Dezert 2002] An introduction to the DSm theory — discussion of
///   counter-intuitive results under high conflict (Zadeh counter-example).
fn aggregate_dempster_shafer(
    per_feature: &[(&'static str, Contribution)],
    conflict_fallback_threshold: f64,
) -> (f64, f64, f64, f64, bool) {
    // Start from the vacuous BPA: total ignorance, zero conflict.
    let mut m_a = 0.0f64; // mass on {accept}
    let mut m_r = 0.0f64; // mass on {reject}
    let mut m_o = 1.0f64; // mass on Ω
    let mut cumulative_k = 0.0f64;

    for (_, c) in per_feature {
        // Sanitise the feature's BPA: clamp negatives, renormalise if
        // the sum drifted. A feature that ships a degenerate BPA
        // contributes vacuous evidence — never poisons the combined
        // belief.
        let b = c.m_belief.max(0.0);
        let d = c.m_disbelief.max(0.0);
        let u = c.m_uncertain.max(0.0);
        let sum = b + d + u;
        let (fb, fd, fu) = if sum > f64::EPSILON {
            (b / sum, d / sum, u / sum)
        } else {
            (0.0, 0.0, 1.0)
        };

        // Conflict mass of this single combination step.
        let step_k = m_a * fd + m_r * fb;
        let norm = 1.0 - step_k;

        if norm < f64::EPSILON {
            // Total conflict — Dempster's rule is undefined. Bump the
            // fallback counter and return the pre-step state with a
            // saturated K signal so the caller treats this as RSS.
            crate::engine::lse_counters::LSE_COUNTERS.inc_policy_scorer_ds_high_conflict_fallback();
            return (m_a, m_r, m_o, 1.0, true);
        }

        let new_a = (m_a * fb + m_a * fu + m_o * fb) / norm;
        let new_r = (m_r * fd + m_r * fu + m_o * fd) / norm;
        let new_o = m_o * fu / norm;

        m_a = new_a;
        m_r = new_r;
        m_o = new_o;
        cumulative_k += step_k;
    }

    if cumulative_k > conflict_fallback_threshold {
        crate::engine::lse_counters::LSE_COUNTERS.inc_policy_scorer_ds_high_conflict_fallback();
        return (m_a, m_r, m_o, cumulative_k, true);
    }

    (m_a, m_r, m_o, cumulative_k, false)
}

/// Builder for [`PolicyScorer`].
#[derive(Default)]
pub struct PolicyScorerBuilder {
    features: Vec<Box<dyn PolicyFeature>>,
    threshold: Option<f64>,
    lambda_cost: Option<f64>,
    lambda_unc: Option<f64>,
    uncertainty_saturation: Option<f64>,
    /// Group C (2026-06-06). DS aggregator mode. Default `Rss`.
    aggregator_mode: Option<AggregatorMode>,
    /// Group C (2026-06-06). DS conflict fallback threshold. Default 0.7.
    ds_conflict_fallback_threshold: Option<f64>,
}

impl PolicyScorerBuilder {
    /// Register a feature. Order is preserved; use for deterministic logging.
    pub fn feature<F: PolicyFeature + 'static>(mut self, f: F) -> Self {
        self.features.push(Box::new(f));
        self
    }

    pub fn threshold(mut self, t: f64) -> Self {
        self.threshold = Some(t);
        self
    }

    pub fn lambda_cost(mut self, l: f64) -> Self {
        self.lambda_cost = Some(l);
        self
    }

    pub fn lambda_unc(mut self, l: f64) -> Self {
        self.lambda_unc = Some(l);
        self
    }

    /// Ceiling for aggregated (post-RSS) uncertainty. Default 1.5.
    ///
    /// Even with many features each reporting `uncertainty=1.0`, the
    /// aggregate cannot exceed this cap — keeping the epistemic penalty
    /// bounded so high-benefit decisions can still clear the threshold.
    pub fn uncertainty_saturation(mut self, f: f64) -> Self {
        self.uncertainty_saturation = Some(f);
        self
    }

    /// Group C (2026-06-06) — select the aggregation strategy. Default
    /// is RSS; pass `AggregatorMode::Dempster` to activate the
    /// evidential layer.
    pub fn aggregator_mode(mut self, m: AggregatorMode) -> Self {
        self.aggregator_mode = Some(m);
        self
    }

    /// Group C (2026-06-06) — Yager conflict mass K above which DS
    /// falls back to RSS for a single scorer call. Default 0.7.
    pub fn ds_conflict_fallback_threshold(mut self, k: f64) -> Self {
        self.ds_conflict_fallback_threshold = Some(k);
        self
    }

    pub fn build(self) -> PolicyScorer {
        PolicyScorer {
            features: self.features,
            threshold: self.threshold.unwrap_or(0.0),
            lambda_cost: self.lambda_cost.unwrap_or(1.0),
            lambda_unc: self.lambda_unc.unwrap_or(0.5),
            uncertainty_saturation: self.uncertainty_saturation.unwrap_or(1.5),
            aggregator_mode: self.aggregator_mode.unwrap_or(AggregatorMode::Rss),
            ds_conflict_fallback_threshold: self.ds_conflict_fallback_threshold.unwrap_or(0.7),
        }
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn action_kind_name(action: &RootAction) -> &'static str {
    match action {
        RootAction::BoostProcess { .. } => "BoostProcess",
        RootAction::ThrottleProcess { .. } => "ThrottleProcess",
        RootAction::FreezeProcess { .. } => "FreezeProcess",
        RootAction::UnfreezeProcess { .. } => "UnfreezeProcess",
        RootAction::SetSysctl(_) => "SetSysctl",
        RootAction::SetMemorystatus { .. } => "SetMemorystatus",
        RootAction::ToggleSpotlight { .. } => "ToggleSpotlight",
        RootAction::QuarantineDaemon { .. } => "QuarantineDaemon",
        RootAction::SetThreadQoS { .. } => "SetThreadQoS",
    }
}

fn is_freeze_or_throttle(action: &RootAction) -> bool {
    matches!(
        action,
        RootAction::FreezeProcess { .. } | RootAction::ThrottleProcess { .. }
    )
}

#[allow(clippy::too_many_arguments)] // dedicated helper; fields are distinct scalars.
fn build_reason(
    action_kind: &'static str,
    accept: bool,
    net: f64,
    threshold: f64,
    total_benefit: f64,
    total_cost: f64,
    total_uncertainty: f64,
    vetoed_by: &Option<String>,
    per_feature: &[(&'static str, Contribution)],
) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(96);
    let verdict = if accept { "accept" } else { "reject" };
    let _ = write!(s, "{action_kind}:{verdict}");
    if let Some(v) = vetoed_by {
        let _ = write!(s, " veto={v}");
    }
    let _ = write!(
        s,
        " net={:.3} (b={:.3} c={:.3} u={:.3} thr={:.3})",
        net, total_benefit, total_cost, total_uncertainty, threshold
    );
    // Always list every feature (even zero contributors) so the reason string
    // deterministically mentions the full registered feature set — callers can
    // audit "silent" features as easily as active ones.
    if !per_feature.is_empty() {
        s.push_str(" [");
        for (i, (name, c)) in per_feature.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            let _ = write!(
                s,
                "{name}:b={:.2},c={:.2},u={:.2}{}",
                c.benefit,
                c.cost,
                c.uncertainty,
                if c.hard_veto { ",veto" } else { "" }
            );
        }
        s.push(']');
    }
    s
}

// -----------------------------------------------------------------------------
// Starter features
// -----------------------------------------------------------------------------

/// F1 — Protection. Veto-only feature: mirrors `classify_protection` semantics.
///
/// - `Unconditional` → veto always.
/// - `ConditionalForeground` + freeze/throttle + in foreground family → veto.
/// - Otherwise contributes zero.
pub struct ProtectionFeature;

impl PolicyFeature for ProtectionFeature {
    fn name(&self) -> &'static str {
        "protection"
    }

    fn contribute(&self, action: &RootAction, ctx: &ActionContext) -> Contribution {
        match ctx.protection_level {
            // Group C (2026-06-06) — under DS the protection feature is
            // the canonical singleton-reject BPA: m_disbelief=1.0 says
            // "this feature is certain the action is wrong" without
            // touching the legacy RSS scalars.
            ProtectionLevel::Unconditional => Contribution {
                benefit: 0.0,
                cost: 0.0,
                uncertainty: 0.0,
                hard_veto: true,
                m_belief: 0.0,
                m_disbelief: 1.0,
                m_uncertain: 0.0,
            },
            ProtectionLevel::ConditionalForeground => {
                if is_freeze_or_throttle(action) && ctx.is_foreground_family {
                    Contribution {
                        benefit: 0.0,
                        cost: 0.0,
                        uncertainty: 0.0,
                        hard_veto: true,
                        m_belief: 0.0,
                        m_disbelief: 1.0,
                        m_uncertain: 0.0,
                    }
                } else {
                    Contribution::zero()
                }
            }
            ProtectionLevel::Unprotected => Contribution::zero(),
        }
    }
}

/// F2 — PressureBenefit. Expected reclaim grows with pressure, thrashing, and
/// predicted OOM. Boosts receive a small positive baseline (cheap insurance).
pub struct PressureBenefitFeature;

impl PolicyFeature for PressureBenefitFeature {
    fn name(&self) -> &'static str {
        "pressure_benefit"
    }

    fn contribute(&self, action: &RootAction, ctx: &ActionContext) -> Contribution {
        match action {
            RootAction::FreezeProcess { .. } | RootAction::ThrottleProcess { .. } => {
                let mut benefit = ctx.pressure * 1.0;
                if ctx.thrashing_score > 5_000.0 {
                    benefit += 0.5;
                }
                if ctx.p_oom_30s.unwrap_or(0.0) > 0.30 {
                    benefit += 1.0;
                }
                Contribution {
                    benefit,
                    cost: 0.0,
                    uncertainty: 0.0,
                    hard_veto: false,
                    ..Contribution::zero()
                }
            }
            RootAction::BoostProcess { .. } => Contribution {
                benefit: 0.1,
                cost: 0.0,
                uncertainty: 0.0,
                hard_veto: false,
                ..Contribution::zero()
            },
            _ => Contribution::zero(),
        }
    }
}

/// F3 — UserDisruptionCost. Charges cost for freeze/throttle during
/// interactive user phases. Reproduces current sleep-assertion bypass
/// semantics: cost contribution is zero when pressure/swap/thrashing are
/// high enough that the existing gate tower would bypass the assertion.
pub struct UserDisruptionCostFeature;

impl PolicyFeature for UserDisruptionCostFeature {
    fn name(&self) -> &'static str {
        "user_cost"
    }

    fn contribute(&self, action: &RootAction, ctx: &ActionContext) -> Contribution {
        match action {
            RootAction::FreezeProcess { .. } | RootAction::ThrottleProcess { .. } => {
                let mut cost = 0.0f64;
                let mut uncertainty = 0.0f64;
                if ctx.call_in_progress {
                    cost += 2.0;
                }
                // Sleep-assertion bypass: match existing freeze_protected/Gate C
                // logic — the assertion stops contributing cost when pressure,
                // swap, or thrashing cross their danger thresholds.
                let bypass_sleep =
                    ctx.pressure >= 0.70 || ctx.swap_gb >= 4.0 || ctx.thrashing_score >= 10_000.0;
                if ctx.has_sleep_assertion && !bypass_sleep {
                    cost += 1.0;
                }
                if ctx.is_recently_active {
                    cost += 0.5;
                }
                if ctx.idle_secs < 5.0 {
                    uncertainty += 0.3;
                }
                Contribution {
                    benefit: 0.0,
                    cost,
                    uncertainty,
                    hard_veto: false,
                    ..Contribution::zero()
                }
            }
            RootAction::BoostProcess { .. } => Contribution::zero(),
            _ => Contribution::zero(),
        }
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::audit_types::DecisionReason;

    fn base_ctx() -> ActionContext {
        ActionContext {
            pressure: 0.40,
            swap_gb: 0.5,
            thrashing_score: 0.0,
            p_oom_30s: None,
            p_jank_60s: None,
            has_sleep_assertion: false,
            call_in_progress: false,
            idle_secs: 60.0,
            foreground_pid: None,
            is_foreground_family: false,
            is_recently_active: false,
            thermal_emergency: false,
            interrupt_phase: 0,
            protection_level: ProtectionLevel::Unprotected,
            hot_page_fraction: None,
            wss_mb: None,
            sensor_age_ms: None,
            epistemic_uncertainty: 0.0,
            is_on_battery: None,
            wakeups_per_sec: None,
            ctx_switches_per_sec: None,
        }
    }

    fn freeze(pid: u32) -> RootAction {
        RootAction::freeze(
            pid,
            format!("p{pid}"),
            "test",
            DecisionReason::PressureContext,
        )
    }

    fn throttle(pid: u32) -> RootAction {
        RootAction::throttle(
            pid,
            format!("p{pid}"),
            false,
            "test",
            DecisionReason::PressureContext,
        )
    }

    fn boost(pid: u32) -> RootAction {
        RootAction::BoostProcess {
            pid,
            name: format!("p{pid}"),
            reason: "test".into(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        }
    }

    /// Sprint patch (2026-06-05) — S11 telemetry: a feature reporting
    /// outsized uncertainty causes the LSE counter to bump and the
    /// `raw_uncertainty` field to exceed the saturated total.
    struct FixedUncFeature(f64);
    impl PolicyFeature for FixedUncFeature {
        fn name(&self) -> &'static str {
            "FixedUnc"
        }
        fn contribute(&self, _action: &RootAction, _ctx: &ActionContext) -> Contribution {
            Contribution {
                benefit: 0.0,
                cost: 0.0,
                uncertainty: self.0,
                hard_veto: false,
                ..Contribution::zero()
            }
        }
    }

    #[test]
    fn saturated_clamp_bumps_counter_and_preserves_raw() {
        use std::sync::atomic::Ordering;
        let pre = crate::engine::lse_counters::LSE_COUNTERS
            .policy_scorer_uncertainty_saturated_total
            .load(Ordering::Relaxed);
        // Uncertainty 2.0 > saturation 1.5 → clamp fires + counter bumps.
        let scorer = PolicyScorer::builder()
            .feature(FixedUncFeature(2.0))
            .uncertainty_saturation(1.5)
            .build();
        let score = scorer.score(&freeze(1), &base_ctx());
        let post = crate::engine::lse_counters::LSE_COUNTERS
            .policy_scorer_uncertainty_saturated_total
            .load(Ordering::Relaxed);
        assert!(post > pre, "saturated clamp must bump the counter");
        assert!(
            score.raw_uncertainty > score.total_uncertainty,
            "raw must exceed clamped: raw={} clamp={}",
            score.raw_uncertainty,
            score.total_uncertainty
        );
        assert!(
            (score.total_uncertainty - 1.5).abs() < 1e-9,
            "clamp == saturation ceiling"
        );
    }

    #[test]
    fn unsaturated_does_not_bump_counter() {
        use std::sync::atomic::Ordering;
        let pre = crate::engine::lse_counters::LSE_COUNTERS
            .policy_scorer_uncertainty_saturated_total
            .load(Ordering::Relaxed);
        let scorer = PolicyScorer::builder()
            .feature(FixedUncFeature(0.5))
            .uncertainty_saturation(1.5)
            .build();
        let score = scorer.score(&freeze(1), &base_ctx());
        let post = crate::engine::lse_counters::LSE_COUNTERS
            .policy_scorer_uncertainty_saturated_total
            .load(Ordering::Relaxed);
        assert_eq!(pre, post, "unsaturated must NOT bump counter");
        assert!(
            (score.raw_uncertainty - score.total_uncertainty).abs() < 1e-9,
            "raw == clamped when below saturation"
        );
    }

    #[test]
    fn zero_features_accepts_by_default() {
        let scorer = PolicyScorer::builder().build();
        let ctx = base_ctx();
        let s = scorer.score(&freeze(1), &ctx);
        assert!(s.accept, "empty scorer should accept (net=0 ≥ threshold=0)");
        assert_eq!(s.total_benefit, 0.0);
        assert_eq!(s.total_cost, 0.0);
        assert!(s.vetoed_by.is_none());
    }

    #[test]
    fn protection_unconditional_vetoes_freeze() {
        let scorer = PolicyScorer::builder()
            .feature(ProtectionFeature)
            .feature(PressureBenefitFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.protection_level = ProtectionLevel::Unconditional;
        ctx.pressure = 0.99;
        let s = scorer.score(&freeze(1), &ctx);
        assert!(!s.accept);
        assert_eq!(s.vetoed_by.as_deref(), Some("protection"));
    }

    #[test]
    fn protection_conditional_foreground_vetoes_only_when_in_family() {
        let scorer = PolicyScorer::builder()
            .feature(ProtectionFeature)
            .feature(PressureBenefitFeature)
            .build();
        // In foreground family → veto.
        let mut ctx = base_ctx();
        ctx.protection_level = ProtectionLevel::ConditionalForeground;
        ctx.is_foreground_family = true;
        ctx.pressure = 0.80;
        let s = scorer.score(&freeze(1), &ctx);
        assert!(!s.accept);
        assert_eq!(s.vetoed_by.as_deref(), Some("protection"));

        // Not in foreground family → no veto, high pressure → accept.
        ctx.is_foreground_family = false;
        let s = scorer.score(&freeze(1), &ctx);
        assert!(s.accept, "reason={}", s.reason);
        assert!(s.vetoed_by.is_none());
    }

    #[test]
    fn pressure_benefit_scales_with_thrashing() {
        let scorer = PolicyScorer::builder()
            .feature(PressureBenefitFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.pressure = 0.50;
        let low = scorer.score(&freeze(1), &ctx);
        ctx.thrashing_score = 6_000.0;
        let hi = scorer.score(&freeze(1), &ctx);
        assert!(
            hi.total_benefit > low.total_benefit,
            "thrashing should raise benefit ({} !> {})",
            hi.total_benefit,
            low.total_benefit
        );
        // +0.5 from thrashing bonus.
        assert!((hi.total_benefit - low.total_benefit - 0.5).abs() < 1e-9);
    }

    #[test]
    fn pressure_benefit_fires_on_high_p_oom_30s() {
        let scorer = PolicyScorer::builder()
            .feature(PressureBenefitFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.pressure = 0.10;
        ctx.p_oom_30s = Some(0.50);
        let s = scorer.score(&freeze(1), &ctx);
        // 0.10 + 1.0 = 1.10.
        assert!(
            (s.total_benefit - 1.10).abs() < 1e-9,
            "benefit={}",
            s.total_benefit
        );
    }

    #[test]
    fn user_cost_blocks_freeze_under_sleep_assertion_at_low_pressure() {
        let scorer = PolicyScorer::builder()
            .feature(PressureBenefitFeature)
            .feature(UserDisruptionCostFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.pressure = 0.30; // benefit = 0.30
        ctx.has_sleep_assertion = true; // cost = 1.0
        let s = scorer.score(&freeze(1), &ctx);
        // net = 0.30 - 1.0*1.0 - 0.5*0 = -0.70 < 0.0 threshold → reject.
        assert!(!s.accept, "reason={}", s.reason);
        assert!(s.vetoed_by.is_none(), "cost-based reject, not veto");
        assert!(s.total_cost >= 1.0);
    }

    #[test]
    fn user_cost_bypassed_at_high_pressure() {
        let scorer = PolicyScorer::builder()
            .feature(UserDisruptionCostFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.pressure = 0.75; // crosses 0.70 threshold → bypass
        ctx.has_sleep_assertion = true;
        let s = scorer.score(&freeze(1), &ctx);
        // Sleep-assertion contribution suppressed; no other cost triggers.
        assert_eq!(s.total_cost, 0.0, "sleep-assertion should be bypassed");
    }

    #[test]
    fn user_cost_bypassed_at_high_thrashing() {
        let scorer = PolicyScorer::builder()
            .feature(UserDisruptionCostFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.pressure = 0.30;
        ctx.swap_gb = 0.5;
        ctx.thrashing_score = 15_000.0; // crosses 10k → bypass
        ctx.has_sleep_assertion = true;
        let s = scorer.score(&freeze(1), &ctx);
        assert_eq!(s.total_cost, 0.0, "thrashing>10k should bypass sleep cost");
    }

    #[test]
    fn call_in_progress_always_vetoes_via_cost_overflow() {
        // "veto via cost overflow": cost is so high benefit can't offset.
        let scorer = PolicyScorer::builder()
            .feature(PressureBenefitFeature)
            .feature(UserDisruptionCostFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.pressure = 0.99; // max benefit ~0.99
        ctx.call_in_progress = true; // cost += 2.0
        let s = scorer.score(&freeze(1), &ctx);
        // net = 0.99 - 2.0 = -1.01 → reject.
        assert!(!s.accept, "reason={}", s.reason);
        assert!(s.total_cost >= 2.0);
        assert!(s.vetoed_by.is_none());
    }

    #[test]
    fn boost_always_accepted_even_under_call() {
        let scorer = PolicyScorer::builder()
            .feature(ProtectionFeature)
            .feature(PressureBenefitFeature)
            .feature(UserDisruptionCostFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.call_in_progress = true;
        ctx.has_sleep_assertion = true;
        ctx.is_recently_active = true;
        let s = scorer.score(&boost(1), &ctx);
        assert!(s.accept, "reason={}", s.reason);
        assert_eq!(s.total_cost, 0.0);
        assert!(s.total_benefit >= 0.1);
    }

    #[test]
    fn score_is_deterministic() {
        let scorer = PolicyScorer::builder()
            .feature(ProtectionFeature)
            .feature(PressureBenefitFeature)
            .feature(UserDisruptionCostFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.pressure = 0.55;
        ctx.has_sleep_assertion = true;
        ctx.thrashing_score = 4_000.0;
        let first = scorer.score(&freeze(42), &ctx);
        for _ in 0..1_000 {
            let s = scorer.score(&freeze(42), &ctx);
            assert_eq!(s.accept, first.accept);
            assert!((s.total_benefit - first.total_benefit).abs() < 1e-12);
            assert!((s.total_cost - first.total_cost).abs() < 1e-12);
            assert!((s.total_uncertainty - first.total_uncertainty).abs() < 1e-12);
            assert_eq!(s.reason, first.reason);
        }
    }

    #[test]
    fn per_feature_contribs_sum_to_totals() {
        let scorer = PolicyScorer::builder()
            .feature(ProtectionFeature)
            .feature(PressureBenefitFeature)
            .feature(UserDisruptionCostFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.pressure = 0.80;
        ctx.thrashing_score = 6_000.0;
        ctx.p_oom_30s = Some(0.40);
        ctx.has_sleep_assertion = true;
        ctx.idle_secs = 2.0;
        let s = scorer.score(&throttle(7), &ctx);
        let sum_b: f64 = s.per_feature.iter().map(|(_, c)| c.benefit.max(0.0)).sum();
        let sum_c: f64 = s.per_feature.iter().map(|(_, c)| c.cost.max(0.0)).sum();
        // RSS (Step 1): total_uncertainty = sqrt(Σ uᵢ²), not Σ uᵢ.
        let rss_u: f64 = s
            .per_feature
            .iter()
            .map(|(_, c)| {
                let u = c.uncertainty.max(0.0);
                u * u
            })
            .sum::<f64>()
            .sqrt()
            .min(1.5);
        assert!((sum_b - s.total_benefit).abs() < 1e-9);
        assert!((sum_c - s.total_cost).abs() < 1e-9);
        assert!((rss_u - s.total_uncertainty).abs() < 1e-9);
    }

    // -------------------------------------------------------------------------
    // RSS uncertainty composition (Step 1) — tests
    // -------------------------------------------------------------------------

    /// Test-only feature emitting a fixed uncertainty contribution.
    struct FixedUncertaintyFeature {
        name: &'static str,
        uncertainty: f64,
    }

    impl PolicyFeature for FixedUncertaintyFeature {
        fn name(&self) -> &'static str {
            self.name
        }
        fn contribute(&self, _action: &RootAction, _ctx: &ActionContext) -> Contribution {
            Contribution {
                benefit: 0.0,
                cost: 0.0,
                uncertainty: self.uncertainty,
                hard_veto: false,
                ..Contribution::zero()
            }
        }
    }

    #[test]
    fn rss_caps_aggregate_uncertainty_below_saturation() {
        // Three features each at uncertainty=1.0 → sqrt(3) ≈ 1.732, capped at 1.5.
        let scorer = PolicyScorer::builder()
            .feature(FixedUncertaintyFeature {
                name: "f1",
                uncertainty: 1.0,
            })
            .feature(FixedUncertaintyFeature {
                name: "f2",
                uncertainty: 1.0,
            })
            .feature(FixedUncertaintyFeature {
                name: "f3",
                uncertainty: 1.0,
            })
            .build();
        let s = scorer.score(&freeze(1), &base_ctx());
        let expected = 3.0f64.sqrt().min(1.5);
        assert!((s.total_uncertainty - expected).abs() < 1e-9);
        assert!(
            (s.total_uncertainty - 1.5).abs() < 1e-9,
            "expected saturated"
        );
    }

    #[test]
    fn rss_single_feature_preserves_input() {
        let scorer = PolicyScorer::builder()
            .feature(FixedUncertaintyFeature {
                name: "f1",
                uncertainty: 0.5,
            })
            .build();
        let s = scorer.score(&freeze(1), &base_ctx());
        assert!((s.total_uncertainty - 0.5).abs() < 1e-9);
    }

    #[test]
    fn rss_below_saturation_matches_sqrt_ss() {
        // Two features at 0.6 each → sqrt(0.72) ≈ 0.8485, NOT 1.2 (linear).
        let scorer = PolicyScorer::builder()
            .feature(FixedUncertaintyFeature {
                name: "f1",
                uncertainty: 0.6,
            })
            .feature(FixedUncertaintyFeature {
                name: "f2",
                uncertainty: 0.6,
            })
            .build();
        let s = scorer.score(&freeze(1), &base_ctx());
        let expected = (0.6f64 * 0.6 + 0.6 * 0.6).sqrt();
        assert!(
            (s.total_uncertainty - expected).abs() < 1e-9,
            "got {}, expected {}",
            s.total_uncertainty,
            expected
        );
        // Guard-rail: ensure we did NOT revert to linear sum.
        assert!(
            (s.total_uncertainty - 1.2).abs() > 1e-3,
            "linear-sum regression detected"
        );
    }

    #[test]
    fn saturation_configurable_via_builder() {
        // Two features at 1.0 each → sqrt(2) ≈ 1.414, but custom cap = 0.5.
        let scorer = PolicyScorer::builder()
            .feature(FixedUncertaintyFeature {
                name: "f1",
                uncertainty: 1.0,
            })
            .feature(FixedUncertaintyFeature {
                name: "f2",
                uncertainty: 1.0,
            })
            .uncertainty_saturation(0.5)
            .build();
        let s = scorer.score(&freeze(1), &base_ctx());
        assert!((s.total_uncertainty - 0.5).abs() < 1e-9);
    }

    #[test]
    fn default_saturation_is_1_5() {
        let built = PolicyScorer::builder().build();
        assert!((built.uncertainty_saturation - 1.5).abs() < 1e-12);
    }

    #[test]
    fn zero_features_zero_uncertainty_regardless_of_saturation() {
        for &sat in &[0.0_f64, 0.5, 1.5, 99.0] {
            let scorer = PolicyScorer::builder().uncertainty_saturation(sat).build();
            let s = scorer.score(&freeze(1), &base_ctx());
            assert_eq!(s.total_uncertainty, 0.0, "sat={}", sat);
        }
    }

    #[test]
    fn rss_preserves_existing_score_test() {
        // Re-run a scenario that asserts total_uncertainty via a single source
        // (idle_secs<5 in UserDisruptionCostFeature → 0.3). RSS of a single
        // 0.3 source is still 0.3 — so existing behaviour is preserved.
        let scorer = PolicyScorer::builder()
            .feature(UserDisruptionCostFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.idle_secs = 2.0;
        let s = scorer.score(&freeze(1), &ctx);
        assert!(
            (s.total_uncertainty - 0.3).abs() < 1e-9,
            "RSS composition (Step 1): single 0.3 source → 0.3, got {}",
            s.total_uncertainty
        );
    }

    #[test]
    fn reason_string_contains_all_contributors() {
        let scorer = PolicyScorer::builder()
            .feature(ProtectionFeature)
            .feature(PressureBenefitFeature)
            .feature(UserDisruptionCostFeature)
            .build();
        let mut ctx = base_ctx();
        ctx.pressure = 0.50;
        let s = scorer.score(&freeze(9), &ctx);
        assert!(s.reason.contains("protection"), "reason={}", s.reason);
        assert!(s.reason.contains("pressure_benefit"), "reason={}", s.reason);
        assert!(s.reason.contains("user_cost"), "reason={}", s.reason);
        assert!(s.reason.contains("FreezeProcess"), "reason={}", s.reason);
    }

    #[test]
    fn builder_composes_correctly() {
        let ctx = {
            let mut c = base_ctx();
            c.pressure = 0.60;
            c
        };
        let built = PolicyScorer::builder()
            .feature(PressureBenefitFeature)
            .threshold(0.5)
            .lambda_cost(2.0)
            .lambda_unc(0.25)
            .build();
        assert!((built.threshold - 0.5).abs() < 1e-12);
        assert!((built.lambda_cost - 2.0).abs() < 1e-12);
        assert!((built.lambda_unc - 0.25).abs() < 1e-12);
        // Functional check: benefit 0.60 passes threshold 0.5 → accept.
        let s = built.score(&freeze(1), &ctx);
        assert!(s.accept, "reason={}", s.reason);
    }

    #[test]
    fn threshold_negative_is_rejected_positive_accepted() {
        let scorer = PolicyScorer::builder()
            .feature(PressureBenefitFeature)
            .feature(UserDisruptionCostFeature)
            .threshold(0.0)
            .build();
        // Negative net: low pressure, sleep assertion costs 1.0.
        let mut ctx = base_ctx();
        ctx.pressure = 0.20;
        ctx.has_sleep_assertion = true;
        let s_neg = scorer.score(&freeze(1), &ctx);
        assert!(!s_neg.accept, "net should be negative");

        // Positive net: high pressure (bypasses sleep cost).
        ctx.pressure = 0.85;
        let s_pos = scorer.score(&freeze(1), &ctx);
        assert!(
            s_pos.accept,
            "net should be positive, reason={}",
            s_pos.reason
        );
    }
}
