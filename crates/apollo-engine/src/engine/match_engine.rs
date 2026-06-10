//! MatchEngine — 3-tier confidence-weighted name matching.
//!
//! Tier 0 FamilyRoot   → conf 1.00  [Brave/Chrome/… substring, Permanent Scar #1 / 26eac06]
//! Tier 1 Exact        → conf 1.00  [HashSet O(1)]
//! Tier 2 WordBoundary → conf 0.85  [≥3 chars, lifted from matches_dev_runtime / 5843bc0]
//! Tier 3 Substring    → conf 0.30  [≥3 chars, DEGRADED — below MIN_FREEZE_CONFIDENCE 0.35]
//!
//! Peer-consult action items (NotebookLM 2026-05-30):
//!  (1) Substring conf lowered 0.40 → 0.30 (≥5pp under MIN_FREEZE_CONFIDENCE);
//!      <3 char patterns rejected from BOTH WordBoundary and Substring tiers.
//!  (2) match_confidence routes through `IdentityUncertaintyFeature` →
//!      PolicyScorer's RSS uncertainty channel (65f310d), NEVER multiplies benefit.
//!  (3) FAMILY_ROOT_PATTERNS short-circuit retains unconditional Chromium-family
//!      substring protection at conf 1.00 (deliberate hierarchy carve-out).
//!
//! Papers: Saltzer & Kaashoek 2009 §3.3 Complete Mediation; Pearl 2009 §1.4
//! evidence weighting; Aho & Ullman 1972 §3.2 lexical boundaries; Shafer 1976 /
//! Gelman BDA 2013 §3 RSS composition (65f310d).

use std::collections::HashMap;
use std::collections::HashSet;

use crate::engine::action_policy::{ActionContext, Contribution, PolicyFeature};
use crate::engine::outcome_tracker::PatternWeight;
use crate::engine::types::RootAction;

pub const TIER_EXACT_CONF: f64 = 1.00;
pub const TIER_WORDBOUND_CONF: f64 = 0.85;
/// Substring is *weak* evidence. 0.30 < MIN_FREEZE_CONFIDENCE (0.35) by design
/// (≥5pp margin) so a lone substring match never independently clears the
/// freeze gate. NotebookLM 2026-05-30 action item #1.
pub const TIER_SUBSTRING_CONF: f64 = 0.30;
/// Minimum pattern length for BOTH word-boundary and substring tiers.
/// `< 3` patterns are rejected outright (e.g., "go" cannot reach "Categories").
pub const MIN_PATTERN_LEN: usize = 3;

/// Family-root patterns: retain unconditional substring semantics at conf
/// 1.00. Honors Permanent Scar #1 (commit 26eac06 — Chromium SIGSTOP IPC
/// contract). Hierarchy carve-out for known macOS process families where the
/// root name is a stable substring of every helper renderer/GPU/utility.
const FAMILY_ROOT_PATTERNS: &[&str] = &[
    "Brave",
    "Google Chrome",
    "Chromium",
    "Electron",
    "Safari",
    "Firefox",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchTier {
    FamilyRoot,
    Exact,
    WordBoundary,
    Substring,
    None,
}

#[derive(Debug, Clone, Copy)]
pub struct MatchResult {
    pub tier: MatchTier,
    pub confidence: f64,
}

impl MatchResult {
    pub const NONE: Self = Self {
        tier: MatchTier::None,
        confidence: 0.0,
    };
    pub fn matched(self) -> bool {
        self.tier != MatchTier::None
    }
}

/// Word-boundary aware case-insensitive contains.
/// Lifted from `safety.rs::matches_dev_runtime` inner loop (commit 5843bc0).
/// Both args MUST already be lowercased by the caller.
pub fn word_boundary_contains(haystack_lower: &str, needle_lower: &str) -> bool {
    if needle_lower.is_empty() || needle_lower.len() > haystack_lower.len() {
        return false;
    }
    let bytes = haystack_lower.as_bytes();
    let mut start = 0usize;
    while let Some(rel) = haystack_lower[start..].find(needle_lower) {
        let abs = start + rel;
        let end = abs + needle_lower.len();
        let lhs_ok = abs == 0 || !(bytes[abs - 1] as char).is_ascii_alphanumeric();
        let rhs_ok = end == bytes.len() || !(bytes[end] as char).is_ascii_alphanumeric();
        if lhs_ok && rhs_ok {
            return true;
        }
        start = abs + 1;
        if start >= haystack_lower.len() {
            break;
        }
    }
    false
}

/// OnceLock AhoCorasick over FAMILY_ROOT_PATTERNS (case-insensitive).
/// Built once at first use; replaces `lc.contains(&root.to_ascii_lowercase())`
/// chain (one alloc per pattern per call). Single-pass O(name.len) scan.
fn family_root_ac() -> &'static aho_corasick::AhoCorasick {
    static AC: std::sync::OnceLock<aho_corasick::AhoCorasick> = std::sync::OnceLock::new();
    AC.get_or_init(|| {
        aho_corasick::AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .build(FAMILY_ROOT_PATTERNS)
            .expect("family root patterns build")
    })
}

/// Returns true if `name` substring-contains any FAMILY_ROOT_PATTERNS entry
/// (case-insensitive). Exposed for callers that want to test the carve-out
/// independent of the full `match_name` dispatch.
pub fn is_family_root(name: &str) -> bool {
    family_root_ac().is_match(name)
}

/// 3-tier dispatch with FAMILY_ROOT carve-out.
///
/// Order: FamilyRoot → Exact → WordBoundary → Substring.
/// Patterns shorter than `MIN_PATTERN_LEN` are rejected from both
/// WordBoundary and Substring tiers — addresses peer-consult item #1
/// ("go" → "Categories" must not match at any tier).
pub fn match_name(
    name: &str,
    exact_set: &HashSet<&'static str>,
    patterns: &[String],
) -> MatchResult {
    // Tier 0: hierarchy carve-out (Brave/Chrome/…) — unconditional substring.
    if is_family_root(name) {
        return MatchResult {
            tier: MatchTier::FamilyRoot,
            confidence: TIER_EXACT_CONF,
        };
    }
    // Tier 1: exact, O(1).
    if exact_set.contains(name) {
        return MatchResult {
            tier: MatchTier::Exact,
            confidence: TIER_EXACT_CONF,
        };
    }
    let name_lc = name.to_ascii_lowercase();
    // Tier 2: word-boundary on patterns ≥ MIN_PATTERN_LEN.
    for pat in patterns.iter().filter(|p| p.len() >= MIN_PATTERN_LEN) {
        let pat_lc = pat.to_ascii_lowercase();
        if word_boundary_contains(&name_lc, &pat_lc) {
            return MatchResult {
                tier: MatchTier::WordBoundary,
                confidence: TIER_WORDBOUND_CONF,
            };
        }
    }
    // Tier 3: substring fallback (degraded). Also gated on MIN_PATTERN_LEN
    // — explicit substring floor per peer-consult action item #1.
    for pat in patterns.iter().filter(|p| p.len() >= MIN_PATTERN_LEN) {
        let pat_lc = pat.to_ascii_lowercase();
        if !pat_lc.is_empty() && name_lc.contains(&pat_lc) {
            return MatchResult {
                tier: MatchTier::Substring,
                confidence: TIER_SUBSTRING_CONF,
            };
        }
    }
    MatchResult::NONE
}

/// Pattern-weighted confidence: `tier_conf × learned_weight` (default 1.0).
///
/// NOTE — peer-consult item #2: the returned scalar is an *epistemic
/// identification confidence*, NOT a utility multiplier. Callers MUST NOT
/// multiply this into a PolicyScorer's benefit/composite. Route it through
/// `IdentityUncertaintyFeature` (RSS-composed via 65f310d) instead.
pub fn confidence_for(
    res: MatchResult,
    pattern: Option<&str>,
    weights: &HashMap<String, PatternWeight>,
) -> f64 {
    if !res.matched() {
        return 0.0;
    }
    let w = pattern
        .and_then(|p| weights.get(p))
        .map(|pw| pw.effectiveness().clamp(0.0, 1.0))
        .unwrap_or(1.0);
    res.confidence * w
}

/// PolicyFeature injecting `1.0 - match_confidence` as RSS-composed uncertainty.
/// ONLY supported path to expose MatchEngine confidence to the policy layer.
/// FORBIDDEN: `score.composite * match_confidence` (peer-consult item #2).
/// CORRECT: `PolicyScorer::builder().feature(IdentityUncertaintyFeature::new(c))`.
pub struct IdentityUncertaintyFeature {
    match_confidence: f64,
}

impl IdentityUncertaintyFeature {
    pub fn new(match_confidence: f64) -> Self {
        Self {
            match_confidence: match_confidence.clamp(0.0, 1.0),
        }
    }
}

impl PolicyFeature for IdentityUncertaintyFeature {
    fn name(&self) -> &'static str {
        "identity_uncertainty"
    }
    fn contribute(&self, _action: &RootAction, _ctx: &ActionContext) -> Contribution {
        Contribution {
            benefit: 0.0,
            cost: 0.0,
            uncertainty: 1.0 - self.match_confidence,
            hard_veto: false,
            ..Contribution::zero()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pats(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    // Peer-consult item #1: word-boundary arc/arcade + 2-char rejection.
    #[test]
    fn word_boundary_semantics() {
        assert!(!word_boundary_contains("arcade", "arc"));
        assert!(word_boundary_contains("arc helper", "arc"));
        assert!(word_boundary_contains("node", "node"));
        assert!(word_boundary_contains("/usr/bin/node", "node"));
        assert!(word_boundary_contains("node-gyp", "node"));
        assert!(!word_boundary_contains("anode", "node"));
    }

    #[test]
    fn tier_confidences() {
        let mut set = HashSet::new();
        set.insert("kernel_task");
        let exact = match_name("kernel_task", &set, &[]);
        assert_eq!(exact.tier, MatchTier::Exact);
        assert_eq!(exact.confidence, 1.00);

        let wb = match_name("rustc", &HashSet::new(), &pats(&["rustc"]));
        assert_eq!(wb.tier, MatchTier::WordBoundary);
        assert_eq!(wb.confidence, 0.85);

        // "test" inside "latest" — substring, ≥3 chars, no word boundary.
        let sub = match_name("latest", &HashSet::new(), &pats(&["test"]));
        assert_eq!(sub.tier, MatchTier::Substring);
        assert_eq!(sub.confidence, 0.30);
        // INVARIANT: substring tier sits below MIN_FREEZE_CONFIDENCE = 0.35
        // so a lone substring match cannot independently authorize a freeze.
        assert!(sub.confidence < 0.35_f64);
    }

    // Peer-consult item #1: substring-tier 2-char floor.
    #[test]
    fn substring_tier_rejects_2char_patterns() {
        // "go" must NOT match "Categories" via ANY tier.
        let r = match_name("Categories", &HashSet::new(), &pats(&["go"]));
        assert_eq!(r.tier, MatchTier::None);
        // Even at a clean word boundary, 2-char patterns are skipped entirely.
        let r2 = match_name("go test", &HashSet::new(), &pats(&["go"]));
        assert_eq!(r2.tier, MatchTier::None);
    }

    #[test]
    fn empty_pattern_does_not_match() {
        // Guard against `.contains("")` = true (mass false-positive bomb,
        // design risk R3).
        let r = match_name("anything", &HashSet::new(), &pats(&[""]));
        assert_eq!(r.tier, MatchTier::None);
    }

    // Migration safety — confidence_for default + scaling.
    #[test]
    fn confidence_for_weight_semantics() {
        let r = MatchResult {
            tier: MatchTier::WordBoundary,
            confidence: TIER_WORDBOUND_CONF,
        };
        let mut weights = HashMap::new();
        // Unknown pattern → multiplier defaults to 1.0 → preserves tier conf.
        assert!((confidence_for(r, Some("rustc"), &weights) - 0.85).abs() < 1e-9);
        // PatternWeight throttle=2/effective=1 → Laplace = 0.5 → halves.
        weights.insert(
            "rustc".to_string(),
            PatternWeight {
                throttle_count: 2,
                effective_count: 1,
            },
        );
        let c = confidence_for(r, Some("rustc"), &weights);
        assert!((c - 0.85 * 0.5).abs() < 1e-9);
    }

    // Peer-consult item #3: Brave/Chromium family carve-out (commit 26eac06).
    #[test]
    fn brave_helper_renderer_still_unconditionally_protected() {
        let r = match_name("Brave Browser Helper (Renderer)", &HashSet::new(), &[]);
        assert_eq!(r.tier, MatchTier::FamilyRoot);
        assert_eq!(r.confidence, 1.00);

        let r2 = match_name("Google Chrome Helper (GPU)", &HashSet::new(), &[]);
        assert_eq!(r2.tier, MatchTier::FamilyRoot);
        assert_eq!(r2.confidence, 1.00);

        assert!(is_family_root("Brave Browser Helper"));
        assert!(!is_family_root("rustc"));
        assert!(!is_family_root("Categories"));
    }

    // Peer-consult item #2: match_confidence is epistemic, NOT utility.
    // We invoke `contribute()` directly with throwaway action+ctx; this proves
    // the Contribution shape (uncertainty-only, no benefit/cost/veto) is what
    // we promised the scorer, regardless of action variant or ctx state.
    #[test]
    fn match_confidence_enters_uncertainty_not_benefit() {
        use crate::engine::audit_types::DecisionReason;
        use crate::engine::safety::ProtectionLevel;
        // Minimal ActionContext — only the type matters for this feature.
        let ctx = ActionContext {
            pressure: 0.0,
            swap_gb: 0.0,
            thrashing_score: 0.0,
            p_oom_30s: None,
            p_jank_60s: None,
            has_sleep_assertion: false,
            call_in_progress: false,
            idle_secs: 0.0,
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
        };
        let action = RootAction::BoostProcess {
            pid: 1,
            name: "x".into(),
            reason: "u".into(),
            decision_reason: DecisionReason::PressureContext,
            start_sec: 0,
            start_usec: 0,
        };
        let c = IdentityUncertaintyFeature::new(0.30).contribute(&action, &ctx);
        assert!((c.uncertainty - 0.70).abs() < 1e-9);
        // CRITICAL invariant: benefit MUST be 0 — confidence is never utility.
        assert_eq!(c.benefit, 0.0);
        assert_eq!(c.cost, 0.0);
        assert!(!c.hard_veto);
    }
}
