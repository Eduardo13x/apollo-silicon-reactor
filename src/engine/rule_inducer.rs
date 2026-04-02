//! Rule Inducer — autonomous skill generation from observed experience.
//!
//! Closes the loop between what Apollo observes and what it acts on.
//!
//! ## The gap it fills
//!
//! The causal graph generates skills from actions that were actually applied.
//! But ExperienceMemory accumulates richer evidence — including pressure context,
//! drop magnitude, and consistency across many cycles. The inducer mines that
//! evidence and crystallizes it into new skills WITHOUT human intervention.
//!
//! ## What it generates
//!
//! 1. **Individual skills** — "throttle process X when pressure ≥ P" derived
//!    from processes with ≥ MIN_OBS experience records and ≥ MIN_RATE effectiveness.
//!
//! 2. **Group skills** — "throttle A + B together" derived from co-occurrence
//!    pairs that spike together reliably (≥ MIN_COOCCUR times).
//!
//! ## When it runs
//!
//! Called from the daemon main loop every 100 cycles (~5 min).
//! Returns only NEW skills (not already in the registry).
//! Caller adds them via `SkillRegistry::register_induced()`.
//!
//! ## Safety
//!
//! - Never generates skills for protected processes (checked by caller).
//! - Caps total induced skills at MAX_INDUCED to prevent explosion.
//! - Skills start with success_rate = 0.0; must prove themselves or get GC'd.

use std::collections::{HashMap, HashSet};

use crate::engine::optimization_skills::OptimizationSkill;
use crate::engine::outcome_tracker::ExperienceMemory;

/// Minimum observations before crystallizing a skill.
const MIN_OBS: usize = 8;

/// Minimum rate of throttles that produced a positive pressure drop.
/// Uses raw pressure_drop > 0 (not the counterfactual-adjusted `effective` flag)
/// so short-lived beneficial effects aren't double-penalized by the baseline filter.
const MIN_EFFECTIVE_RATE: f64 = 0.50;

/// Minimum mean pressure drop to consider a throttle meaningful.
const MIN_MEAN_DROP: f64 = 0.015;

/// Minimum co-occurrence count to induce a group skill.
const MIN_COOCCUR: u32 = 10;

/// Co-occurrence count above which a_ok/b_ok individual evidence is not required.
/// Two processes that co-spike ≥500 times are worth trying to throttle together
/// even without individual effectiveness records.
const HIGH_COOCCUR_BYPASS: u32 = 500;

/// Maximum induced skills total (prevents explosion under noisy data).
const MAX_INDUCED: usize = 60;

/// Prefix for auto-induced skill names — distinguishes from causal-graph skills.
const INDUCED_PREFIX: &str = "induced:";
const GROUP_PREFIX: &str = "group:";

// ── Public API ────────────────────────────────────────────────────────────────

/// Mine experience memory and co-occurrence graph for new skills.
///
/// Returns skills that are not already present in `existing_names`.
/// The caller is responsible for adding them to the registry and persisting.
pub fn induce(
    experience: &ExperienceMemory,
    top_pairs: &[(&str, &str, u32)],
    existing_names: &HashSet<String>,
    protected: &[&str],
) -> Vec<OptimizationSkill> {
    let mut result = Vec::new();

    // ── 1. Individual skills from experience memory ───────────────────────────

    // Aggregate records per process.
    // Use pressure_drop > 0 (not rec.effective) to avoid double-penalizing
    // beneficial throttles that fell below the global counterfactual baseline.
    let mut by_process: HashMap<&str, ProcessStats> = HashMap::new();
    for rec in experience.records() {
        let e = by_process.entry(rec.process_name.as_str()).or_default();
        e.total += 1;
        if rec.pressure_drop > 0.0 {
            e.positive_drops += 1;
        }
        e.sum_drop += rec.pressure_drop as f64;
        e.sum_pressure += rec.pressure_at_action as f64;
    }

    for (name, stats) in &by_process {
        if result.len() >= MAX_INDUCED {
            break;
        }
        if stats.total < MIN_OBS {
            continue;
        }
        let rate = stats.positive_drops as f64 / stats.total as f64;
        let mean_drop = stats.sum_drop / stats.total as f64;
        let mean_pressure = stats.sum_pressure / stats.total as f64;

        if rate < MIN_EFFECTIVE_RATE || mean_drop < MIN_MEAN_DROP {
            continue;
        }
        if is_protected(name, protected) {
            continue;
        }

        let skill_name = format!("{}{}", INDUCED_PREFIX, name);
        if existing_names.contains(&skill_name) {
            continue;
        }

        // Trigger slightly below the mean observed pressure so we act
        // proactively before the situation peaks.
        let trigger = ((mean_pressure - 0.05) as f32).clamp(0.40, 0.85);

        result.push(OptimizationSkill {
            name: skill_name,
            min_pressure: trigger,
            workload_hint: "any".to_string(),
            throttle_targets: vec![name.to_string()],
            success_rate: 0.0,
            apply_count: 0,
            success_count: 0,
        });
    }

    // ── 2. Group skills from co-occurrence pairs ──────────────────────────────

    for (a, b, count) in top_pairs {
        if result.len() >= MAX_INDUCED {
            break;
        }
        if *count < MIN_COOCCUR {
            continue; // pairs are sorted descending; can break early
        }
        if is_protected(a, protected) || is_protected(b, protected) {
            continue;
        }

        // For very high co-occurrence counts, skip individual effectiveness
        // check — the co-spike frequency alone is sufficient evidence.
        // Otherwise require at least one process to have some individual
        // effectiveness (relaxed from AND → OR to avoid blocking all pairs
        // where one process lacks experience records).
        if *count < HIGH_COOCCUR_BYPASS {
            let a_stats = by_process.get(a);
            let b_stats = by_process.get(b);
            let a_ok = a_stats.map_or(false, |s| {
                s.total >= 3 && s.positive_drops as f64 / s.total as f64 >= 0.40
            });
            let b_ok = b_stats.map_or(false, |s| {
                s.total >= 3 && s.positive_drops as f64 / s.total as f64 >= 0.40
            });
            if !a_ok && !b_ok {
                continue;
            }
        }

        // Sort names so the skill name is deterministic regardless of pair order.
        let (first, second) = if a <= b { (a, b) } else { (b, a) };
        let skill_name = format!("{}{}+{}", GROUP_PREFIX, first, second);
        if existing_names.contains(&skill_name) {
            continue;
        }

        // Group skills trigger at moderate pressure — they're proactive.
        let trigger = 0.60_f32;

        result.push(OptimizationSkill {
            name: skill_name,
            min_pressure: trigger,
            workload_hint: "any".to_string(),
            throttle_targets: vec![first.to_string(), second.to_string()],
            success_rate: 0.0,
            apply_count: 0,
            success_count: 0,
        });
    }

    result
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[derive(Default)]
struct ProcessStats {
    total: usize,
    /// Records where pressure_drop > 0 (raw positive outcomes).
    positive_drops: usize,
    sum_drop: f64,
    sum_pressure: f64,
}

fn is_protected(name: &str, protected: &[&str]) -> bool {
    let nl = name.to_ascii_lowercase();
    protected
        .iter()
        .any(|p| nl.contains(&p.to_ascii_lowercase()))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::outcome_tracker::{ExperienceMemory, ExperienceRecord};

    fn make_experience(process: &str, n: usize, effective_rate: f64, pressure: f64) -> ExperienceMemory {
        let mut mem = ExperienceMemory::new(300);
        for i in 0..n {
            let effective = (i as f64 / n as f64) < effective_rate;
            mem.push(ExperienceRecord {
                process_name: process.to_string(),
                pressure_at_action: pressure,
                pressure_drop: if effective { 0.05 } else { 0.0 },
                effective,
            });
        }
        mem
    }

    #[test]
    fn induces_high_confidence_process() {
        let mem = make_experience("mediaanalysisd", 20, 0.80, 0.70);
        let skills = induce(&mem, &[], &HashSet::new(), &[]);
        assert_eq!(skills.len(), 1);
        assert!(skills[0].name.starts_with("induced:"));
        assert!(skills[0].throttle_targets.contains(&"mediaanalysisd".to_string()));
    }

    #[test]
    fn does_not_induce_low_confidence() {
        let mem = make_experience("someprocess", 20, 0.40, 0.70);
        let skills = induce(&mem, &[], &HashSet::new(), &[]);
        assert!(skills.is_empty());
    }

    #[test]
    fn does_not_induce_insufficient_obs() {
        // MIN_OBS = 8; n=5 is genuinely insufficient.
        let mem = make_experience("mediaanalysisd", 5, 0.90, 0.70);
        let skills = induce(&mem, &[], &HashSet::new(), &[]);
        assert!(skills.is_empty());
    }

    #[test]
    fn induces_from_positive_drops_bypassing_effective_flag() {
        // Records where effective=false but pressure_drop > 0 (below counterfactual
        // baseline) should still count toward induction — avoids double-penalizing.
        let mut mem = ExperienceMemory::new(300);
        for _ in 0..10 {
            mem.push(ExperienceRecord {
                process_name: "corespeechd".to_string(),
                pressure_at_action: 0.65,
                pressure_drop: 0.03, // positive but below natural_drift → effective=false in tracker
                effective: false,    // counterfactual said "not good enough"
            });
        }
        let skills = induce(&mem, &[], &HashSet::new(), &[]);
        // Should induce: 10 records, 100% positive_drops, mean_drop=0.03 > 0.015
        assert_eq!(skills.len(), 1, "should induce despite effective=false");
        assert!(skills[0].name.starts_with("induced:"));
    }

    #[test]
    fn does_not_induce_negative_mean_drop() {
        // Processes where throttling makes things worse (negative drop) must not
        // be induced even if n and rate would otherwise qualify.
        let mut mem = ExperienceMemory::new(300);
        for _ in 0..10 {
            mem.push(ExperienceRecord {
                process_name: "suggestd".to_string(),
                pressure_at_action: 0.62,
                pressure_drop: -0.011, // pressure went UP after throttle
                effective: false,
            });
        }
        let skills = induce(&mem, &[], &HashSet::new(), &[]);
        assert!(skills.is_empty(), "must not induce skill for process that increases pressure");
    }

    #[test]
    fn group_skill_from_high_cooccur_bypass() {
        // Very high co-occurrence bypasses a_ok/b_ok individual evidence requirement.
        let mem = ExperienceMemory::new(300); // empty — no individual evidence
        let pairs = vec![("coreaudiod", "corespeechd", HIGH_COOCCUR_BYPASS + 10)];
        let skills = induce(&mem, &pairs, &HashSet::new(), &[]);
        let group = skills.iter().find(|s| s.name.starts_with("group:"));
        assert!(group.is_some(), "high co-occurrence should bypass individual evidence check");
    }

    #[test]
    fn skips_already_existing() {
        let mem = make_experience("mediaanalysisd", 20, 0.80, 0.70);
        let mut existing = HashSet::new();
        existing.insert("induced:mediaanalysisd".to_string());
        let skills = induce(&mem, &[], &existing, &[]);
        assert!(skills.is_empty());
    }

    #[test]
    fn skips_protected() {
        let mem = make_experience("mds_stores", 20, 0.80, 0.70);
        let skills = induce(&mem, &[], &HashSet::new(), &["mds_stores"]);
        assert!(skills.is_empty());
    }

    #[test]
    fn induces_group_skill_from_cooccurrence() {
        let mem = ExperienceMemory::new(300);
        // Both processes have some individual effectiveness
        let mut mem2 = ExperienceMemory::new(300);
        for i in 0..5 {
            mem2.push(ExperienceRecord {
                process_name: "photoanalysisd".to_string(),
                pressure_at_action: 0.70,
                pressure_drop: if i % 2 == 0 { 0.04 } else { 0.0 },
                effective: i % 2 == 0,
            });
            mem2.push(ExperienceRecord {
                process_name: "mediaanalysisd".to_string(),
                pressure_at_action: 0.70,
                pressure_drop: if i % 2 == 0 { 0.04 } else { 0.0 },
                effective: i % 2 == 0,
            });
        }
        let pairs = vec![("mediaanalysisd", "photoanalysisd", 25u32)];
        let _ = mem;
        let skills = induce(&mem2, &pairs, &HashSet::new(), &[]);
        let group = skills.iter().find(|s| s.name.starts_with("group:"));
        assert!(group.is_some(), "should induce group skill");
        let g = group.unwrap();
        assert!(g.throttle_targets.contains(&"mediaanalysisd".to_string()));
        assert!(g.throttle_targets.contains(&"photoanalysisd".to_string()));
    }

    #[test]
    fn trigger_pressure_is_below_mean_observation() {
        let mem = make_experience("suggestd", 20, 0.75, 0.72);
        let skills = induce(&mem, &[], &HashSet::new(), &[]);
        assert!(!skills.is_empty());
        // trigger should be ~0.67 (0.72 - 0.05), clamped to [0.40, 0.85]
        assert!(skills[0].min_pressure < 0.72);
    }
}
