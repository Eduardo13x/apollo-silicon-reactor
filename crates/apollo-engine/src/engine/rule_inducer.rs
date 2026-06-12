//! Rule Inducer — group and batch skill generation from observed experience.
//!
//! ## Division of labor with causal_graph
//!
//! `causal_graph` already handles **individual process skills** — it records each
//! throttle action and measures pressure N cycles later, generating
//! `"throttle process X when pressure ≥ P"` skills automatically.
//!
//! `rule_inducer` handles what causal_graph cannot:
//!
//! 1. **Group skills** — "throttle A + B together" derived from co-occurrence
//!    pairs that spike together reliably (≥ MIN_COOCCUR times).
//!
//! 2. **Batch skills** — pairs detected from coincident pressure drops in
//!    ExperienceMemory: records with identical drop values were resolved in
//!    the same daemon cycle, revealing which processes were co-throttled
//!    during a large-drop event.
//!
//! ## When it runs
//!
//! Called from the daemon main loop every 100 cycles (~50s).
//! Returns only NEW skills (not already in the registry).
//! Caller adds them via `SkillRegistry::register_induced()`.
//!
//! ## Safety
//!
//! - Never generates skills for protected processes.
//! - Caps total induced skills at MAX_INDUCED to prevent explosion.
//! - Skills start with success_rate = 0.0; must prove themselves or get GC'd.

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};

use crate::engine::optimization_skills::OptimizationSkill;
use crate::engine::outcome_tracker::ExperienceMemory;
use crate::engine::safety::is_protected_name;

/// Only use experience records above this pressure threshold.
/// Low-pressure throttles add noise: background daemons often show negative
/// drops when the real culprit (browser, Spotlight) dominates and is protected.
const MIN_PRESSURE_AT_ACTION: f64 = 0.50;

/// Minimum co-occurrence count to induce a group skill.
const MIN_COOCCUR: u32 = 10;

/// Co-occurrence count above which a_ok/b_ok individual evidence is not required.
/// Two processes that co-spike ≥500 times are worth trying to throttle together
/// even without individual effectiveness records.
const HIGH_COOCCUR_BYPASS: u32 = 500;

/// Maximum induced skills total (prevents explosion under noisy data).
const MAX_INDUCED: usize = 60;

/// Minimum pressure drop (absolute) to count an event as a "large drop".
/// Events where pressure fell by at least 5% — meaningful batch signal.
const BATCH_MIN_DROP: f64 = 0.05;

/// Minimum number of large-drop batch events two processes must share
/// before inducing a batch group skill.
const BATCH_MIN_EVENTS: usize = 4;

/// Maximum batch size to consider for attribution.
/// Mass-throttle events (>5 processes at once) spread across too many
/// pairs and dilute the causal signal — skip them.
const BATCH_MAX_SIZE: usize = 5;

const GROUP_PREFIX: &str = "group:";
const BATCH_PREFIX: &str = "batch:";

// ── Public API ────────────────────────────────────────────────────────────────

/// Mine experience memory and co-occurrence graph for group/batch skills.
///
/// Individual process skills are handled by causal_graph — this function
/// only generates skills that require multi-process correlation evidence.
///
/// Returns skills not already present in `existing_names`.
/// The caller is responsible for adding them to the registry and persisting.
/// Induce new skills from observed experience.
///
/// `workload` is the current workload mode string (e.g., "build", "browsing").
/// - **Group skills** (co-occurrence) use `"any"` — they represent structural
///   pairs that co-spike across all workloads.
/// - **Batch skills** (coincident events) use the provided `workload` — they
///   represent session-specific bursts likely tied to the active context.
pub fn induce(
    experience: &ExperienceMemory,
    top_pairs: &[(&str, &str, u32)],
    existing_names: &HashSet<String>,
    protected: &[&str],
    workload: &str,
) -> Vec<OptimizationSkill> {
    let mut result = Vec::new();

    // Build per-process stats from high-pressure records.
    // Used by group skill a_ok/b_ok individual evidence check.
    let mut by_process: HashMap<&str, ProcessStats> = HashMap::new();
    for rec in experience.records() {
        if rec.pressure_at_action < MIN_PRESSURE_AT_ACTION {
            continue;
        }
        let e = by_process.entry(rec.process_name.as_str()).or_default();
        e.total += 1;
        if rec.pressure_drop > 0.0 {
            e.positive_drops += 1;
        }
        e.sum_pressure += rec.pressure_at_action;
    }

    // ── 1. Group skills from co-occurrence pairs ──────────────────────────────

    for (a, b, count) in top_pairs {
        if result.len() >= MAX_INDUCED {
            break;
        }
        if *count < MIN_COOCCUR {
            continue;
        }
        if is_protected(a, protected) || is_protected(b, protected) {
            continue;
        }
        // Skip self-pairs (same process co-occurring with itself — e.g., multiple
        // instances). A single-target "group" skill offers no synergy over throttle:X.
        if a == b {
            continue;
        }

        // For very high co-occurrence counts, skip individual effectiveness
        // check — the co-spike frequency alone is sufficient evidence.
        // Otherwise require at least one process to have some positive signal.
        if *count < HIGH_COOCCUR_BYPASS {
            let a_ok = by_process
                .get(a)
                .is_some_and(|s| s.total >= 3 && s.positive_drops as f64 / s.total as f64 >= 0.40);
            let b_ok = by_process
                .get(b)
                .is_some_and(|s| s.total >= 3 && s.positive_drops as f64 / s.total as f64 >= 0.40);
            if !a_ok && !b_ok {
                continue;
            }
        }

        let (first, second) = if a <= b { (a, b) } else { (b, a) };
        let skill_name = format!("{}{}+{}", GROUP_PREFIX, first, second);
        if existing_names.contains(&skill_name) {
            continue;
        }

        result.push(OptimizationSkill {
            name: skill_name,
            min_pressure: 0.60,
            workload_hint: "any".to_string(),
            throttle_targets: vec![first.to_string(), second.to_string()],
            success_rate: 0.0,
            apply_count: 0,
            success_count: 0,
        });
    }

    // ── 2. Batch skills from coincident experience records ────────────────────
    //
    // Records with identical (pressure_drop, pressure_at_action) values were
    // resolved in the same daemon cycle — they share a causal event.
    // Pairs that appear together in ≥BATCH_MIN_EVENTS large-drop batches
    // are worth throttling as a unit.

    let mut batch_pair_counts: HashMap<(&str, &str), usize> = HashMap::new();
    {
        let mut buckets: HashMap<(i32, i32), Vec<&str>> = HashMap::new();
        for rec in experience.records() {
            if rec.pressure_drop < BATCH_MIN_DROP {
                continue;
            }
            if rec.pressure_at_action < MIN_PRESSURE_AT_ACTION {
                continue;
            }
            if is_protected(rec.process_name.as_str(), protected) {
                continue;
            }
            let drop_k = (rec.pressure_drop * 1000.0) as i32;
            let pres_k = (rec.pressure_at_action * 100.0) as i32;
            buckets
                .entry((drop_k, pres_k))
                .or_default()
                .push(rec.process_name.as_str());
        }

        for processes in buckets.values() {
            if processes.len() < 2 || processes.len() > BATCH_MAX_SIZE {
                continue;
            }
            for i in 0..processes.len() {
                for j in (i + 1)..processes.len() {
                    if processes[i] == processes[j] {
                        continue;
                    }
                    let (a, b) = if processes[i] <= processes[j] {
                        (processes[i], processes[j])
                    } else {
                        (processes[j], processes[i])
                    };
                    *batch_pair_counts.entry((a, b)).or_insert(0) += 1;
                }
            }
        }
    }

    let mut batch_pairs: Vec<((&str, &str), usize)> = batch_pair_counts.into_iter().collect();
    batch_pairs.sort_by_key(|(_, count)| Reverse(*count));

    for ((a, b), count) in &batch_pairs {
        if result.len() >= MAX_INDUCED {
            break;
        }
        if *count < BATCH_MIN_EVENTS {
            continue;
        }
        let skill_name = format!("{}{}+{}", BATCH_PREFIX, a, b);
        let group_name = format!("{}{}+{}", GROUP_PREFIX, a, b);
        if existing_names.contains(&skill_name)
            || existing_names.contains(&group_name)
            || result.iter().any(|s| s.name == group_name)
        {
            continue;
        }

        let mean_pressure = by_process
            .get(a)
            .map(|s| s.sum_pressure / s.total.max(1) as f64)
            .unwrap_or(0.65);
        let trigger = ((mean_pressure - 0.05) as f32).clamp(0.45, 0.80);

        result.push(OptimizationSkill {
            name: skill_name,
            min_pressure: trigger,
            // Batch skills tag the active workload — they come from coincident
            // events in the current session, so they're context-specific.
            workload_hint: workload.to_string(),
            throttle_targets: vec![a.to_string(), b.to_string()],
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
    positive_drops: usize,
    sum_pressure: f64,
}

fn is_protected(name: &str, protected: &[&str]) -> bool {
    // Hard protection from the unified oracle wins unconditionally
    // [Saltzer & Kaashoek 2009] Complete Mediation — single choke-point.
    if is_protected_name(name) {
        return true;
    }
    // Then the caller-provided policy/override list (substring, case-insensitive).
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

    #[test]
    fn group_skill_from_cooccurrence_with_individual_evidence() {
        let mut mem = ExperienceMemory::new(300);
        for i in 0..5 {
            for name in &["photoanalysisd", "mediaanalysisd"] {
                mem.push(ExperienceRecord {
                    process_name: name.to_string(),
                    pressure_at_action: 0.70,
                    pressure_drop: if i % 2 == 0 { 0.04 } else { 0.0 },
                    effective: i % 2 == 0,
                    workload: 0,
                });
            }
        }
        let pairs = vec![("mediaanalysisd", "photoanalysisd", 25u32)];
        let skills = induce(&mem, &pairs, &HashSet::new(), &[], "any");
        let group = skills.iter().find(|s| s.name.starts_with("group:"));
        assert!(group.is_some(), "should induce group skill");
        let g = group.unwrap();
        assert!(g.throttle_targets.contains(&"mediaanalysisd".to_string()));
        assert!(g.throttle_targets.contains(&"photoanalysisd".to_string()));
    }

    #[test]
    fn group_skill_from_high_cooccur_bypass() {
        // Very high co-occurrence bypasses a_ok/b_ok individual evidence requirement.
        // Use non-protected background daemons — the unified oracle now hard-rejects
        // names like `coreaudiod` before the co-occurrence path is even considered.
        let mem = ExperienceMemory::new(300);
        let pairs = vec![("corespeechd", "suggestd", HIGH_COOCCUR_BYPASS + 10)];
        let skills = induce(&mem, &pairs, &HashSet::new(), &[], "any");
        let group = skills.iter().find(|s| s.name.starts_with("group:"));
        assert!(
            group.is_some(),
            "high co-occurrence should bypass individual evidence check"
        );
    }

    #[test]
    fn skips_protected_in_group() {
        let mem = ExperienceMemory::new(300);
        let pairs = vec![("mds_stores", "photoanalysisd", HIGH_COOCCUR_BYPASS + 10)];
        let skills = induce(&mem, &pairs, &HashSet::new(), &["mds_stores"], "any");
        assert!(
            skills.is_empty(),
            "protected process must not appear in group skill"
        );
    }

    #[test]
    fn skips_already_existing_group() {
        // Use non-protected background daemons so protection isn't what short-circuits
        // induction — the test subject is the existing-skill dedup path.
        let mem = ExperienceMemory::new(300);
        let pairs = vec![("corespeechd", "suggestd", HIGH_COOCCUR_BYPASS + 10)];
        let mut existing = HashSet::new();
        existing.insert("group:corespeechd+suggestd".to_string());
        let skills = induce(&mem, &pairs, &existing, &[], "any");
        assert!(skills.is_empty(), "must not re-induce existing skill");
    }

    #[test]
    fn batch_detector_induces_group_from_coincident_drops() {
        let mut mem = ExperienceMemory::new(300);
        // 4 distinct large-drop events with the same pair
        let drops = [0.15_f64, 0.18, 0.12, 0.20];
        let pressures = [0.70_f64, 0.72, 0.68, 0.75];
        for (drop, pressure) in drops.iter().zip(pressures.iter()) {
            for name in &["corespeechd", "suggestd"] {
                mem.push(ExperienceRecord {
                    process_name: name.to_string(),
                    pressure_at_action: *pressure,
                    pressure_drop: *drop,
                    effective: true,
                    workload: 0,
                });
            }
        }
        let skills = induce(&mem, &[], &HashSet::new(), &[], "any");
        let batch = skills.iter().find(|s| s.name.starts_with("batch:"));
        assert!(
            batch.is_some(),
            "batch detector should find coincident pairs after {} events",
            BATCH_MIN_EVENTS
        );
        let b = batch.unwrap();
        assert!(b.throttle_targets.contains(&"corespeechd".to_string()));
        assert!(b.throttle_targets.contains(&"suggestd".to_string()));
    }

    #[test]
    fn batch_detector_requires_large_drop() {
        let mut mem = ExperienceMemory::new(300);
        for _ in 0..5 {
            for name in &["alpha", "beta"] {
                mem.push(ExperienceRecord {
                    process_name: name.to_string(),
                    pressure_at_action: 0.65,
                    pressure_drop: 0.02, // below BATCH_MIN_DROP=0.05
                    effective: true,
                    workload: 0,
                });
            }
        }
        let skills = induce(&mem, &[], &HashSet::new(), &[], "any");
        let batch = skills.iter().find(|s| s.name.starts_with("batch:"));
        assert!(
            batch.is_none(),
            "small drops must not trigger batch induction"
        );
    }

    #[test]
    fn no_duplicate_between_group_and_batch() {
        // If a group: skill already exists for a pair, batch: must not re-create it.
        let mut mem = ExperienceMemory::new(300);
        let drops = [0.15_f64, 0.18, 0.12, 0.20];
        let pressures = [0.70_f64, 0.72, 0.68, 0.75];
        for (drop, pressure) in drops.iter().zip(pressures.iter()) {
            for name in &["corespeechd", "suggestd"] {
                mem.push(ExperienceRecord {
                    process_name: name.to_string(),
                    pressure_at_action: *pressure,
                    pressure_drop: *drop,
                    effective: true,
                    workload: 0,
                });
            }
        }
        let mut existing = HashSet::new();
        existing.insert("group:corespeechd+suggestd".to_string());
        let skills = induce(&mem, &[], &existing, &[], "any");
        assert!(
            skills
                .iter()
                .all(|s| s.name != "batch:corespeechd+suggestd"),
            "batch: must not duplicate an existing group: skill"
        );
    }

    #[test]
    fn group_skills_always_tagged_any() {
        // Group skills are structural (cross-workload) — always tagged "any".
        // Use non-protected background daemons (unified oracle hard-rejects `coreaudiod`).
        let mem = ExperienceMemory::new(300);
        let pairs = vec![("corespeechd", "suggestd", HIGH_COOCCUR_BYPASS + 10)];
        let skills = induce(&mem, &pairs, &HashSet::new(), &[], "build");
        let group = skills
            .iter()
            .find(|s| s.name.starts_with("group:"))
            .unwrap();
        assert_eq!(
            group.workload_hint, "any",
            "group skills must be tagged 'any' regardless of workload"
        );
    }

    #[test]
    fn batch_skills_tagged_with_active_workload() {
        // Batch skills come from session-specific bursts — tagged with current workload.
        let mut mem = ExperienceMemory::new(300);
        let drops = [0.15_f64, 0.18, 0.12, 0.20];
        let pressures = [0.70_f64, 0.72, 0.68, 0.75];
        for (drop, pressure) in drops.iter().zip(pressures.iter()) {
            for name in &["corespeechd", "suggestd"] {
                mem.push(ExperienceRecord {
                    process_name: name.to_string(),
                    pressure_at_action: *pressure,
                    pressure_drop: *drop,
                    effective: true,
                    workload: 0,
                });
            }
        }
        let skills = induce(&mem, &[], &HashSet::new(), &[], "browsing");
        let batch = skills
            .iter()
            .find(|s| s.name.starts_with("batch:"))
            .unwrap();
        assert_eq!(
            batch.workload_hint, "browsing",
            "batch skills must be tagged with active workload"
        );
    }

    #[test]
    fn batch_skills_tagged_idle_when_workload_idle() {
        let mut mem = ExperienceMemory::new(300);
        for drop in &[0.15_f64, 0.18, 0.12, 0.20] {
            for name in &["alpha", "beta"] {
                mem.push(ExperienceRecord {
                    process_name: name.to_string(),
                    pressure_at_action: 0.70,
                    pressure_drop: *drop,
                    effective: true,
                    workload: 0,
                });
            }
        }
        let skills = induce(&mem, &[], &HashSet::new(), &[], "idle");
        if let Some(batch) = skills.iter().find(|s| s.name.starts_with("batch:")) {
            assert_eq!(batch.workload_hint, "idle");
        }
    }
}
