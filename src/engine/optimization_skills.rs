//! Optimization Skills — persistent recipes learned from experience.
//!
//! Inspired by Hermes Agent's self-improving skills system.
//! When Apollo discovers a sequence of actions that reliably reduces pressure,
//! it saves it as a "skill" — a reusable recipe with measured success rate.
//!
//! Skills are persisted to disk and loaded on startup, surviving restarts.
//! Each skill has a trigger condition and a measured effectiveness.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// A learned optimization recipe.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizationSkill {
    /// Human-readable name (e.g., "cloud_sync_throttle").
    pub name: String,
    /// Trigger: minimum pressure to activate.
    pub min_pressure: f32,
    /// Trigger: workload pattern (e.g., "Browser", "Build").
    pub workload_hint: String,
    /// Action: process names to throttle.
    pub throttle_targets: Vec<String>,
    /// Measured success rate [0, 1].
    pub success_rate: f32,
    /// Total times applied.
    pub apply_count: u32,
    /// Total times pressure dropped after applying.
    pub success_count: u32,
}

impl OptimizationSkill {
    /// Record an application result.
    pub fn record(&mut self, was_effective: bool) {
        self.apply_count += 1;
        if was_effective {
            self.success_count += 1;
        }
        self.success_rate = self.success_count as f32 / self.apply_count.max(1) as f32;
    }

    /// Is this skill reliable? (≥5 applications, ≥60% success rate)
    pub fn is_reliable(&self) -> bool {
        self.apply_count >= 5 && self.success_rate >= 0.60
    }

    /// Should this skill be retired? (≥10 applications, <35% success rate)
    /// 35% threshold: skills in the 20-35% range are barely above noise and
    /// consume a slot that a better skill could occupy. The reliable threshold
    /// is 60%, so anything below 35% is clearly not useful in practice.
    pub fn should_retire(&self) -> bool {
        self.apply_count >= 10 && self.success_rate < 0.35
    }
}

/// Registry of learned optimization skills.
pub struct SkillRegistry {
    skills: HashMap<String, OptimizationSkill>,
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self {
            skills: HashMap::new(),
        }
    }

    /// Learn a new skill from observed effective throttle patterns.
    pub fn learn(&mut self, name: &str, pressure: f32, workload: &str, targets: Vec<String>) {
        let skill = self.skills.entry(name.to_string()).or_insert_with(|| {
            OptimizationSkill {
                name: name.to_string(),
                min_pressure: pressure,
                workload_hint: workload.to_string(),
                throttle_targets: targets.clone(),
                success_rate: 0.5,
                apply_count: 0,
                success_count: 0,
            }
        });
        // Update pressure threshold with EMA.
        skill.min_pressure = skill.min_pressure * 0.9 + pressure * 0.1;
    }

    /// Record result for a skill.
    pub fn record_result(&mut self, name: &str, was_effective: bool) {
        if let Some(skill) = self.skills.get_mut(name) {
            skill.record(was_effective);
        }
    }

    /// Get reliable skills matching current conditions.
    pub fn matching_skills(&self, pressure: f32, workload: &str) -> Vec<&OptimizationSkill> {
        self.skills
            .values()
            .filter(|s| {
                s.is_reliable()
                    && pressure >= s.min_pressure
                    && (s.workload_hint == workload || s.workload_hint == "any")
            })
            .collect()
    }

    /// Pick one unproven induced skill to trial this cycle.
    ///
    /// Induced skills (group:/batch:) start with apply_count=0 and can never
    /// reach is_reliable() without being applied first. This method returns
    /// the single highest-priority unproven skill at current pressure so it
    /// gets one real observation per call. Caller records the result via
    /// record_result() after measuring pressure delta.
    ///
    /// Criteria: apply_count < 5, pressure >= min_pressure, not retired.
    /// Priority: fewest applications first (round-robin exploration).
    pub fn next_trial_skill(&self, pressure: f32) -> Option<&OptimizationSkill> {
        self.skills
            .values()
            .filter(|s| {
                s.apply_count < 10
                    && !s.should_retire()
                    && pressure >= s.min_pressure
                    && (s.name.starts_with("group:") || s.name.starts_with("batch:"))
            })
            .min_by_key(|s| s.apply_count)
    }

    /// Remove induced skills whose ALL throttle targets are protected.
    /// These skills can never execute and would spin forever in the trial loop.
    /// `protected` is the combined hard + policy protected set.
    pub fn purge_unexecutable(&mut self, protected: &[&str]) {
        self.skills.retain(|name, skill| {
            if !name.starts_with("group:") && !name.starts_with("batch:") {
                return true; // keep individual skills unconditionally
            }
            // Keep if at least one target is NOT protected.
            skill.throttle_targets.iter().any(|target| {
                let tl = target.to_ascii_lowercase();
                !protected.iter().any(|p| tl.contains(&p.to_ascii_lowercase()))
            })
        });
    }

    /// Register an autonomously induced skill (from rule_inducer).
    /// Skips if a skill with the same name already exists.
    /// Returns true if the skill was added.
    pub fn register_induced(&mut self, skill: OptimizationSkill) -> bool {
        if self.skills.contains_key(&skill.name) {
            return false;
        }
        self.skills.insert(skill.name.clone(), skill);
        true
    }

    /// Snapshot of all skill names (for duplicate checking in rule_inducer).
    pub fn name_set(&self) -> std::collections::HashSet<String> {
        self.skills.keys().cloned().collect()
    }

    /// Retire ineffective skills.
    pub fn gc(&mut self) {
        self.skills.retain(|_, s| !s.should_retire());
    }

    /// Number of skills.
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Number of reliable skills.
    pub fn reliable_count(&self) -> usize {
        self.skills.values().filter(|s| s.is_reliable()).count()
    }

    /// Persist to disk.
    pub fn persist(&self, path: &Path) {
        if let Ok(json) = serde_json::to_string(&self.skills) {
            let _ = std::fs::write(path, json);
        }
    }

    /// Load from disk.
    pub fn load(&mut self, path: &Path) {
        if let Ok(data) = std::fs::read_to_string(path) {
            if let Ok(skills) = serde_json::from_str::<HashMap<String, OptimizationSkill>>(&data) {
                self.skills = skills;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_reliability() {
        let mut s = OptimizationSkill {
            name: "test".into(),
            min_pressure: 0.70,
            workload_hint: "any".into(),
            throttle_targets: vec!["Dropbox".into()],
            success_rate: 0.0,
            apply_count: 0,
            success_count: 0,
        };
        for _ in 0..8 {
            s.record(true);
        }
        for _ in 0..2 {
            s.record(false);
        }
        assert!(s.is_reliable()); // 80% success
        assert!(!s.should_retire());
    }

    #[test]
    fn test_skill_retirement() {
        let mut s = OptimizationSkill {
            name: "bad".into(),
            min_pressure: 0.50,
            workload_hint: "any".into(),
            throttle_targets: vec![],
            success_rate: 0.0,
            apply_count: 0,
            success_count: 0,
        };
        for _ in 0..10 {
            s.record(false);
        }
        assert!(s.should_retire());
    }

    #[test]
    fn test_registry_learn_and_match() {
        let mut reg = SkillRegistry::new();
        reg.learn("cloud_throttle", 0.70, "any", vec!["Dropbox".into()]);
        for _ in 0..10 {
            reg.record_result("cloud_throttle", true);
        }
        let matches = reg.matching_skills(0.75, "any");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].name, "cloud_throttle");
    }

    #[test]
    fn test_registry_gc() {
        let mut reg = SkillRegistry::new();
        reg.learn("bad_skill", 0.50, "any", vec![]);
        for _ in 0..15 {
            reg.record_result("bad_skill", false);
        }
        reg.gc();
        assert_eq!(reg.len(), 0);
    }
}
