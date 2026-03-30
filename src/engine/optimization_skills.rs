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

    /// Should this skill be retired? (≥10 applications, <20% success rate)
    pub fn should_retire(&self) -> bool {
        self.apply_count >= 10 && self.success_rate < 0.20
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
