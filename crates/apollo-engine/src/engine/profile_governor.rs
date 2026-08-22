use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use crate::engine::memory_regime::{MemoryRegime, MemoryRegimeEvidence};
use crate::engine::types::{
    GovernorState, ManualOverride, OptimizationProfile, OverrideOrigin, ProfileTransition,
};
use crate::engine::workload_classifier::WorkloadMode;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernorPersisted {
    pub auto_profile_enabled: bool,
    pub base_profile: OptimizationProfile,
    pub governor_state: GovernorState,
    pub manual_override: Option<ManualOverride>,
    pub balanced_lock_until: Option<DateTime<Utc>>,
    pub transition_reason: String,
    pub transition_times: Vec<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct GovernorInput {
    pub cpu_pressure: f64,
    pub ram_pressure: f64,
    pub interactive_wait_ratio: f64,
    pub reactor_event_weight: f64,
    pub thermal_constrained: bool,
    pub dev_session_active: bool,
    pub interactive_heavy: bool,
    /// Usuario cambió de app 3+ veces en los últimos 5 min → modo burst de cambio de contexto.
    pub context_switch_burst: bool,
    /// Live arousal estimate (0.0..1.0). Context-switch bursts only justify
    /// AggressiveRoot while the user is actually active.
    pub arousal_level: f64,
    /// Workload mode from Phase 3 feature-based classifier (None for backward compat).
    pub workload_mode: Option<WorkloadMode>,
    /// True when workload just transitioned INTO Build mode (cargo/rustc/swift detected).
    /// Triggers proactive AggressiveRoot before RAM pressure builds —
    /// faster than any reactive pressure-based trigger.
    pub workload_onset: bool,
    /// Capacity-normalized physical memory evidence from the previous fresh
    /// collector generation. `None` and `Unknown` cannot escalate policy.
    pub memory_evidence: Option<MemoryRegimeEvidence>,
}

#[derive(Debug, Clone)]
pub struct GovernorDecision {
    pub effective_profile: OptimizationProfile,
    pub throttle_level: String,
    pub pressure_score: f64,
    pub transition_reason: String,
    pub transition: Option<ProfileTransition>,
    pub override_expired: bool,
    pub override_active: bool,
}

#[derive(Debug, Clone)]
pub struct ProfileGovernor {
    pub auto_profile_enabled: bool,
    pub base_profile: OptimizationProfile,
    pub state: GovernorState,
    pub manual_override: Option<ManualOverride>,
    pub balanced_lock_until: Option<DateTime<Utc>>,
    pub transition_reason: String,
    transition_times: Vec<DateTime<Utc>>,
    safe_low_consecutive: u32,
    safe_rise_consecutive: u32,
}

impl ProfileGovernor {
    pub fn new(base_profile: OptimizationProfile) -> Self {
        Self {
            auto_profile_enabled: true,
            base_profile,
            state: GovernorState {
                effective_profile: base_profile,
                cooldown_until: None,
                consecutive_high: 0,
                consecutive_low: 0,
            },
            manual_override: None,
            balanced_lock_until: None,
            transition_reason: "startup".to_string(),
            transition_times: Vec::new(),
            safe_low_consecutive: 0,
            safe_rise_consecutive: 0,
        }
    }

    pub fn from_persisted(mut p: GovernorPersisted) -> Self {
        // Startup sanity: clear stale persisted state that would otherwise
        // silently block auto-management.
        //
        // 1. Expired manual_override — if the override deadline already passed,
        //    it has no semantic effect at decide-time but (a) confuses operators
        //    inspecting governor_state.json and (b) can leave auto_profile_enabled
        //    disabled under the mistaken assumption that the override is still
        //    steering. Clear it.
        // 2. Ancient cooldown_until — prod observation 2026-04-16: cooldown_until
        //    was frozen at 2026-04-07 (9 days stale). Harmless at runtime
        //    (in_cooldown compares to now) but pollutes diagnostics. Clear any
        //    cooldown whose deadline already passed.
        // 3. auto_profile_enabled re-enable — if auto was disabled solely because
        //    a manual override was taking over, clearing the override means we
        //    should resume auto management. Re-enable unless the persisted state
        //    was explicitly without a pending override when disabled (i.e. user
        //    asked for manual mode deliberately).
        let now = Utc::now();
        let had_expired_override = p
            .manual_override
            .as_ref()
            .map(|o| o.expires_at <= now)
            .unwrap_or(false);
        if had_expired_override {
            p.manual_override = None;
            if !p.auto_profile_enabled {
                // Auto was off solely because the override was active; resume auto.
                p.auto_profile_enabled = true;
            }
        }
        if p.manual_override
            .as_ref()
            .is_some_and(|active| active.origin == OverrideOrigin::Predictive)
        {
            p.manual_override = None;
            p.transition_reason = "predictive-override-reset-on-startup".to_string();
        }
        if let Some(t) = p.governor_state.cooldown_until {
            if t <= now {
                p.governor_state.cooldown_until = None;
            }
        }

        Self {
            auto_profile_enabled: p.auto_profile_enabled,
            base_profile: p.base_profile,
            state: p.governor_state,
            manual_override: p.manual_override,
            balanced_lock_until: p.balanced_lock_until,
            transition_reason: p.transition_reason,
            transition_times: p.transition_times,
            safe_low_consecutive: 0,
            safe_rise_consecutive: 0,
        }
    }

    pub fn to_persisted(&self) -> GovernorPersisted {
        GovernorPersisted {
            auto_profile_enabled: self.auto_profile_enabled,
            base_profile: self.base_profile,
            governor_state: self.state.clone(),
            manual_override: self.manual_override.clone(),
            balanced_lock_until: self.balanced_lock_until,
            transition_reason: self.transition_reason.clone(),
            transition_times: self.transition_times.clone(),
        }
    }

    pub fn set_manual_override(
        &mut self,
        profile: OptimizationProfile,
        ttl_minutes: u64,
        reason: String,
    ) {
        self.manual_override = Some(ManualOverride {
            profile,
            expires_at: Utc::now() + Duration::minutes(ttl_minutes as i64),
            reason,
            origin: OverrideOrigin::Operator,
        });
        self.transition_reason = "manual-override".to_string();
    }

    pub fn set_predictive_override(
        &mut self,
        profile: OptimizationProfile,
        ttl_seconds: i64,
        reason: String,
    ) -> bool {
        if self
            .manual_override
            .as_ref()
            .is_some_and(|active| active.origin == OverrideOrigin::Operator)
        {
            return false;
        }
        self.manual_override = Some(ManualOverride {
            profile,
            expires_at: Utc::now() + Duration::seconds(ttl_seconds.clamp(1, 30)),
            reason,
            origin: OverrideOrigin::Predictive,
        });
        self.transition_reason = "predictive-override".to_string();
        true
    }

    pub fn clear_predictive_override(&mut self) -> bool {
        if self
            .manual_override
            .as_ref()
            .is_some_and(|active| active.origin == OverrideOrigin::Predictive)
        {
            self.manual_override = None;
            self.transition_reason = "predictive-override-released".to_string();
            true
        } else {
            false
        }
    }

    pub fn clear_manual_override(&mut self) {
        self.manual_override = None;
        self.transition_reason = "manual-override-cleared".to_string();
    }

    pub fn set_auto_profile(&mut self, enabled: bool) {
        self.auto_profile_enabled = enabled;
        self.transition_reason = if enabled {
            "auto-profile-enabled".to_string()
        } else {
            "auto-profile-disabled".to_string()
        };
    }

    pub fn force_safe_on_errors(&mut self) {
        self.state.effective_profile = OptimizationProfile::SafeRoot;
        self.state.cooldown_until = Some(Utc::now() + Duration::seconds(90));
        self.transition_reason = "critical-errors-temporary-safe".to_string();
    }

    pub fn evaluate(&mut self, input: GovernorInput) -> GovernorDecision {
        let now = Utc::now();
        let pressure_score = pressure_score(&input);
        let mut override_expired = false;

        if let Some(ov) = &self.manual_override {
            if ov.expires_at <= now {
                self.manual_override = None;
                override_expired = true;
                self.transition_reason = "manual-override-expired".to_string();
            }
        }

        if let Some(ov) = &self.manual_override {
            self.state.effective_profile = ov.profile;
            return GovernorDecision {
                effective_profile: ov.profile,
                throttle_level: throttle_level(pressure_score),
                pressure_score,
                transition_reason: self.transition_reason.clone(),
                transition: None,
                override_expired,
                override_active: true,
            };
        }

        if !self.auto_profile_enabled {
            self.state.effective_profile = self.base_profile;
            return GovernorDecision {
                effective_profile: self.base_profile,
                throttle_level: throttle_level(pressure_score),
                pressure_score,
                transition_reason: self.transition_reason.clone(),
                transition: None,
                override_expired,
                override_active: false,
            };
        }

        let current = self.state.effective_profile;
        let mut target = current;
        let mut reason = "steady".to_string();

        self.transition_times
            .retain(|t| *t >= now - Duration::minutes(10));

        let mem_thrash_crisis = input
            .memory_evidence
            .as_ref()
            .is_some_and(MemoryRegimeEvidence::physically_correlated_crisis);

        let balanced_lock_active = self
            .balanced_lock_until
            .is_some_and(|lock_until| lock_until > now);
        if let Some(lock_until) = self.balanced_lock_until {
            if lock_until > now && !mem_thrash_crisis {
                // Anti-thrash lock active — hold balanced unless the machine is genuinely
                // thrashing (high RAM + high swap).  In a real memory crisis the lock
                // was protecting against oscillation, not against a sustained emergency.
                target = OptimizationProfile::BalancedRoot;
                reason = "anti-thrash-balanced-lock".to_string();
                self.transition_reason = reason.clone();
            } else {
                self.balanced_lock_until = None;
            }
        }

        if target == current {
            if input.thermal_constrained {
                if current == OptimizationProfile::AggressiveRoot {
                    target = OptimizationProfile::BalancedRoot;
                    reason = "thermal-cap-immediate".to_string();
                }
            } else {
                self.accumulate_counters(pressure_score);
                if !self.in_cooldown(now) {
                    target = self.transition_target(current, pressure_score, input.workload_mode);
                    if target != current {
                        reason = format!(
                            "pressure {:.2} transition {} -> {}",
                            pressure_score,
                            current.as_str(),
                            target.as_str()
                        );
                    }
                }
            }
        }

        if input.thermal_constrained && target == OptimizationProfile::AggressiveRoot {
            target = OptimizationProfile::BalancedRoot;
            reason = "thermal-cap".to_string();
        }

        // Workload-onset proactive boost: jump to AggressiveRoot the moment a heavy build
        // starts, before pressure climbs. Pre-freezes backgrounds BEFORE the compiler needs
        // the RAM rather than reacting after swap starts. Skip if thermal is constrained.
        if input.workload_onset
            && !input.thermal_constrained
            && (!balanced_lock_active || mem_thrash_crisis)
            && target != OptimizationProfile::AggressiveRoot
        {
            target = OptimizationProfile::AggressiveRoot;
            reason = "build-onset-proactive".to_string();
        }

        // Dev/interactive floor: don't drop to safe-root when the user is actively developing
        // or running heavy interactive workloads; safe-root feels like "less punch".
        if (input.dev_session_active || input.interactive_heavy)
            && target == OptimizationProfile::SafeRoot
        {
            target = OptimizationProfile::BalancedRoot;
            reason = if input.dev_session_active {
                "dev-floor balanced-root".to_string()
            } else {
                "interactive-floor balanced-root".to_string()
            };
        }

        // Context-switch burst: usuario cambia de app frecuentemente (TDA-aware).
        // Empuja hacia aggressive-root para maximizar la respuesta del app en primer plano.
        // Cede ante thermal-constrained, ante el mecanismo anti-thrash (balanced_lock_until),
        // y ante presión de RAM alta (>70%): con muchas ventanas y RAM alta, subir a
        // aggressive causaría más freezes/throttles justo cuando el sistema más lo necesita.
        if input.context_switch_burst
            && input.arousal_level >= 0.25
            && !input.thermal_constrained
            && input.ram_pressure < 0.70
            && self.balanced_lock_until.is_none_or(|t| t <= now)
            && target != OptimizationProfile::AggressiveRoot
        {
            target = OptimizationProfile::AggressiveRoot;
            reason = "context-switch-burst".to_string();
        } else if input.context_switch_burst
            && input.arousal_level < 0.25
            && !input.thermal_constrained
            && input.ram_pressure < 0.70
            && self.balanced_lock_until.is_none_or(|t| t <= now)
            && target != OptimizationProfile::AggressiveRoot
        {
            reason = "context-switch-burst-suppressed-calm".to_string();
        }

        let transition = if target != current {
            self.state.cooldown_until = Some(now + Duration::seconds(90));
            let entry = ProfileTransition {
                from: current,
                to: target,
                at: now,
                reason: reason.clone(),
                pressure_score,
            };
            self.transition_times.push(now);

            if self.transition_times.len() > 4 {
                self.balanced_lock_until = Some(now + Duration::minutes(5));
                // Start a fresh observation window after the lock. Keeping the
                // triggering timestamps made the governor immediately relock
                // when the five-minute hold expired.
                self.transition_times.clear();
                self.state.effective_profile = OptimizationProfile::BalancedRoot;
                self.transition_reason = "anti-thrash-balanced-lock".to_string();
                // If Balanced was already effective, the proposed escalation
                // was suppressed; reporting target -> Balanced fabricated a
                // switch on every cycle even though the machine never moved.
                (current != OptimizationProfile::BalancedRoot).then_some(ProfileTransition {
                    from: current,
                    to: OptimizationProfile::BalancedRoot,
                    at: now,
                    reason: "anti-thrash-balanced-lock".to_string(),
                    pressure_score,
                })
            } else {
                self.state.effective_profile = target;
                self.transition_reason = reason.clone();
                Some(entry)
            }
        } else {
            self.state.effective_profile = current;
            self.transition_reason = reason.clone();
            None
        };

        GovernorDecision {
            effective_profile: self.state.effective_profile,
            throttle_level: throttle_level(pressure_score),
            pressure_score,
            transition_reason: self.transition_reason.clone(),
            transition,
            override_expired,
            override_active: false,
        }
    }

    fn in_cooldown(&self, now: DateTime<Utc>) -> bool {
        self.state.cooldown_until.map(|t| t > now).unwrap_or(false)
    }

    fn accumulate_counters(&mut self, pressure_score: f64) {
        if pressure_score >= 0.72 {
            self.state.consecutive_high += 1;
        } else {
            self.state.consecutive_high = 0;
        }

        if pressure_score <= 0.55 {
            self.state.consecutive_low += 1;
        } else {
            self.state.consecutive_low = 0;
        }

        if pressure_score <= 0.28 {
            self.safe_low_consecutive += 1;
        } else {
            self.safe_low_consecutive = 0;
        }

        if pressure_score >= 0.40 {
            self.safe_rise_consecutive += 1;
        } else {
            self.safe_rise_consecutive = 0;
        }
    }

    fn transition_target(
        &self,
        current: OptimizationProfile,
        pressure_score: f64,
        workload_mode: Option<WorkloadMode>,
    ) -> OptimizationProfile {
        let consecutive_high_threshold = match workload_mode {
            Some(WorkloadMode::Build) => 2,
            _ => 3,
        };
        let safe_low_threshold = match workload_mode {
            Some(WorkloadMode::Idle) => 4,
            _ => 6,
        };

        match current {
            OptimizationProfile::BalancedRoot => {
                if pressure_score >= 0.72
                    && self.state.consecutive_high >= consecutive_high_threshold
                {
                    OptimizationProfile::AggressiveRoot
                } else if pressure_score <= 0.28 && self.safe_low_consecutive >= safe_low_threshold
                {
                    OptimizationProfile::SafeRoot
                } else {
                    current
                }
            }
            OptimizationProfile::AggressiveRoot => {
                if pressure_score <= 0.55 && self.state.consecutive_low >= 6 {
                    OptimizationProfile::BalancedRoot
                } else {
                    current
                }
            }
            OptimizationProfile::SafeRoot => {
                if pressure_score >= 0.40 && self.safe_rise_consecutive >= 3 {
                    OptimizationProfile::BalancedRoot
                } else {
                    current
                }
            }
        }
    }
}

fn pressure_score(input: &GovernorInput) -> f64 {
    let cpu = input.cpu_pressure.clamp(0.0, 1.0);
    let ram = input.ram_pressure.clamp(0.0, 1.0);
    let wait = input.interactive_wait_ratio.clamp(0.0, 1.0);
    let reactor = input.reactor_event_weight.clamp(0.0, 1.0);
    let memory = input.memory_evidence.as_ref().filter(|evidence| {
        evidence.confidence > 0.0
            && evidence.sustained
            && evidence.adverse_flow
            && evidence.regime != MemoryRegime::Unknown
    });
    let memory_boost = memory
        .map(|evidence| evidence.normalized_score.clamp(0.0, 1.0) * 0.12)
        .unwrap_or(0.0);
    let base =
        (0.35 * cpu + 0.35 * ram + 0.20 * wait + 0.10 * reactor + memory_boost).clamp(0.0, 1.0);

    match memory.map(|evidence| evidence.regime) {
        Some(MemoryRegime::Crisis) => base.max(0.75),
        Some(MemoryRegime::Contended) => base.max(0.60),
        _ => base,
    }
}

fn throttle_level(pressure_score: f64) -> String {
    if pressure_score >= 0.72 {
        "high".to_string()
    } else if pressure_score >= 0.40 {
        "medium".to_string()
    } else {
        "low".to_string()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::types::OptimizationProfile;

    fn low_pressure_input(workload_onset: bool) -> GovernorInput {
        GovernorInput {
            cpu_pressure: 0.20,
            ram_pressure: 0.30,
            interactive_wait_ratio: 0.10,
            reactor_event_weight: 0.0,
            thermal_constrained: false,
            dev_session_active: false,
            interactive_heavy: false,
            context_switch_burst: false,
            arousal_level: 0.0,
            workload_mode: None,
            workload_onset,
            memory_evidence: None,
        }
    }

    fn make_governor(profile: OptimizationProfile) -> ProfileGovernor {
        ProfileGovernor::new(profile)
    }

    #[test]
    fn workload_onset_jumps_to_aggressive_at_low_pressure() {
        // Feature 3: governor should proactively jump to AggressiveRoot
        // when a build starts — without waiting for pressure to climb.
        let mut gov = make_governor(OptimizationProfile::BalancedRoot);
        let decision = gov.evaluate(low_pressure_input(true));
        assert_eq!(
            decision.effective_profile,
            OptimizationProfile::AggressiveRoot,
            "build-onset should jump to AggressiveRoot immediately; got {:?}",
            decision.effective_profile
        );
        assert!(
            decision.transition_reason.contains("build-onset"),
            "reason should mention build-onset; got '{}'",
            decision.transition_reason
        );
    }

    #[test]
    fn workload_onset_blocked_by_thermal() {
        // Thermal cap must override onset — we can't heat up more during a thermal event.
        let mut gov = make_governor(OptimizationProfile::BalancedRoot);
        let mut input = low_pressure_input(true);
        input.thermal_constrained = true;
        let decision = gov.evaluate(input);
        assert_ne!(
            decision.effective_profile,
            OptimizationProfile::AggressiveRoot,
            "thermal constraint must block onset boost"
        );
    }

    #[test]
    fn no_onset_flag_does_not_boost() {
        // Without onset, low-pressure BalancedRoot should stay BalancedRoot.
        let mut gov = make_governor(OptimizationProfile::BalancedRoot);
        let decision = gov.evaluate(low_pressure_input(false));
        assert_eq!(
            decision.effective_profile,
            OptimizationProfile::BalancedRoot,
            "without onset flag, profile should not change at low pressure"
        );
    }

    #[test]
    fn onset_noop_when_already_aggressive() {
        // If already in AggressiveRoot, onset should be a no-op.
        let mut gov = make_governor(OptimizationProfile::AggressiveRoot);
        let decision = gov.evaluate(low_pressure_input(true));
        assert_eq!(
            decision.effective_profile,
            OptimizationProfile::AggressiveRoot
        );
    }

    #[test]
    fn calm_context_switch_burst_does_not_escalate() {
        let mut gov = make_governor(OptimizationProfile::BalancedRoot);
        let mut input = low_pressure_input(false);
        input.context_switch_burst = true;
        input.arousal_level = 0.15;

        let decision = gov.evaluate(input);

        assert_eq!(
            decision.effective_profile,
            OptimizationProfile::BalancedRoot
        );
        assert_eq!(
            decision.transition_reason,
            "context-switch-burst-suppressed-calm"
        );
    }

    #[test]
    fn active_context_switch_burst_escalates() {
        let mut gov = make_governor(OptimizationProfile::BalancedRoot);
        let mut input = low_pressure_input(false);
        input.context_switch_burst = true;
        input.arousal_level = 0.60;

        let decision = gov.evaluate(input);

        assert_eq!(
            decision.effective_profile,
            OptimizationProfile::AggressiveRoot
        );
        assert_eq!(decision.transition_reason, "context-switch-burst");
    }

    #[test]
    fn calm_burst_releases_aggressive_after_low_pressure_hysteresis() {
        let mut gov = make_governor(OptimizationProfile::AggressiveRoot);
        let mut input = low_pressure_input(false);
        input.context_switch_burst = true;
        input.arousal_level = 0.15;

        let mut decision = gov.evaluate(input.clone());
        for _ in 1..6 {
            decision = gov.evaluate(input.clone());
        }

        assert_eq!(
            decision.effective_profile,
            OptimizationProfile::BalancedRoot
        );
        assert_eq!(
            decision.transition_reason,
            "context-switch-burst-suppressed-calm"
        );
    }

    #[test]
    fn calm_burst_does_not_hide_build_onset_reason() {
        let mut gov = make_governor(OptimizationProfile::BalancedRoot);
        let mut input = low_pressure_input(true);
        input.context_switch_burst = true;
        input.arousal_level = 0.15;

        let decision = gov.evaluate(input);

        assert_eq!(
            decision.effective_profile,
            OptimizationProfile::AggressiveRoot
        );
        assert_eq!(decision.transition_reason, "build-onset-proactive");
    }

    #[test]
    fn calm_burst_does_not_hide_thermal_release_reason() {
        let mut gov = make_governor(OptimizationProfile::AggressiveRoot);
        let mut input = low_pressure_input(false);
        input.context_switch_burst = true;
        input.arousal_level = 0.15;
        input.thermal_constrained = true;

        let decision = gov.evaluate(input);

        assert_eq!(
            decision.effective_profile,
            OptimizationProfile::BalancedRoot
        );
        assert_eq!(decision.transition_reason, "thermal-cap-immediate");
    }

    #[test]
    fn memory_thrash_crisis_score_clears_aggressive_threshold() {
        // Sustained, capacity-normalized crisis must clear the aggressive threshold.
        let input = GovernorInput {
            cpu_pressure: 0.05,
            ram_pressure: 0.67,
            interactive_wait_ratio: 0.0,
            reactor_event_weight: 0.0,
            thermal_constrained: false,
            dev_session_active: false,
            interactive_heavy: false,
            context_switch_burst: false,
            arousal_level: 0.0,
            workload_mode: None,
            workload_onset: false,
            memory_evidence: Some(MemoryRegimeEvidence {
                regime: MemoryRegime::Crisis,
                generation: 1,
                confidence: 1.0,
                normalized_score: 0.85,
                swap_fraction_of_ram: 0.31,
                swap_growth_fraction_per_minute: 0.01,
                adverse_flow: true,
                sustained: true,
            }),
        };
        let score = pressure_score(&input);
        assert!(
            score >= 0.72,
            "memory thrash crisis should yield score >= 0.72, got {:.3}",
            score
        );
    }

    #[test]
    fn memory_thrash_bypasses_anti_thrash_lock() {
        // Even if balanced_lock is active, a genuine memory crisis should be able
        // to escalate to AggressiveRoot.
        let mut gov = make_governor(OptimizationProfile::BalancedRoot);
        gov.balanced_lock_until = Some(Utc::now() + Duration::minutes(5));
        gov.state.consecutive_high = 2;
        // Now evaluate with memory crisis — should break the lock
        let crisis_input = GovernorInput {
            cpu_pressure: 0.05,
            ram_pressure: 0.70,
            interactive_wait_ratio: 0.0,
            reactor_event_weight: 0.0,
            thermal_constrained: false,
            dev_session_active: false,
            interactive_heavy: false,
            context_switch_burst: false,
            arousal_level: 0.0,
            workload_mode: None,
            workload_onset: false,
            memory_evidence: Some(MemoryRegimeEvidence {
                regime: MemoryRegime::Crisis,
                generation: 1,
                confidence: 1.0,
                normalized_score: 0.85,
                swap_fraction_of_ram: 0.31,
                swap_growth_fraction_per_minute: 0.01,
                adverse_flow: true,
                sustained: true,
            }),
        };
        let decision = gov.evaluate(crisis_input);
        assert_eq!(
            decision.effective_profile,
            OptimizationProfile::AggressiveRoot,
            "memory thrash crisis must bypass anti-thrash balanced_lock; got {:?} (reason: {})",
            decision.effective_profile,
            decision.transition_reason
        );
    }

    #[test]
    fn unknown_memory_evidence_cannot_escalate_policy() {
        let mut input = low_pressure_input(false);
        input.memory_evidence = Some(MemoryRegimeEvidence::default());

        assert!(pressure_score(&input) < 0.40);
    }

    #[test]
    fn predictive_override_cannot_replace_operator_override() {
        let mut gov = make_governor(OptimizationProfile::BalancedRoot);
        gov.set_manual_override(OptimizationProfile::SafeRoot, 5, "operator".to_string());

        assert!(!gov.set_predictive_override(
            OptimizationProfile::AggressiveRoot,
            30,
            "predictive".to_string(),
        ));
        assert_eq!(
            gov.manual_override.as_ref().map(|active| active.origin),
            Some(OverrideOrigin::Operator)
        );
        assert_eq!(
            gov.manual_override.as_ref().map(|active| active.profile),
            Some(OptimizationProfile::SafeRoot)
        );
    }

    #[test]
    fn predictive_override_releases_without_touching_operator_override() {
        let mut gov = make_governor(OptimizationProfile::BalancedRoot);
        assert!(gov.set_predictive_override(
            OptimizationProfile::AggressiveRoot,
            30,
            "predictive".to_string(),
        ));
        assert!(gov.clear_predictive_override());
        assert!(gov.manual_override.is_none());

        gov.set_manual_override(OptimizationProfile::SafeRoot, 5, "operator".to_string());
        assert!(!gov.clear_predictive_override());
        assert!(gov.manual_override.is_some());
    }

    #[test]
    fn legacy_override_without_origin_defaults_to_operator() {
        let override_state = ManualOverride {
            profile: OptimizationProfile::BalancedRoot,
            expires_at: Utc::now() + Duration::minutes(5),
            reason: "legacy".to_string(),
            origin: OverrideOrigin::Operator,
        };
        let mut json = serde_json::to_value(override_state).unwrap();
        json.as_object_mut().unwrap().remove("origin");

        let restored: ManualOverride = serde_json::from_value(json).unwrap();

        assert_eq!(restored.origin, OverrideOrigin::Operator);
    }

    #[test]
    fn unsustained_memory_burst_cannot_bypass_anti_thrash_lock() {
        let mut gov = make_governor(OptimizationProfile::BalancedRoot);
        gov.balanced_lock_until = Some(Utc::now() + Duration::minutes(5));
        let mut input = low_pressure_input(false);
        input.memory_evidence = Some(MemoryRegimeEvidence {
            regime: MemoryRegime::Crisis,
            generation: 1,
            confidence: 1.0,
            normalized_score: 1.0,
            swap_fraction_of_ram: 0.60,
            swap_growth_fraction_per_minute: 0.20,
            adverse_flow: true,
            sustained: false,
        });

        let decision = gov.evaluate(input);

        assert_eq!(
            decision.effective_profile,
            OptimizationProfile::BalancedRoot
        );
        assert_eq!(decision.transition_reason, "anti-thrash-balanced-lock");
    }

    #[test]
    fn unsustained_crisis_cannot_raise_pressure_score_without_a_lock() {
        let mut input = low_pressure_input(false);
        input.memory_evidence = Some(MemoryRegimeEvidence {
            regime: MemoryRegime::Crisis,
            generation: 1,
            confidence: 1.0,
            normalized_score: 1.0,
            swap_fraction_of_ram: 0.60,
            swap_growth_fraction_per_minute: 0.20,
            adverse_flow: true,
            sustained: false,
        });

        assert!(pressure_score(&input) < 0.40);
    }

    #[test]
    fn restore_discards_predictive_override_but_preserves_operator_override() {
        let mut predictive = make_governor(OptimizationProfile::BalancedRoot);
        predictive.set_predictive_override(
            OptimizationProfile::AggressiveRoot,
            30,
            "predictive".to_string(),
        );
        let restored = ProfileGovernor::from_persisted(predictive.to_persisted());
        assert!(restored.manual_override.is_none());

        let mut operator = make_governor(OptimizationProfile::BalancedRoot);
        operator.set_manual_override(OptimizationProfile::SafeRoot, 5, "operator".to_string());
        let restored = ProfileGovernor::from_persisted(operator.to_persisted());
        assert_eq!(
            restored.manual_override.as_ref().map(|value| value.origin),
            Some(OverrideOrigin::Operator)
        );
    }

    #[test]
    fn anti_thrash_lock_blocks_latched_onset_without_fake_transition() {
        let mut gov = make_governor(OptimizationProfile::BalancedRoot);
        gov.balanced_lock_until = Some(Utc::now() + Duration::minutes(5));

        let decision = gov.evaluate(low_pressure_input(true));

        assert_eq!(
            decision.effective_profile,
            OptimizationProfile::BalancedRoot
        );
        assert!(decision.transition.is_none());
        assert_eq!(decision.transition_reason, "anti-thrash-balanced-lock");
    }

    #[test]
    fn anti_thrash_activation_does_not_report_suppressed_target_as_switch() {
        let mut gov = make_governor(OptimizationProfile::BalancedRoot);
        gov.transition_times = vec![Utc::now(); 4];

        let decision = gov.evaluate(low_pressure_input(true));

        assert_eq!(
            decision.effective_profile,
            OptimizationProfile::BalancedRoot
        );
        assert!(decision.transition.is_none());
        assert!(gov.transition_times.is_empty());
        assert!(gov.balanced_lock_until.is_some());
    }
}
