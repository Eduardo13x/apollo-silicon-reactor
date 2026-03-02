use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use crate::engine::types::{GovernorState, ManualOverride, OptimizationProfile, ProfileTransition};

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

    pub fn from_persisted(p: GovernorPersisted) -> Self {
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
        });
        self.transition_reason = "manual-override".to_string();
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

        if let Some(lock_until) = self.balanced_lock_until {
            if lock_until > now {
                target = OptimizationProfile::BalancedRoot;
                reason = "anti-thrash-balanced-lock".to_string();
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
                    target = self.transition_target(current, pressure_score);
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
            self.state.effective_profile = target;
            self.transition_reason = reason.clone();

            if self.transition_times.len() > 4 {
                self.balanced_lock_until = Some(now + Duration::minutes(5));
                self.state.effective_profile = OptimizationProfile::BalancedRoot;
                self.transition_reason = "anti-thrash-balanced-lock".to_string();
                Some(ProfileTransition {
                    from: target,
                    to: OptimizationProfile::BalancedRoot,
                    at: now,
                    reason: "anti-thrash-balanced-lock".to_string(),
                    pressure_score,
                })
            } else {
                Some(entry)
            }
        } else {
            self.state.effective_profile = current;
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
    ) -> OptimizationProfile {
        match current {
            OptimizationProfile::BalancedRoot => {
                if pressure_score >= 0.72 && self.state.consecutive_high >= 3 {
                    OptimizationProfile::AggressiveRoot
                } else if pressure_score <= 0.28 && self.safe_low_consecutive >= 6 {
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
    (0.35 * cpu + 0.35 * ram + 0.20 * wait + 0.10 * reactor).clamp(0.0, 1.0)
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
