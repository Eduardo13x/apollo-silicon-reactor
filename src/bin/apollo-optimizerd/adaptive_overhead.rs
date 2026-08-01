#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OverheadLevel {
    Nominal,
    Guarded,
    Constrained,
}

impl OverheadLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nominal => "nominal",
            Self::Guarded => "guarded",
            Self::Constrained => "constrained",
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OverheadInput {
    pub p95_cycle_ms: f64,
    pub reason_avg_ms: f64,
    pub memory_pressure: f64,
    pub fluidity_degraded: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverheadBudget {
    pub level: OverheadLevel,
    pub full_refresh_cadence: u64,
    pub optional_reason_budget_ms: u64,
    pub allow_speculation: bool,
}

impl OverheadBudget {
    fn for_level(level: OverheadLevel) -> Self {
        match level {
            OverheadLevel::Nominal => Self {
                level,
                full_refresh_cadence: 30,
                optional_reason_budget_ms: 150,
                allow_speculation: true,
            },
            OverheadLevel::Guarded => Self {
                level,
                full_refresh_cadence: 60,
                optional_reason_budget_ms: 100,
                allow_speculation: true,
            },
            OverheadLevel::Constrained => Self {
                level,
                full_refresh_cadence: 120,
                optional_reason_budget_ms: 60,
                allow_speculation: false,
            },
        }
    }
}

#[derive(Debug)]
pub struct AdaptiveOverheadGovernor {
    level: OverheadLevel,
    recovery_streak: u32,
}

impl Default for AdaptiveOverheadGovernor {
    fn default() -> Self {
        Self {
            level: OverheadLevel::Nominal,
            recovery_streak: 0,
        }
    }
}

impl AdaptiveOverheadGovernor {
    const RECOVERY_CYCLES: u32 = 120;

    pub fn observe(&mut self, input: OverheadInput) -> OverheadBudget {
        let requested = requested_level(input);
        if requested > self.level {
            self.level = requested;
            self.recovery_streak = 0;
        } else if requested < self.level {
            self.recovery_streak = self.recovery_streak.saturating_add(1);
            if self.recovery_streak >= Self::RECOVERY_CYCLES {
                self.level = match self.level {
                    OverheadLevel::Constrained => OverheadLevel::Guarded,
                    OverheadLevel::Guarded | OverheadLevel::Nominal => OverheadLevel::Nominal,
                };
                self.recovery_streak = 0;
            }
        } else {
            self.recovery_streak = 0;
        }
        OverheadBudget::for_level(self.level)
    }
}

fn requested_level(input: OverheadInput) -> OverheadLevel {
    if input.p95_cycle_ms >= 120.0 || input.reason_avg_ms >= 100.0 || input.memory_pressure >= 0.80
    {
        OverheadLevel::Constrained
    } else if input.p95_cycle_ms >= 60.0
        || input.reason_avg_ms >= 50.0
        || input.memory_pressure >= 0.65
        || input.fluidity_degraded
    {
        OverheadLevel::Guarded
    } else {
        OverheadLevel::Nominal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m4_steady_state_keeps_full_capability_budget() {
        let mut governor = AdaptiveOverheadGovernor::default();
        let budget = governor.observe(OverheadInput {
            p95_cycle_ms: 37.0,
            reason_avg_ms: 12.0,
            memory_pressure: 0.39,
            fluidity_degraded: false,
        });
        assert_eq!(budget.level, OverheadLevel::Nominal);
        assert_eq!(budget.full_refresh_cadence, 30);
        assert!(budget.allow_speculation);
    }

    #[test]
    fn overload_sheds_optional_work_immediately_and_recovers_slowly() {
        let mut governor = AdaptiveOverheadGovernor::default();
        let overloaded = governor.observe(OverheadInput {
            p95_cycle_ms: 140.0,
            ..OverheadInput::default()
        });
        assert_eq!(overloaded.level, OverheadLevel::Constrained);
        assert!(!overloaded.allow_speculation);
        assert_eq!(overloaded.full_refresh_cadence, 120);

        let healthy = OverheadInput {
            p95_cycle_ms: 20.0,
            ..OverheadInput::default()
        };
        for _ in 0..(AdaptiveOverheadGovernor::RECOVERY_CYCLES - 1) {
            assert_eq!(governor.observe(healthy).level, OverheadLevel::Constrained);
        }
        assert_eq!(governor.observe(healthy).level, OverheadLevel::Guarded);
    }
}
