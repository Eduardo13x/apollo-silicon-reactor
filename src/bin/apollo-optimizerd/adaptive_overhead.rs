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
    pub pressure_sample_stale: bool,
    pub fluidity_degraded: bool,
    pub realtime_media_active: bool,
    pub media_output_active: bool,
    pub interaction_q: f64,
    pub cpu_max_busy: f64,
    pub gpu_render_load: f64,
    pub hardware_sample_stale: bool,
    pub compositor_cpu_pct: f64,
    pub predicted_fluidity_3s: f64,
    pub swap_delta_bps: f64,
    pub thrashing_score: f64,
    pub refault_delta_per_sec: f64,
    pub vm_page_size_bytes: u64,
    pub stall_fraction: f64,
    pub adaptive_stutter_guarded: bool,
    pub adaptive_stutter_constrained: bool,
    pub holt_winters_avg_ms: f64,
    pub page_reclaim_avg_ms: f64,
}

const SWAP_GROWTH_GUARDED_BPS: f64 = 512.0 * 1024.0;
const SWAP_GROWTH_CONSTRAINED_BPS: f64 = 1024.0 * 1024.0;
const THRASHING_GUARDED: f64 = 1_500.0;
const THRASHING_CONSTRAINED: f64 = 5_000.0;
const REFAULT_GUARDED_BPS: f64 = 384.0 * 1024.0 * 1024.0;
const REFAULT_CONSTRAINED_BPS: f64 = 1.25 * 1024.0 * 1024.0 * 1024.0;
const STALL_FRACTION_GUARDED: f64 = 0.15;
const STALL_FRACTION_CONSTRAINED: f64 = 0.35;
const REALTIME_PRESSURE_CONSTRAINED: f64 = 0.50;
const CPU_BUSY_GUARDED: f64 = 0.80;
const CPU_BUSY_CONSTRAINED: f64 = 0.95;
const COMPOSITOR_CPU_GUARDED_PCT: f64 = 35.0;
const COMPOSITOR_CPU_CONSTRAINED_PCT: f64 = 60.0;
const INTERACTION_ACTIVE_Q: f64 = 0.05;
const PREDICTED_FLUIDITY_GUARDED: f64 = 0.65;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionalSubsystem {
    HoltWinters,
    PageReclaim,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubsystemBudget {
    pub cadence: u64,
    pub deadline_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverheadBudget {
    pub level: OverheadLevel,
    pub full_refresh_cadence: u64,
    pub optional_reason_budget_ms: u64,
    pub allow_speculation: bool,
    pub sensor_interval_secs: u64,
    pub holt_winters: SubsystemBudget,
    pub page_reclaim: SubsystemBudget,
}

impl OverheadBudget {
    fn for_level(level: OverheadLevel, input: OverheadInput) -> Self {
        let mut budget = match level {
            OverheadLevel::Nominal => Self {
                level,
                full_refresh_cadence: 30,
                optional_reason_budget_ms: 150,
                allow_speculation: true,
                sensor_interval_secs: 3,
                holt_winters: SubsystemBudget {
                    cadence: 1,
                    deadline_ms: 150,
                },
                page_reclaim: SubsystemBudget {
                    cadence: 10,
                    deadline_ms: 170,
                },
            },
            OverheadLevel::Guarded => Self {
                level,
                full_refresh_cadence: 60,
                optional_reason_budget_ms: 100,
                allow_speculation: true,
                sensor_interval_secs: 6,
                holt_winters: SubsystemBudget {
                    cadence: 2,
                    deadline_ms: 100,
                },
                page_reclaim: SubsystemBudget {
                    cadence: 20,
                    deadline_ms: 115,
                },
            },
            OverheadLevel::Constrained => Self {
                level,
                full_refresh_cadence: 120,
                optional_reason_budget_ms: 60,
                allow_speculation: false,
                sensor_interval_secs: 9,
                holt_winters: SubsystemBudget {
                    cadence: 4,
                    deadline_ms: 60,
                },
                page_reclaim: SubsystemBudget {
                    cadence: 30,
                    deadline_ms: 75,
                },
            },
        };
        // A lane that exceeds its own expected cost pays a lower cadence even
        // when the aggregate daemon remains healthy. This prevents one noisy
        // optional subsystem from consuming every other lane's budget.
        if input.holt_winters_avg_ms > 5.0 {
            budget.holt_winters.cadence = budget.holt_winters.cadence.saturating_mul(2);
        }
        if input.page_reclaim_avg_ms > 5.0 {
            budget.page_reclaim.cadence = budget.page_reclaim.cadence.saturating_mul(2);
        }
        // Preserve foreground latency before aggregate p95 reports the loss.
        // Audio output by itself is intentionally neutral: it only contributes
        // when the compositor or compute lanes also show contention.
        if should_shed_speculation(input) {
            budget.allow_speculation = false;
        }
        budget
    }

    pub fn admits(self, subsystem: OptionalSubsystem, cycle: u64, elapsed_ms: u128) -> bool {
        let lane = match subsystem {
            OptionalSubsystem::HoltWinters => self.holt_winters,
            OptionalSubsystem::PageReclaim => self.page_reclaim,
        };
        cycle.is_multiple_of(lane.cadence.max(1)) && elapsed_ms <= lane.deadline_ms as u128
    }
}

#[derive(Debug)]
pub struct AdaptiveOverheadGovernor {
    hard: LevelHold,
    adaptive: LevelHold,
}

#[derive(Debug)]
struct LevelHold {
    level: OverheadLevel,
    recovery_streak: u32,
    recovery_started_at: Option<Instant>,
}

impl Default for LevelHold {
    fn default() -> Self {
        Self {
            level: OverheadLevel::Nominal,
            recovery_streak: 0,
            recovery_started_at: None,
        }
    }
}

impl LevelHold {
    fn observe(
        &mut self,
        requested: OverheadLevel,
        now: Instant,
        recovery_cycles: u32,
        recovery_max: Duration,
    ) {
        if requested > self.level {
            self.level = requested;
            self.recovery_streak = 0;
            self.recovery_started_at = None;
        } else if requested < self.level {
            let recovery_started_at = *self.recovery_started_at.get_or_insert(now);
            self.recovery_streak = self.recovery_streak.saturating_add(1);
            if self.recovery_streak >= recovery_cycles
                || now.saturating_duration_since(recovery_started_at) >= recovery_max
            {
                self.level = match self.level {
                    OverheadLevel::Constrained => OverheadLevel::Guarded,
                    OverheadLevel::Guarded | OverheadLevel::Nominal => OverheadLevel::Nominal,
                };
                self.recovery_streak = 0;
                self.recovery_started_at = None;
            }
        } else {
            self.recovery_streak = 0;
            self.recovery_started_at = None;
        }
    }
}

impl Default for AdaptiveOverheadGovernor {
    fn default() -> Self {
        Self {
            hard: LevelHold::default(),
            adaptive: LevelHold::default(),
        }
    }
}

impl AdaptiveOverheadGovernor {
    const HARD_RECOVERY_CYCLES: u32 = 120;
    const ADAPTIVE_RECOVERY_CYCLES: u32 = 15;
    const HARD_RECOVERY_MAX: Duration = Duration::from_secs(240);
    const ADAPTIVE_RECOVERY_MAX: Duration = Duration::from_secs(30);

    pub fn observe(&mut self, input: OverheadInput) -> OverheadBudget {
        self.observe_at(input, Instant::now())
    }

    fn observe_at(&mut self, input: OverheadInput, now: Instant) -> OverheadBudget {
        let requested = requested_level(input);
        let fixed_input = if input.pressure_sample_stale {
            OverheadInput {
                memory_pressure: 0.0,
                pressure_sample_stale: false,
                cpu_max_busy: 0.0,
                swap_delta_bps: 0.0,
                thrashing_score: 0.0,
                refault_delta_per_sec: 0.0,
                stall_fraction: 0.0,
                adaptive_stutter_guarded: false,
                adaptive_stutter_constrained: false,
                ..input
            }
        } else {
            OverheadInput {
                adaptive_stutter_guarded: false,
                adaptive_stutter_constrained: false,
                ..input
            }
        };
        let fixed_requested = requested_level(fixed_input);
        let adaptive_requested = if requested > fixed_requested {
            requested
        } else {
            OverheadLevel::Nominal
        };

        self.hard.observe(
            fixed_requested,
            now,
            Self::HARD_RECOVERY_CYCLES,
            Self::HARD_RECOVERY_MAX,
        );
        self.adaptive.observe(
            adaptive_requested,
            now,
            Self::ADAPTIVE_RECOVERY_CYCLES,
            Self::ADAPTIVE_RECOVERY_MAX,
        );

        OverheadBudget::for_level(self.hard.level.max(self.adaptive.level), input)
    }
}

fn requested_level(input: OverheadInput) -> OverheadLevel {
    let memory_flow_guarded = memory_flow_guarded(input);
    let memory_flow_constrained = memory_flow_constrained(input);
    let compute_guarded = compute_guarded(input);
    let compute_constrained = compute_constrained(input);
    let interactive_visual_contention = input.interaction_q >= INTERACTION_ACTIVE_Q
        && input.compositor_cpu_pct >= COMPOSITOR_CPU_GUARDED_PCT;
    let realtime_memory_contention = !input.pressure_sample_stale
        && input.realtime_media_active
        && (input.memory_pressure >= REALTIME_PRESSURE_CONSTRAINED || memory_flow_guarded);

    if input.p95_cycle_ms >= 120.0
        || input.reason_avg_ms >= 100.0
        || (!input.pressure_sample_stale && input.memory_pressure >= 0.80)
        || input.adaptive_stutter_constrained
        || memory_flow_constrained
        || compute_constrained
        || interactive_visual_contention
        || realtime_memory_contention
    {
        OverheadLevel::Constrained
    } else if input.p95_cycle_ms >= 60.0
        || input.reason_avg_ms >= 50.0
        || (!input.pressure_sample_stale && input.memory_pressure >= 0.65)
        || input.fluidity_degraded
        || input.pressure_sample_stale
        || input.adaptive_stutter_guarded
        || input.realtime_media_active
        || memory_flow_guarded
        || compute_guarded
    {
        OverheadLevel::Guarded
    } else {
        OverheadLevel::Nominal
    }
}

fn memory_flow_guarded(input: OverheadInput) -> bool {
    !input.pressure_sample_stale
        && (input.swap_delta_bps > SWAP_GROWTH_GUARDED_BPS
            || input.thrashing_score > THRASHING_GUARDED
            || refault_bytes_per_sec(input) > REFAULT_GUARDED_BPS
            || input.stall_fraction >= STALL_FRACTION_GUARDED)
}

fn memory_flow_constrained(input: OverheadInput) -> bool {
    !input.pressure_sample_stale
        && (input.swap_delta_bps > SWAP_GROWTH_CONSTRAINED_BPS
            || input.thrashing_score > THRASHING_CONSTRAINED
            || refault_bytes_per_sec(input) > REFAULT_CONSTRAINED_BPS
            || input.stall_fraction >= STALL_FRACTION_CONSTRAINED)
}

fn refault_bytes_per_sec(input: OverheadInput) -> f64 {
    if !input.refault_delta_per_sec.is_finite() || input.refault_delta_per_sec <= 0.0 {
        return 0.0;
    }
    let page_size = input.vm_page_size_bytes.clamp(4 * 1024, 64 * 1024) as f64;
    input.refault_delta_per_sec * page_size
}

fn predicted_fluidity_guarded(input: OverheadInput) -> bool {
    input.predicted_fluidity_3s > 0.0 && input.predicted_fluidity_3s < PREDICTED_FLUIDITY_GUARDED
}

fn compute_guarded(input: OverheadInput) -> bool {
    (!input.pressure_sample_stale && input.cpu_max_busy >= CPU_BUSY_GUARDED)
        || input.compositor_cpu_pct >= COMPOSITOR_CPU_GUARDED_PCT
        || predicted_fluidity_guarded(input)
}

fn compute_constrained(input: OverheadInput) -> bool {
    (!input.pressure_sample_stale && input.cpu_max_busy >= CPU_BUSY_CONSTRAINED)
        || input.compositor_cpu_pct >= COMPOSITOR_CPU_CONSTRAINED_PCT
}

fn should_shed_speculation(input: OverheadInput) -> bool {
    input.realtime_media_active
        || input.adaptive_stutter_guarded
        || input.adaptive_stutter_constrained
        || memory_flow_guarded(input)
        || compute_guarded(input)
        || (input.media_output_active && input.compositor_cpu_pct >= COMPOSITOR_CPU_GUARDED_PCT)
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
            ..OverheadInput::default()
        });
        assert_eq!(budget.level, OverheadLevel::Nominal);
        assert_eq!(budget.full_refresh_cadence, 30);
        assert!(budget.allow_speculation);
        assert_eq!(budget.sensor_interval_secs, 3);
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
        for _ in 0..(AdaptiveOverheadGovernor::HARD_RECOVERY_CYCLES - 1) {
            assert_eq!(governor.observe(healthy).level, OverheadLevel::Constrained);
        }
        assert_eq!(governor.observe(healthy).level, OverheadLevel::Guarded);
    }

    #[test]
    fn expensive_optional_lane_pays_lower_cadence_independently() {
        let mut governor = AdaptiveOverheadGovernor::default();
        let budget = governor.observe(OverheadInput {
            holt_winters_avg_ms: 8.0,
            page_reclaim_avg_ms: 1.0,
            ..OverheadInput::default()
        });
        assert_eq!(budget.holt_winters.cadence, 2);
        assert_eq!(budget.page_reclaim.cadence, 10);
        assert!(!budget.admits(OptionalSubsystem::HoltWinters, 1, 10));
        assert!(budget.admits(OptionalSubsystem::PageReclaim, 10, 10));
    }

    #[test]
    fn realtime_call_sheds_speculation_before_fluidity_degrades() {
        let mut governor = AdaptiveOverheadGovernor::default();
        let budget = governor.observe(OverheadInput {
            p95_cycle_ms: 35.0,
            reason_avg_ms: 12.0,
            memory_pressure: 0.35,
            realtime_media_active: true,
            ..OverheadInput::default()
        });

        assert_eq!(budget.level, OverheadLevel::Guarded);
        assert!(!budget.allow_speculation);
        assert_eq!(budget.full_refresh_cadence, 60);
    }

    #[test]
    fn swap_growth_constrains_overhead_while_level_pressure_lags() {
        let mut governor = AdaptiveOverheadGovernor::default();
        let budget = governor.observe(OverheadInput {
            p95_cycle_ms: 35.0,
            reason_avg_ms: 12.0,
            memory_pressure: 0.45,
            swap_delta_bps: 1_048_577.0,
            ..OverheadInput::default()
        });

        assert_eq!(budget.level, OverheadLevel::Constrained);
        assert!(!budget.allow_speculation);
        assert_eq!(budget.full_refresh_cadence, 120);
    }

    #[test]
    fn call_plus_early_memory_flow_constrains_before_stutter() {
        let mut governor = AdaptiveOverheadGovernor::default();
        let budget = governor.observe(OverheadInput {
            p95_cycle_ms: 35.0,
            reason_avg_ms: 12.0,
            memory_pressure: 0.50,
            realtime_media_active: true,
            swap_delta_bps: 600_000.0,
            ..OverheadInput::default()
        });

        assert_eq!(budget.level, OverheadLevel::Constrained);
        assert!(!budget.allow_speculation);
    }

    #[test]
    fn thrashing_constrains_overhead_while_level_pressure_lags() {
        let mut governor = AdaptiveOverheadGovernor::default();
        let budget = governor.observe(OverheadInput {
            p95_cycle_ms: 35.0,
            reason_avg_ms: 12.0,
            memory_pressure: 0.45,
            thrashing_score: 5_001.0,
            ..OverheadInput::default()
        });

        assert_eq!(budget.level, OverheadLevel::Constrained);
        assert!(!budget.allow_speculation);
    }

    #[test]
    fn refault_bandwidth_guards_before_level_pressure_rises() {
        let mut governor = AdaptiveOverheadGovernor::default();
        let budget = governor.observe(OverheadInput {
            memory_pressure: 0.35,
            refault_delta_per_sec: 30_000.0,
            vm_page_size_bytes: 16 * 1024,
            ..OverheadInput::default()
        });

        assert_eq!(budget.level, OverheadLevel::Guarded);
        assert!(!budget.allow_speculation);
    }

    #[test]
    fn refault_storm_constrains_on_any_page_size() {
        let mut apple_silicon = AdaptiveOverheadGovernor::default();
        let apple_budget = apple_silicon.observe(OverheadInput {
            refault_delta_per_sec: 100_000.0,
            vm_page_size_bytes: 16 * 1024,
            ..OverheadInput::default()
        });
        let mut intel = AdaptiveOverheadGovernor::default();
        let intel_budget = intel.observe(OverheadInput {
            refault_delta_per_sec: 400_000.0,
            vm_page_size_bytes: 4 * 1024,
            ..OverheadInput::default()
        });

        assert_eq!(apple_budget.level, OverheadLevel::Constrained);
        assert_eq!(intel_budget.level, OverheadLevel::Constrained);
        assert!(!apple_budget.allow_speculation);
        assert!(!intel_budget.allow_speculation);
    }

    #[test]
    fn scheduler_stalls_shed_optional_work() {
        let mut guarded = AdaptiveOverheadGovernor::default();
        assert_eq!(
            guarded
                .observe(OverheadInput {
                    stall_fraction: 0.20,
                    ..OverheadInput::default()
                })
                .level,
            OverheadLevel::Guarded
        );

        let mut constrained = AdaptiveOverheadGovernor::default();
        let budget = constrained.observe(OverheadInput {
            stall_fraction: 0.40,
            ..OverheadInput::default()
        });
        assert_eq!(budget.level, OverheadLevel::Constrained);
        assert!(!budget.allow_speculation);
    }

    #[test]
    fn quiet_memory_flow_remains_nominal() {
        let mut governor = AdaptiveOverheadGovernor::default();
        let budget = governor.observe(OverheadInput {
            refault_delta_per_sec: 2_000.0,
            vm_page_size_bytes: 16 * 1024,
            stall_fraction: 0.02,
            ..OverheadInput::default()
        });

        assert_eq!(budget.level, OverheadLevel::Nominal);
        assert!(budget.allow_speculation);
    }

    #[test]
    fn stale_pressure_sample_guards_without_replaying_a_cached_storm() {
        let mut governor = AdaptiveOverheadGovernor::default();
        let budget = governor.observe(OverheadInput {
            pressure_sample_stale: true,
            memory_pressure: 0.95,
            swap_delta_bps: 8.0 * 1024.0 * 1024.0,
            thrashing_score: 20_000.0,
            refault_delta_per_sec: 100_000.0,
            vm_page_size_bytes: 16 * 1024,
            cpu_max_busy: 1.0,
            ..OverheadInput::default()
        });

        assert_eq!(budget.level, OverheadLevel::Guarded);
        for _ in 0..14 {
            assert_eq!(
                governor.observe(OverheadInput::default()).level,
                OverheadLevel::Guarded
            );
        }
        assert_eq!(
            governor.observe(OverheadInput::default()).level,
            OverheadLevel::Nominal
        );
    }

    #[test]
    fn adaptive_stutter_warning_can_guard_before_a_hard_limit() {
        let mut governor = AdaptiveOverheadGovernor::default();
        let budget = governor.observe(OverheadInput {
            adaptive_stutter_guarded: true,
            ..OverheadInput::default()
        });

        assert_eq!(budget.level, OverheadLevel::Guarded);
        assert!(!budget.allow_speculation);
    }

    #[test]
    fn adaptive_stutter_emergency_can_constrain_before_a_hard_limit() {
        let mut governor = AdaptiveOverheadGovernor::default();
        let budget = governor.observe(OverheadInput {
            adaptive_stutter_constrained: true,
            ..OverheadInput::default()
        });

        assert_eq!(budget.level, OverheadLevel::Constrained);
        assert!(!budget.allow_speculation);
    }

    #[test]
    fn adaptive_only_protection_recovers_faster_than_a_hard_overload() {
        let mut governor = AdaptiveOverheadGovernor::default();
        assert_eq!(
            governor
                .observe(OverheadInput {
                    adaptive_stutter_constrained: true,
                    ..OverheadInput::default()
                })
                .level,
            OverheadLevel::Constrained
        );

        for _ in 0..14 {
            assert_eq!(
                governor.observe(OverheadInput::default()).level,
                OverheadLevel::Constrained
            );
        }
        assert_eq!(
            governor.observe(OverheadInput::default()).level,
            OverheadLevel::Guarded
        );
    }

    #[test]
    fn adaptive_signal_never_shortens_an_existing_hard_hold() {
        let mut governor = AdaptiveOverheadGovernor::default();
        assert_eq!(
            governor
                .observe(OverheadInput {
                    p95_cycle_ms: 140.0,
                    ..OverheadInput::default()
                })
                .level,
            OverheadLevel::Constrained
        );
        assert_eq!(
            governor
                .observe(OverheadInput {
                    adaptive_stutter_constrained: true,
                    ..OverheadInput::default()
                })
                .level,
            OverheadLevel::Constrained
        );

        for _ in 0..15 {
            assert_eq!(
                governor.observe(OverheadInput::default()).level,
                OverheadLevel::Constrained
            );
        }
    }

    #[test]
    fn adaptive_escalation_preserves_a_lower_hard_hold() {
        let mut governor = AdaptiveOverheadGovernor::default();
        assert_eq!(
            governor
                .observe(OverheadInput {
                    p95_cycle_ms: 70.0,
                    ..OverheadInput::default()
                })
                .level,
            OverheadLevel::Guarded
        );
        assert_eq!(
            governor
                .observe(OverheadInput {
                    adaptive_stutter_constrained: true,
                    ..OverheadInput::default()
                })
                .level,
            OverheadLevel::Constrained
        );

        for _ in 0..14 {
            assert_eq!(
                governor.observe(OverheadInput::default()).level,
                OverheadLevel::Constrained
            );
        }
        assert_eq!(
            governor.observe(OverheadInput::default()).level,
            OverheadLevel::Guarded
        );
    }

    #[test]
    fn adaptive_recovery_has_a_wall_clock_bound_at_slow_cadence() {
        let started = std::time::Instant::now();
        let mut governor = AdaptiveOverheadGovernor::default();
        assert_eq!(
            governor
                .observe_at(
                    OverheadInput {
                        adaptive_stutter_constrained: true,
                        ..OverheadInput::default()
                    },
                    started,
                )
                .level,
            OverheadLevel::Constrained
        );
        governor.observe_at(OverheadInput::default(), started);
        assert_eq!(
            governor
                .observe_at(
                    OverheadInput::default(),
                    started + std::time::Duration::from_secs(30),
                )
                .level,
            OverheadLevel::Guarded
        );
        governor.observe_at(
            OverheadInput::default(),
            started + std::time::Duration::from_secs(30),
        );
        assert_eq!(
            governor
                .observe_at(
                    OverheadInput::default(),
                    started + std::time::Duration::from_secs(60),
                )
                .level,
            OverheadLevel::Nominal
        );
    }

    #[test]
    fn uncalibrated_gpu_power_alone_never_hard_guards() {
        let mut governor = AdaptiveOverheadGovernor::default();
        let budget = governor.observe(OverheadInput {
            gpu_render_load: 0.90,
            ..OverheadInput::default()
        });

        assert_eq!(budget.level, OverheadLevel::Nominal);
        assert!(budget.allow_speculation);
    }

    #[test]
    fn interaction_does_not_turn_uncalibrated_gpu_power_into_a_hard_limit() {
        let mut governor = AdaptiveOverheadGovernor::default();
        let budget = governor.observe(OverheadInput {
            gpu_render_load: 0.90,
            interaction_q: 0.40,
            ..OverheadInput::default()
        });

        assert_eq!(budget.level, OverheadLevel::Nominal);
        assert!(budget.allow_speculation);
    }

    #[test]
    fn saturated_cpu_core_sheds_optional_work() {
        let mut governor = AdaptiveOverheadGovernor::default();
        let budget = governor.observe(OverheadInput {
            cpu_max_busy: 0.82,
            ..OverheadInput::default()
        });

        assert_eq!(budget.level, OverheadLevel::Guarded);
        assert!(!budget.allow_speculation);
    }

    #[test]
    fn media_plus_compositor_load_is_latency_sensitive() {
        let mut governor = AdaptiveOverheadGovernor::default();
        let budget = governor.observe(OverheadInput {
            media_output_active: true,
            compositor_cpu_pct: 36.0,
            ..OverheadInput::default()
        });

        assert_eq!(budget.level, OverheadLevel::Guarded);
        assert!(!budget.allow_speculation);
    }

    #[test]
    fn predicted_jank_sheds_work_before_measured_p95_regresses() {
        let mut governor = AdaptiveOverheadGovernor::default();
        let budget = governor.observe(OverheadInput {
            predicted_fluidity_3s: 0.55,
            ..OverheadInput::default()
        });

        assert_eq!(budget.level, OverheadLevel::Guarded);
        assert!(!budget.allow_speculation);
    }

    #[test]
    fn light_audio_without_visual_or_compute_load_stays_nominal() {
        let mut governor = AdaptiveOverheadGovernor::default();
        let budget = governor.observe(OverheadInput {
            media_output_active: true,
            gpu_render_load: 0.05,
            compositor_cpu_pct: 5.0,
            cpu_max_busy: 0.20,
            predicted_fluidity_3s: 0.95,
            ..OverheadInput::default()
        });

        assert_eq!(budget.level, OverheadLevel::Nominal);
        assert!(budget.allow_speculation);
    }
}
use std::time::{Duration, Instant};
