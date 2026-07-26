//! Runtime workload-phase duration learning.
//!
//! This layer complements the coarse workload classifier: it separates media
//! and rapid multitasking, then learns an EWMA duration for each phase. It is
//! intentionally in-memory so imported hardware state cannot seed M1 timing
//! assumptions on a new Mac.

use std::time::Instant;

use crate::engine::workload_classifier::WorkloadMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadPhase {
    Idle,
    Browsing,
    Build,
    LocalAi,
    Multimedia,
    Multitask,
}

impl WorkloadPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Browsing => "browsing",
            Self::Build => "build",
            Self::LocalAi => "local-ai",
            Self::Multimedia => "multimedia",
            Self::Multitask => "multitask",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::Idle => 0,
            Self::Browsing => 1,
            Self::Build => 2,
            Self::LocalAi => 3,
            Self::Multimedia => 4,
            Self::Multitask => 5,
        }
    }
}

pub fn classify_phase(
    mode: WorkloadMode,
    media_active: bool,
    context_switch_burst: bool,
) -> WorkloadPhase {
    match mode {
        WorkloadMode::Build => WorkloadPhase::Build,
        WorkloadMode::LlmInference => WorkloadPhase::LocalAi,
        _ if media_active => WorkloadPhase::Multimedia,
        _ if context_switch_burst => WorkloadPhase::Multitask,
        WorkloadMode::Browsing => WorkloadPhase::Browsing,
        WorkloadMode::Idle => WorkloadPhase::Idle,
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct DurationEstimate {
    ewma_secs: f64,
    observations: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct PhaseObservation {
    pub phase: WorkloadPhase,
    pub expected_duration_secs: f64,
    pub duration_observations: u64,
}

#[derive(Debug)]
pub struct WorkloadPhaseTracker {
    current: Option<WorkloadPhase>,
    started_at: Instant,
    estimates: [DurationEstimate; 6],
}

impl WorkloadPhaseTracker {
    pub fn new(now: Instant) -> Self {
        Self {
            current: None,
            started_at: now,
            estimates: [DurationEstimate::default(); 6],
        }
    }

    pub fn observe(&mut self, phase: WorkloadPhase, now: Instant) -> PhaseObservation {
        if let Some(previous) = self.current {
            if previous != phase {
                let elapsed = now.saturating_duration_since(self.started_at).as_secs_f64();
                let estimate = &mut self.estimates[previous.index()];
                estimate.ewma_secs = if estimate.observations == 0 {
                    elapsed
                } else {
                    0.25 * elapsed + 0.75 * estimate.ewma_secs
                };
                estimate.observations = estimate.observations.saturating_add(1);
                self.started_at = now;
            }
        } else {
            self.started_at = now;
        }
        self.current = Some(phase);
        let estimate = self.estimates[phase.index()];
        PhaseObservation {
            phase,
            expected_duration_secs: estimate.ewma_secs,
            duration_observations: estimate.observations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn classifier_exposes_media_and_multitask_without_hiding_build_or_ai() {
        assert_eq!(
            classify_phase(WorkloadMode::Build, true, true),
            WorkloadPhase::Build
        );
        assert_eq!(
            classify_phase(WorkloadMode::LlmInference, true, true),
            WorkloadPhase::LocalAi
        );
        assert_eq!(
            classify_phase(WorkloadMode::Browsing, true, false),
            WorkloadPhase::Multimedia
        );
        assert_eq!(
            classify_phase(WorkloadMode::Browsing, false, true),
            WorkloadPhase::Multitask
        );
    }

    #[test]
    fn tracker_learns_phase_duration_only_after_completed_transition() {
        let t0 = Instant::now();
        let mut tracker = WorkloadPhaseTracker::new(t0);
        let first = tracker.observe(WorkloadPhase::Build, t0);
        assert_eq!(first.duration_observations, 0);

        tracker.observe(WorkloadPhase::Idle, t0 + Duration::from_secs(20));
        let learned = tracker.observe(WorkloadPhase::Build, t0 + Duration::from_secs(30));
        assert_eq!(learned.duration_observations, 1);
        assert_eq!(learned.expected_duration_secs, 20.0);
    }
}
