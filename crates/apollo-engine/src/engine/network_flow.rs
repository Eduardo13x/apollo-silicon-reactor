//! Bounded, content-blind inference for interactive network work in any app.

use serde::{Deserialize, Serialize};

pub const GENERIC_FLOW_INITIAL_TTL_MS: u64 = 1_200;
pub const GENERIC_FLOW_HARD_CAP_MS: u64 = 12_000;
pub const GENERIC_FLOW_COOLDOWN_MS: u64 = 2_000;
pub const GENERIC_FLOW_MAX_TCP_AGE_MS: u64 = 2_000;
pub const GENERIC_FLOW_MIN_TRAFFIC_BPS: u64 = 32 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NetworkFlowTickInput {
    pub now_ms: u64,
    pub session_revision: u64,
    pub foreground_pid: Option<u32>,
    pub identity_available: bool,
    pub interaction_active: bool,
    pub foreground_socket_active: bool,
    pub tcp_sample_age_ms: u64,
    pub send_bps: u64,
    pub recv_bps: u64,
    pub new_connections: u32,
    pub exact_web_active: bool,
    pub pressure_constrained: bool,
    pub thermal_constrained: bool,
    pub low_power: bool,
    pub sleeping: bool,
    pub kill_switch: bool,
}

impl NetworkFlowTickInput {
    fn constrained(self) -> bool {
        self.pressure_constrained
            || self.thermal_constrained
            || self.low_power
            || self.sleeping
            || self.kill_switch
    }

    fn traffic_bps(self) -> u64 {
        self.send_bps.saturating_add(self.recv_bps)
    }

    fn fresh_traffic(self) -> bool {
        self.tcp_sample_age_ms <= GENERIC_FLOW_MAX_TCP_AGE_MS
            && (self.traffic_bps() >= GENERIC_FLOW_MIN_TRAFFIC_BPS || self.new_connections > 0)
    }

    fn qualifies(self) -> bool {
        self.foreground_pid.is_some()
            && self.identity_available
            && self.interaction_active
            && self.foreground_socket_active
            && self.fresh_traffic()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NetworkFlowReason {
    ForegroundTransfer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NetworkFlowIntent {
    pub target_pid: u32,
    pub confidence_q: u16,
    pub intensity_q: u16,
    pub ttl_ms: u32,
    pub reason: NetworkFlowReason,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkWorldObservation {
    pub active: bool,
    pub inferred: bool,
    pub target_available: bool,
    pub socket_active: bool,
    pub sample_fresh: bool,
    pub interaction_active: bool,
    pub traffic_bps: u64,
    pub confidence_q: u16,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NetworkFlowCounters {
    pub proposals: u64,
    pub starts: u64,
    pub renewals: u64,
    pub skipped: u64,
    pub suppressed_exact: u64,
    pub constrained: u64,
    pub expirations: u64,
    pub hard_cap_expirations: u64,
    pub target_changes: u64,
    pub session_resets: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NetworkFlowOutput {
    pub intent: Option<NetworkFlowIntent>,
    pub observation: NetworkWorldObservation,
    pub counters: NetworkFlowCounters,
}

#[derive(Debug, Clone, Copy)]
struct ActiveFlow {
    target_pid: u32,
    started_at_ms: u64,
    expires_at_ms: u64,
}

#[derive(Debug, Default)]
pub struct NetworkFlowController {
    active: Option<ActiveFlow>,
    session_revision: Option<u64>,
    cooldown_until_ms: u64,
    counters: NetworkFlowCounters,
}

impl NetworkFlowController {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tick(&mut self, input: NetworkFlowTickInput) -> NetworkFlowOutput {
        if self
            .session_revision
            .is_some_and(|revision| revision != input.session_revision)
        {
            self.active = None;
            self.cooldown_until_ms = 0;
            self.session_revision = Some(input.session_revision);
            self.counters.session_resets = self.counters.session_resets.saturating_add(1);
            return self.output(input, None);
        }
        self.session_revision = Some(input.session_revision);

        if input.exact_web_active {
            self.active = None;
            self.cooldown_until_ms = 0;
            self.counters.suppressed_exact = self.counters.suppressed_exact.saturating_add(1);
            return self.output(input, None);
        }

        if input.constrained() {
            if self.active.take().is_some() {
                self.counters.constrained = self.counters.constrained.saturating_add(1);
            }
            return self.output(input, None);
        }

        if self
            .active
            .is_some_and(|active| input.now_ms >= active.started_at_ms + GENERIC_FLOW_HARD_CAP_MS)
        {
            self.active = None;
            self.cooldown_until_ms = input.now_ms.saturating_add(GENERIC_FLOW_COOLDOWN_MS);
            self.counters.hard_cap_expirations =
                self.counters.hard_cap_expirations.saturating_add(1);
            return self.output(input, None);
        }

        if self
            .active
            .is_some_and(|active| Some(active.target_pid) != input.foreground_pid)
        {
            self.active = None;
            self.counters.target_changes = self.counters.target_changes.saturating_add(1);
        }

        if input.now_ms < self.cooldown_until_ms {
            self.counters.skipped = self.counters.skipped.saturating_add(1);
            return self.output(input, None);
        }

        if !input.qualifies() {
            if self
                .active
                .is_some_and(|active| input.now_ms >= active.expires_at_ms)
            {
                self.active = None;
                self.counters.expirations = self.counters.expirations.saturating_add(1);
            }
            self.counters.skipped = self.counters.skipped.saturating_add(1);
            return self.output(input, None);
        }

        let target_pid = input.foreground_pid.expect("qualifies requires a PID");
        let hard_deadline_ms = self.active.map_or(
            input.now_ms.saturating_add(GENERIC_FLOW_HARD_CAP_MS),
            |active| {
                active
                    .started_at_ms
                    .saturating_add(GENERIC_FLOW_HARD_CAP_MS)
            },
        );
        let expires_at_ms = input
            .now_ms
            .saturating_add(GENERIC_FLOW_INITIAL_TTL_MS)
            .min(hard_deadline_ms);
        match self.active.as_mut() {
            Some(active) => {
                active.expires_at_ms = expires_at_ms;
                self.counters.renewals = self.counters.renewals.saturating_add(1);
            }
            None => {
                self.active = Some(ActiveFlow {
                    target_pid,
                    started_at_ms: input.now_ms,
                    expires_at_ms,
                });
                self.counters.starts = self.counters.starts.saturating_add(1);
            }
        }
        self.counters.proposals = self.counters.proposals.saturating_add(1);

        let ttl_ms = expires_at_ms
            .saturating_sub(input.now_ms)
            .min(u32::MAX as u64) as u32;
        let traffic = input.traffic_bps();
        let intensity_q = if traffic >= 4 * 1024 * 1024 {
            7_000
        } else if traffic >= 512 * 1024 {
            5_500
        } else {
            4_000
        };
        let confidence_q = if input.new_connections > 0 {
            5_000
        } else {
            4_500
        };
        self.output(
            input,
            Some(NetworkFlowIntent {
                target_pid,
                confidence_q,
                intensity_q,
                ttl_ms,
                reason: NetworkFlowReason::ForegroundTransfer,
            }),
        )
    }

    fn output(
        &self,
        input: NetworkFlowTickInput,
        intent: Option<NetworkFlowIntent>,
    ) -> NetworkFlowOutput {
        NetworkFlowOutput {
            intent,
            observation: NetworkWorldObservation {
                active: self.active.is_some(),
                inferred: true,
                target_available: input.foreground_pid.is_some() && input.identity_available,
                socket_active: input.foreground_socket_active,
                sample_fresh: input.tcp_sample_age_ms <= GENERIC_FLOW_MAX_TCP_AGE_MS,
                interaction_active: input.interaction_active,
                traffic_bps: input.traffic_bps(),
                confidence_q: intent.map_or(0, |intent| intent.confidence_q),
            },
            counters: self.counters,
        }
    }
}
