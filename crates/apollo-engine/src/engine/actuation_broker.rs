//! Typed privilege boundary for host mutations.
//!
//! The current broker is in-process so existing mediator effectors remain the
//! syscall authority. Keeping the request surface closed and authority checked
//! here allows a future out-of-process root broker without changing planners.

use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::engine::active_coalition_envelope::CoalitionGuard;
use crate::engine::execute_actions::{execute_actions, ExecuteOutcomes};
use crate::engine::mach_qos::MachQoSManager;
use crate::engine::types::{CapabilityReport, RootAction};

const MAX_ACTIONS_PER_BATCH: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrokerMode {
    InProcessRoot,
    DryRun,
    Denied,
}

impl BrokerMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InProcessRoot => "in_process_root",
            Self::DryRun => "dry_run",
            Self::Denied => "denied",
        }
    }
}

pub struct ActuationRequest<'a> {
    pub actions: Vec<RootAction>,
    pub caps: &'a CapabilityReport,
    pub journal_path: &'a Path,
    pub frozen: &'a mut HashSet<u32>,
    pub learned_protected: &'a [String],
    pub learned_interactive: &'a [String],
    pub qos_mgr: Option<&'a Arc<Mutex<MachQoSManager>>>,
    pub memory_pressure: f64,
    pub thrashing_score: f64,
    pub coalition_guard: Option<&'a CoalitionGuard<'a>>,
    pub cpu_pegged_fraction: f64,
}

#[derive(Debug)]
pub struct BrokerExecution {
    pub outcomes: ExecuteOutcomes,
    pub requests: u64,
    pub rejected: u64,
    pub mode: BrokerMode,
}

#[derive(Debug, Clone, Copy)]
pub struct ActuationBroker {
    mode: BrokerMode,
}

impl ActuationBroker {
    pub fn from_runtime(caps: &CapabilityReport, dry_run: bool) -> Self {
        let actual_root = {
            #[cfg(unix)]
            {
                unsafe { libc::geteuid() == 0 }
            }
            #[cfg(not(unix))]
            {
                caps.is_root
            }
        };
        let mode = if dry_run {
            BrokerMode::DryRun
        } else if caps.is_root && actual_root {
            BrokerMode::InProcessRoot
        } else {
            BrokerMode::Denied
        };
        Self { mode }
    }

    pub fn mode(self) -> BrokerMode {
        self.mode
    }

    pub fn execute(self, request: ActuationRequest<'_>) -> BrokerExecution {
        let requests = request.actions.len() as u64;
        if self.mode == BrokerMode::Denied || request.actions.len() > MAX_ACTIONS_PER_BATCH {
            let reason = if self.mode == BrokerMode::Denied {
                "actuation broker denied: root authority mismatch"
            } else {
                "actuation broker denied: batch exceeds 512 actions"
            };
            return BrokerExecution {
                outcomes: ExecuteOutcomes {
                    failures: requests,
                    last_error: Some(reason.to_string()),
                    ..ExecuteOutcomes::default()
                },
                requests,
                rejected: requests,
                mode: BrokerMode::Denied,
            };
        }

        let outcomes = execute_actions(
            request.actions,
            request.caps,
            request.journal_path,
            request.frozen,
            request.learned_protected,
            request.learned_interactive,
            request.qos_mgr,
            self.mode == BrokerMode::DryRun,
            request.memory_pressure,
            request.thrashing_score,
            request.coalition_guard,
            request.cpu_pegged_fraction,
        );
        BrokerExecution {
            outcomes,
            requests,
            rejected: 0,
            mode: self.mode,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(is_root: bool) -> CapabilityReport {
        CapabilityReport {
            can_taskpolicy: false,
            can_sysctl: false,
            can_memorystatus: false,
            can_memory_pressure_send: false,
            can_mdutil: false,
            can_tmutil: false,
            is_root,
            p_core_count: None,
            e_core_count: None,
            unavailable: Vec::new(),
            memorystatus_probe: None,
            task_for_pid_probe: None,
        }
    }

    #[test]
    fn non_root_mutation_authority_fails_closed() {
        assert_eq!(
            ActuationBroker::from_runtime(&caps(false), false).mode(),
            BrokerMode::Denied
        );
    }

    #[test]
    fn dry_run_authority_is_available_without_root() {
        assert_eq!(
            ActuationBroker::from_runtime(&caps(false), true).mode(),
            BrokerMode::DryRun
        );
    }
}
