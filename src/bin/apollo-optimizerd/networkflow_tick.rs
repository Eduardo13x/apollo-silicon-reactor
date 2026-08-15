use apollo_engine::engine::network_flow::{
    NetworkFlowController, NetworkFlowOutput, NetworkFlowTickInput,
};
use apollo_engine::engine::network_monitor::NetworkTrafficSample;
use apollo_engine::engine::process_tree::ProcessTree;

const MAX_SOCKET_CANDIDATES: usize = 16;

#[derive(Debug, Clone, Copy)]
pub struct NetworkFlowCycleInput {
    pub now_ms: u64,
    pub session_revision: u64,
    pub foreground_pid: Option<u32>,
    pub interaction_active: bool,
    pub foreground_socket_active: bool,
    pub traffic: NetworkTrafficSample,
    pub tcp_sample_age_ms: u64,
    pub exact_web_active: bool,
    pub pressure_constrained: bool,
    pub thermal_constrained: bool,
    pub low_power: bool,
    pub sleeping: bool,
    pub kill_switch: bool,
}

pub struct NetworkFlowRuntime {
    controller: NetworkFlowController,
}

impl NetworkFlowRuntime {
    pub fn new() -> Self {
        Self {
            controller: NetworkFlowController::new(),
        }
    }

    pub fn tick(&mut self, input: NetworkFlowCycleInput) -> NetworkFlowOutput {
        self.controller.tick(NetworkFlowTickInput {
            now_ms: input.now_ms,
            session_revision: input.session_revision,
            foreground_pid: input.foreground_pid,
            identity_available: input.foreground_pid.is_some(),
            interaction_active: input.interaction_active,
            foreground_socket_active: input.foreground_socket_active,
            tcp_sample_age_ms: input.tcp_sample_age_ms,
            send_bps: input.traffic.send_bps,
            recv_bps: input.traffic.recv_bps,
            new_connections: input.traffic.new_connections,
            exact_web_active: input.exact_web_active,
            pressure_constrained: input.pressure_constrained,
            thermal_constrained: input.thermal_constrained,
            low_power: input.low_power,
            sleeping: input.sleeping,
            kill_switch: input.kill_switch,
        })
    }
}

pub fn bounded_family_candidates(process_tree: &ProcessTree, pid: u32) -> Vec<u32> {
    let mut candidates = process_tree.cascade_pids(pid);
    if candidates.is_empty() {
        candidates.push(pid);
    }
    if let Some(index) = candidates.iter().position(|candidate| *candidate == pid) {
        candidates.swap(0, index);
    } else {
        candidates.insert(0, pid);
    }
    candidates.truncate(MAX_SOCKET_CANDIDATES);
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_engine::engine::network_monitor::NetworkTrafficSample;
    use apollo_engine::engine::process_tree::{ProcessEntry, ProcessTree};

    fn tree() -> ProcessTree {
        let mut entries = vec![ProcessEntry {
            pid: 100,
            ppid: 1,
            name: "Any Internet App".into(),
            cpu_usage: 1.0,
            memory_bytes: 1,
        }];
        for pid in 101..125 {
            entries.push(ProcessEntry {
                pid,
                ppid: 100,
                name: format!("helper-{pid}"),
                cpu_usage: 0.1,
                memory_bytes: 1,
            });
        }
        entries.push(ProcessEntry {
            pid: 900,
            ppid: 1,
            name: "Unrelated".into(),
            cpu_usage: 1.0,
            memory_bytes: 1,
        });
        ProcessTree::build(&entries)
    }

    #[test]
    fn socket_candidates_are_bounded_to_the_foreground_family() {
        let candidates = bounded_family_candidates(&tree(), 100);
        assert_eq!(candidates.len(), 16);
        assert!(candidates.contains(&100));
        assert!(!candidates.contains(&900));
    }

    #[test]
    fn universal_runtime_accepts_a_non_browser_foreground_app() {
        let mut runtime = NetworkFlowRuntime::new();
        let output = runtime.tick(NetworkFlowCycleInput {
            now_ms: 1_000,
            session_revision: 1,
            foreground_pid: Some(100),
            interaction_active: true,
            foreground_socket_active: true,
            traffic: NetworkTrafficSample {
                send_bps: 10_000,
                recv_bps: 100_000,
                new_connections: 1,
            },
            tcp_sample_age_ms: 100,
            exact_web_active: false,
            pressure_constrained: false,
            thermal_constrained: false,
            low_power: false,
            sleeping: false,
            kill_switch: false,
        });
        assert_eq!(output.intent.expect("universal intent").target_pid, 100);
    }

    #[test]
    fn exact_web_activity_wins_over_universal_inference() {
        let mut runtime = NetworkFlowRuntime::new();
        let output = runtime.tick(NetworkFlowCycleInput {
            now_ms: 1_000,
            session_revision: 1,
            foreground_pid: Some(100),
            interaction_active: true,
            foreground_socket_active: true,
            traffic: NetworkTrafficSample {
                send_bps: 100_000,
                recv_bps: 100_000,
                new_connections: 1,
            },
            tcp_sample_age_ms: 0,
            exact_web_active: true,
            pressure_constrained: false,
            thermal_constrained: false,
            low_power: false,
            sleeping: false,
            kill_switch: false,
        });
        assert!(output.intent.is_none());
        assert_eq!(output.counters.suppressed_exact, 1);
    }
}
