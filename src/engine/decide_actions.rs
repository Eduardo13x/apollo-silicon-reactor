use sysinfo::System;

use crate::collector::SystemSnapshot;
use crate::engine::safety::critical_background_processes;
use crate::engine::types::{
    BlockerScore, InteractiveContext, LatencyTarget, OptimizationProfile, RootAction,
};
use std::collections::HashSet;

const INTERACTIVE_APPS: [&str; 9] = [
    "Code",
    "Arc",
    "Google Chrome",
    "Terminal",
    "iTerm",
    "Warp",
    "Antigravity",
    "Cursor",
    "LM Studio",
];

const NOISE_APPS: [&str; 6] = [
    "Dropbox",
    "Google Drive",
    "OneDrive",
    "corespeechd",
    "logioptionsplus",
    "suggestd",
];

const BLOCKER_APPS: [&str; 7] = [
    "WindowServer",
    "accountsd",
    "cfprefsd",
    "distnoted",
    "cloudd",
    "coreaudiod",
    "runningboardd",
];

#[derive(Debug, Clone)]
pub struct DecisionOutput {
    pub context: InteractiveContext,
    pub reactor_event_weight: f64,
    pub blockers: Vec<BlockerScore>,
    pub actions: Vec<RootAction>,
}

fn is_interactive(name: &str) -> bool {
    INTERACTIVE_APPS.iter().any(|n| name.contains(n))
}

fn is_background_noise(name: &str) -> bool {
    NOISE_APPS.iter().any(|n| name.contains(n))
}

fn is_known_blocker(name: &str) -> bool {
    BLOCKER_APPS.iter().any(|n| name.contains(n))
}

fn context_from_pressure(snapshot: &SystemSnapshot) -> InteractiveContext {
    let ram_pressure = snapshot.pressure.memory_pressure * 100.0;
    let cpu_pressure = snapshot.cpu.global_usage as f64;

    if cpu_pressure > 88.0 || ram_pressure > 90.0 {
        InteractiveContext::ThermalConstrained
    } else if cpu_pressure > 72.0 || ram_pressure > 78.0 {
        InteractiveContext::BackgroundPressure
    } else {
        InteractiveContext::InteractiveFocus
    }
}

fn top_blockers(
    sys: &System,
    snapshot: &SystemSnapshot,
    reactor_event_weight: f64,
) -> Vec<BlockerScore> {
    let mut blockers = Vec::new();

    let interactive_waiters = snapshot
        .top_processes
        .iter()
        .filter(|p| {
            is_interactive(&p.name) && p.cpu_usage < 8.0 && p.memory_usage > 100 * 1024 * 1024
        })
        .count() as f64;

    let interactive_wait_ratio = (interactive_waiters / 5.0).clamp(0.0, 1.0);

    for (pid, process) in sys.processes() {
        let name = process.name().to_string();
        if !is_known_blocker(&name) {
            continue;
        }

        let cpu = process.cpu_usage();
        let blocker_cpu_spike = (cpu / 100.0).clamp(0.0, 1.0);
        let blocker_seen_recently = cpu > 8.0;
        let score = (interactive_wait_ratio * 0.45)
            + (blocker_cpu_spike as f64 * 0.35)
            + (if blocker_seen_recently { 0.1 } else { 0.0 })
            + (reactor_event_weight * 0.1);

        if score > 0.30 {
            blockers.push(BlockerScore {
                name,
                pid: pid.as_u32(),
                score,
                blocker_cpu_spike: cpu,
                interactive_wait_ratio,
                blocker_seen_recently,
                reactor_event_weight,
            });
        }
    }

    blockers.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    blockers.truncate(8);
    blockers
}

pub fn decide_actions(
    snapshot: &SystemSnapshot,
    sys: &System,
    profile: OptimizationProfile,
    latency_target: LatencyTarget,
    reactor_event_weight: f64,
) -> DecisionOutput {
    let mut actions = Vec::new();

    // Dev-first: protect critical background workloads and their children.
    let critical_patterns = critical_background_processes();
    let mut critical_pids: HashSet<u32> = HashSet::new();
    for (pid, process) in sys.processes() {
        let name = process.name().to_string();
        if critical_patterns.iter().any(|p| name.contains(p)) {
            critical_pids.insert(pid.as_u32());
        }
    }
    // Add children-of-critical by walking parent chain once.
    // Depth-limited to prevent infinite loops from PID recycling (BUG 10 fix).
    const MAX_PARENT_DEPTH: usize = 20;
    for pid in sys.processes().keys() {
        let mut cur = Some(*pid);
        let mut is_child = false;
        let mut depth = 0usize;
        while let Some(p) = cur {
            if depth >= MAX_PARENT_DEPTH {
                break;
            }
            if critical_pids.contains(&p.as_u32()) {
                is_child = true;
                break;
            }
            cur = sys.process(p).and_then(|pp| pp.parent());
            depth += 1;
        }
        if is_child {
            critical_pids.insert(pid.as_u32());
        }
    }

    let context = context_from_pressure(snapshot);
    let blockers = top_blockers(sys, snapshot, reactor_event_weight);

    // 1) Wait-graph practical: temporary boost for top blockers.
    let blocker_boost_count = match latency_target {
        LatencyTarget::Max => 3,
        LatencyTarget::Low => 1,
        LatencyTarget::Normal => 2,
    };
    for blocker in blockers.iter().take(blocker_boost_count) {
        actions.push(RootAction::BoostProcess {
            pid: blocker.pid,
            name: blocker.name.clone(),
            reason: format!("wait-graph blocker score {:.2}", blocker.score),
        });
    }

    // 2) Context-aware scheduling.
    for (pid, process) in sys.processes() {
        let name = process.name().to_string();
        let pid = pid.as_u32();

        if critical_pids.contains(&pid) {
            continue;
        }

        if is_interactive(&name) {
            actions.push(RootAction::BoostProcess {
                pid,
                name,
                reason: "interactive focus boost".to_string(),
            });
            continue;
        }

        if is_background_noise(&name) {
            // BUG 11 fix: ThermalConstrained was less aggressive than BackgroundPressure.
            // Under thermal constraint, throttle aggressively; under BackgroundPressure,
            // also throttle aggressively. InteractiveFocus uses profile-driven policy.
            let aggressive = match context {
                InteractiveContext::ThermalConstrained => true,
                InteractiveContext::BackgroundPressure => true,
                InteractiveContext::InteractiveFocus => {
                    matches!(profile, OptimizationProfile::AggressiveRoot)
                }
            };
            actions.push(RootAction::ThrottleProcess {
                pid,
                name,
                aggressive,
                reason: format!("context-aware throttle ({:?})", context),
            });
        }
    }

    // 3) Pressure actions with hysteresis-ish behavior by context.
    match context {
        InteractiveContext::BackgroundPressure | InteractiveContext::ThermalConstrained => {
            // Dev-first: no-freeze by default. Only consider freeze under extreme memory pressure
            // AND swap growth, and never for protected/critical workloads.
            let extreme_freeze_ok = snapshot.pressure.memory_pressure >= 0.90
                && snapshot.pressure.swap_delta_bytes_per_sec > (5.0 * 1024.0 * 1024.0);
            if extreme_freeze_ok {
                for (pid, process) in sys.processes() {
                    let pid = pid.as_u32();
                    if critical_pids.contains(&pid) {
                        continue;
                    }
                    let name = process.name().to_string();
                    if ["Slack", "Discord", "Spotify", "Teams"]
                        .iter()
                        .any(|n| name.contains(n))
                    {
                        actions.push(RootAction::FreezeProcess {
                            pid,
                            name,
                            reason: format!("extreme pressure quarantine under {:?}", context),
                        });
                    }
                }
            }

            actions.push(RootAction::SetSysctl {
                key: "vm.compressor_poll_interval".to_string(),
                value: "20".to_string(),
                reason: "pre-emptive paging tune".to_string(),
            });
        }
        InteractiveContext::InteractiveFocus => {
            actions.push(RootAction::SetSysctl {
                key: "debug.lowpri_throttle_enabled".to_string(),
                value: "0".to_string(),
                reason: "favor interactive I/O latency".to_string(),
            });
        }
    }

    DecisionOutput {
        context,
        reactor_event_weight,
        blockers,
        actions,
    }
}
