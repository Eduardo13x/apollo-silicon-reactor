use std::process::Command;

use crate::engine::types::CapabilityReport;

fn can_run_with_exit_codes(program: &str, args: &[&str], ok: &[i32]) -> bool {
    Command::new(program)
        .args(args)
        .output()
        .map_or(false, |out| {
            if out.status.success() {
                return true;
            }
            out.status.code().is_some_and(|c| ok.contains(&c))
        })
}

pub fn detect_capabilities() -> CapabilityReport {
    let mut unavailable = Vec::new();

    // On macOS `taskpolicy -h` returns EX_USAGE (64) on some builds.
    let can_taskpolicy = can_run_with_exit_codes("taskpolicy", &["-h"], &[64]);
    if !can_taskpolicy {
        unavailable.push("taskpolicy".to_string());
    }

    let can_sysctl = can_run_with_exit_codes("sysctl", &["-a"], &[]);
    if !can_sysctl {
        unavailable.push("sysctl".to_string());
    }

    let can_mdutil = can_run_with_exit_codes("mdutil", &["-s", "/"], &[]);
    if !can_mdutil {
        unavailable.push("mdutil".to_string());
    }

    let can_tmutil = can_run_with_exit_codes("tmutil", &["listlocalsnapshots", "/"], &[]);
    if !can_tmutil {
        unavailable.push("tmutil".to_string());
    }

    let is_root = unsafe { libc::geteuid() == 0 };

    CapabilityReport {
        can_taskpolicy,
        can_sysctl,
        can_memorystatus: is_root,
        can_mdutil,
        can_tmutil,
        is_root,
        unavailable,
    }
}
