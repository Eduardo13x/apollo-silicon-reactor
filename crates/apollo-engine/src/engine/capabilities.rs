use crate::engine::types::CapabilityReport;
use std::sync::OnceLock;

pub fn detect_capabilities() -> CapabilityReport {
    static PASSIVE_CAPABILITIES: OnceLock<CapabilityReport> = OnceLock::new();
    PASSIVE_CAPABILITIES
        .get_or_init(|| detect_capabilities_inner(false))
        .clone()
}

/// Detect capabilities and execute the explicit write probes used by `doctor`.
///
/// Keep this out of daemon hot paths: the memorystatus probe writes the
/// daemon's current limit back to the kernel and `task_for_pid` acquires a Mach
/// task port. Normal action dispatch only needs the passive capability flags.
pub fn detect_capabilities_with_write_probes() -> CapabilityReport {
    detect_capabilities_inner(true)
}

fn detect_capabilities_inner(run_write_probes: bool) -> CapabilityReport {
    let mut unavailable = Vec::new();

    // taskpolicy: check if setpriority works (always available on macOS).
    let can_taskpolicy = cfg!(target_os = "macos");
    if !can_taskpolicy {
        unavailable.push("taskpolicy".to_string());
    }

    // sysctl: probe via direct sysctlbyname.
    let can_sysctl = crate::engine::sysctl_direct::exists("kern.ostype");
    if !can_sysctl {
        unavailable.push("sysctl".to_string());
    }

    // mdutil: check if binary exists (Spotlight control).
    let can_mdutil = std::path::Path::new("/usr/bin/mdutil").exists();
    if !can_mdutil {
        unavailable.push("mdutil".to_string());
    }

    // tmutil: check if binary exists (Time Machine).
    let can_tmutil = std::path::Path::new("/usr/bin/tmutil").exists();
    if !can_tmutil {
        unavailable.push("tmutil".to_string());
    }

    let is_root = unsafe { libc::geteuid() == 0 };
    let can_memory_pressure_send =
        is_root && crate::engine::sysctl_direct::exists("kern.memorystatus_vm_pressure_send");

    // Core counts (Apple Silicon clusters)
    // perflevel0 = P-cores (Firestorm/Avalanche/etc.)
    // perflevel1 = E-cores (Icestorm/Blizzard/etc.)
    let p_core_count = crate::engine::sysctl_direct::read_u32_val("hw.perflevel0.logicalcpu");
    let e_core_count = crate::engine::sysctl_direct::read_u32_val("hw.perflevel1.logicalcpu");

    // ── Live write probes ────────────────────────────────────────────────────
    // These actually call the kernel API to prove the write path works,
    // rather than inferring capability from euid or binary existence.

    let memorystatus_probe = if is_root && run_write_probes {
        match crate::engine::jetsam_control::probe_write() {
            Ok(()) => Some("ok".to_string()),
            Err(e) => Some(format!("fail: {}", e)),
        }
    } else {
        None
    };

    let task_for_pid_probe = if run_write_probes {
        match probe_task_for_pid() {
            Ok(()) => Some("ok".to_string()),
            Err(e) => Some(format!("fail: {}", e)),
        }
    } else {
        None
    };

    CapabilityReport {
        can_taskpolicy,
        can_sysctl,
        can_memorystatus: is_root,
        can_memory_pressure_send,
        can_mdutil,
        can_tmutil,
        is_root,
        p_core_count,
        e_core_count,
        unavailable,
        memorystatus_probe,
        task_for_pid_probe,
    }
}

#[cfg(target_os = "macos")]
fn probe_task_for_pid() -> Result<(), String> {
    type MachPortT = libc::c_uint;
    const KERN_SUCCESS: i32 = 0;

    extern "C" {
        fn mach_task_self() -> MachPortT;
        fn task_for_pid(target_tport: MachPortT, pid: libc::pid_t, t: *mut MachPortT) -> i32;
        fn mach_port_deallocate(target_task: MachPortT, name: MachPortT) -> i32;
    }

    let pid = std::process::id() as libc::pid_t;
    let mut task_port: MachPortT = 0;
    let kr = unsafe { task_for_pid(mach_task_self(), pid, &mut task_port) };
    if kr != KERN_SUCCESS {
        return Err(format!("kern_return={}", kr));
    }
    unsafe {
        mach_port_deallocate(mach_task_self(), task_port);
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn probe_task_for_pid() -> Result<(), String> {
    Err("not macOS".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_capabilities_does_not_panic() {
        let _cap = detect_capabilities();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn can_taskpolicy_is_true_on_macos() {
        let cap = detect_capabilities();
        assert!(cap.can_taskpolicy, "can_taskpolicy should be true on macOS");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn can_sysctl_is_true_on_macos() {
        let cap = detect_capabilities();
        assert!(cap.can_sysctl, "can_sysctl should be true on macOS");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn unavailable_does_not_contain_taskpolicy_on_macos() {
        let cap = detect_capabilities();
        assert!(
            !cap.unavailable.contains(&"taskpolicy".to_string()),
            "unavailable should not contain 'taskpolicy' on macOS, got: {:?}",
            cap.unavailable
        );
    }

    #[test]
    fn capability_report_fields_are_bool() {
        let cap = detect_capabilities();
        // Implicit type check: these are all bool fields used in assertions
        let _ = cap.can_taskpolicy as u8;
        let _ = cap.can_sysctl as u8;
        let _ = cap.can_memorystatus as u8;
        let _ = cap.can_memory_pressure_send as u8;
        let _ = cap.can_mdutil as u8;
        let _ = cap.can_tmutil as u8;
        let _ = cap.is_root as u8;
    }

    #[test]
    fn passive_detection_never_runs_write_probes() {
        let cap = detect_capabilities();
        assert!(cap.memorystatus_probe.is_none());
        assert!(cap.task_for_pid_probe.is_none());
    }
}
