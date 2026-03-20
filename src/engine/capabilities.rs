use crate::engine::types::CapabilityReport;

pub fn detect_capabilities() -> CapabilityReport {
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
