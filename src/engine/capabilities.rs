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
        let _ = cap.can_mdutil as u8;
        let _ = cap.can_tmutil as u8;
        let _ = cap.is_root as u8;
    }
}
