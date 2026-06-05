//! RAM Phase G — disallowed-methods regression gate.
//!
//! This test snapshots the current set of files that call raw `libc::kill(`
//! or `sysctlbyname` directly. Any file added to that set in a future commit
//! that is NOT already on the allowlist causes the test to fail — forcing
//! authors to route the new mutation through the typed `Effector` trait
//! (`engine::mediator::SignalEffector` / `SysctlEffector`).
//!
//! Existing call sites remain allowed; the switch-over sprint migrates each
//! one through the mediator chokepoint and removes it from the allowlist.
//!
//! This is NOT a clippy `disallowed-methods` lint (which would either
//! refuse all sites or require per-call `#[allow]`); a grep-style snapshot
//! test gives the same regression-prevention property with a single
//! file-level allowlist that's trivial to amend during migration.

use std::fs;
use std::path::{Path, PathBuf};

/// Recursively collect every `.rs` file under `root`.
fn walk_rs_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    fn recurse(dir: &Path, acc: &mut Vec<PathBuf>) {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                recurse(&p, acc);
            } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
                acc.push(p);
            }
        }
    }
    recurse(root, &mut out);
    out
}

/// Repo root resolved from `CARGO_MANIFEST_DIR`. Falls back to walking up
/// from the test binary's CWD when run outside cargo.
fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR points at the apollo-engine crate; go up 2 to repo.
    let manifest = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest)
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Files allowed to call `libc::kill(`. Add a file to this list ONLY after
/// the switch-over sprint cannot yet route it through SignalEffector.
/// Remove a file when its raw call sites are migrated to the mediator.
const LIBC_KILL_ALLOWLIST: &[&str] = &[
    "crates/apollo-engine/src/engine/mediator.rs",
    "crates/apollo-engine/src/engine/mach_qos.rs",
    "crates/apollo-engine/src/engine/daemon_helpers.rs",
    "crates/apollo-engine/src/engine/thermal_interrupt.rs",
    "crates/apollo-engine/src/engine/process_identity.rs",
    "crates/apollo-engine/src/engine/execute_actions.rs",
    "crates/apollo-engine/src/engine/chromium_manager.rs",
    "src/bin/apollo-optimizerd/daemon_process_collector.rs",
    "src/bin/apollo-optimizerd/main.rs",
    "src/bin/apollo-optimizerd/daemon_thermal_freeze.rs",
    "src/bin/apollo-optimizerd/daemon_turbo_manager.rs",
];

/// Files allowed to call `sysctlbyname` directly. Same migration semantics.
const SYSCTLBYNAME_ALLOWLIST: &[&str] = &[
    "crates/apollo-engine/src/collector.rs",
    "crates/apollo-engine/src/engine/mediator.rs",
    "crates/apollo-engine/src/engine/host_vm_info.rs",
    "crates/apollo-engine/src/engine/daemon_helpers.rs",
    "crates/apollo-engine/src/engine/network_monitor.rs",
    "crates/apollo-engine/src/engine/sysctl_direct.rs",
    "crates/apollo-engine/src/engine/dispatch_pressure.rs",
    "crates/apollo-engine/src/engine/execute_actions.rs",
    "crates/apollo-engine/src/engine/capabilities.rs",
    "crates/apollo-engine/src/engine/kqueue_pressure.rs",
    "crates/apollo-engine/src/engine/silicon_probe.rs",
];

/// Run a grep-style check: find every `.rs` file under `root` (less the test
/// tree) that contains `needle`, normalize to repo-relative path strings,
/// and assert the set is a subset of `allowlist`.
fn check_needle(needle: &str, allowlist: &[&str], roots: &[&str]) {
    let repo = repo_root();
    let allowset: std::collections::HashSet<&&str> = allowlist.iter().collect();
    let mut violations: Vec<String> = Vec::new();
    for root_str in roots {
        let root = repo.join(root_str);
        for path in walk_rs_files(&root) {
            // Skip test files — they are allowed to construct raw syscalls
            // for fixture purposes, and tests already opt into the trait
            // surface they want to exercise.
            if path.components().any(|c| c.as_os_str() == "tests") {
                continue;
            }
            let txt = match fs::read_to_string(&path) {
                Ok(t) => t,
                Err(_) => continue,
            };
            if !txt.contains(needle) {
                continue;
            }
            let rel = path
                .strip_prefix(&repo)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();
            // Normalise to forward slashes for cross-platform parity (tests
            // run on macOS today; future Linux CI parity is cheap).
            let rel_norm = rel.replace('\\', "/");
            let rel_ref: &str = rel_norm.as_ref();
            if !allowset.iter().any(|a| **a == rel_ref) {
                violations.push(rel_norm);
            }
        }
    }
    assert!(
        violations.is_empty(),
        "RAM Phase G — disallowed-methods regression: new {} call site(s) not on allowlist: {:?}. \
         Either migrate the new code through engine::mediator::{{SignalEffector|SysctlEffector}}, \
         or add the file to the allowlist constant in this test if a switch-over is in flight.",
        needle,
        violations,
    );
}

#[test]
fn raw_libc_kill_only_in_allowlisted_files() {
    check_needle(
        "libc::kill(",
        LIBC_KILL_ALLOWLIST,
        &["crates/apollo-engine/src", "src/bin"],
    );
}

#[test]
fn raw_sysctlbyname_only_in_allowlisted_files() {
    check_needle(
        "sysctlbyname",
        SYSCTLBYNAME_ALLOWLIST,
        &["crates/apollo-engine/src", "src/bin"],
    );
}
