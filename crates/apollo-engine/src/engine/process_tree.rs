//! Process Tree Tracking — groups child processes under their root application.
//!
//! Chrome spawns 20+ helper processes, Electron apps have renderer/plugin/GPU
//! children.  Today each process is treated independently: if "Google Chrome"
//! is boosted to P-cores but "Google Chrome Helper (Renderer)" PID 5432 isn't
//! matched by name patterns, the actual render thread that matters gets missed.
//!
//! This module builds a process tree from the current snapshot and provides:
//!
//!   - `resolve_app_name(pid)` — maps any child PID to its root app name
//!   - `cascade_tier(pid, tier)` — returns all PIDs that should inherit a tier
//!   - `app_groups()` — grouped view with aggregate CPU/mem per application
//!
//! Performance target: <1 ms for 500 processes (single HashMap pass to build).

use std::collections::HashMap;

use crate::collector::ProcessStats;

// ── Input ────────────────────────────────────────────────────────────────────

/// Lightweight process entry with parent PID.
///
/// `ProcessStats` in `collector.rs` does not include ppid (it only stores the
/// top-10 snapshot).  The daemon loop already extracts ppid from `sysinfo`, so
/// callers construct `ProcessEntry` from the full process table.
#[derive(Debug, Clone)]
pub struct ProcessEntry {
    pub pid: u32,
    pub ppid: u32,
    pub name: String,
    pub cpu_usage: f32,
    pub memory_bytes: u64,
}

impl ProcessEntry {
    /// Convenience: build from a `ProcessStats` plus a known ppid.
    pub fn from_stats(stats: &ProcessStats, ppid: u32) -> Self {
        Self {
            pid: stats.pid,
            ppid,
            name: stats.name.clone(),
            cpu_usage: stats.cpu_usage,
            memory_bytes: stats.memory_usage,
        }
    }
}

// ── App Group ────────────────────────────────────────────────────────────────

/// A group of processes rooted at one application.
#[derive(Debug, Clone)]
pub struct AppGroup {
    /// PID of the root application process.
    pub root_pid: u32,
    /// Name of the root application (e.g., "Google Chrome").
    pub root_name: String,
    /// All child PIDs (does NOT include root_pid itself).
    pub child_pids: Vec<u32>,
    /// Aggregate CPU usage across root + all children.
    pub total_cpu: f32,
    /// Aggregate memory usage across root + all children.
    pub total_memory: u64,
    /// Number of processes in this group (root + children).
    pub process_count: usize,
}

// ── Constants ────────────────────────────────────────────────────────────────

/// PID 1 on macOS is launchd.  It is the parent of all daemons and adopted
/// orphans.  We never cascade tiers from launchd — doing so would affect the
/// entire system.
const LAUNCHD_PID: u32 = 1;

/// PID 0 is kernel_task.  Never cascade from it either.
const KERNEL_PID: u32 = 0;

/// Maximum depth when walking the parent chain to find the root app.
/// Guards against cycles caused by PID recycling (a child's ppid might
/// point to a recycled PID that happens to be another child).
const MAX_ANCESTOR_DEPTH: usize = 32;

// ── Process Tree ─────────────────────────────────────────────────────────────

/// An in-memory process tree built from a single snapshot.
///
/// Constructed once per daemon cycle via `ProcessTree::build()`, then queried
/// multiple times.  The tree is immutable after construction — if the process
/// table changes, build a new tree.
pub struct ProcessTree {
    /// pid -> (ppid, name) for every known process.
    entries: HashMap<u32, EntryData>,

    /// pid -> root app pid.  Lazily computed during `build()`.
    root_map: HashMap<u32, u32>,

    /// root_pid -> AppGroup.  One entry per root application.
    groups: HashMap<u32, AppGroup>,

    /// pid -> cpu_usage snapshot (from ProcessEntry input).
    cpu_map: HashMap<u32, f32>,
}

struct EntryData {
    ppid: u32,
    name: String,
}

impl ProcessTree {
    /// Build the process tree from a list of process entries.
    ///
    /// Complexity: O(n * d) where d <= MAX_ANCESTOR_DEPTH (effectively O(n)).
    pub fn build(entries: &[ProcessEntry]) -> Self {
        // Phase 1: index all entries by PID.
        let mut entry_map: HashMap<u32, EntryData> = HashMap::with_capacity(entries.len());
        for e in entries {
            entry_map.insert(
                e.pid,
                EntryData {
                    ppid: e.ppid,
                    name: e.name.clone(),
                },
            );
        }

        // Phase 2: for every PID, walk the parent chain to find the root app.
        let mut root_map: HashMap<u32, u32> = HashMap::with_capacity(entries.len());
        for e in entries {
            let root = find_root(e.pid, &entry_map);
            root_map.insert(e.pid, root);
        }

        // Phase 3: build AppGroup for each root.
        let mut groups: HashMap<u32, AppGroup> = HashMap::new();
        for e in entries {
            let root_pid = root_map.get(&e.pid).copied().unwrap_or(e.pid);
            let group = groups.entry(root_pid).or_insert_with(|| {
                let root_name = entry_map
                    .get(&root_pid)
                    .map(|d| d.name.clone())
                    .unwrap_or_default();
                AppGroup {
                    root_pid,
                    root_name,
                    child_pids: Vec::new(),
                    total_cpu: 0.0,
                    total_memory: 0,
                    process_count: 0,
                }
            });

            // Skip NaN/Inf CPU values so one bad reading doesn't poison the sum.
            if e.cpu_usage.is_finite() {
                group.total_cpu += e.cpu_usage;
            }
            // Saturating add for memory to prevent overflow.
            group.total_memory = group.total_memory.saturating_add(e.memory_bytes);
            group.process_count += 1;

            if e.pid != root_pid {
                group.child_pids.push(e.pid);
            }
        }

        // Build cpu_map for idle_children queries.
        let cpu_map: HashMap<u32, f32> = entries.iter().map(|e| (e.pid, e.cpu_usage)).collect();

        Self {
            entries: entry_map,
            root_map,
            groups,
            cpu_map,
        }
    }

    // ── Queries ──────────────────────────────────────────────────────────────

    /// Resolve any PID to the name of its root application.
    ///
    /// Returns `None` if the PID is not in the tree (it may have exited between
    /// snapshot and query, or was never collected).
    ///
    /// Examples:
    ///   - PID of "Google Chrome Helper (Renderer)" -> Some("Google Chrome")
    ///   - PID of "Google Chrome" itself -> Some("Google Chrome")
    ///   - PID of launchd -> Some("launchd")
    pub fn resolve_app_name(&self, pid: u32) -> Option<&str> {
        let root_pid = self.root_map.get(&pid)?;
        self.entries.get(root_pid).map(|d| d.name.as_str())
    }

    /// Get the root PID for any process in the tree.
    pub fn resolve_root_pid(&self, pid: u32) -> Option<u32> {
        self.root_map.get(&pid).copied()
    }

    /// Get all PIDs that should inherit a tier when `pid` is classified.
    ///
    /// If `pid` is a root app, returns all its children.
    /// If `pid` is a child, returns all siblings (other children of the same root)
    /// plus the root itself — the entire app group.
    ///
    /// Never cascades from launchd (PID 1) or kernel_task (PID 0).
    pub fn cascade_pids(&self, pid: u32) -> Vec<u32> {
        let root_pid = match self.root_map.get(&pid) {
            Some(&r) => r,
            None => return Vec::new(),
        };

        // Never cascade from system roots.
        if root_pid == LAUNCHD_PID || root_pid == KERNEL_PID {
            return Vec::new();
        }

        let group = match self.groups.get(&root_pid) {
            Some(g) => g,
            None => return Vec::new(),
        };

        // Return all PIDs in the group (root + children).
        let mut pids = Vec::with_capacity(group.process_count);
        pids.push(root_pid);
        pids.extend_from_slice(&group.child_pids);
        pids
    }

    /// Get the app group for a given PID (resolves to the root first).
    pub fn app_group(&self, pid: u32) -> Option<&AppGroup> {
        let root_pid = self.root_map.get(&pid)?;
        self.groups.get(root_pid)
    }

    /// Iterate over all app groups.
    pub fn app_groups(&self) -> impl Iterator<Item = &AppGroup> {
        self.groups.values()
    }

    /// Get the direct parent PID of a process.
    pub fn parent_pid(&self, pid: u32) -> Option<u32> {
        self.entries.get(&pid).map(|d| d.ppid)
    }

    /// Get the direct children of a PID.
    ///
    /// This is an O(n) scan; prefer `cascade_pids()` for the common case.
    pub fn children_of(&self, pid: u32) -> Vec<u32> {
        self.entries
            .iter()
            .filter(|(_, data)| data.ppid == pid)
            .map(|(&child_pid, _)| child_pid)
            .collect()
    }

    /// Number of processes in the tree.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of app groups (root applications).
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Check if a PID is the root of its app group (not a child).
    pub fn is_root(&self, pid: u32) -> bool {
        self.root_map.get(&pid) == Some(&pid)
    }

    /// Returns child PIDs of `parent_pid` that are idle:
    /// - cpu_usage < `cpu_threshold` (not actively computing)
    /// - not present in `active_set` (no power assertion, no open sockets, etc.)
    ///
    /// Used for subprocess-selective freeze: freeze idle renderer/worker children
    /// while leaving active audio workers, download helpers, and visible renderers
    /// untouched.
    ///
    /// Does NOT include the parent itself — only direct children.
    pub fn idle_children(
        &self,
        parent_pid: u32,
        cpu_threshold: f32,
        active_set: &std::collections::HashSet<u32>,
    ) -> Vec<u32> {
        self.entries
            .iter()
            .filter(|(&child_pid, data)| {
                data.ppid == parent_pid
                    && child_pid != parent_pid
                    && !active_set.contains(&child_pid)
                    && self
                        .cpu_map
                        .get(&child_pid)
                        .map(|&cpu| cpu < cpu_threshold)
                        .unwrap_or(true) // unknown CPU = treat as idle
            })
            .map(|(&pid, _)| pid)
            .collect()
    }
}

// ── Private helpers ──────────────────────────────────────────────────────────

/// Walk the parent chain from `pid` to find the root application.
///
/// Stops at:
///   - launchd (PID 1) or kernel_task (PID 0): the child of launchd is the root
///   - a PID whose parent is not in the entry map (orphan or system process)
///   - MAX_ANCESTOR_DEPTH to guard against cycles from PID recycling
///   - a PID that points to itself (ppid == pid)
fn find_root(pid: u32, entries: &HashMap<u32, EntryData>) -> u32 {
    let mut current = pid;
    let mut visited = 0usize;

    loop {
        if visited >= MAX_ANCESTOR_DEPTH {
            // Depth limit hit — treat current as the root to avoid infinite loops.
            return current;
        }

        let entry = match entries.get(&current) {
            Some(e) => e,
            // Parent not in the process table — current is the effective root.
            None => return current,
        };

        let parent = entry.ppid;

        // Stop conditions: parent is launchd/kernel, parent is self, or parent
        // is not in the table.
        if parent == LAUNCHD_PID || parent == KERNEL_PID {
            return current;
        }
        if parent == current {
            // Self-referencing ppid (should only happen for PID 0/1, but guard).
            return current;
        }
        if !entries.contains_key(&parent) {
            // Parent not in our snapshot — current is the effective root.
            return current;
        }

        current = parent;
        visited += 1;
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(pid: u32, ppid: u32, name: &str, cpu: f32, mem: u64) -> ProcessEntry {
        ProcessEntry {
            pid,
            ppid,
            name: name.to_string(),
            cpu_usage: cpu,
            memory_bytes: mem,
        }
    }

    #[test]
    fn empty_process_list() {
        let tree = ProcessTree::build(&[]);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.group_count(), 0);
        assert!(tree.resolve_app_name(1234).is_none());
        assert!(tree.cascade_pids(1234).is_empty());
    }

    #[test]
    fn single_process_under_launchd() {
        let entries = vec![entry(100, LAUNCHD_PID, "Safari", 5.0, 1024)];
        let tree = ProcessTree::build(&entries);

        assert_eq!(tree.len(), 1);
        assert_eq!(tree.group_count(), 1);
        assert_eq!(tree.resolve_app_name(100), Some("Safari"));
        assert!(tree.is_root(100));
    }

    #[test]
    fn chrome_with_helpers() {
        let entries = vec![
            entry(200, LAUNCHD_PID, "Google Chrome", 10.0, 500_000),
            entry(201, 200, "Google Chrome Helper (Renderer)", 25.0, 300_000),
            entry(202, 200, "Google Chrome Helper (GPU)", 8.0, 200_000),
            entry(203, 201, "Google Chrome Helper (Plugin)", 2.0, 50_000),
        ];

        let tree = ProcessTree::build(&entries);

        // All resolve to Chrome.
        assert_eq!(tree.resolve_app_name(200), Some("Google Chrome"));
        assert_eq!(tree.resolve_app_name(201), Some("Google Chrome"));
        assert_eq!(tree.resolve_app_name(202), Some("Google Chrome"));
        assert_eq!(tree.resolve_app_name(203), Some("Google Chrome"));

        // Root PID resolution.
        assert_eq!(tree.resolve_root_pid(203), Some(200));
        assert_eq!(tree.resolve_root_pid(200), Some(200));

        // One group.
        assert_eq!(tree.group_count(), 1);
        let group = tree.app_group(201).expect("group should exist");
        assert_eq!(group.root_pid, 200);
        assert_eq!(group.root_name, "Google Chrome");
        assert_eq!(group.process_count, 4);
        assert_eq!(group.child_pids.len(), 3);

        // Aggregate stats.
        let expected_cpu = 10.0 + 25.0 + 8.0 + 2.0;
        assert!((group.total_cpu - expected_cpu).abs() < 0.01);
        assert_eq!(group.total_memory, 500_000 + 300_000 + 200_000 + 50_000);

        // Cascade from any PID in the group returns all.
        let cascaded = tree.cascade_pids(203);
        assert_eq!(cascaded.len(), 4);
        assert!(cascaded.contains(&200));
        assert!(cascaded.contains(&201));
        assert!(cascaded.contains(&202));
        assert!(cascaded.contains(&203));

        // Root check.
        assert!(tree.is_root(200));
        assert!(!tree.is_root(201));
    }

    #[test]
    fn launchd_children_do_not_cascade() {
        // Two independent daemons under launchd.
        let entries = vec![
            entry(100, LAUNCHD_PID, "configd", 1.0, 1000),
            entry(200, LAUNCHD_PID, "mDNSResponder", 0.5, 2000),
        ];

        let tree = ProcessTree::build(&entries);

        // Each is its own root.
        assert_eq!(tree.group_count(), 2);
        assert!(tree.is_root(100));
        assert!(tree.is_root(200));

        // Cascade returns only the single process (no siblings).
        assert_eq!(tree.cascade_pids(100), vec![100]);
        assert_eq!(tree.cascade_pids(200), vec![200]);
    }

    #[test]
    fn cascade_never_returns_launchd_group() {
        // If someone passes PID 1 itself, cascade should be empty.
        let entries = vec![
            entry(LAUNCHD_PID, 0, "launchd", 0.1, 500),
            entry(100, LAUNCHD_PID, "Safari", 5.0, 1000),
        ];
        let tree = ProcessTree::build(&entries);

        // launchd's root is itself (ppid=0 -> kernel, stop).
        // But cascade from launchd should return empty (special case).
        assert!(tree.cascade_pids(LAUNCHD_PID).is_empty());
    }

    #[test]
    fn orphan_process_parent_not_in_table() {
        // PID 300 has ppid 999 which is not in the table.
        let entries = vec![
            entry(300, 999, "orphan_worker", 3.0, 5000),
            entry(301, 300, "orphan_child", 1.0, 1000),
        ];

        let tree = ProcessTree::build(&entries);

        // 300 becomes its own root (parent 999 not found).
        assert!(tree.is_root(300));
        assert_eq!(tree.resolve_app_name(301), Some("orphan_worker"));
        assert_eq!(tree.group_count(), 1);
    }

    #[test]
    fn pid_recycling_cycle_detection() {
        // Simulate a cycle: 400 -> 401 -> 402 -> 400 (impossible in real life,
        // but could happen if PPIDs are stale due to PID recycling).
        let entries = vec![
            entry(400, 402, "cycle_a", 1.0, 100),
            entry(401, 400, "cycle_b", 1.0, 100),
            entry(402, 401, "cycle_c", 1.0, 100),
        ];

        let tree = ProcessTree::build(&entries);

        // Should not panic or loop forever.  All three will resolve to some
        // root (whichever the depth limit picks).
        assert!(tree.resolve_app_name(400).is_some());
        assert!(tree.resolve_app_name(401).is_some());
        assert!(tree.resolve_app_name(402).is_some());
        assert!(tree.len() == 3);
    }

    #[test]
    fn self_referencing_ppid() {
        let entries = vec![entry(500, 500, "stuck", 0.0, 0)];
        let tree = ProcessTree::build(&entries);

        assert_eq!(tree.resolve_app_name(500), Some("stuck"));
        assert!(tree.is_root(500));
    }

    #[test]
    fn multiple_app_groups() {
        let entries = vec![
            entry(100, LAUNCHD_PID, "Safari", 10.0, 500_000),
            entry(101, 100, "Safari Networking", 2.0, 50_000),
            entry(200, LAUNCHD_PID, "Code", 15.0, 800_000),
            entry(201, 200, "Code Helper (Renderer)", 20.0, 400_000),
            entry(202, 200, "Code Helper (Plugin)", 5.0, 100_000),
            entry(300, LAUNCHD_PID, "Finder", 1.0, 100_000),
        ];

        let tree = ProcessTree::build(&entries);
        assert_eq!(tree.group_count(), 3);

        // Safari group.
        let safari = tree.app_group(101).expect("safari group");
        assert_eq!(safari.root_pid, 100);
        assert_eq!(safari.process_count, 2);

        // Code group.
        let code = tree.app_group(202).expect("code group");
        assert_eq!(code.root_pid, 200);
        assert_eq!(code.process_count, 3);

        // Finder is alone.
        let finder = tree.app_group(300).expect("finder group");
        assert_eq!(finder.root_pid, 300);
        assert_eq!(finder.process_count, 1);
        assert!(finder.child_pids.is_empty());
    }

    #[test]
    fn deep_hierarchy_respects_depth_limit() {
        // Build a chain of MAX_ANCESTOR_DEPTH + 10 processes.
        let depth = MAX_ANCESTOR_DEPTH + 10;
        let mut entries = Vec::with_capacity(depth);

        // PID 1000 is the root (under launchd).
        entries.push(entry(1000, LAUNCHD_PID, "deep_root", 1.0, 100));
        for i in 1..depth {
            let pid = 1000 + i as u32;
            let ppid = pid - 1;
            entries.push(entry(pid, ppid, &format!("deep_{}", i), 0.1, 10));
        }

        let tree = ProcessTree::build(&entries);

        // Should not panic.  The deepest process may not resolve all the way
        // to 1000, but should resolve to *something*.
        let deepest_pid = 1000 + (depth - 1) as u32;
        assert!(tree.resolve_app_name(deepest_pid).is_some());
    }

    #[test]
    fn nan_cpu_does_not_poison_group() {
        let entries = vec![
            entry(600, LAUNCHD_PID, "app", 5.0, 1000),
            ProcessEntry {
                pid: 601,
                ppid: 600,
                name: "helper".to_string(),
                cpu_usage: f32::NAN,
                memory_bytes: 500,
            },
        ];

        let tree = ProcessTree::build(&entries);
        let group = tree.app_group(600).expect("group exists");

        // NaN is skipped, so only the valid 5.0 contributes.
        assert!(
            (group.total_cpu - 5.0).abs() < 0.01,
            "total_cpu should be 5.0, got {}",
            group.total_cpu
        );
    }

    #[test]
    fn memory_overflow_saturates() {
        let entries = vec![
            entry(700, LAUNCHD_PID, "big_app", 1.0, u64::MAX - 10),
            entry(701, 700, "helper", 1.0, 100),
        ];

        let tree = ProcessTree::build(&entries);
        let group = tree.app_group(700).expect("group exists");

        // Should saturate to u64::MAX, not wrap around.
        assert_eq!(group.total_memory, u64::MAX);
    }

    #[test]
    fn from_stats_convenience() {
        let stats = ProcessStats {
            pid: 42,
            name: "test_proc".to_string(),
            cpu_usage: 3.5,
            memory_usage: 9999,
            cpu_wall_ratio: None,
        };
        let pe = ProcessEntry::from_stats(&stats, 1);
        assert_eq!(pe.pid, 42);
        assert_eq!(pe.ppid, 1);
        assert_eq!(pe.name, "test_proc");
        assert!((pe.cpu_usage - 3.5).abs() < f32::EPSILON);
        assert_eq!(pe.memory_bytes, 9999);
    }

    #[test]
    fn children_of_query() {
        let entries = vec![
            entry(100, LAUNCHD_PID, "parent", 1.0, 100),
            entry(101, 100, "child_a", 1.0, 100),
            entry(102, 100, "child_b", 1.0, 100),
            entry(103, 101, "grandchild", 1.0, 100),
        ];

        let tree = ProcessTree::build(&entries);
        let mut children = tree.children_of(100);
        children.sort();
        assert_eq!(children, vec![101, 102]);

        let grandchildren = tree.children_of(101);
        assert_eq!(grandchildren, vec![103]);

        assert!(tree.children_of(103).is_empty());
    }

    #[test]
    fn unknown_pid_queries_return_none() {
        let entries = vec![entry(100, LAUNCHD_PID, "app", 1.0, 100)];
        let tree = ProcessTree::build(&entries);

        assert!(tree.resolve_app_name(9999).is_none());
        assert!(tree.resolve_root_pid(9999).is_none());
        assert!(tree.app_group(9999).is_none());
        assert!(tree.parent_pid(9999).is_none());
        assert!(tree.cascade_pids(9999).is_empty());
    }

    #[test]
    fn kernel_task_does_not_cascade() {
        let entries = vec![
            entry(KERNEL_PID, 0, "kernel_task", 0.0, 0),
            entry(LAUNCHD_PID, KERNEL_PID, "launchd", 0.1, 500),
        ];
        let tree = ProcessTree::build(&entries);

        // kernel_task's root is itself (ppid=0 = KERNEL_PID, self-ref stop).
        // cascade from kernel_task should be empty.
        assert!(tree.cascade_pids(KERNEL_PID).is_empty());
    }

    #[test]
    fn duplicate_pid_last_wins() {
        // If the input has duplicate PIDs (shouldn't happen, but defensive).
        let entries = vec![
            entry(100, LAUNCHD_PID, "first", 1.0, 100),
            entry(100, LAUNCHD_PID, "second", 2.0, 200),
        ];
        let tree = ProcessTree::build(&entries);

        // HashMap insert overwrites, so the second entry wins for entry data.
        // But both entries contribute to the group aggregation since we iterate
        // the input slice.
        assert_eq!(tree.len(), 1);
        // The name in the map is "second" (last insert wins).
        assert_eq!(tree.resolve_app_name(100), Some("second"));
    }
}
