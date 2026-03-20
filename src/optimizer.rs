use crate::collector::SystemSnapshot;
use crate::engine::sysctl_governor::SysctlGovernor;
use crate::engine::types::HardPath;
use libc::{kill, SIGCONT, SIGSTOP};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};
use sysinfo::System;

enum DefaultsValue {
    Bool(bool),
    Float(f64),
}

/// Write a macOS user defaults value via CoreFoundation CFPreferences.
/// Replaces `defaults write domain key [-type] value`.
fn defaults_write(domain: &str, key: &str, value: DefaultsValue) {
    #[cfg(not(target_os = "macos"))]
    { let _ = (domain, key, value); return; }

    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn CFStringCreateWithCString(
                alloc: *const std::ffi::c_void,
                cstr: *const i8,
                encoding: u32,
            ) -> *const std::ffi::c_void;
            fn CFPreferencesSetAppValue(
                key: *const std::ffi::c_void,
                value: *const std::ffi::c_void,
                app_id: *const std::ffi::c_void,
            );
            fn CFPreferencesAppSynchronize(app_id: *const std::ffi::c_void) -> bool;
            fn CFRelease(cf: *const std::ffi::c_void);

            // kCFBooleanTrue / kCFBooleanFalse
            static kCFBooleanTrue: *const std::ffi::c_void;
            static kCFBooleanFalse: *const std::ffi::c_void;

            fn CFNumberCreate(
                alloc: *const std::ffi::c_void,
                the_type: i64,
                value_ptr: *const std::ffi::c_void,
            ) -> *const std::ffi::c_void;
        }

        const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
        const K_CF_NUMBER_FLOAT64_TYPE: i64 = 13;

        unsafe {
            let cf_domain = CFStringCreateWithCString(
                std::ptr::null(),
                std::ffi::CString::new(domain).unwrap().as_ptr(),
                K_CF_STRING_ENCODING_UTF8,
            );
            let cf_key = CFStringCreateWithCString(
                std::ptr::null(),
                std::ffi::CString::new(key).unwrap().as_ptr(),
                K_CF_STRING_ENCODING_UTF8,
            );
            if cf_domain.is_null() || cf_key.is_null() {
                if !cf_domain.is_null() { CFRelease(cf_domain); }
                if !cf_key.is_null() { CFRelease(cf_key); }
                return;
            }

            match value {
                DefaultsValue::Bool(b) => {
                    let cf_val = if b { kCFBooleanTrue } else { kCFBooleanFalse };
                    CFPreferencesSetAppValue(cf_key, cf_val, cf_domain);
                }
                DefaultsValue::Float(f) => {
                    let cf_num = CFNumberCreate(
                        std::ptr::null(),
                        K_CF_NUMBER_FLOAT64_TYPE,
                        &f as *const f64 as *const _,
                    );
                    if !cf_num.is_null() {
                        CFPreferencesSetAppValue(cf_key, cf_num, cf_domain);
                        CFRelease(cf_num);
                    }
                }
            }

            CFPreferencesAppSynchronize(cf_domain);
            CFRelease(cf_domain);
            CFRelease(cf_key);
        }
    }
}

/// Send a signal to all processes matching a name. Replaces `killall`.
fn signal_process_by_name(name: &str, signal: i32) {
    let sys = sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::new().with_processes(sysinfo::ProcessRefreshKind::new()),
    );
    for (pid, proc_info) in sys.processes() {
        if proc_info.name() == name {
            unsafe { libc::kill(pid.as_u32() as i32, signal); }
        }
    }
}

pub struct HeuristicEngine {
    // Process Name -> (consecutive high-CPU count, last_seen Instant)
    // BUG 15 fix: store timestamp so stale entries can be purged.
    noise_count: Mutex<HashMap<String, (u32, Instant)>>,
    // Known patterns of 'good' vs 'bad'
    pro_patterns: Vec<String>,
    noise_patterns: Vec<String>,
}

#[allow(clippy::new_without_default)]
impl HeuristicEngine {
    pub fn new() -> Self {
        Self {
            noise_count: Mutex::new(HashMap::new()),
            pro_patterns: vec![
                "rust".into(),
                "cargo".into(),
                "node".into(),
                "compiler".into(),
                "studio".into(),
                "engine".into(),
                "chrome".into(),
                "brave".into(),
            ],
            noise_patterns: vec![
                "agent".into(),
                "daemon".into(),
                "updater".into(),
                "sync".into(),
                "helper".into(),
                "analytics".into(),
            ],
        }
    }

    pub fn analyze(&self, snapshot: &crate::collector::SystemSnapshot) -> Vec<String> {
        let mut discovered_noise = Vec::new();
        let mut noise_map = safe_lock(&self.noise_count);
        let now = Instant::now();

        // BUG 15 fix: purge entries not seen in the last 10 minutes to prevent unbounded growth.
        noise_map
            .retain(|_, (_, last_seen)| now.duration_since(*last_seen) < Duration::from_secs(600));

        for process in &snapshot.top_processes {
            // Heuristic 1: Constant CPU (>5%) in a daemon/agent/helper
            if process.cpu_usage > 5.0 {
                let entry = noise_map.entry(process.name.clone()).or_insert((0, now));
                entry.0 += 1;
                entry.1 = now; // refresh last_seen

                // Heuristic: Is it noise? (matches noise patterns)
                let looks_like_noise = self
                    .noise_patterns
                    .iter()
                    .any(|p| process.name.to_lowercase().contains(p));
                // Protection: Does it look like a pro tool?
                let looks_like_pro = self
                    .pro_patterns
                    .iter()
                    .any(|p| process.name.to_lowercase().contains(p));

                // If seen in 5 consecutive snapshots (5 minutes) and matches noise but NOT pro patterns
                if entry.0 >= 5 && looks_like_noise && !looks_like_pro {
                    discovered_noise.push(process.name.clone());
                }
            } else {
                // BUG 15 fix: reset counter if process is no longer noisy.
                if let Some(entry) = noise_map.get_mut(&process.name) {
                    entry.0 = entry.0.saturating_sub(1);
                    entry.1 = now;
                }
            }
        }
        discovered_noise
    }
}

pub struct OptimizerEngine {
    high_priority_apps: Vec<String>,
    low_priority_apps: Vec<String>,
    freezable_apps: Vec<String>,
    llm_apps: Vec<String>,
    dev_apps: Vec<String>,
    media_apps: Vec<String>,
    wait_graph_blockers: Vec<String>,
    uma_texture_apps: Vec<String>,
    frozen_processes: Mutex<HashSet<u32>>,
    current_mode: Mutex<String>,
    heuristics: HeuristicEngine,
    auto_discovered_noise: Mutex<HashSet<String>>,
    system_critical: HashSet<String>,
    wait_boost_cooldowns: Mutex<HashMap<String, Instant>>,
    last_preemptive_paging: Mutex<Option<Instant>>,
    history: Mutex<VecDeque<SystemSnapshot>>, // Rolling 1-hour history
    app_profiles: Mutex<HashMap<String, AppProfile>>, // Learned behaviors
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct AppProfile {
    pub peak_memory: u64,
    pub avg_cpu: f32,
    pub launch_count: u32,
}

fn safe_lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            eprintln!("⚠️ Recovered from a poisoned mutex; continuing with inner state.");
            poisoned.into_inner()
        }
    }
}

#[allow(clippy::new_without_default)]
impl OptimizerEngine {
    pub fn new() -> Self {
        let mut system_critical = HashSet::new();
        let critical_names = vec![
            "kernel_task",
            "launchd",
            "WindowServer",
            "loginwindow",
            "hidd",
            "configd",
            "opendirectoryd",
            "notifyd",
            "UserEventAgent",
            "securityd",
            "syspolicyd",
            "tccd",
            "CoreServicesUIAgent",
            // Spotlight stack (never touch: search/indexing must remain reliable).
            "Spotlight",
            "mds",
            "mds_stores",
            "mdworker",
            "mdworker_shared",
        ];
        for name in critical_names {
            system_critical.insert(name.to_string());
        }

        Self {
            app_profiles: Mutex::new(Self::load_profiles()),
            history: Mutex::new(VecDeque::with_capacity(240)), // ~1 hour at 15s ticks
            high_priority_apps: vec![
                "Code".to_string(),
                "Terminal".to_string(),
                "iTerm2".to_string(),
                "Zoom".to_string(),
                "Obsidian".to_string(),
                "Arc".to_string(),
                "Google Chrome".to_string(),
                "Warp".to_string(),
                "Lapce".to_string(),
                "iTerm".to_string(),
                "pgAdmin 4".to_string(),
            ],
            low_priority_apps: vec![
                "Dropbox".to_string(),
                "Google Drive".to_string(),
                "Backup and Sync".to_string(),
                "OneDrive".to_string(),
                "Docker".to_string(),
                // Noise Daemons (Badly coded background agents)
                "knowledge-agent".to_string(),
                "suggestd".to_string(),
                "triald".to_string(),
                "corefollowupd".to_string(),
                "parsec-fbf".to_string(),
                "translationd".to_string(),
                // User-Specific Noise (Detected from ps/ls)
                "logioptionsplus".to_string(),     // Logitech Bloat
                "corespeechd".to_string(),         // Siri/Dictation Waste
                "postgres".to_string(),            // Database background when idle
                "STExtractionService".to_string(), // Screen Time/Text Extract
            ],
            freezable_apps: vec![
                "Slack".to_string(),
                "Discord".to_string(),
                "Spotify".to_string(),
                "Webex".to_string(),
                "Microsoft Teams".to_string(),
            ],
            llm_apps: vec![
                "ollama".to_string(),
                "llama".to_string(),
                "python".to_string(),
                "Ollama".to_string(),
                "LM Studio".to_string(),
                "Cursor".to_string(),
            ],
            dev_apps: vec![
                "rustc".to_string(),
                "cargo".to_string(),
                "node".to_string(),
                "docker".to_string(),
                "go".to_string(),
                "python3".to_string(),
                "Xcode".to_string(),
                "antigravity".to_string(),
                "Visual Studio Code".to_string(),
                "Code".to_string(),
            ],
            media_apps: vec![
                "Final Cut Pro".to_string(),
                "Adobe Premiere Pro".to_string(),
                "DaVinci Resolve".to_string(),
                "OBS".to_string(),
                "ffmpeg".to_string(),
            ],
            wait_graph_blockers: vec![
                "WindowServer".to_string(),
                "accountsd".to_string(),
                "cfprefsd".to_string(),
                "distnoted".to_string(),
                "cloudd".to_string(),
                "photolibraryd".to_string(),
            ],
            uma_texture_apps: vec![
                "Final Cut Pro".to_string(),
                "Adobe Premiere Pro".to_string(),
                "DaVinci Resolve".to_string(),
                "OBS".to_string(),
                "Google Chrome".to_string(),
                "Arc".to_string(),
                "Safari".to_string(),
                "Discord".to_string(),
                "Slack".to_string(),
            ],
            frozen_processes: Mutex::new(HashSet::new()),
            current_mode: Mutex::new("normal".to_string()),
            heuristics: HeuristicEngine::new(),
            auto_discovered_noise: Mutex::new(HashSet::new()),
            system_critical,
            wait_boost_cooldowns: Mutex::new(HashMap::new()),
            last_preemptive_paging: Mutex::new(None),
        }
    }

    pub fn optimize(&self, snapshot: &SystemSnapshot) {
        // 0. Autonomous Discovery
        let new_noise = self.heuristics.analyze(snapshot);
        if !new_noise.is_empty() {
            let mut auto_noise = safe_lock(&self.auto_discovered_noise);
            for n in new_noise {
                if !auto_noise.contains(&n) {
                    println!(
                        "🧠 AUTONOMOUS DISCOVERY: Identified '{}' as persistent background noise.",
                        n
                    );
                    auto_noise.insert(n);
                }
            }
        }

        // (Periodic save moved to end of method)

        println!(
            "Starting optimization based on snapshot from {}",
            snapshot.timestamp
        );

        // --- TRACE & LEARN: Push to history ---
        {
            let mut history = safe_lock(&self.history);
            if history.len() >= 240 {
                // Cap at 1 hour
                history.pop_front();
            }
            history.push_back(snapshot.clone());
        }

        // --- TRACE & LEARN: Analyze trends ---
        self.analyze_resource_trends();

        let mut sys = System::new_all();
        sys.refresh_processes();

        // Calculate overall load
        let total_ram = snapshot.memory.total_ram;
        let used_ram = snapshot.memory.used_ram;
        let ram_pressure = (used_ram as f64 / total_ram as f64) * 100.0;
        let cpu_load = snapshot.cpu.global_usage;

        let is_heavy_load = ram_pressure > 85.0 || cpu_load > 80.0;

        if is_heavy_load {
            println!(
                "⚠️ HEAVY LOAD DETECTED (RAM: {:.1}%, CPU: {:.1}%) - Engaging TURBO optimizations",
                ram_pressure, cpu_load
            );
        }

        for (pid, process) in sys.processes() {
            let name = process.name();
            let pid_u32 = pid.as_u32();

            // 0. SAFETY GUARD: Never touch system-critical kernel processes
            if self.system_critical.contains(name) {
                continue;
            }

            // 0.5 PROTECTION: Google Chrome & Sub-processes (Never throttle, always interactive)
            if name.contains("Google Chrome") || name.contains("Chrome") {
                // Ensure Chrome gets the best cores and high priority
                println!(
                    "🛡️  PROTECTED: Ensuring Google Chrome ({}) is interactive.",
                    name
                );
                self.set_high_priority(pid_u32, name);
                continue;
            }

            // 1. SILENT GROOMING: Always keep noise in efficiency policy.
            if self.is_noise_process(name) {
                self.apply_silent_grooming(pid_u32, name, is_heavy_load);
            }

            // 2. INTERACTIVE BOOSTING: keep professional foreground flows responsive.
            if self.is_interactive_process(name) {
                self.apply_interactive_boosting(pid_u32, name);
            }

            // 3. QUARANTINE (EXTREME): freeze non-essential chat/media apps when pressure is high.
            if self.freezable_apps.iter().any(|app| name.contains(app)) {
                self.apply_quarantine_extreme(pid_u32, name, is_heavy_load);
            }
        }

        // Phase 8 - Holographic optimizer layer:
        // Wait-Graph + UMA scavenging + pre-emptive paging.
        self.run_holographic_optimizer(snapshot, &sys, is_heavy_load);

        // Check for WindowServer high usage
        if let Some(ws_process) = snapshot
            .top_processes
            .iter()
            .find(|p| p.name == "WindowServer")
        {
            if ws_process.cpu_usage > 40.0 {
                println!(
                    "ALERT: WindowServer high CPU usage: {}%",
                    ws_process.cpu_usage
                );
            }
        }

        // Check RAM Pressure
        self.check_ram_pressure(snapshot, is_heavy_load);

        // --- NEW: Dynamic Pro Performance Detection ---
        let mut is_performance_workload = false;
        let pro_apps = [&self.llm_apps, &self.dev_apps, &self.media_apps];

        for process in &snapshot.top_processes {
            if pro_apps
                .iter()
                .any(|list| list.iter().any(|app| process.name.contains(app)))
            {
                is_performance_workload = true;
                break;
            }
        }

        if is_performance_workload {
            let mut mode = safe_lock(&self.current_mode);
            if *mode != "pro" {
                println!("🚀 Pro-Workload detected (LLM/Dev/Media)! Activating Apollo Performance Mode...");
                let gov = SysctlGovernor::new(unsafe { libc::geteuid() } == 0);
                gov.apply_tuning_direct();
                self.quarantine_apple_bloat(true);
                self.apply_gpu_eco_mode(true);
                *mode = "pro".to_string();
            }
        } else {
            let mut mode = safe_lock(&self.current_mode);
            if *mode != "normal" {
                println!("🍃 Silent Efficiency Active. Background noise suppressed.");
                *mode = "normal".to_string();
                self.restore_background_noise();
            }
        }

        // --- Periodic Persistence ---
        self.run_periodic_save();
    }

    fn is_interactive_process(&self, name: &str) -> bool {
        self.high_priority_apps.iter().any(|app| name.contains(app))
            || self.dev_apps.iter().any(|app| name.contains(app))
            || self.llm_apps.iter().any(|app| name.contains(app))
    }

    fn is_noise_process(&self, name: &str) -> bool {
        let is_discovered_noise = {
            let auto_noise = safe_lock(&self.auto_discovered_noise);
            auto_noise.contains(name)
        };
        self.low_priority_apps.iter().any(|app| name.contains(app)) || is_discovered_noise
    }

    fn apply_interactive_boosting(&self, pid: u32, name: &str) {
        self.set_high_priority(pid, name);
    }

    fn apply_silent_grooming(&self, pid: u32, name: &str, aggressive: bool) {
        // This keeps background daemons on efficiency-biased scheduling.
        self.set_low_priority(pid, name, aggressive);
    }

    fn apply_quarantine_extreme(&self, pid: u32, name: &str, active: bool) {
        if active {
            self.freeze_process(pid, name);
        } else {
            self.unfreeze_process(pid, name);
        }
    }

    fn run_holographic_optimizer(
        &self,
        snapshot: &SystemSnapshot,
        sys: &System,
        is_heavy_load: bool,
    ) {
        self.run_wait_graph_analysis(snapshot, sys);
        self.run_uma_scavenging(snapshot, sys, is_heavy_load);
        self.run_preemptive_paging(snapshot, sys);
    }

    fn run_wait_graph_analysis(&self, snapshot: &SystemSnapshot, sys: &System) {
        let interactive_waiters = snapshot
            .top_processes
            .iter()
            .filter(|proc| {
                self.is_interactive_process(&proc.name)
                    && proc.cpu_usage < 8.0
                    && proc.memory_usage > 128 * 1024 * 1024
            })
            .count();

        let runtime_wait_ratio = self.sample_runtime_wait_ratio().unwrap_or(0.0);
        if interactive_waiters == 0 && runtime_wait_ratio < 0.6 {
            return;
        }

        let mut best_blocker: Option<(u32, String, f32)> = None;
        for (pid, process) in sys.processes() {
            let name = process.name();
            if !self
                .wait_graph_blockers
                .iter()
                .any(|item| name.contains(item))
            {
                continue;
            }

            let cpu = process.cpu_usage();
            if cpu < 10.0 {
                continue;
            }

            match &best_blocker {
                Some((_, _, best_cpu)) if cpu <= *best_cpu => {}
                _ => best_blocker = Some((pid.as_u32(), name.to_string(), cpu)),
            }
        }

        if let Some((pid, name, cpu)) = best_blocker {
            if self.can_boost_blocker_now(&name) {
                println!(
                    "🕸️ WAIT-GRAPH: '{}' is a likely flow blocker (CPU {:.1}%). Applying temporary boost.",
                    name, cpu
                );
                self.set_high_priority(pid, &name);
            }
        }
    }

    fn sample_runtime_wait_ratio(&self) -> Option<f32> {
        unsafe {
            let mut thread_list: *mut u32 = std::ptr::null_mut();
            let mut thread_count: u32 = 0;
            let task = mach_task_self();

            if task_threads(task, &mut thread_list, &mut thread_count) != 0
                || thread_count == 0
                || thread_list.is_null()
            {
                return None;
            }

            let threads = std::slice::from_raw_parts(thread_list, thread_count as usize);
            let mut waiting = 0u32;

            for thread in threads {
                let mut info: ThreadBasicInfo = std::mem::zeroed();
                let mut count = THREAD_BASIC_INFO_COUNT;

                let result = thread_info(
                    *thread,
                    THREAD_BASIC_INFO,
                    &mut info as *mut _ as *mut i32,
                    &mut count,
                );

                if result == 0 && info.run_state == TH_STATE_WAITING {
                    waiting += 1;
                }

                let _ = mach_port_deallocate(task, *thread);
            }

            let _ = mach_vm_deallocate(
                task,
                thread_list as u64,
                (thread_count as usize * std::mem::size_of::<u32>()) as u64,
            );

            Some(waiting as f32 / thread_count as f32)
        }
    }

    fn can_boost_blocker_now(&self, name: &str) -> bool {
        let mut cooldowns = safe_lock(&self.wait_boost_cooldowns);
        let now = Instant::now();
        let cooldown_window = Duration::from_secs(25);

        if let Some(last) = cooldowns.get(name) {
            if now.duration_since(*last) < cooldown_window {
                return false;
            }
        }

        cooldowns.insert(name.to_string(), now);
        cooldowns.retain(|_, ts| now.duration_since(*ts) < Duration::from_secs(300));
        true
    }

    fn run_uma_scavenging(&self, snapshot: &SystemSnapshot, sys: &System, is_heavy_load: bool) {
        if !cfg!(target_arch = "aarch64") {
            return;
        }

        let ram_pressure =
            (snapshot.memory.used_ram as f64 / snapshot.memory.total_ram as f64) * 100.0;
        if ram_pressure < 72.0 && !is_heavy_load {
            return;
        }

        let mut shifted = 0u32;
        for (pid, process) in sys.processes() {
            let name = process.name();

            if self.is_interactive_process(name) {
                continue;
            }

            let is_gpu_weighted = self.media_apps.iter().any(|app| name.contains(app))
                || self.uma_texture_apps.iter().any(|app| name.contains(app));
            if !is_gpu_weighted {
                continue;
            }

            if process.cpu_usage() > 6.0 {
                continue;
            }

            self.set_low_priority(pid.as_u32(), name, true);
            shifted += 1;
        }

        if shifted > 0 {
            println!(
                "🧬 UMA SCAVENGING: moved {} background GPU-heavy processes to efficiency policy.",
                shifted
            );
        }
    }

    fn run_preemptive_paging(&self, snapshot: &SystemSnapshot, sys: &System) {
        let current_pressure =
            (snapshot.memory.used_ram as f64 / snapshot.memory.total_ram as f64) * 100.0;
        let projected_pressure = self.get_pressure_projection().unwrap_or(current_pressure);

        if current_pressure < 75.0 && projected_pressure < 80.0 {
            return;
        }

        if !self.can_run_preemptive_paging() {
            return;
        }

        println!(
            "🧊 PRE-EMPTIVE PAGING: pressure {:.1}% (projected {:.1}%). Cooling cold processes.",
            current_pressure, projected_pressure
        );

        let mut hinted = 0u32;
        for (pid, process) in sys.processes() {
            let name = process.name();
            if self.system_critical.contains(name) || self.is_interactive_process(name) {
                continue;
            }

            if process.cpu_usage() > 2.0 {
                continue;
            }

            if process.memory() < 250 * 1024 * 1024 {
                continue;
            }

            self.set_low_priority(pid.as_u32(), name, true);
            let _ = self.set_memorystatus_priority(pid.as_u32(), JETSAM_PRIORITY_BACKGROUND);
            hinted += 1;

            if hinted >= 6 {
                break;
            }
        }

        if projected_pressure > 86.0 {
            crate::engine::host_vm_info::trigger_purge();
        }
    }

    fn get_pressure_projection(&self) -> Option<f64> {
        let history = safe_lock(&self.history);
        if history.len() < 4 {
            return None;
        }

        let latest = history.back()?;
        let oldest = history.iter().rev().nth(3)?;

        let latest_pressure =
            (latest.memory.used_ram as f64 / latest.memory.total_ram as f64) * 100.0;
        let oldest_pressure =
            (oldest.memory.used_ram as f64 / oldest.memory.total_ram as f64) * 100.0;
        let growth = latest_pressure - oldest_pressure;

        if growth <= 0.0 {
            return Some(latest_pressure);
        }

        Some((latest_pressure + growth * 2.0).min(100.0))
    }

    fn can_run_preemptive_paging(&self) -> bool {
        let mut state = safe_lock(&self.last_preemptive_paging);
        let now = Instant::now();

        if let Some(last) = *state {
            if now.duration_since(last) < Duration::from_secs(90) {
                return false;
            }
        }

        *state = Some(now);
        true
    }

    fn set_memorystatus_priority(&self, pid: u32, priority: i32) -> bool {
        unsafe {
            let mut props = MemorystatusPriorityProperties {
                priority,
                user_data: APOLLO_MEMORY_HINT,
            };

            let status = memorystatus_control(
                MEMORYSTATUS_CMD_SET_PRIORITY_PROPERTIES,
                pid as i32,
                0,
                &mut props as *mut _ as *mut libc::c_void,
                std::mem::size_of::<MemorystatusPriorityProperties>(),
            );

            if status == 0 {
                return true;
            }

            memorystatus_control(
                MEMORYSTATUS_CMD_SET_JETSAM_HIGH_WATER_MARK,
                pid as i32,
                1024,
                std::ptr::null_mut(),
                0,
            ) == 0
        }
    }

    fn analyze_resource_trends(&self) {
        let history = safe_lock(&self.history);
        if history.len() < 10 {
            return; // Not enough data for trend analysis
        }

        // Detect Memory Leaks: Look for processes with consistent growth
        let mut process_traces: HashMap<String, Vec<u64>> = HashMap::new();

        for snapshot in history.iter() {
            for proc in &snapshot.top_processes {
                process_traces
                    .entry(proc.name.clone())
                    .or_default()
                    .push(proc.memory_usage);
            }
        }

        for (name, trace) in process_traces {
            if trace.len() < 10 {
                continue;
            }

            // Heuristic: Is it growing in at least 80% of samples?
            let mut growth_count = 0;
            for i in 1..trace.len() {
                if trace[i] > trace[i - 1] {
                    growth_count += 1;
                }
            }

            let growth_rate = growth_count as f64 / (trace.len() - 1) as f64;
            if let Some(last) = trace.last() {
                if !(growth_rate > 0.8 && *last > (100 * 1024 * 1024)) {
                    continue;
                }
                println!("🔍 TRACE ALERT: Possible Memory Leak in '{}' (Growth Rate: {:.1}% over {} samples)", 
                    name, growth_rate * 100.0, trace.len());

                // If it's a known noise process, we could be more aggressive
                // Or if it's hitting a limit, we could restart it?
            }
        }

        // --- BEHAVIOR PROFILING: Update signatures ---
        if let Some(last_snapshot) = history.back() {
            for proc in &last_snapshot.top_processes {
                let mut profiles = safe_lock(&self.app_profiles);
                let profile = profiles.entry(proc.name.clone()).or_insert(AppProfile {
                    peak_memory: 0,
                    avg_cpu: 0.0,
                    launch_count: 0,
                });

                if proc.memory_usage > profile.peak_memory {
                    profile.peak_memory = proc.memory_usage;
                }
                // BUG 13 fix: use proper EMA instead of the biased two-point average.
                const CPU_EMA_ALPHA: f32 = 0.1;
                profile.avg_cpu =
                    profile.avg_cpu * (1.0 - CPU_EMA_ALPHA) + proc.cpu_usage * CPU_EMA_ALPHA;

                // PREDICTIVE SCALING: If this app is known to be heavy, boost it now
                // This catches apps that are just starting or are in a known heavy phase
                if profile.peak_memory > 1024 * 1024 * 1024 || profile.avg_cpu > 50.0 {
                    // We don't log this every time to avoid spam, but it runs
                    self.set_high_priority(proc.pid, &proc.name);
                }
            }
        }
    }

    fn run_periodic_save(&self) {
        static ITERATION_COUNT: AtomicU32 = AtomicU32::new(0);
        let count = ITERATION_COUNT.fetch_add(1, Ordering::SeqCst);
        if count % 20 == 0 {
            self.save_profiles();
        }
    }

    pub fn get_tick_rate(&self) -> u64 {
        let mode = safe_lock(&self.current_mode);
        match mode.as_str() {
            "pro" | "llm" => 15,
            _ => 60,
        }
    }

    fn check_ram_pressure(&self, snapshot: &SystemSnapshot, force_purge: bool) {
        let total_ram = snapshot.memory.total_ram;
        let used_ram = snapshot.memory.used_ram;
        let pressure_percentage = (used_ram as f64 / total_ram as f64) * 100.0;

        println!("Memory Pressure: {:.2}%", pressure_percentage);

        if pressure_percentage > 80.0 || force_purge {
            println!("High memory pressure detected (or Turbo engaged)! Attempting to purge...");
            // Attempt to purge inactive memory
            crate::engine::host_vm_info::trigger_purge();

            // Port from ram-guardian.sh: Guard against memory-hungry system apps
            for process in &snapshot.top_processes {
                if process.name == "Finder" && process.memory_usage > 500 * 1024 * 1024 {
                    println!(
                        "🚀 RAM Guardian: Restarting Finder (High Memory: {}MB)",
                        process.memory_usage / 1024 / 1024
                    );
                    signal_process_by_name("Finder", libc::SIGHUP);
                }
                if process.name == "Dock" && process.memory_usage > 300 * 1024 * 1024 {
                    println!(
                        "🚀 RAM Guardian: Restarting Dock (High Memory: {}MB)",
                        process.memory_usage / 1024 / 1024
                    );
                    signal_process_by_name("Dock", libc::SIGHUP);
                }
            }
        }
    }

    pub fn configure_startup(&self) {
        println!("Configuring Smart Startup...");
        // Disable "Reopen windows when logging back in"
        defaults_write("com.apple.loginwindow", "TALLogoutSavesState", DefaultsValue::Bool(false));
        defaults_write("com.apple.loginwindow", "LoginwindowLaunchesRelaunchApps", DefaultsValue::Bool(false));
    }

    pub fn apply_turbo_mode(&self) {
        println!("Activating TURBO MODE (Faster than Native)...");

        // 1. Disable Animations (Speed up UI)
        println!("Disabling UI animations...");
        defaults_write("NSGlobalDomain", "NSAutomaticWindowAnimationsEnabled", DefaultsValue::Bool(false));
        defaults_write("NSGlobalDomain", "NSWindowResizeTime", DefaultsValue::Float(0.001));
        defaults_write("com.apple.finder", "DisableAllAnimations", DefaultsValue::Bool(true));
        defaults_write("com.apple.dock", "launchanim", DefaultsValue::Bool(false));
        defaults_write("com.apple.dock", "expose-animation-duration", DefaultsValue::Float(0.1));

        // Fix for WindowServer "Bad Coding" - Disable Window Shadows (Performance Workaround)
        println!("Applying WindowServer performance workaround (Disable shadows)...");
        defaults_write("com.apple.WindowManager", "AppWindowShadows", DefaultsValue::Bool(false));

        // 2. Kernel performance tuning (via SysctlGovernor — captures defaults for safe revert)
        let gov = SysctlGovernor::new(unsafe { libc::geteuid() } == 0);
        gov.apply_tuning_direct();

        // 3. Disable Dashboard / Widgets (Save RAM)
        defaults_write("com.apple.dashboard", "mcx-disabled", DefaultsValue::Bool(true));

        // 4. Restart Dock and Finder to apply
        println!("Restarting UI components...");
        signal_process_by_name("Dock", libc::SIGHUP);
        signal_process_by_name("Finder", libc::SIGHUP);

        // 5. GPU Eco Mode
        self.apply_gpu_eco_mode(true);

        println!("TURBO MODE Active. UI should feel snappier.");
    }

    pub fn apply_llm_mode(&self) {
        println!("🚀 ENGAGING LLM MODE (Max GPU & AI Priority)...");

        // 1. Apply Performance Kernel Tuning (via SysctlGovernor — captures defaults for safe revert)
        let gov = SysctlGovernor::new(unsafe { libc::geteuid() } == 0);
        gov.apply_tuning_direct();
        SysctlGovernor::check_server_mode();

        // 2. Reduce Background Noise & Quarantine
        self.disable_background_noise();
        self.quarantine_apple_bloat(true);
        self.apply_gpu_eco_mode(true);

        // 3. Hard RAM Purge to make room for models
        println!("🧹 Clearing inactive RAM for models...");
        crate::engine::host_vm_info::trigger_purge();

        // 3. Identify and boost AI processes
        let mut sys = System::new_all();
        sys.refresh_processes();

        for (pid, process) in sys.processes() {
            let name = process.name();
            let pid_u32 = pid.as_u32();

            if self.llm_apps.iter().any(|app| name.contains(app)) {
                println!("🔝 BOOSTING AI PROCESS: {} (PID {})", name, pid_u32);

                unsafe {
                    libc::setpriority(libc::PRIO_PROCESS, pid_u32, -5);
                }
            }
        }

        let mut mode = safe_lock(&self.current_mode);
        *mode = "llm".to_string();
        println!("LLM MODE Fully Engaged.");
    }

    fn run_best_effort(&self, program: &str, args: &[&str]) {
        match std::process::Command::new(program).args(args).output() {
            Ok(out) if out.status.success() => {}
            Ok(out) => {
                let err = String::from_utf8_lossy(&out.stderr);
                if !err.trim().is_empty() {
                    println!("⚠️ Command '{}' failed: {}", program, err.trim());
                }
            }
            Err(e) => {
                println!("⚠️ Command '{}' not executed: {}", program, e);
            }
        }
    }

    fn clear_directory_contents(&self, dir: &PathBuf, preserve_prefixes: &[&str]) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_string();

                if preserve_prefixes
                    .iter()
                    .any(|prefix| name.starts_with(prefix))
                {
                    continue;
                }

                let metadata = match std::fs::symlink_metadata(&path) {
                    Ok(m) => m,
                    Err(_) => continue,
                };

                // Avoid following symlinks during cleanup.
                if metadata.file_type().is_symlink() {
                    continue;
                }

                if metadata.is_dir() {
                    let _ = std::fs::remove_dir_all(path);
                } else {
                    let _ = std::fs::remove_file(path);
                }
            }
        }
    }

    pub fn disable_background_noise(&self) {
        println!("🔇 Reducing Background Noise (Extreme Mode)...");

        // 1. Disable Apple Intelligence Reporting (Background Agent)
        defaults_write("com.apple.Siri", "AppleIntelligenceReportEnabled", DefaultsValue::Bool(false));

        // 2. Close non-essential background daemons (if possible)
        signal_process_by_name("SocialLayerAgent", libc::SIGTERM);
        signal_process_by_name("CloudPhotoLibrary", libc::SIGTERM);
    }

    pub fn restore_background_noise(&self) {
        println!("🔔 Restoring Background Services...");

        // BUG 9 fix: also unfreeze processes frozen by apply_quarantine_extreme.
        {
            let mut frozen = safe_lock(&self.frozen_processes);
            for &pid in frozen.iter() {
                let exists = unsafe { libc::kill(pid as i32, 0) == 0 };
                if exists {
                    unsafe {
                        kill(pid as i32, SIGCONT);
                    }
                }
            }
            frozen.clear();
        }

        self.quarantine_apple_bloat(false);
        self.apply_gpu_eco_mode(false);
    }

    pub fn quarantine_apple_bloat(&self, active: bool) {
        let daemons = vec![
            "mediaanalysisd",                // Photos/Video analysis
            "intelligenceflowd",             // Apple Intelligence Core
            "photoanalysisd",                // Face recognition
            "remindd",                       // Indexing reminders
            "homed",                         // HomeKit background
            "cloudphotod",                   // iCloud Photo Sync
            "generative-experiences-daemon", // AI Assets
        ];

        let action = if active {
            "🔒 QUARANTINING"
        } else {
            "🔓 RELEASING"
        };

        for daemon in daemons {
            println!("{} device service: {}", action, daemon);
            signal_process_by_name(daemon, if active { libc::SIGSTOP } else { libc::SIGCONT });
        }
    }

    pub fn apply_gpu_eco_mode(&self, active: bool) {
        let action = if active { "Applying" } else { "Restoring" };

        println!("{} GPU Eco-Mode (UI Optimization)...", action);

        // BUG 3 fix: AppWindowShadows semantics are inverted relative to reduceTransparency.
        // Setting AppWindowShadows=false DISABLES shadows (saves GPU). Eco mode ON → shadows OFF.
        defaults_write("com.apple.WindowManager", "AppWindowShadows", DefaultsValue::Bool(!active));
        defaults_write("com.apple.universalaccess", "reduceTransparency", DefaultsValue::Bool(active));
        defaults_write("com.apple.universalaccess", "reduceMotion", DefaultsValue::Bool(active));
    }

    pub fn clean_disk(&self) {
        println!("Starting Disk Cleanup...");

        let home_dir = std::env::var("HOME").unwrap_or_else(|_| "/".to_string());
        let home_path = PathBuf::from(&home_dir);

        // 1. Clean User Caches
        // BUG 14 fix: preserve caches for browsers and active build tools to avoid
        // destroying gigabytes of data that take hours to rebuild.
        let caches_path = home_path.join("Library/Caches");
        println!("Cleaning {:?}", caches_path);
        self.clear_directory_contents(
            &caches_path,
            &[
                "com.apple.Safari",
                "com.google.Chrome",
                "com.brave.Browser",
                "org.mozilla.firefox",
                "com.apple.dt.Xcode", // DerivedData cache
                "com.apple.metal",    // GPU shader cache
            ],
        );

        // 2. Clean User Logs
        let logs_path = home_path.join("Library/Logs");
        println!("Cleaning {:?}", logs_path);
        // Keep active optimizer logs intact.
        self.clear_directory_contents(&logs_path, &["system-optimizer"]);

        // 3. Empty Trash
        let trash_path = home_path.join(".Trash");
        println!("Emptying Trash...");
        self.clear_directory_contents(&trash_path, &[]);

        // 4. Docker Prune
        println!("Pruning Docker...");
        self.run_best_effort("docker", &["system", "prune", "-f"]);

        // 5. Homebrew Cleanup
        println!("Cleaning Homebrew...");
        self.run_best_effort("brew", &["cleanup"]);

        // 6. Telegram Media Cleanup (Moon-Shot priority)
        println!("Cleaning Telegram Media caches...");
        let telegram_path =
            home_path.join("Library/Group Containers/6N38VWS5BX.ru.keepcoder.Telegram");
        if telegram_path.exists() {
            // Ported from disk-cleanup.sh
            if let Some(path_str) = telegram_path.to_str() {
                self.run_best_effort(
                    "/usr/bin/find",
                    &[
                        path_str,
                        "-path",
                        "*/postbox/media",
                        "-type",
                        "f",
                        "-mtime",
                        "+7",
                        "-delete",
                    ],
                );
            }
        }

        // 7. Developer Cache Cleanup
        println!("Cleaning Developer Caches (Go & NPM)...");
        self.run_best_effort("go", &["clean", "-cache"]);
        self.run_best_effort("go", &["clean", "-modcache"]);
        self.run_best_effort("npm", &["cache", "clean", "--force"]);

        // 8. APFS Snapshot Management (Apple Silicon Efficiency)
        println!("Purging old APFS local snapshots...");
        self.run_best_effort(
            "/usr/bin/tmutil",
            &["thinlocalsnapshots", "/", "10000000000", "4"],
        );

        println!("Disk Cleanup Complete.");
    }

    /// Boost the optimizer process's own QoS once on startup; subsequent calls are no-ops.
    /// BUG 16 fix: previously called on every set_high_priority invocation (per process per cycle).
    pub fn boost_self_once(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static DONE: AtomicBool = AtomicBool::new(false);
        if DONE.swap(true, Ordering::SeqCst) {
            return;
        }
        unsafe {
            let mut qos_policy = TaskQosPolicy {
                task_latency_qos_tier: LATENCY_QOS_TIER_0,
                task_throughput_qos_tier: THROUGHPUT_QOS_TIER_0,
            };
            task_policy_set(
                mach_task_self(),
                TASK_QOS_POLICY,
                &mut qos_policy as *mut _ as *mut i32,
                2,
            );
            setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_PROCESS, IOPOL_NORMAL);
        }
    }

    fn set_high_priority(&self, pid: u32, _name: &str) {
        // --- CHIP-LEVEL PEAK SCALING (NATIVE) ---
        unsafe {
            // BUG 16 fix: self-boost moved to boost_self_once(), called once at startup.
            // Scheduling Priority (Boosted) - only for the target PID.
            libc::setpriority(libc::PRIO_PROCESS, pid, -20);
        }
    }

    fn set_low_priority(&self, pid: u32, _name: &str, aggressive: bool) {
        // --- CHIP-LEVEL EFFICIENCY SCALING (NATIVE) ---
        unsafe {
            let val = if aggressive { 20 } else { 10 };
            libc::setpriority(libc::PRIO_PROCESS, pid, val);
        }
    }

    fn freeze_process(&self, pid: u32, name: &str) {
        let mut frozen = safe_lock(&self.frozen_processes);
        if !frozen.contains(&pid) {
            // BUG 4 fix: validate the PID still belongs to a live process before SIGSTOP.
            // kill(pid, 0) returns 0 if the process exists and we have permission.
            let exists = unsafe { libc::kill(pid as i32, 0) == 0 };
            if !exists {
                return;
            }
            println!("❄️  Freezing {} (PID {}) to save CPU/RAM", name, pid);
            unsafe {
                kill(pid as i32, SIGSTOP);
            }
            frozen.insert(pid);
        }
    }

    fn unfreeze_process(&self, pid: u32, _name: &str) {
        let mut frozen = safe_lock(&self.frozen_processes);
        if frozen.contains(&pid) {
            // BUG 4 fix: only send SIGCONT if the PID still exists.
            let exists = unsafe { libc::kill(pid as i32, 0) == 0 };
            if exists {
                unsafe {
                    kill(pid as i32, SIGCONT);
                }
            }
            frozen.remove(&pid);
        }
    }

    pub fn cleanup(&self) {
        println!("🧹 Optimizer shutting down. Cleaning up...");
        let mut frozen = safe_lock(&self.frozen_processes);
        for pid in frozen.iter() {
            unsafe {
                kill(*pid as i32, SIGCONT);
            }
        }
        frozen.clear();

        // Restore background services
        self.save_profiles();
        self.restore_background_noise();
    }

    fn save_profiles(&self) {
        let home_dir = std::env::var("HOME").unwrap_or_else(|_| "/".to_string());
        let path = PathBuf::from(home_dir).join(".apollo_profiles.json");
        println!("💾 Saving Behavioral Profiles to {:?}...", path);
        let profiles = safe_lock(&self.app_profiles);
        if let Ok(json) = serde_json::to_string_pretty(&*profiles) {
            match std::fs::write(&path, json) {
                Ok(_) => println!("✅ Successfully saved {} profiles.", profiles.len()),
                Err(e) => println!("❌ Failed to save profiles: {}", e),
            }
        } else {
            println!("❌ Failed to serialize profiles to JSON.");
        }
    }

    fn load_profiles() -> HashMap<String, AppProfile> {
        let home_dir = std::env::var("HOME").unwrap_or_else(|_| "/".to_string());
        let path = PathBuf::from(home_dir).join(".apollo_profiles.json");
        if let Ok(content) = HardPath::read_to_string_limited(&path, 5 * 1024 * 1024) {
            if let Ok(profiles) = serde_json::from_str(&content) {
                return profiles;
            }
        }
        HashMap::new()
    }
}

#[allow(clashing_extern_declarations)]
extern "C" {
    fn setiopolicy_np(iotype: i32, scope: i32, policy: i32) -> i32;
    fn task_policy_set(task: u32, flavor: u32, policy_info: *mut i32, count: u32) -> i32;
    fn mach_task_self() -> u32;

    // --- Phase 8: Deep Kernel Flow ---
    fn task_threads(task: u32, thread_list: *mut *mut u32, thread_count: *mut u32) -> i32;
    fn thread_info(thread: u32, flavor: u32, thread_info_out: *mut i32, count: *mut u32) -> i32;
    fn mach_port_deallocate(target_task: u32, name: u32) -> i32;
    fn mach_vm_deallocate(target: u32, address: u64, size: u64) -> i32;
    fn memorystatus_control(
        command: u32,
        pid: i32,
        flags: u32,
        buffer: *mut libc::c_void,
        buffersize: usize,
    ) -> i32;
}

/// BUG 2 fix: macOS kernel uses time_value_t = {integer_t seconds; integer_t microseconds},
/// both i32. libc::timeval uses tv_sec: i64 on arm64, producing a 16-byte struct instead of
/// the 8-byte time_value_t, which completely corrupts the thread_basic_info layout.
#[repr(C)]
struct TimeValueT {
    seconds: i32,
    microseconds: i32,
}

#[repr(C)]
struct ThreadBasicInfo {
    user_time: TimeValueT,
    system_time: TimeValueT,
    cpu_usage: i32,
    policy: i32,
    run_state: i32,
    flags: i32,
    suspend_count: i32,
    sleep_time: i32,
}

const THREAD_BASIC_INFO: u32 = 3;
// BUG 2 fix: count in units of natural_t (i32). With TimeValueT (2×i32 each) the struct is
// 2×8 + 6×4 = 40 bytes = 10 i32s, matching the kernel's THREAD_BASIC_INFO_COUNT = 10.
const THREAD_BASIC_INFO_COUNT: u32 =
    (std::mem::size_of::<ThreadBasicInfo>() / std::mem::size_of::<i32>()) as u32;
const TH_STATE_WAITING: i32 = 3;

const MEMORYSTATUS_CMD_SET_PRIORITY_PROPERTIES: u32 = 2;
const MEMORYSTATUS_CMD_SET_JETSAM_HIGH_WATER_MARK: u32 = 5;
const JETSAM_PRIORITY_BACKGROUND: i32 = 40;
const APOLLO_MEMORY_HINT: u64 = 0x4150_4f4c_4c4f;

#[repr(C)]
struct MemorystatusPriorityProperties {
    priority: i32,
    user_data: u64,
}

#[repr(C)]
struct TaskQosPolicy {
    task_latency_qos_tier: i32,
    task_throughput_qos_tier: i32,
}

const TASK_QOS_POLICY: u32 = 6;
const LATENCY_QOS_TIER_0: i32 = (0xFF << 16) | 1;
const THROUGHPUT_QOS_TIER_0: i32 = (0xFF << 16) | 1;

const IOPOL_TYPE_DISK: i32 = 0;
const IOPOL_SCOPE_PROCESS: i32 = 0;
const IOPOL_NORMAL: i32 = 0;

// ── Internal unit tests (BUG 2 regression) ───────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that TimeValueT is exactly 8 bytes (two i32 fields), matching
    /// the kernel's time_value_t. If libc::timeval were used instead, this
    /// would be 16 bytes on arm64 and corrupt the ThreadBasicInfo layout.
    #[test]
    fn time_value_t_is_8_bytes() {
        assert_eq!(
            std::mem::size_of::<TimeValueT>(),
            8,
            "TimeValueT must be 8 bytes (two i32 fields)"
        );
    }

    /// Verify ThreadBasicInfo is exactly 40 bytes: 2×TimeValueT(8) + 6×i32(4) = 40.
    /// The kernel's THREAD_BASIC_INFO_COUNT = 10 (in units of natural_t = i32 = 4 bytes),
    /// so the struct must be 10 × 4 = 40 bytes.
    #[test]
    fn thread_basic_info_is_40_bytes() {
        assert_eq!(
            std::mem::size_of::<ThreadBasicInfo>(),
            40,
            "ThreadBasicInfo must be 40 bytes to match kernel THREAD_BASIC_INFO_COUNT=10"
        );
    }

    /// Verify THREAD_BASIC_INFO_COUNT is exactly 10 (natural_t units).
    #[test]
    fn thread_basic_info_count_is_10() {
        assert_eq!(
            THREAD_BASIC_INFO_COUNT, 10,
            "THREAD_BASIC_INFO_COUNT must equal 10 to match the kernel ABI"
        );
    }
}
