use std::process::Command;

pub struct SysctlTuner;

#[allow(clippy::new_without_default)]
impl SysctlTuner {
    pub fn new() -> Self {
        Self
    }

    pub fn apply_performance_tuning(&self) {
        println!("🚀 Applying Kernel Performance Tuning...");

        // 1. I/O Throttling - Disable to allow full SSD speed for background/low-priority tasks
        self.set_sysctl("debug.lowpri_throttle_enabled", "0");

        // 2. Network Stack Tuning (TCP)
        // Values from research: Default 128KB -> 1MB (1048576 bytes)
        self.set_sysctl("net.inet.tcp.sendspace", "1048576");
        self.set_sysctl("net.inet.tcp.recvspace", "1048576");

        // Reduce latency by disabling delayed ACKs
        self.set_sysctl("net.inet.tcp.delayed_ack", "0");

        // Optimize TCP for local low-latency communication (useful for LLM APIs)
        self.set_sysctl("net.inet.tcp.min_iaj_win", "4");

        // --- Phase 3: Extreme Tuning ---
        // High bandwidth scaling
        self.set_sysctl("net.inet.tcp.win_scale_factor", "8");
        // 32MB Max Buffers
        self.set_sysctl("net.inet.tcp.autorcvbufmax", "33554432");
        self.set_sysctl("net.inet.tcp.autosndbufmax", "33554432");

        // --- Memory Compression Tuning (M1 Scaler) ---
        // Increase compression efficiency
        self.set_sysctl("vm.compressor_poll_interval", "20");
        self.set_sysctl("vm.compressor_sample_min", "10");

        // --- Phase 4: Apollo Moon-Shot (Extreme Kernel Scaling) ---
        // 1. File System Cache 'Stickiness' (VNodes)
        // High VNode count prevents the kernel from recycling file descriptors
        // during heavy dev work (npm/cargo).
        self.set_sysctl("kern.maxvnodes", "300000");
        self.set_sysctl("kern.maxfiles", "100000");
        self.set_sysctl("kern.maxfilesperproc", "50000");

        // 2. High-Throughput IPC & Networking
        // Increase the maximum number of pending connections (useful for microservices/LLMs)
        self.set_sysctl("kern.ipc.somaxconn", "2048");
        // Increase maximum socket buffer size
        self.set_sysctl("kern.ipc.maxsockbuf", "4194304");

        // 3. Hardware Fault Mitigation: VRAM Allocation
        self.boost_vram_if_needed();
    }

    fn boost_vram_if_needed(&self) {
        println!("🚀 Boosting GPU Wired Memory Limit (VRAM Boost)...");
        // macOS Sonoma (14.x) and later
        self.set_sysctl("iogpu.wired_limit_mb", "12288");
        // macOS Ventura (13.x)
        self.set_sysctl("debug.iogpu.wired_limit", "12288");
    }

    pub fn check_server_mode(&self) {
        println!("🔍 Checking Server Performance Mode...");
        let output = Command::new("serverinfo").arg("--perfmode").output();

        match output {
            Ok(out) => {
                let status = String::from_utf8_lossy(&out.stdout);
                if status.contains("enabled") {
                    println!("🚀 Server Performance Mode: ENABLED");
                } else {
                    println!("ℹ️  Server Performance Mode: DISABLED");
                    println!("💡 To enable (Extreme Efficiency):");
                    println!("   1. Boot into Recovery (Hold Power)");
                    println!("   2. Terminal: csrutil disable");
                    println!("   3. Reboot & run: sudo nvram boot-args=\"serverperfmode=1\"");
                }
            }
            Err(_) => {
                println!("⚠️  'serverinfo' command not found. Might be a standard macOS install.");
            }
        }
    }

    #[allow(dead_code)]
    pub fn reset_to_defaults(&self) {
        println!("🔄 Resetting Kernel Tuning to macOS defaults...");
        self.set_sysctl("debug.lowpri_throttle_enabled", "1");
        self.set_sysctl("net.inet.tcp.sendspace", "131072");
        self.set_sysctl("net.inet.tcp.recvspace", "131072");
        self.set_sysctl("net.inet.tcp.delayed_ack", "3");

        // Reset VRAM limit (typically 0 or default is hidden)
        // macOS will usually ignore setting it to 0 or low value if it's managed, but we try anyway.
        // Better yet, just notify that it resets on reboot.
        println!(
            "ℹ️  VRAM Limit and Boot-Args require a reboot to fully reset to factory defaults."
        );
    }

    fn set_sysctl(&self, key: &str, value: &str) {
        // Attempt to set sysctl. This might fail without sudo.
        let output = Command::new("sysctl")
            .args(["-w", &format!("{}={}", key, value)])
            .output();

        match output {
            Ok(out) if out.status.success() => {
                println!("✅ {} = {}", key, value);
            }
            Ok(out) => {
                let err = String::from_utf8_lossy(&out.stderr);
                if err.contains("Permission denied") {
                    println!(
                        "❌ Failed to set {}: Permission denied (requires sudo)",
                        key
                    );
                } else {
                    println!("❌ Failed to set {}: {}", key, err.trim());
                }
            }
            Err(e) => {
                println!("❌ System error while setting {}: {}", key, e);
            }
        }
    }
}
