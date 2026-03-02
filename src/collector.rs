use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::process::Command;
use std::time::Instant;
use sysinfo::{Disks, Networks, System};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SystemSnapshot {
    pub timestamp: DateTime<Utc>,
    pub cpu: CpuStats,
    pub memory: MemoryStats,
    pub pressure: PressureStats,
    pub disks: Vec<DiskStats>,
    pub networks: Vec<NetworkStats>,
    pub top_processes: Vec<ProcessStats>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CpuStats {
    pub global_usage: f32,
    pub core_count: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MemoryStats {
    pub total_ram: u64,
    pub used_ram: u64,
    pub free_ram: u64,
    pub total_swap: u64,
    pub used_swap: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PressureStats {
    // 0..1 where 1 == high pressure.
    pub memory_pressure: f64,
    pub swap_used_bytes: u64,
    pub swap_total_bytes: u64,
    pub swap_delta_bytes_per_sec: f64,
    pub thermal_level: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DiskStats {
    pub name: String,
    pub mount_point: String,
    pub total_space: u64,
    pub available_space: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NetworkStats {
    pub interface_name: String,
    pub received: u64,
    pub transmitted: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProcessStats {
    pub pid: u32,
    pub name: String,
    pub cpu_usage: f32,
    pub memory_usage: u64,
}

pub struct SystemCollector {
    sys: System,
    disks: Disks,
    networks: Networks,
    prev_swap_used_bytes: Option<u64>,
    prev_swap_at: Option<Instant>,
}

impl SystemCollector {
    pub fn new() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();
        let disks = Disks::new_with_refreshed_list();
        let networks = Networks::new_with_refreshed_list();
        Self {
            sys,
            disks,
            networks,
            prev_swap_used_bytes: None,
            prev_swap_at: None,
        }
    }

    pub fn collect_snapshot(&mut self) -> SystemSnapshot {
        // Refresh system stats
        self.sys.refresh_all();
        self.disks.refresh_list();
        self.networks.refresh();

        // CPU
        let global_cpu = self.sys.global_cpu_info().cpu_usage();
        let core_count = self.sys.cpus().len();

        // Memory
        let total_ram = self.sys.total_memory();
        let used_ram = self.sys.used_memory();
        let free_ram = self.sys.free_memory();
        let total_swap = self.sys.total_swap();
        let used_swap = self.sys.used_swap();

        // Pressure (public commands, no private APIs)
        let (mem_pressure, swap_used_bytes, swap_total_bytes) = collect_pressure_facts();
        let nowi = Instant::now();
        let swap_delta_bps = match (self.prev_swap_used_bytes, self.prev_swap_at) {
            (Some(prev_used), Some(prev_at)) => {
                let dt = nowi.duration_since(prev_at).as_secs_f64().max(0.001);
                (swap_used_bytes.saturating_sub(prev_used) as f64) / dt
            }
            _ => 0.0,
        };
        self.prev_swap_used_bytes = Some(swap_used_bytes);
        self.prev_swap_at = Some(nowi);

        // Disks
        let disks = self
            .disks
            .iter()
            .map(|disk| DiskStats {
                name: disk.name().to_string_lossy().into_owned(),
                mount_point: disk.mount_point().to_string_lossy().into_owned(),
                total_space: disk.total_space(),
                available_space: disk.available_space(),
            })
            .collect();

        // Networks
        let networks = self
            .networks
            .iter()
            .map(|(name, data)| NetworkStats {
                interface_name: name.clone(),
                received: data.received(),
                transmitted: data.transmitted(),
            })
            .collect();

        // Processes - Get top 10 by CPU usage
        let mut processes: Vec<ProcessStats> = self
            .sys
            .processes()
            .iter()
            .map(|(pid, process)| ProcessStats {
                pid: pid.as_u32(),
                name: process.name().to_string(),
                cpu_usage: process.cpu_usage(),
                memory_usage: process.memory(),
            })
            .collect();

        // Sort by CPU usage descending
        processes.sort_by(|a, b| {
            b.cpu_usage
                .partial_cmp(&a.cpu_usage)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_processes = processes.into_iter().take(10).collect();

        SystemSnapshot {
            timestamp: Utc::now(),
            cpu: CpuStats {
                global_usage: global_cpu,
                core_count,
            },
            memory: MemoryStats {
                total_ram,
                used_ram,
                free_ram,
                total_swap,
                used_swap,
            },
            pressure: PressureStats {
                memory_pressure: mem_pressure,
                swap_used_bytes,
                swap_total_bytes,
                swap_delta_bytes_per_sec: swap_delta_bps,
                thermal_level: "unknown".to_string(),
            },
            disks,
            networks,
            top_processes,
        }
    }
}

fn collect_pressure_facts() -> (f64, u64, u64) {
    let mut memory_pressure = 0.0;
    if let Ok(out) = Command::new("memory_pressure").args(["-Q"]).output() {
        if out.status.success() {
            let text = String::from_utf8_lossy(&out.stdout);
            // Example: "System-wide memory free percentage: 44%"
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("System-wide memory free percentage:") {
                    let s = rest.trim().trim_end_matches('%').trim();
                    if let Ok(free_pct) = s.parse::<f64>() {
                        memory_pressure = (1.0 - (free_pct / 100.0)).clamp(0.0, 1.0);
                    }
                }
            }
        }
    }

    let mut swap_used_bytes = 0_u64;
    let mut swap_total_bytes = 0_u64;
    if let Ok(out) = Command::new("sysctl").args(["vm.swapusage"]).output() {
        if out.status.success() {
            let text = String::from_utf8_lossy(&out.stdout);
            // Example: "vm.swapusage: total = 3072.00M  used = 2251.25M  free = 820.75M  (encrypted)"
            swap_total_bytes = parse_sysctl_size(&text, "total").unwrap_or(0);
            swap_used_bytes = parse_sysctl_size(&text, "used").unwrap_or(0);
        }
    }

    (memory_pressure, swap_used_bytes, swap_total_bytes)
}

fn parse_sysctl_size(s: &str, key: &str) -> Option<u64> {
    // Find "<key> = <num><unit>" where unit is K/M/G (binary-ish but close enough).
    let needle = format!("{key} =");
    let idx = s.find(&needle)?;
    let rest = s[idx + needle.len()..].trim_start();
    let mut num = String::new();
    let mut unit = None;
    for ch in rest.chars() {
        if ch.is_ascii_digit() || ch == '.' {
            num.push(ch);
        } else if ch.is_ascii_alphabetic() {
            unit = Some(ch);
            break;
        } else if !num.is_empty() {
            break;
        }
    }
    let val = num.parse::<f64>().ok()?;
    let mul = match unit.unwrap_or('B') {
        'K' | 'k' => 1024_f64,
        'M' | 'm' => 1024_f64 * 1024_f64,
        'G' | 'g' => 1024_f64 * 1024_f64 * 1024_f64,
        _ => 1_f64,
    };
    Some((val * mul) as u64)
}
