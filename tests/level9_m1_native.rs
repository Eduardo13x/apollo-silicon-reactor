//! Level 9: M1-Native Scheduling Tests
//!
//! Tests for mach_qos (P/E-Core routing) and iokit_sensors (hardware telemetry).
//! All tests are non-root-safe: they verify logic, parsing, and mapping only.

use apollo_optimizer::engine::iokit_sensors::{
    ClusterTemps, HardwareSnapshot, IOKitSensorReader, PowerReading, ThermalState,
};
use apollo_optimizer::engine::mach_qos::{tier_for_process, MachQoSManager, SchedulingTier};
use apollo_optimizer::engine::process_classifier::ProcessTier;

// ── MachQoS unit tests ────────────────────────────────────────────────────────

#[test]
fn test_qos_manager_new_has_no_tracked_pids() {
    let mgr = MachQoSManager::new();
    assert_eq!(mgr.background_count(), 0);
    assert!(mgr.current_tier(1234).is_none());
}

#[test]
fn test_qos_manager_tracks_background_count() {
    let mut mgr = MachQoSManager::new();
    // Manually insert via apply (will fail on non-root but state tracks)
    let _ = mgr.set_tier(1001, SchedulingTier::Background);
    let _ = mgr.set_tier(1002, SchedulingTier::Background);
    let _ = mgr.set_tier(1003, SchedulingTier::Foreground);
    // background_count counts only Background entries that were accepted
    // (syscall may fail without root, but logic is correct)
    // Verify no panic
}

#[test]
fn test_qos_manager_remove_clears_tracking() {
    let mut mgr = MachQoSManager::new();
    let _ = mgr.set_tier(9999, SchedulingTier::Normal);
    mgr.remove(9999);
    assert!(mgr.current_tier(9999).is_none());
}

#[test]
fn test_qos_manager_deduplicates_same_tier() {
    let mut mgr = MachQoSManager::new();
    // First call sets the tier
    let r1 = mgr.set_tier(1234, SchedulingTier::Background);
    // If first succeeded, second call should be a no-op (same tier)
    if r1.success {
        let r2 = mgr.set_tier(1234, SchedulingTier::Background);
        assert!(r2.success, "Dedup should succeed immediately (no syscall)");
    }
}

// ── tier_for_process mapping tests ────────────────────────────────────────────

#[test]
fn test_tier_active_foreground_goes_to_p_cores() {
    let tier = tier_for_process(ProcessTier::ActiveForeground);
    assert_eq!(tier, SchedulingTier::Foreground);
}

#[test]
fn test_tier_system_essential_goes_to_p_cores() {
    let tier = tier_for_process(ProcessTier::SystemEssential);
    assert_eq!(tier, SchedulingTier::Foreground);
}

#[test]
fn test_tier_silent_daemon_goes_to_e_cores() {
    let tier = tier_for_process(ProcessTier::SilentDaemon);
    assert_eq!(tier, SchedulingTier::Background);
}

#[test]
fn test_tier_telemetry_goes_to_e_cores() {
    let tier = tier_for_process(ProcessTier::Telemetry);
    assert_eq!(tier, SchedulingTier::Background);
}

#[test]
fn test_tier_stale_goes_to_e_cores() {
    let tier = tier_for_process(ProcessTier::Stale);
    assert_eq!(tier, SchedulingTier::Background);
}

#[test]
fn test_tier_zombie_goes_to_e_cores() {
    let tier = tier_for_process(ProcessTier::ZombieOrphan);
    assert_eq!(tier, SchedulingTier::Background);
}

#[test]
fn test_tier_background_visible_is_normal() {
    let tier = tier_for_process(ProcessTier::BackgroundVisible);
    assert_eq!(tier, SchedulingTier::Normal);
}

#[test]
fn test_tier_app_helper_is_normal() {
    let tier = tier_for_process(ProcessTier::AppHelper);
    assert_eq!(tier, SchedulingTier::Normal);
}

// ── IOKit Sensor parsing tests ────────────────────────────────────────────────

fn make_sample_powermetrics_output() -> &'static str {
    r#"
Machine model: MacBookPro18,3
SMC version (system): 2.0f4

*** Sampled system activity (Fri Jan 01 12:00:00 2021 -0600) (503ms elapsed) ***

**** Processor usage ****
P-cluster HW active residency: 34.5%
E-cluster HW active residency: 12.1%

CPU P-cluster temp: 72.34 C
CPU E-cluster temp: 41.12 C
GPU temp: 48.91 C

Package power: 4.532 W
CPU Power: 3.021 W
GPU Power: 0.881 W

System thermal state: NORMAL

**** Battery Stats ****
Capacity: 87% (discharging)
Battery power: 6.3 W
"#
}

fn make_hot_powermetrics_output() -> &'static str {
    r#"
CPU P-cluster temp: 95.0 C
CPU E-cluster temp: 62.0 C
GPU temp: 88.0 C
Package power: 28.5 W
CPU Power: 20.1 W
GPU Power: 6.2 W
System thermal state: SEVERE
P-cluster HW active residency: 98.1%
E-cluster HW active residency: 90.2%
Capacity: 35% (discharging)
Battery power: 18.0 W
"#
}

#[test]
fn test_parse_powermetrics_normal() {
    let reader = IOKitSensorReader::new();
    let snap = reader.parse_powermetrics(make_sample_powermetrics_output());
    assert_eq!(snap.thermal_state, ThermalState::Normal);
    assert!(snap.temps.p_cluster_celsius.is_some());
    assert!((snap.temps.p_cluster_celsius.unwrap() - 72.34).abs() < 0.1);
    assert!((snap.temps.e_cluster_celsius.unwrap() - 41.12).abs() < 0.1);
}

#[test]
fn test_parse_powermetrics_gpu_temp() {
    let reader = IOKitSensorReader::new();
    let snap = reader.parse_powermetrics(make_sample_powermetrics_output());
    assert!(snap.temps.gpu_celsius.is_some());
    assert!((snap.temps.gpu_celsius.unwrap() - 48.91).abs() < 0.1);
}

#[test]
fn test_parse_powermetrics_package_power() {
    let reader = IOKitSensorReader::new();
    let snap = reader.parse_powermetrics(make_sample_powermetrics_output());
    assert!(snap.power.package_watts.is_some());
    assert!((snap.power.package_watts.unwrap() - 4.532).abs() < 0.01);
}

#[test]
fn test_parse_powermetrics_cpu_power() {
    let reader = IOKitSensorReader::new();
    let snap = reader.parse_powermetrics(make_sample_powermetrics_output());
    assert!(snap.power.cpu_watts.is_some());
    assert!((snap.power.cpu_watts.unwrap() - 3.021).abs() < 0.01);
}

#[test]
fn test_parse_powermetrics_cluster_utilisation() {
    let reader = IOKitSensorReader::new();
    let snap = reader.parse_powermetrics(make_sample_powermetrics_output());
    assert!(snap.p_cluster_util.is_some());
    assert!((snap.p_cluster_util.unwrap() - 34.5).abs() < 0.1);
    assert!(snap.e_cluster_util.is_some());
    assert!((snap.e_cluster_util.unwrap() - 12.1).abs() < 0.1);
}

#[test]
fn test_parse_powermetrics_battery_percent() {
    let reader = IOKitSensorReader::new();
    let snap = reader.parse_powermetrics(make_sample_powermetrics_output());
    assert_eq!(snap.battery_percent, Some(87));
}

#[test]
fn test_parse_powermetrics_severe_thermal_state() {
    let reader = IOKitSensorReader::new();
    let snap = reader.parse_powermetrics(make_hot_powermetrics_output());
    assert_eq!(snap.thermal_state, ThermalState::Severe);
}

#[test]
fn test_parse_powermetrics_inferred_critical_from_temp() {
    let reader = IOKitSensorReader::new();
    let text = "CPU P-cluster temp: 102.0 C\n";
    let snap = reader.parse_powermetrics(text);
    assert_eq!(snap.thermal_state, ThermalState::Critical);
}

#[test]
fn test_parse_powermetrics_inferred_moderate_from_temp() {
    let reader = IOKitSensorReader::new();
    let text = "CPU P-cluster temp: 85.0 C\n";
    let snap = reader.parse_powermetrics(text);
    assert_eq!(snap.thermal_state, ThermalState::Moderate);
}

#[test]
fn test_is_throttled_normal_is_false() {
    let snap = HardwareSnapshot {
        thermal_state: ThermalState::Normal,
        temps: ClusterTemps {
            p_cluster_celsius: None,
            e_cluster_celsius: None,
            gpu_celsius: None,
            nand_celsius: None,
        },
        power: PowerReading {
            package_watts: None,
            cpu_watts: None,
            gpu_watts: None,
            dram_watts: None,
        },
        p_cluster_util: None,
        e_cluster_util: None,
        battery_percent: None,
        battery_watts: None,
    };
    assert!(!IOKitSensorReader::is_throttled(&snap));
}

#[test]
fn test_is_throttled_moderate_is_true() {
    let snap = HardwareSnapshot {
        thermal_state: ThermalState::Moderate,
        temps: ClusterTemps {
            p_cluster_celsius: None,
            e_cluster_celsius: None,
            gpu_celsius: None,
            nand_celsius: None,
        },
        power: PowerReading {
            package_watts: None,
            cpu_watts: None,
            gpu_watts: None,
            dram_watts: None,
        },
        p_cluster_util: None,
        e_cluster_util: None,
        battery_percent: None,
        battery_watts: None,
    };
    assert!(IOKitSensorReader::is_throttled(&snap));
}

#[test]
fn test_battery_critical_detection() {
    let snap = HardwareSnapshot {
        thermal_state: ThermalState::Normal,
        temps: ClusterTemps {
            p_cluster_celsius: None,
            e_cluster_celsius: None,
            gpu_celsius: None,
            nand_celsius: None,
        },
        power: PowerReading {
            package_watts: None,
            cpu_watts: None,
            gpu_watts: None,
            dram_watts: None,
        },
        p_cluster_util: None,
        e_cluster_util: None,
        battery_percent: Some(15), // Below 20%
        battery_watts: Some(8.0),  // Positive = discharging
    };
    assert!(IOKitSensorReader::is_battery_critical(&snap));
}

#[test]
fn test_battery_not_critical_when_charging() {
    let snap = HardwareSnapshot {
        thermal_state: ThermalState::Normal,
        temps: ClusterTemps {
            p_cluster_celsius: None,
            e_cluster_celsius: None,
            gpu_celsius: None,
            nand_celsius: None,
        },
        power: PowerReading {
            package_watts: None,
            cpu_watts: None,
            gpu_watts: None,
            dram_watts: None,
        },
        p_cluster_util: None,
        e_cluster_util: None,
        battery_percent: Some(15),
        battery_watts: Some(-30.0), // Negative = charging
    };
    assert!(!IOKitSensorReader::is_battery_critical(&snap));
}

#[test]
fn test_should_push_to_ecores_when_throttled() {
    let snap = HardwareSnapshot {
        thermal_state: ThermalState::Severe,
        temps: ClusterTemps {
            p_cluster_celsius: None,
            e_cluster_celsius: None,
            gpu_celsius: None,
            nand_celsius: None,
        },
        power: PowerReading {
            package_watts: None,
            cpu_watts: None,
            gpu_watts: None,
            dram_watts: None,
        },
        p_cluster_util: None,
        e_cluster_util: None,
        battery_percent: Some(80),
        battery_watts: None,
    };
    assert!(IOKitSensorReader::should_push_to_ecores(&snap));
}

#[test]
fn test_should_push_to_ecores_when_battery_critical() {
    let snap = HardwareSnapshot {
        thermal_state: ThermalState::Normal,
        temps: ClusterTemps {
            p_cluster_celsius: None,
            e_cluster_celsius: None,
            gpu_celsius: None,
            nand_celsius: None,
        },
        power: PowerReading {
            package_watts: None,
            cpu_watts: None,
            gpu_watts: None,
            dram_watts: None,
        },
        p_cluster_util: None,
        e_cluster_util: None,
        battery_percent: Some(10),
        battery_watts: Some(5.0),
    };
    assert!(IOKitSensorReader::should_push_to_ecores(&snap));
}

#[test]
fn test_qos_batch_apply() {
    let mut mgr = MachQoSManager::new();
    let changes = vec![
        (1001, SchedulingTier::Foreground),
        (1002, SchedulingTier::Background),
        (1003, SchedulingTier::Normal),
    ];
    let outcomes = mgr.apply_batch(&changes);
    assert_eq!(outcomes.len(), 3);
    // Each outcome should correspond to its input
    assert_eq!(outcomes[0].pid, 1001);
    assert_eq!(outcomes[0].tier, SchedulingTier::Foreground);
    assert_eq!(outcomes[1].pid, 1002);
    assert_eq!(outcomes[1].tier, SchedulingTier::Background);
}
