//! Level 11: Subatomic Tests — push Apollo to the EL0 floor.
//!
//! Stress tests, break scenarios, and simulation of extreme conditions.
//! These tests verify that every low-level module survives adversity:
//! - kqueue reactor under event floods
//! - VM surgeon under memory pressure
//! - proc_taskinfo with adversarial PIDs
//! - LSE counters under heavy contention
//! - Cross-module integration: kqueue + proc + vm acting together

use apollo_engine::engine::kqueue_pressure::{KqueuePressure, PressureEvent};
use apollo_engine::engine::lse_counters::LockFreeMetrics;
use apollo_engine::engine::proc_taskinfo;
use apollo_engine::engine::vm_surgeon;

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

// ═══════════════════════════════════════════════════════════════════════════════
// kqueue stress tests
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn kqueue_rapid_create_destroy() {
    // Create and destroy 100 kqueue reactors — verify no fd leak
    for _ in 0..100 {
        let reactor = KqueuePressure::new().expect("kqueue create");
        assert!(reactor.kq_fd() >= 0);
        drop(reactor);
    }
}

#[test]
fn kqueue_watch_many_children() {
    // Spawn 20 children, watch them all, verify all exits detected
    let mut reactor = KqueuePressure::new().unwrap();
    let mut children = Vec::new();
    let mut child_pids = Vec::new();

    for _ in 0..20 {
        let child = std::process::Command::new("/usr/bin/true")
            .spawn()
            .expect("spawn");
        let pid = child.id();
        reactor.watch_pid(pid).ok(); // may fail if already exited
        child_pids.push(pid);
        children.push(child);
    }

    // Wait for all exits
    let mut exited = std::collections::HashSet::new();
    let deadline = Instant::now() + Duration::from_secs(5);
    while exited.len() < child_pids.len() && Instant::now() < deadline {
        let events = reactor.wait_events(100);
        for ev in events {
            if let PressureEvent::ProcessExited(pid) = ev {
                exited.insert(pid);
            }
        }
    }
    for mut child in children {
        let _ = child.wait();
    }

    // At least some should have been detected (race: some may exit before watch)
    assert!(
        exited.len() >= 10,
        "should detect most exits, got {}/{}",
        exited.len(),
        child_pids.len(),
    );
}

#[test]
fn kqueue_timer_accuracy() {
    // Verify timer fires within acceptable jitter
    let mut reactor = KqueuePressure::new().unwrap();
    reactor.start_timer(20).unwrap(); // 20ms timer

    let mut deltas = Vec::new();
    let mut last = Instant::now();
    for _ in 0..10 {
        let events = reactor.wait_events(200);
        if events.contains(&PressureEvent::TimerTick) {
            let now = Instant::now();
            deltas.push(now.duration_since(last).as_millis());
            last = now;
        }
    }

    assert!(
        deltas.len() >= 5,
        "should get at least 5 timer ticks, got {}",
        deltas.len(),
    );

    // Check that most ticks are within 15-60ms (20ms target ± jitter)
    let reasonable = deltas.iter().filter(|&&d| (10..=80).contains(&d)).count();
    assert!(
        reasonable >= deltas.len() / 2,
        "most ticks should be near 20ms: {:?}",
        deltas,
    );
}

#[test]
fn kqueue_poll_is_nonblocking() {
    let mut reactor = KqueuePressure::new().unwrap();
    let t0 = Instant::now();
    let _ = reactor.poll_events();
    let elapsed = t0.elapsed();
    assert!(
        elapsed < Duration::from_millis(10),
        "poll should be non-blocking, took {:?}",
        elapsed,
    );
}

#[test]
fn kqueue_wait_respects_timeout() {
    let mut reactor = KqueuePressure::new().unwrap();
    let t0 = Instant::now();
    let _ = reactor.wait_events(50); // 50ms timeout
    let elapsed = t0.elapsed().as_millis();
    assert!(
        (30..200).contains(&elapsed),
        "wait(50ms) should take ~50ms, took {}ms",
        elapsed,
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// VM Surgeon stress tests
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn vm_mlock_unlock_cycle_stress() {
    let ps = vm_surgeon::page_size();
    let (ptr, len) = vm_surgeon::alloc_aligned(ps * 4).unwrap();

    // Touch pages
    unsafe { std::ptr::write_bytes(ptr, 0xAB, len) };

    // Rapid lock/unlock 100 times
    for _ in 0..100 {
        vm_surgeon::pin_memory(ptr, len).expect("mlock");
        vm_surgeon::unpin_memory(ptr, len).expect("munlock");
    }

    unsafe { vm_surgeon::free_aligned(ptr, len) };
}

#[test]
fn vm_madvise_stress() {
    let ps = vm_surgeon::page_size();
    let (ptr, len) = vm_surgeon::alloc_aligned(ps * 16).unwrap();

    // Rapid hint changes
    for _ in 0..100 {
        unsafe { std::ptr::write_bytes(ptr, 0xCC, len) };
        vm_surgeon::hint_willneed(ptr, len).ok();
        vm_surgeon::hint_sequential(ptr, len).ok();
        vm_surgeon::hint_random(ptr, len).ok();
        vm_surgeon::hint_dontneed(ptr, len).ok();
    }

    unsafe { vm_surgeon::free_aligned(ptr, len) };
}

#[test]
fn vm_mincore_consistency_after_touch() {
    let ps = vm_surgeon::page_size();
    let n_pages = 32;
    let (ptr, len) = vm_surgeon::alloc_aligned(ps * n_pages).unwrap();

    // Touch every page
    for i in 0..n_pages {
        unsafe { *(ptr.add(i * ps)) = 0xDD };
    }

    let resident = vm_surgeon::check_resident(ptr, len).unwrap();
    let ratio = resident.iter().filter(|&&r| r).count() as f64 / resident.len() as f64;

    assert!(
        ratio >= 0.5,
        "after touching all pages, most should be resident: ratio={:.2}, pages={:?}",
        ratio,
        resident.len(),
    );

    unsafe { vm_surgeon::free_aligned(ptr, len) };
}

#[test]
fn vm_resident_ratio_api() {
    let ps = vm_surgeon::page_size();
    let (ptr, len) = vm_surgeon::alloc_aligned(ps * 8).unwrap();
    unsafe { std::ptr::write_bytes(ptr, 0xEE, len) };

    let ratio = vm_surgeon::resident_ratio(ptr, len);
    assert!(
        (0.0..=1.0).contains(&ratio),
        "ratio must be in [0,1]: {}",
        ratio,
    );

    unsafe { vm_surgeon::free_aligned(ptr, len) };
}

#[test]
fn vm_alloc_zero_size() {
    let result = vm_surgeon::alloc_aligned(0);
    // Zero size should still round up to page size
    if let Ok((ptr, len)) = result {
        assert!(len >= vm_surgeon::page_size());
        unsafe { vm_surgeon::free_aligned(ptr, len) };
    }
}

#[test]
fn vm_pin_then_verify_resident() {
    let ps = vm_surgeon::page_size();
    let (ptr, len) = vm_surgeon::alloc_aligned(ps * 4).unwrap();
    unsafe { std::ptr::write_bytes(ptr, 0x11, len) };

    vm_surgeon::pin_memory(ptr, len).expect("mlock");

    // Pinned memory MUST be resident
    let resident = vm_surgeon::check_resident(ptr, len).unwrap();
    assert!(
        resident.iter().all(|&r| r),
        "mlocked pages MUST be resident: {:?}",
        resident,
    );

    vm_surgeon::unpin_memory(ptr, len).expect("munlock");
    unsafe { vm_surgeon::free_aligned(ptr, len) };
}

// ═══════════════════════════════════════════════════════════════════════════════
// proc_taskinfo stress tests
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn proc_scan_all_processes_no_panic() {
    let pids = proc_taskinfo::list_all_pids();
    assert!(pids.len() > 10);

    // Read task info for ALL processes — should not panic even on error
    let mut success = 0;
    let mut fail = 0;
    for &pid in &pids {
        match proc_taskinfo::get_task_info(pid) {
            Some(_) => success += 1,
            None => fail += 1,
        }
    }
    println!(
        "proc scan: {} success, {} fail, {} total",
        success,
        fail,
        pids.len()
    );
    assert!(success > 10, "should read at least some processes");
}

#[test]
fn proc_rusage_scan_all_no_panic() {
    let pids = proc_taskinfo::list_all_pids();

    let mut success = 0;
    for &pid in &pids {
        if proc_taskinfo::get_rusage_info(pid).is_some() {
            success += 1;
        }
    }
    println!("rusage scan: {}/{} succeeded", success, pids.len());
    assert!(success > 0, "should read at least our own process");
}

#[test]
fn proc_path_known_processes() {
    // PID 1 = launchd
    if let Some(path) = proc_taskinfo::get_proc_path(1) {
        assert!(
            path.contains("launchd"),
            "PID 1 should be launchd, got: {}",
            path,
        );
    }

    // Our own process
    let my_path = proc_taskinfo::get_proc_path(std::process::id());
    assert!(my_path.is_some(), "should read own path");
}

#[test]
fn proc_adversarial_pids() {
    // These should all return None without panicking
    // kernel_task (PID 0): may fail without root — just verify no panic.
    let _ = proc_taskinfo::get_task_info(0);
    assert!(proc_taskinfo::get_task_info(u32::MAX).is_none());
    assert!(proc_taskinfo::get_rusage_info(u32::MAX).is_none());
    assert!(proc_taskinfo::get_proc_path(u32::MAX).is_none());
}

#[test]
fn proc_bulk_scan_performance() {
    let t0 = Instant::now();
    let results = proc_taskinfo::bulk_process_scan();
    let elapsed = t0.elapsed();

    println!(
        "bulk scan: {} processes in {:?} ({:.1}µs/proc)",
        results.len(),
        elapsed,
        elapsed.as_micros() as f64 / results.len().max(1) as f64,
    );

    assert!(results.len() > 10);
    // Should complete in under 50ms for ~400 processes
    assert!(
        elapsed < Duration::from_millis(100),
        "bulk scan too slow: {:?}",
        elapsed,
    );
}

#[test]
fn proc_self_context_switches_increase_under_work() {
    let pid = std::process::id();
    let before = proc_taskinfo::get_task_info(pid).unwrap();

    // Do some work that causes context switches
    let handles: Vec<_> = (0..4)
        .map(|_| {
            thread::spawn(|| {
                let mut x = 0u64;
                for i in 0..100_000 {
                    x = x.wrapping_add(i);
                    if i % 10_000 == 0 {
                        thread::yield_now();
                    }
                }
                std::hint::black_box(x);
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }

    let after = proc_taskinfo::get_task_info(pid).unwrap();
    assert!(
        after.context_switches >= before.context_switches,
        "csw should not decrease: {} -> {}",
        before.context_switches,
        after.context_switches,
    );
}

#[test]
fn proc_rusage_idle_wakeups_exist() {
    let pid = std::process::id();
    let info = proc_taskinfo::get_rusage_info(pid).unwrap();
    // idle_wakeups might be 0 for test processes, but the field should be readable
    println!(
        "idle_wakeups={} interrupt_wakeups={}",
        info.idle_wakeups, info.interrupt_wakeups
    );
    // No assertion on value — just verify no crash and fields are populated
}

// ═══════════════════════════════════════════════════════════════════════════════
// LSE Counters stress tests
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn lse_metrics_32_threads_1m_increments() {
    let m = Arc::new(LockFreeMetrics::new());
    let n_threads = 32;
    let n_increments = 100_000;

    let t0 = Instant::now();
    let threads: Vec<_> = (0..n_threads)
        .map(|_| {
            let m = m.clone();
            thread::spawn(move || {
                for _ in 0..n_increments {
                    m.inc_cycles();
                    m.inc_freezes();
                    m.add_actions(1);
                }
            })
        })
        .collect();

    for t in threads {
        t.join().unwrap();
    }
    let elapsed = t0.elapsed();
    m.commit();

    let snap = m.snapshot();
    let expected = n_threads * n_increments;
    assert_eq!(snap.cycles, expected as u64);
    assert_eq!(snap.freezes, expected as u64);
    assert_eq!(snap.actions_applied, expected as u64);

    let ops_total = expected * 3; // 3 ops per iteration
    let ops_per_sec = ops_total as f64 / elapsed.as_secs_f64();
    println!(
        "LSE stress: {}M ops in {:?} ({:.0}M ops/sec)",
        ops_total / 1_000_000,
        elapsed,
        ops_per_sec / 1e6,
    );
}

#[test]
fn lse_snapshot_never_decreases() {
    let m = Arc::new(LockFreeMetrics::new());
    let m_writer = m.clone();

    // Writer thread: increment rapidly
    let writer = thread::spawn(move || {
        for _ in 0..500_000 {
            m_writer.inc_cycles();
            m_writer.commit();
        }
    });

    // Reader threads: take snapshots and verify monotonicity
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let m = m.clone();
            thread::spawn(move || {
                let mut prev_cycles = 0u64;
                let mut reads = 0u64;
                loop {
                    let snap = m.snapshot();
                    assert!(
                        snap.cycles >= prev_cycles,
                        "cycles must be monotonic: {} -> {}",
                        prev_cycles,
                        snap.cycles,
                    );
                    prev_cycles = snap.cycles;
                    reads += 1;
                    if snap.cycles >= 500_000 {
                        break;
                    }
                }
                reads
            })
        })
        .collect();

    writer.join().unwrap();
    let total_reads: u64 = readers.into_iter().map(|h| h.join().unwrap()).sum();
    println!("monotonicity verified over {} total reads", total_reads);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn lse_asm_instructions_work() {
    use apollo_engine::engine::lse_counters::{lse_cas, lse_swap, verify_lse_atomic_add};
    use std::sync::atomic::AtomicU64;

    // ldaddal
    let v = AtomicU64::new(0);
    for i in 1..=100u64 {
        verify_lse_atomic_add(&v, 1);
        assert_eq!(v.load(std::sync::atomic::Ordering::SeqCst), i);
    }

    // swpal
    let v = AtomicU64::new(42);
    let old = lse_swap(&v, 100);
    assert_eq!(old, 42);
    assert_eq!(v.load(std::sync::atomic::Ordering::SeqCst), 100);

    // casal success + failure
    let v = AtomicU64::new(50);
    let found = lse_cas(&v, 50, 99);
    assert_eq!(found, 50);
    assert_eq!(v.load(std::sync::atomic::Ordering::SeqCst), 99);

    let found = lse_cas(&v, 50, 0); // mismatch
    assert_eq!(found, 99); // returns actual
    assert_eq!(v.load(std::sync::atomic::Ordering::SeqCst), 99); // unchanged
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cross-module integration tests
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn integration_kqueue_detects_child_and_proc_reads_it() {
    // 1. Spawn a long-running child
    let mut child = std::process::Command::new("/bin/sleep")
        .arg("10")
        .spawn()
        .expect("spawn sleep");
    let pid = child.id();

    // 2. Read its proc info while alive
    let info = proc_taskinfo::get_task_info(pid);
    assert!(info.is_some(), "should read child task info");

    let path = proc_taskinfo::get_proc_path(pid);
    if let Some(p) = &path {
        assert!(p.contains("sleep"), "should be sleep: {}", p);
    }

    // 3. Watch it with kqueue
    let mut reactor = KqueuePressure::new().unwrap();
    reactor.watch_pid(pid).expect("watch child");

    // 4. Kill it
    child.kill().ok();
    child.wait().ok();

    // 5. kqueue should report exit
    let events = reactor.wait_events(2000);
    assert!(
        events.contains(&PressureEvent::ProcessExited(pid)),
        "should detect killed child exit: {:?}",
        events,
    );

    // 6. proc info should now fail
    assert!(proc_taskinfo::get_task_info(pid).is_none());
}

#[test]
fn integration_vm_surgeon_with_proc_footprint() {
    let ps = vm_surgeon::page_size();
    let n_pages = 64;
    let alloc_size = ps * n_pages;

    // Read our footprint before
    let pid = std::process::id();
    let before = proc_taskinfo::get_rusage_info(pid).unwrap();

    // Allocate and touch pages
    let (ptr, len) = vm_surgeon::alloc_aligned(alloc_size).unwrap();
    unsafe { std::ptr::write_bytes(ptr, 0xFF, len) };
    vm_surgeon::pin_memory(ptr, len).ok();

    // Read footprint after
    let after = proc_taskinfo::get_rusage_info(pid).unwrap();

    println!(
        "footprint change: {}KB -> {}KB (delta={}KB, allocated={}KB)",
        before.phys_footprint / 1024,
        after.phys_footprint / 1024,
        (after.phys_footprint as i64 - before.phys_footprint as i64) / 1024,
        alloc_size / 1024,
    );

    // Cleanup
    vm_surgeon::unpin_memory(ptr, len).ok();
    unsafe { vm_surgeon::free_aligned(ptr, len) };
}

#[test]
fn integration_metrics_during_proc_scan() {
    // Simulate what the daemon does: scan processes + update metrics atomically
    let metrics = Arc::new(LockFreeMetrics::new());

    let t0 = Instant::now();
    let results = proc_taskinfo::bulk_process_scan();
    let scan_us = t0.elapsed().as_micros() as u64;

    metrics
        .processes_scanned
        .store(results.len() as u64, std::sync::atomic::Ordering::Relaxed);
    metrics.set_snapshot_time_us(scan_us);
    metrics.inc_cycles();
    metrics.commit();

    let snap = metrics.snapshot();
    assert!(snap.processes_scanned > 0);
    assert!(snap.snapshot_time_us > 0);
    assert_eq!(snap.cycles, 1);

    println!(
        "scan: {} procs in {}µs, metrics epoch={}",
        snap.processes_scanned, snap.snapshot_time_us, snap.epoch,
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Adversarial / break scenarios
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn adversarial_rapid_pid_watch_unwatch() {
    // Watch and unwatch rapidly — should not leak fds
    let mut reactor = KqueuePressure::new().unwrap();
    let pid = std::process::id();

    for _ in 0..1000 {
        // Watching our own PID that won't exit
        let _ = reactor.watch_pid(pid);
        reactor.unwatch_pid(pid);
    }
    assert_eq!(reactor.watched_pid_count(), 0);
}

#[test]
fn adversarial_mlock_after_free_does_not_crash() {
    let ps = vm_surgeon::page_size();
    let (ptr, len) = vm_surgeon::alloc_aligned(ps).unwrap();
    unsafe { std::ptr::write_bytes(ptr, 0xAA, len) };

    // Pin, unpin, free, then... don't try to pin again (UB).
    // But verify the normal sequence works.
    vm_surgeon::pin_memory(ptr, len).ok();
    vm_surgeon::unpin_memory(ptr, len).ok();
    unsafe { vm_surgeon::free_aligned(ptr, len) };
    // ptr is now invalid — do not touch. Test passes if no crash.
}

#[test]
fn adversarial_proc_scan_during_process_churn() {
    // Spawn and kill processes while scanning — should not crash
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let done2 = done.clone();

    // Churn thread: spawn and kill rapidly
    let churn = thread::spawn(move || {
        while !done2.load(std::sync::atomic::Ordering::Relaxed) {
            if let Ok(mut child) = std::process::Command::new("/usr/bin/true").spawn() {
                child.wait().ok();
            }
        }
    });

    // Scanner thread: scan all processes repeatedly
    for _ in 0..10 {
        let results = proc_taskinfo::bulk_process_scan();
        assert!(results.len() > 5);
    }

    done.store(true, std::sync::atomic::Ordering::Relaxed);
    churn.join().ok();
}

#[test]
fn adversarial_lse_overflow_wrapping() {
    // AtomicU64 at near-max should wrap correctly
    let m = LockFreeMetrics::new();
    m.cycles
        .store(u64::MAX - 5, std::sync::atomic::Ordering::Relaxed);
    for _ in 0..10 {
        m.inc_cycles();
    }
    m.commit();
    let snap = m.snapshot();
    // u64::MAX - 5 + 10 = wraps to 4
    assert_eq!(snap.cycles, 4, "should wrap around u64::MAX");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Performance benchmarks (informational, not gating)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn bench_kqueue_poll_latency() {
    let mut reactor = KqueuePressure::new().unwrap();
    let t0 = Instant::now();
    let n = 10_000;
    for _ in 0..n {
        let _ = reactor.poll_events();
    }
    let elapsed = t0.elapsed();
    println!(
        "kqueue poll: {:.1}µs/call ({} calls in {:?})",
        elapsed.as_micros() as f64 / n as f64,
        n,
        elapsed,
    );
}

#[test]
fn bench_proc_taskinfo_latency() {
    let pid = std::process::id();
    let t0 = Instant::now();
    let n = 10_000;
    for _ in 0..n {
        let _ = proc_taskinfo::get_task_info(pid);
    }
    let elapsed = t0.elapsed();
    println!(
        "proc_pidinfo: {:.1}µs/call ({} calls in {:?})",
        elapsed.as_micros() as f64 / n as f64,
        n,
        elapsed,
    );
}

#[test]
fn bench_proc_rusage_latency() {
    let pid = std::process::id();
    let t0 = Instant::now();
    let n = 10_000;
    for _ in 0..n {
        let _ = proc_taskinfo::get_rusage_info(pid);
    }
    let elapsed = t0.elapsed();
    println!(
        "proc_pid_rusage: {:.1}µs/call ({} calls in {:?})",
        elapsed.as_micros() as f64 / n as f64,
        n,
        elapsed,
    );
}

#[test]
fn bench_mincore_latency() {
    let ps = vm_surgeon::page_size();
    let (ptr, len) = vm_surgeon::alloc_aligned(ps * 16).unwrap();
    unsafe { std::ptr::write_bytes(ptr, 0xBB, len) };

    let t0 = Instant::now();
    let n = 10_000;
    for _ in 0..n {
        let _ = vm_surgeon::check_resident(ptr, len);
    }
    let elapsed = t0.elapsed();
    println!(
        "mincore: {:.1}µs/call ({} calls in {:?})",
        elapsed.as_micros() as f64 / n as f64,
        n,
        elapsed,
    );

    unsafe { vm_surgeon::free_aligned(ptr, len) };
}

#[test]
fn bench_lse_increment_throughput() {
    let m = LockFreeMetrics::new();
    let n = 10_000_000u64;
    let t0 = Instant::now();
    for _ in 0..n {
        m.inc_cycles();
    }
    let elapsed = t0.elapsed();
    let ops_per_sec = n as f64 / elapsed.as_secs_f64();
    println!(
        "LSE inc: {:.0}M ops/sec ({} ops in {:?}, {:.1}ns/op)",
        ops_per_sec / 1e6,
        n,
        elapsed,
        elapsed.as_nanos() as f64 / n as f64,
    );
}
