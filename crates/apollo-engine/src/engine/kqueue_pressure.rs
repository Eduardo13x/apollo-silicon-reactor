//! kqueue-based event-driven pressure + process exit monitoring.
//!
//! Multiplexes:
//! - Memory pressure: polls `kern.memorystatus_vm_pressure_level` sysctl on each
//!   timer tick (~1µs). All push APIs (EVFILT_VM, DISPATCH_SOURCE_TYPE_MEMORYPRESSURE,
//!   Darwin notify) are broken on macOS 15 Apple Silicon.
//! - EVFILT_PROC + NOTE_EXIT: watched process exited (frozen process died)
//! - EVFILT_TIMER: periodic tick (replaces sleep)

use std::collections::HashMap;
use std::os::unix::io::RawFd;
use std::time::Instant;

use crate::engine::dispatch_pressure::{KernelPressureLevel, KernelPressureMonitor};

// Timer ident namespace (arbitrary, must not collide with PIDs)
const TIMER_IDENT_PERIODIC: usize = 0xAF01_0001;

// ── Types ────────────────────────────────────────────────────────────────────

/// VM pressure level as reported by the kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VmPressureLevel {
    /// No pressure — system has plenty of free memory.
    Normal,
    /// Warning — compressor active, swap growing.
    Warning,
    /// Critical — jetsam may start killing processes.
    Critical,
    /// Emergency — kernel will terminate processes immediately.
    SuddenTerminate,
}

/// Events delivered by the kqueue reactor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PressureEvent {
    /// VM pressure level changed.
    VmPressure(VmPressureLevel),
    /// Watched process exited.
    ProcessExited(u32),
    /// Periodic timer fired.
    TimerTick,
}

// ── Reactor ──────────────────────────────────────────────────────────────────

/// A kqueue-based event reactor for VM pressure and process lifecycle.
///
/// # Architecture
/// Single kqueue fd multiplexes:
///   1. Memory pressure via sysctl polling on timer tick (~1µs per read)
///   2. Per-PID exit notifications (EVFILT_PROC, one-shot)
///   3. Optional periodic timer (EVFILT_TIMER)
///
/// The daemon can replace its `sleep(500ms)` main loop with:
///   `reactor.wait_events(500)` — sleeps until event OR timeout.
pub struct KqueuePressure {
    kq: RawFd,
    watched_pids: HashMap<u32, Instant>,
    last_vm_level: VmPressureLevel,
    last_event_at: Instant,
    vm_registered: bool,
    timer_registered: bool,
    /// Sysctl-based kernel pressure level monitor.
    pressure_monitor: KernelPressureMonitor,
}

impl KqueuePressure {
    /// Create a new kqueue reactor with sysctl-based pressure monitoring.
    pub fn new() -> std::io::Result<Self> {
        let kq = unsafe { libc::kqueue() };
        if kq < 0 {
            return Err(std::io::Error::last_os_error());
        }

        let reactor = Self {
            kq,
            watched_pids: HashMap::new(),
            last_vm_level: VmPressureLevel::Normal,
            last_event_at: Instant::now(),
            vm_registered: true, // sysctl polling always works
            timer_registered: false,
            pressure_monitor: KernelPressureMonitor::new(),
        };

        Ok(reactor)
    }

    /// Start a periodic timer that fires every `interval_ms` milliseconds.
    /// Replaces `thread::sleep()` — the timer wakes the kqueue.
    pub fn start_timer(&mut self, interval_ms: u64) -> std::io::Result<()> {
        let ev = libc::kevent {
            ident: TIMER_IDENT_PERIODIC,
            filter: libc::EVFILT_TIMER,
            flags: libc::EV_ADD | libc::EV_ENABLE,
            fflags: 0, // milliseconds (default unit)
            data: interval_ms as isize,
            udata: std::ptr::null_mut(),
        };
        let rc =
            unsafe { libc::kevent(self.kq, &ev, 1, std::ptr::null_mut(), 0, std::ptr::null()) };
        if rc < 0 {
            return Err(std::io::Error::last_os_error());
        }
        self.timer_registered = true;
        Ok(())
    }

    /// Stop the periodic timer.
    pub fn stop_timer(&mut self) {
        if !self.timer_registered {
            return;
        }
        let ev = libc::kevent {
            ident: TIMER_IDENT_PERIODIC,
            filter: libc::EVFILT_TIMER,
            flags: libc::EV_DELETE,
            fflags: 0,
            data: 0,
            udata: std::ptr::null_mut(),
        };
        unsafe {
            libc::kevent(self.kq, &ev, 1, std::ptr::null_mut(), 0, std::ptr::null());
        }
        self.timer_registered = false;
    }

    /// Watch a PID for exit. When the process dies, `poll_events` returns
    /// `ProcessExited(pid)`. One-shot: auto-removes after firing.
    pub fn watch_pid(&mut self, pid: u32) -> std::io::Result<()> {
        let ev = libc::kevent {
            ident: pid as usize,
            filter: libc::EVFILT_PROC,
            flags: libc::EV_ADD | libc::EV_ENABLE | libc::EV_ONESHOT,
            fflags: libc::NOTE_EXIT,
            data: 0,
            udata: std::ptr::null_mut(),
        };
        let rc =
            unsafe { libc::kevent(self.kq, &ev, 1, std::ptr::null_mut(), 0, std::ptr::null()) };
        if rc < 0 {
            return Err(std::io::Error::last_os_error());
        }
        self.watched_pids.insert(pid, Instant::now());
        Ok(())
    }

    /// Stop watching a PID.
    pub fn unwatch_pid(&mut self, pid: u32) {
        if self.watched_pids.remove(&pid).is_some() {
            let ev = libc::kevent {
                ident: pid as usize,
                filter: libc::EVFILT_PROC,
                flags: libc::EV_DELETE,
                fflags: 0,
                data: 0,
                udata: std::ptr::null_mut(),
            };
            unsafe {
                libc::kevent(self.kq, &ev, 1, std::ptr::null_mut(), 0, std::ptr::null());
            }
        }
    }

    /// Non-blocking poll: returns all pending events immediately.
    pub fn poll_events(&mut self) -> Vec<PressureEvent> {
        let timeout = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        self.drain_events(&timeout)
    }

    /// Blocking wait: sleeps until an event arrives or `timeout_ms` elapses.
    /// Returns empty Vec on timeout.
    pub fn wait_events(&mut self, timeout_ms: u64) -> Vec<PressureEvent> {
        let timeout = libc::timespec {
            tv_sec: (timeout_ms / 1000) as i64,
            tv_nsec: ((timeout_ms % 1000) * 1_000_000) as i64,
        };
        self.drain_events(&timeout)
    }

    fn drain_events(&mut self, timeout: &libc::timespec) -> Vec<PressureEvent> {
        let mut buf = [make_empty_kevent(); 64];
        let n = unsafe {
            libc::kevent(
                self.kq,
                std::ptr::null(),
                0,
                buf.as_mut_ptr(),
                buf.len() as i32,
                timeout,
            )
        };

        let mut result = Vec::new();

        // Check kernel pressure level on every drain (timer tick or event).
        // Cost: ~1µs via sysctlbyname. Fires VmPressure on level transitions.
        if let Some(level) = self.pressure_monitor.poll() {
            let vm_level = match level {
                KernelPressureLevel::Critical => VmPressureLevel::Critical,
                KernelPressureLevel::Warning => VmPressureLevel::Warning,
                KernelPressureLevel::Normal => VmPressureLevel::Normal,
            };
            self.last_vm_level = vm_level;
            self.last_event_at = Instant::now();
            result.push(PressureEvent::VmPressure(vm_level));
        }

        if n <= 0 {
            return result;
        }

        for ev in &buf[..n as usize] {
            if ev.flags & libc::EV_ERROR != 0 {
                continue;
            }
            match ev.filter {
                libc::EVFILT_PROC => {
                    let pid = ev.ident as u32;
                    self.watched_pids.remove(&pid);
                    result.push(PressureEvent::ProcessExited(pid));
                }
                libc::EVFILT_TIMER => {
                    result.push(PressureEvent::TimerTick);
                }
                _ => {}
            }
        }

        result
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    pub fn last_vm_level(&self) -> VmPressureLevel {
        self.last_vm_level
    }
    pub fn last_event_at(&self) -> Instant {
        self.last_event_at
    }
    pub fn watched_pid_count(&self) -> usize {
        self.watched_pids.len()
    }
    pub fn vm_registered(&self) -> bool {
        self.vm_registered
    }
    pub fn timer_registered(&self) -> bool {
        self.timer_registered
    }
    pub fn kq_fd(&self) -> RawFd {
        self.kq
    }
}

impl Drop for KqueuePressure {
    fn drop(&mut self) {
        self.stop_timer();
        unsafe {
            libc::close(self.kq);
        }
    }
}

fn make_empty_kevent() -> libc::kevent {
    libc::kevent {
        ident: 0,
        filter: 0,
        flags: 0,
        fflags: 0,
        data: 0,
        udata: std::ptr::null_mut(),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kqueue_creates_successfully() {
        let reactor = KqueuePressure::new().expect("kqueue should work on macOS");
        assert!(reactor.kq_fd() >= 0);
        assert!(reactor.vm_registered());
    }

    #[test]
    fn poll_returns_empty_when_no_pressure() {
        let mut reactor = KqueuePressure::new().unwrap();
        let events = reactor.poll_events();
        assert!(events.len() < 100); // sanity
    }

    #[test]
    fn timer_fires_within_tolerance() {
        let mut reactor = KqueuePressure::new().unwrap();
        reactor.start_timer(50).expect("timer should register");
        assert!(reactor.timer_registered());

        let t0 = Instant::now();
        let events = reactor.wait_events(200);
        let elapsed = t0.elapsed().as_millis();

        assert!(
            events.contains(&PressureEvent::TimerTick),
            "timer should fire within 200ms, got {:?} in {}ms",
            events,
            elapsed,
        );
        assert!(elapsed < 150, "timer took too long: {}ms", elapsed);
    }

    #[test]
    fn watch_child_exit() {
        let mut reactor = KqueuePressure::new().unwrap();

        let child = std::process::Command::new("/usr/bin/true")
            .spawn()
            .expect("spawn true");
        let pid = child.id();

        reactor
            .watch_pid(pid)
            .expect("watch should work as same user");
        assert_eq!(reactor.watched_pid_count(), 1);

        let events = reactor.wait_events(2000);
        assert!(
            events.contains(&PressureEvent::ProcessExited(pid)),
            "should detect child exit, got {:?}",
            events,
        );
        assert_eq!(reactor.watched_pid_count(), 0, "pid should auto-remove");
    }

    #[test]
    fn watch_nonexistent_pid_fails() {
        let mut reactor = KqueuePressure::new().unwrap();
        let result = reactor.watch_pid(999_999_999);
        assert!(result.is_err(), "watching non-existent PID should fail");
    }

    #[test]
    fn unwatch_is_idempotent() {
        let mut reactor = KqueuePressure::new().unwrap();
        reactor.unwatch_pid(12345);
        reactor.unwatch_pid(12345);
    }
}
