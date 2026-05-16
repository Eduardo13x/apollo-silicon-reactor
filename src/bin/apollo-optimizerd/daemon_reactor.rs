//! # Daemon Reactor
//!
//! kqueue-based event reactor extracted from main.rs (Wave 12).
//! [Fowler 2004] Strangler Fig — pure move, no semantic change.
//!
//! ## Responsibilities
//! - Registers kqueue notifications: thermal, launchd-fork, power
//! - Polls kernel memory pressure level each iteration (~1µs sysctl)
//! - Pulses `reactor_pulses` counter each iteration (liveness proof for watchdog)
//! - Wakes main loop via condvar on any event
//! - Signals resource_interrupt flags (thermal/power/memory)
//!
//! ## Thread model
//! Runs in a dedicated background thread spawned by main().
//! Terminates when `state.stop` OR the passed `stop_requested` flag is set.

use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use apollo_engine::engine::daemon_state::SharedState;
use apollo_engine::engine::lock_ext::LockRecover;
use chrono::Utc;

/// Fast-tick window after a reactor event (seconds).
pub const REACTOR_FAST_TICK_SECS: u64 = 30;

#[link(name = "System")]
extern "C" {
    fn notify_register_file_descriptor(
        name: *const libc::c_char,
        out_fd: *mut libc::c_int,
        flags: libc::c_int,
        out_token: *mut libc::c_int,
    ) -> u32;
}

/// kqueue reactor loop — runs until `state.stop` or `stop_requested` is set.
///
/// Registers OS notifications for thermal, power, and process-fork events.
/// Each loop iteration pulses `reactor_pulses` so the main loop watchdog can
/// distinguish a live-but-quiet reactor from a crashed one.
///
/// # Safety
/// Uses raw `libc` kqueue / kevent / notify APIs. All fd lifetimes are
/// confined to this function; fds are closed before return. The `unsafe`
/// invariant is: every `kevent` fd registered here is either a valid OS fd
/// returned by `notify_register_file_descriptor` (positive) or -1 (sentinel
/// that is never read from).
pub fn run_reactor(state: SharedState, stop_requested: &AtomicBool) -> anyhow::Result<()> {
    unsafe {
        let kq = libc::kqueue();
        if kq == -1 {
            state.metrics.lock_recover().reactor_status.last_error =
                Some("kqueue failed".to_string());
            return Ok(());
        }

        // Memory pressure via sysctl polling (all push APIs are broken on macOS 15).
        // Polls kern.memorystatus_vm_pressure_level (~1µs) on each loop iteration.
        let mut pressure_monitor =
            apollo_engine::engine::dispatch_pressure::KernelPressureMonitor::new();

        // notify → thermal
        let mut thermal_fd: libc::c_int = 0;
        let mut thermal_token: libc::c_int = 0;
        let thermal_name = CString::new("com.apple.system.thermalpressurelevel")
            .expect("static string should not contain NUL");
        let thermal_reg = notify_register_file_descriptor(
            thermal_name.as_ptr(),
            &mut thermal_fd,
            0,
            &mut thermal_token,
        );
        if thermal_reg != 0 {
            state.metrics.lock_recover().reactor_status.last_error = Some(format!(
                "thermal notify_register_file_descriptor failed: {}",
                thermal_reg
            ));
        }
        if thermal_fd > 0 {
            let kev = libc::kevent {
                ident: thermal_fd as usize,
                filter: libc::EVFILT_READ,
                flags: libc::EV_ADD | libc::EV_ENABLE,
                fflags: 0,
                data: 0,
                udata: 2 as *mut libc::c_void, // ID 2 = Thermal
            };
            let _ = libc::kevent(kq, &kev, 1, std::ptr::null_mut(), 0, std::ptr::null());
        }

        // notify → lifecycle spawn
        // NOTE: com.apple.launchd.spawn is a private notification and never
        // delivers to external processes (reactor_events_spawn stays 0).
        // Replaced with EVFILT_PROC NOTE_FORK on launchd PID 1 — fires on
        // every process fork from launchd, which is the actual mechanism we
        // wanted to observe.
        let launch_fd: libc::c_int = -1;
        let launchd_kev = libc::kevent {
            ident: 1, // launchd PID
            filter: libc::EVFILT_PROC,
            flags: libc::EV_ADD | libc::EV_ENABLE | libc::EV_CLEAR,
            fflags: libc::NOTE_FORK as u32,
            data: 0,
            udata: 3 as *mut libc::c_void, // ID 3 = Lifecycle
        };
        let fork_rc = libc::kevent(
            kq,
            &launchd_kev,
            1,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
        );
        if fork_rc < 0 {
            let errno = *libc::__error();
            state.metrics.lock_recover().reactor_status.last_error = Some(format!(
                "EVFILT_PROC NOTE_FORK on launchd failed errno={}",
                errno
            ));
        }

        // notify → power
        let mut power_fd: libc::c_int = 0;
        let mut power_token: libc::c_int = 0;
        let power_name = CString::new("com.apple.system.powersources.source")
            .expect("static string should not contain NUL");
        let power_reg = notify_register_file_descriptor(
            power_name.as_ptr(),
            &mut power_fd,
            0,
            &mut power_token,
        );
        if power_reg != 0 {
            state.metrics.lock_recover().reactor_status.last_error = Some(format!(
                "power notify_register_file_descriptor failed: {}",
                power_reg
            ));
        }
        if power_fd > 0 {
            let kev = libc::kevent {
                ident: power_fd as usize,
                filter: libc::EVFILT_READ,
                flags: libc::EV_ADD | libc::EV_ENABLE,
                fflags: 0,
                data: 0,
                udata: 4 as *mut libc::c_void, // ID 4 = Power
            };
            let _ = libc::kevent(kq, &kev, 1, std::ptr::null_mut(), 0, std::ptr::null());
        }

        let mut out_ev = std::mem::zeroed::<libc::kevent>();
        let timeout = libc::timespec {
            tv_sec: 1,
            tv_nsec: 0,
        };
        while !state.stop.load(Ordering::Acquire) && !stop_requested.load(Ordering::Acquire) {
            let n = libc::kevent(kq, std::ptr::null(), 0, &mut out_ev, 1, &timeout);
            // Pulse on every iteration (event or timeout) so main loop can
            // distinguish a live-but-quiet reactor from a dead one.
            {
                let mut m = state.metrics.lock_recover();
                apollo_engine::engine::lse_counters::LSE_COUNTERS.increment_reactor_pulses();
            }
            // Poll kernel pressure level on every iteration (~1µs sysctl read).
            // Fires memory signal on level transitions (Normal↔Warning↔Critical).
            if let Some(level) = pressure_monitor.poll() {
                use apollo_engine::engine::dispatch_pressure::KernelPressureLevel;
                if level >= KernelPressureLevel::Warning {
                    state
                        .resource_interrupt
                        .memory_signal
                        .store(true, Ordering::Release);
                }
                state.metrics.lock_recover().reactor_status.events_mem += 1;
                // Wake main loop for pressure transition.
                {
                    let (lock, cvar) = &*state.cycle_condvar;
                    let mut triggered = lock.lock_recover();
                    *triggered = true;
                    cvar.notify_one();
                }
            }
            if n == 0 {
                // Timeout — no events within 1 second. Continue so the condvar
                // pulse above keeps the main loop aware the reactor is alive.
                continue;
            }
            if n < 0 {
                // kevent error (e.g. EINTR). Record and retry.
                let errno = *libc::__error();
                if errno != libc::EINTR {
                    state.metrics.lock_recover().reactor_status.last_error =
                        Some(format!("kevent error errno={}", errno));
                }
                continue;
            }

            let id = out_ev.udata as usize;
            // Update shared counters + status in one lock acquisition.
            let reactor_mode = {
                let mut m = state.metrics.lock_recover();
                m.reactor_status.events_total += 1;
                m.reactor_status.last_event_at = Some(Utc::now());
                m.reactor_status.health = "ok".to_string();
                m.reactor_status.mode.clone()
            };
            if id == 2 {
                // Drain thermal pipe
                let mut dummy: i32 = 0;
                let _ = libc::read(thermal_fd, &mut dummy as *mut _ as *mut libc::c_void, 4);
                state.metrics.lock_recover().reactor_status.events_thermal += 1;
                let level = match dummy {
                    0 => "nominal",
                    1 => "moderate",
                    2 => "serious",
                    _ => "critical",
                };
                state.metrics.lock_recover().thermal_level_real = level.to_string();
                // Signal resource sentinel for thermal ≥ serious.
                if dummy >= 2 {
                    state
                        .resource_interrupt
                        .thermal_signal
                        .store(true, Ordering::Release);
                }
            } else if id == 3 {
                // EVFILT_PROC NOTE_FORK on launchd (pid 1) — no pipe to drain.
                // launch_fd == -1 (sentinel); reading from it is always EBADF.
                state.metrics.lock_recover().reactor_status.events_spawn += 1;
            } else if id == 4 {
                let mut dummy: i32 = 0;
                let _ = libc::read(power_fd, &mut dummy as *mut _ as *mut libc::c_void, 4);
                state.metrics.lock_recover().reactor_status.events_power += 1;
                state
                    .resource_interrupt
                    .power_signal
                    .store(true, Ordering::Release);
            } else if id == 1 {
                state.metrics.lock_recover().reactor_status.events_mem += 1;
                state
                    .resource_interrupt
                    .memory_signal
                    .store(true, Ordering::Release);
            }

            apollo_engine::engine::lse_counters::LSE_COUNTERS.set_reactor_event_weight(1.0);
            if reactor_mode.as_str() == "normal" {
                state.metrics.lock_recover().fast_tick_until =
                    Some(Instant::now() + Duration::from_secs(REACTOR_FAST_TICK_SECS));
            }

            // NOTE: reactor_pulses is already incremented once per loop
            // iteration at the top of the loop (including timeouts). Do not
            // increment again here — that would double-count real events.

            // Wake the main loop immediately via condvar.
            {
                let (lock, cvar) = &*state.cycle_condvar;
                let mut triggered = lock.lock_recover();
                *triggered = true;
                cvar.notify_one();
            }
        }

        if thermal_fd > 0 {
            libc::close(thermal_fd);
        }
        // launch_fd == -1 (EVFILT_PROC on launchd PID 1): no fd to close.
        let _ = launch_fd;
        if power_fd > 0 {
            libc::close(power_fd);
        }
        libc::close(kq);
    }

    Ok(())
}
