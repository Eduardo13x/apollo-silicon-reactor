//! Rosetta AOT Compilation Monitor
//!
//! Watches `/var/db/oah/` via kqueue `EVFILT_VNODE` for write events that
//! indicate `oahd-helper` is compiling x86→ARM AOT translations.
//!
//! When active AOT compilation is detected, Apollo should:
//! - Never freeze `oahd` or `oahd-helper`
//! - Suppress background freezing for 30s (oahd needs filesystem access)
//!
//! Pure Rust — no C bridge needed.

use std::time::Instant;

/// Cooldown after last event: suppress background freezing for this long.
const AOT_COOLDOWN_SECS: u64 = 30;

/// Path where Rosetta stores AOT cache.
const OAH_DIR: &str = "/var/db/oah";

/// Rosetta AOT compilation monitor via kqueue.
pub struct RosettaMonitor {
    /// kqueue file descriptor (-1 if unavailable).
    kq: i32,
    /// Open fd on /var/db/oah (-1 if unavailable).
    dir_fd: i32,
    /// Last time a write event was detected.
    last_event: Option<Instant>,
    /// Whether the monitor is operational.
    pub available: bool,
}

impl RosettaMonitor {
    /// Create a new monitor. Safe to call even if Rosetta is not installed.
    pub fn new() -> Self {
        #[cfg(target_os = "macos")]
        {
            use std::ffi::CString;

            let path = match CString::new(OAH_DIR) {
                Ok(p) => p,
                Err(_) => {
                    return Self {
                        kq: -1,
                        dir_fd: -1,
                        last_event: None,
                        available: false,
                    };
                }
            };

            let dir_fd = unsafe { libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_EVTONLY) };
            if dir_fd < 0 {
                return Self {
                    kq: -1,
                    dir_fd: -1,
                    last_event: None,
                    available: false,
                };
            }

            let kq = unsafe { libc::kqueue() };
            if kq < 0 {
                unsafe { libc::close(dir_fd) };
                return Self {
                    kq: -1,
                    dir_fd: -1,
                    last_event: None,
                    available: false,
                };
            }

            // Register EVFILT_VNODE for NOTE_WRITE on the oah directory.
            let changelist = libc::kevent {
                ident: dir_fd as usize,
                filter: libc::EVFILT_VNODE,
                flags: libc::EV_ADD | libc::EV_CLEAR,
                fflags: libc::NOTE_WRITE,
                data: 0,
                udata: std::ptr::null_mut(),
            };

            let ret = unsafe {
                libc::kevent(
                    kq,
                    &changelist as *const libc::kevent,
                    1,
                    std::ptr::null_mut(),
                    0,
                    std::ptr::null(), // don't block
                )
            };

            if ret < 0 {
                unsafe {
                    libc::close(dir_fd);
                    libc::close(kq);
                }
                return Self {
                    kq: -1,
                    dir_fd: -1,
                    last_event: None,
                    available: false,
                };
            }

            Self {
                kq,
                dir_fd,
                last_event: None,
                available: true,
            }
        }

        #[cfg(not(target_os = "macos"))]
        Self {
            kq: -1,
            dir_fd: -1,
            last_event: None,
            available: false,
        }
    }

    /// Non-blocking poll for new write events in /var/db/oah/.
    /// Returns `true` if a new AOT compilation event was detected.
    #[cfg(target_os = "macos")]
    pub fn poll(&mut self) -> bool {
        if !self.available {
            return false;
        }

        let mut event = libc::kevent {
            ident: 0,
            filter: 0,
            flags: 0,
            fflags: 0,
            data: 0,
            udata: std::ptr::null_mut(),
        };

        // Zero timeout = non-blocking poll.
        let timeout = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };

        let n = unsafe {
            libc::kevent(
                self.kq,
                std::ptr::null(),
                0,
                &mut event as *mut libc::kevent,
                1,
                &timeout,
            )
        };

        if n > 0 && (event.fflags & libc::NOTE_WRITE) != 0 {
            self.last_event = Some(Instant::now());
            return true;
        }

        false
    }

    #[cfg(not(target_os = "macos"))]
    pub fn poll(&mut self) -> bool {
        false
    }

    /// Whether AOT compilation is considered active (event within cooldown window).
    pub fn is_compiling(&self) -> bool {
        if let Some(last) = self.last_event {
            last.elapsed().as_secs() < AOT_COOLDOWN_SECS
        } else {
            false
        }
    }

    /// Process names that should be immune when AOT compilation is active.
    pub fn immune_processes() -> &'static [&'static str] {
        &["oahd", "oahd-helper"]
    }
}

impl Drop for RosettaMonitor {
    fn drop(&mut self) {
        if self.dir_fd >= 0 {
            unsafe { libc::close(self.dir_fd) };
        }
        if self.kq >= 0 {
            unsafe { libc::close(self.kq) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_missing_dir() {
        // On systems without Rosetta, should gracefully set available=false.
        let mon = RosettaMonitor::new();
        // Either available (Rosetta installed) or not — no panic.
        if !mon.available {
            assert_eq!(mon.kq, -1);
            assert_eq!(mon.dir_fd, -1);
        }
    }

    #[test]
    fn no_fd_leak() {
        // Create and drop — should not leak file descriptors.
        for _ in 0..10 {
            let _mon = RosettaMonitor::new();
        }
    }

    #[test]
    fn poll_false_when_quiet() {
        let mut mon = RosettaMonitor::new();
        // Even if available, a quiet directory should not trigger.
        assert!(!mon.poll());
        assert!(!mon.is_compiling());
    }
}
