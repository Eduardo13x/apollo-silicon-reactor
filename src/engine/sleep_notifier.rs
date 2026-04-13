//! Sleep Notifier — IOKit power callback for pre-sleep memory flush.
//!
//! Registers with IOKit's `IORegisterForSystemPower` to receive
//! `kIOMessageSystemWillSleep` notifications. This gives us ~30s of
//! grace time before the kernel actually suspends — enough to purge
//! purgeable memory regions and free RAM for a clean hibernate.
//!
//! On 8GB M1 without AC and high compressor load, this prevents macOS
//! from doing a dirty shutdown instead of clean hibernate/resume.
//!
//! ## Why not poll SMC lid_closed?
//! The SMC MSLD key updates, but macOS suspends the daemon within <1s
//! of lid close — faster than the ~5s polling cycle. The IOKit callback
//! fires *before* sleep, with guaranteed time to respond.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Shared flag: set to `true` when the system is about to sleep.
/// The daemon main loop checks this each cycle and performs memory flush.
/// Reset to `false` after the flush is performed.
#[derive(Clone)]
pub struct SleepNotifier {
    will_sleep: Arc<AtomicBool>,
    /// True from `kIOMessageSystemWillSleep` until `kIOMessageSystemHasPoweredOn`.
    /// Stays true across the entire sleep period — unlike `will_sleep` which is
    /// cleared by `acknowledge()` right after the pre-sleep flush.
    /// Use this to gate disk writes: no point flushing while the system is asleep.
    in_sleep: Arc<AtomicBool>,
    /// Whether we successfully registered with IOKit.
    pub available: bool,
}

impl SleepNotifier {
    /// Create and register the IOKit power notification listener.
    /// Spawns a background thread running a CFRunLoop to receive callbacks.
    /// Safe to call without root — registration itself doesn't require privileges.
    pub fn new() -> Self {
        let will_sleep = Arc::new(AtomicBool::new(false));
        let in_sleep = Arc::new(AtomicBool::new(false));

        #[cfg(target_os = "macos")]
        {
            let flag = will_sleep.clone();
            let sleep_flag = in_sleep.clone();
            let available = spawn_iokit_listener(flag, sleep_flag);
            Self {
                will_sleep,
                in_sleep,
                available,
            }
        }

        #[cfg(not(target_os = "macos"))]
        Self {
            will_sleep,
            in_sleep,
            available: false,
        }
    }

    /// Check if a sleep event is pending. Non-blocking.
    pub fn will_sleep_pending(&self) -> bool {
        self.will_sleep.load(Ordering::Acquire)
    }

    /// True from pre-sleep until system has powered on after wake.
    /// Gate non-critical disk writes on `!is_sleeping()` to avoid burning
    /// the macOS daily disk-write budget (~2GB/day) during idle sleep periods.
    pub fn is_sleeping(&self) -> bool {
        self.in_sleep.load(Ordering::Acquire)
    }

    /// Acknowledge the sleep event after performing the flush.
    pub fn acknowledge(&self) {
        self.will_sleep.store(false, Ordering::Release);
    }
}

// ── IOKit FFI ────────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn spawn_iokit_listener(flag: Arc<AtomicBool>, in_sleep: Arc<AtomicBool>) -> bool {
    use std::os::raw::c_void;

    // IOKit power management types.
    type IONotificationPortRef = *mut c_void;
    type IOReturn = i32;

    const K_IO_MESSAGE_SYSTEM_WILL_SLEEP: u32 = 0xe0000280;
    const K_IO_MESSAGE_CAN_SYSTEM_SLEEP: u32 = 0xe0000240;
    // CoreFoundation types for run loop.
    type CFRunLoopRef = *mut c_void;
    type CFRunLoopSourceRef = *mut c_void;
    type CFStringRef = *const c_void;

    const K_IO_MESSAGE_SYSTEM_HAS_POWERED_ON: u32 = 0xe0000300;

    extern "C" {
        fn IORegisterForSystemPower(
            refcon: *mut c_void,
            notify_port: *mut IONotificationPortRef,
            callback: unsafe extern "C" fn(
                refcon: *mut c_void,
                service: u32,
                message_type: u32,
                message_argument: *mut c_void,
            ),
            notifier: *mut u32,
        ) -> u32;

        fn IONotificationPortGetRunLoopSource(
            notify_port: IONotificationPortRef,
        ) -> CFRunLoopSourceRef;

        fn IOAllowPowerChange(kernel_port: u32, notification_id: isize) -> IOReturn;
        fn CFRunLoopGetCurrent() -> CFRunLoopRef;
        fn CFRunLoopAddSource(rl: CFRunLoopRef, source: CFRunLoopSourceRef, mode: CFStringRef);
        fn CFRunLoopRun();

        static kCFRunLoopDefaultMode: CFStringRef;
    }

    // The callback context: pointers to our AtomicBool flags.
    struct CallbackCtx {
        flag: Arc<AtomicBool>,
        /// Set true on WILL_SLEEP, cleared on HAS_POWERED_ON.
        /// Stays true across the full sleep period so callers can gate I/O.
        in_sleep: Arc<AtomicBool>,
        root_port: u32,
    }

    unsafe extern "C" fn power_callback(
        refcon: *mut c_void,
        _service: u32,
        message_type: u32,
        message_argument: *mut c_void,
    ) {
        let ctx = &*(refcon as *const CallbackCtx);
        match message_type {
            K_IO_MESSAGE_SYSTEM_WILL_SLEEP => {
                // System is about to sleep — signal the main loop.
                ctx.flag.store(true, Ordering::Release);
                // Mark as sleeping; stays true until HAS_POWERED_ON.
                ctx.in_sleep.store(true, Ordering::Release);
                // We MUST acknowledge within 30s or the kernel proceeds anyway.
                // The main loop will do the flush and acknowledge, but we also
                // allow here as a safety net (double-allow is harmless).
                // Small delay to give the main loop a chance to see the flag.
                std::thread::sleep(std::time::Duration::from_millis(500));
                IOAllowPowerChange(ctx.root_port, message_argument as isize);
            }
            K_IO_MESSAGE_CAN_SYSTEM_SLEEP => {
                // We always allow sleep — we just want the pre-notification.
                IOAllowPowerChange(ctx.root_port, message_argument as isize);
            }
            K_IO_MESSAGE_SYSTEM_HAS_POWERED_ON => {
                // System woke up — clear the sleep gate so disk writes resume.
                ctx.in_sleep.store(false, Ordering::Release);
            }
            _ => {}
        }
    }

    // Spawn a dedicated thread with its own CFRunLoop for IOKit callbacks.
    let (tx, rx) = std::sync::mpsc::channel::<bool>();

    std::thread::Builder::new()
        .name("apollo-sleep-notifier".to_string())
        .spawn(move || unsafe {
            let mut notify_port: IONotificationPortRef = std::ptr::null_mut();
            let mut notifier: u32 = 0;

            // Leak the context — it lives for the process lifetime.
            let ctx = Box::new(CallbackCtx {
                flag,
                in_sleep,
                root_port: 0, // will be set after registration
            });
            let ctx_ptr = Box::into_raw(ctx);

            let root_port = IORegisterForSystemPower(
                ctx_ptr as *mut c_void,
                &mut notify_port,
                power_callback,
                &mut notifier,
            );

            if root_port == 0 {
                let _ = tx.send(false);
                // Clean up leaked box on failure.
                drop(Box::from_raw(ctx_ptr));
                return;
            }

            // Store root_port in the context so the callback can use it.
            (*ctx_ptr).root_port = root_port;

            let rls = IONotificationPortGetRunLoopSource(notify_port);
            if rls.is_null() {
                let _ = tx.send(false);
                return;
            }

            CFRunLoopAddSource(CFRunLoopGetCurrent(), rls, kCFRunLoopDefaultMode);
            let _ = tx.send(true);

            // Block forever — this thread's run loop processes IOKit callbacks.
            CFRunLoopRun();
        })
        .ok();

    // Wait for registration result (timeout 2s).
    rx.recv_timeout(std::time::Duration::from_secs(2))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notifier_creation() {
        let notifier = SleepNotifier::new();
        // Should not crash. On macOS it may or may not be available
        // depending on whether we can register with IOKit.
        assert!(!notifier.will_sleep_pending());
    }

    #[test]
    fn test_acknowledge() {
        let notifier = SleepNotifier::new();
        notifier.will_sleep.store(true, Ordering::Release);
        assert!(notifier.will_sleep_pending());
        notifier.acknowledge();
        assert!(!notifier.will_sleep_pending());
    }
}
