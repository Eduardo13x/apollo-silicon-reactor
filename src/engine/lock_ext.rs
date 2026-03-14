use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Extension trait for `Mutex<T>` that recovers from poisoning.
///
/// In a long-running daemon, a poisoned mutex (caused by a panic in another
/// thread while holding the lock) should not bring down the entire process.
/// This trait provides `.lock_recover()` as a drop-in replacement for
/// `.lock().unwrap_or_else(|e| e.into_inner())`.
pub trait LockRecover<T> {
    fn lock_recover(&self) -> MutexGuard<'_, T>;
}

impl<T> LockRecover<T> for Mutex<T> {
    #[inline]
    fn lock_recover(&self) -> MutexGuard<'_, T> {
        self.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Extension trait for `RwLock<T>` that recovers from poisoning.
///
/// Read-heavy fields (metrics, profile, thermal_state) benefit from
/// `RwLock` over `Mutex`: multiple socket handler threads can read
/// concurrently without blocking each other, while the main loop
/// takes a write lock once per cycle.
pub trait RwLockRecover<T> {
    fn read_recover(&self) -> RwLockReadGuard<'_, T>;
    fn write_recover(&self) -> RwLockWriteGuard<'_, T>;
}

impl<T> RwLockRecover<T> for RwLock<T> {
    #[inline]
    fn read_recover(&self) -> RwLockReadGuard<'_, T> {
        self.read().unwrap_or_else(|e| e.into_inner())
    }

    #[inline]
    fn write_recover(&self) -> RwLockWriteGuard<'_, T> {
        self.write().unwrap_or_else(|e| e.into_inner())
    }
}
