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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex, RwLock};

    #[test]
    fn lock_recover_normal_mutex() {
        let m = Mutex::new(42u32);
        let guard = m.lock_recover();
        assert_eq!(*guard, 42);
    }

    #[test]
    fn lock_recover_poisoned_mutex() {
        let m = Arc::new(Mutex::new(99u32));
        let m2 = Arc::clone(&m);

        // Poison the mutex from another thread
        let handle = std::thread::spawn(move || {
            let _guard = m2.lock().expect("lock in spawned thread");
            panic!("intentional poison");
        });
        let _ = handle.join(); // will be Err — expected

        // lock_recover should not panic and should return the original value
        let guard = m.lock_recover();
        assert_eq!(*guard, 99, "poisoned mutex should still yield the value");
    }

    #[test]
    fn read_recover_normal_rwlock() {
        let rw = RwLock::new(7u32);
        let guard = rw.read_recover();
        assert_eq!(*guard, 7);
    }

    #[test]
    fn write_recover_normal_rwlock() {
        let rw = RwLock::new(0u32);
        {
            let mut guard = rw.write_recover();
            *guard = 55;
        }
        let guard = rw.read_recover();
        assert_eq!(*guard, 55);
    }
}
