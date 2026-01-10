//! Multi-Version Concurrency Control for hot path data.
//!
//! Reduces contention on frequently updated data like VSIDS scores.

use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

/// A version in the MVCC chain.
struct Version<T> {
    /// Transaction ID that created this version.
    txid: u64,
    /// The data.
    data: T,
    /// Previous version.
    prev: AtomicPtr<Version<T>>,
}

/// An MVCC cell providing snapshot isolation.
pub struct MvccCell<T> {
    /// Current version.
    current: AtomicPtr<Version<T>>,
    /// Minimum active transaction ID (for GC).
    min_active_txid: AtomicU64,
}

impl<T: Clone> MvccCell<T> {
    /// Creates a new MVCC cell with initial value.
    pub fn new(initial: T, txid: u64) -> Self {
        let version = Box::into_raw(Box::new(Version {
            txid,
            data: initial,
            prev: AtomicPtr::new(ptr::null_mut()),
        }));

        Self {
            current: AtomicPtr::new(version),
            min_active_txid: AtomicU64::new(txid),
        }
    }

    /// Reads the value visible to the given transaction.
    pub fn read(&self, reader_txid: u64) -> T {
        let mut current = self.current.load(Ordering::Acquire);

        // Find the appropriate version for this reader
        while !current.is_null() {
            let version = unsafe { &*current };
            if version.txid <= reader_txid {
                return version.data.clone();
            }
            current = version.prev.load(Ordering::Acquire);
        }

        // Should not reach here if properly initialized
        panic!("No visible version found");
    }

    /// Writes a new value, creating a new version.
    ///
    /// Uses copy-on-write semantics.
    pub fn write(&self, new_value: T, writer_txid: u64) {
        let current = self.current.load(Ordering::Acquire);

        let new_version = Box::into_raw(Box::new(Version {
            txid: writer_txid,
            data: new_value,
            prev: AtomicPtr::new(current),
        }));

        // CAS to install new version
        loop {
            let old = self.current.load(Ordering::Acquire);
            unsafe { (*new_version).prev.store(old, Ordering::Relaxed) };

            if self
                .current
                .compare_exchange_weak(old, new_version, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Updates the minimum active transaction ID for garbage collection.
    pub fn update_min_txid(&self, new_min: u64) {
        self.min_active_txid.fetch_max(new_min, Ordering::Relaxed);
        self.gc();
    }

    /// Garbage collects old versions.
    fn gc(&self) {
        let min_txid = self.min_active_txid.load(Ordering::Relaxed);
        let current = self.current.load(Ordering::Acquire);

        if current.is_null() {
            return;
        }

        // Find the last version needed and truncate the rest
        let mut prev_ptr = unsafe { &(*current).prev };
        loop {
            let prev = prev_ptr.load(Ordering::Acquire);
            if prev.is_null() {
                break;
            }

            let prev_version = unsafe { &*prev };
            if prev_version.txid < min_txid {
                // This and all previous can be garbage collected
                if prev_ptr
                    .compare_exchange(prev, ptr::null_mut(), Ordering::Release, Ordering::Relaxed)
                    .is_ok()
                {
                    // Free the chain
                    let mut to_free = prev;
                    while !to_free.is_null() {
                        let v = unsafe { Box::from_raw(to_free) };
                        to_free = v.prev.load(Ordering::Relaxed);
                    }
                }
                break;
            }

            prev_ptr = &prev_version.prev;
        }
    }
}

impl<T> Drop for MvccCell<T> {
    fn drop(&mut self) {
        let mut current = self.current.load(Ordering::Relaxed);
        while !current.is_null() {
            let version = unsafe { Box::from_raw(current) };
            current = version.prev.load(Ordering::Relaxed);
        }
    }
}

// Safety: MvccCell is safe to share across threads
unsafe impl<T: Send> Send for MvccCell<T> {}
unsafe impl<T: Send + Sync> Sync for MvccCell<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_read_write() {
        let cell = MvccCell::new(42, 1);
        assert_eq!(cell.read(1), 42);

        cell.write(100, 2);
        assert_eq!(cell.read(2), 100);
        assert_eq!(cell.read(1), 42); // Old reader still sees old value
    }
}
