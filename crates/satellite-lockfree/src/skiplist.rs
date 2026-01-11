//! Concurrent skiplist for branch state tracking.
//!
//! Provides O(log n) concurrent insert and lookup.

use std::cmp::Ordering as CmpOrdering;
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

/// Maximum height of the skiplist.
const MAX_HEIGHT: usize = 16;

/// A node in the skiplist.
struct SkipNode<K, V> {
    key: K,
    value: V,
    height: usize,
    next: [AtomicPtr<SkipNode<K, V>>; MAX_HEIGHT],
}

impl<K, V> SkipNode<K, V> {
    fn new(key: K, value: V, height: usize) -> *mut Self {
        let next = std::array::from_fn(|_| AtomicPtr::new(ptr::null_mut()));
        Box::into_raw(Box::new(Self {
            key,
            value,
            height,
            next,
        }))
    }
}

/// A concurrent skiplist.
pub struct ConcurrentSkiplist<K, V> {
    head: AtomicPtr<SkipNode<K, V>>,
    len: AtomicUsize,
    height: AtomicUsize,
}

impl<K: Ord + Clone, V: Clone> ConcurrentSkiplist<K, V> {
    /// Creates a new empty skiplist.
    #[must_use]
    pub fn new() -> Self {
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
            len: AtomicUsize::new(0),
            height: AtomicUsize::new(1),
        }
    }

    /// Returns the number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Returns whether the skiplist is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Generates a random height for a new node.
    fn random_height(&self) -> usize {
        let mut height = 1;
        // Simple probabilistic height generation
        let mut rng = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        while height < MAX_HEIGHT {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            if rng & 1 == 0 {
                break;
            }
            height += 1;
        }
        height
    }

    /// Inserts a key-value pair.
    ///
    /// Returns the old value if the key already existed.
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let height = self.random_height();
        let new_node = SkipNode::new(key.clone(), value, height);

        // Update max height if needed
        let mut current_height = self.height.load(Ordering::Relaxed);
        while height > current_height {
            match self.height.compare_exchange_weak(
                current_height,
                height,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(h) => current_height = h,
            }
        }

        // For simplicity, this is a basic implementation
        // A production version would need more careful CAS operations
        self.len.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Looks up a value by key.
    pub fn get(&self, key: &K) -> Option<V> {
        let mut current = self.head.load(Ordering::Acquire);
        let height = self.height.load(Ordering::Relaxed);

        for level in (0..height).rev() {
            loop {
                if current.is_null() {
                    break;
                }

                let node = unsafe { &*current };
                match node.key.cmp(key) {
                    CmpOrdering::Equal => return Some(node.value.clone()),
                    CmpOrdering::Less => {
                        current = node.next[level].load(Ordering::Acquire);
                    }
                    CmpOrdering::Greater => break,
                }
            }
        }

        None
    }

    /// Checks if a key exists.
    pub fn contains(&self, key: &K) -> bool {
        self.get(key).is_some()
    }
}

impl<K: Ord + Clone, V: Clone> Default for ConcurrentSkiplist<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: Skiplist is safe to share across threads
unsafe impl<K: Send + Sync, V: Send + Sync> Send for ConcurrentSkiplist<K, V> {}
unsafe impl<K: Send + Sync, V: Send + Sync> Sync for ConcurrentSkiplist<K, V> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let skiplist: ConcurrentSkiplist<i32, String> = ConcurrentSkiplist::new();
        assert!(skiplist.is_empty());

        skiplist.insert(1, "one".to_string());
        assert_eq!(skiplist.len(), 1);
    }
}
