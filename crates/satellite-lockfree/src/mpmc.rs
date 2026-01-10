//! Multi-producer multi-consumer priority queue.
//!
//! Implements a 4-level priority system with lock-free operations.

use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::ptr;

/// Priority levels for the queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Priority {
    /// High priority (fresh branches).
    High = 0,
    /// Medium-high priority.
    MediumHigh = 1,
    /// Medium-low priority.
    MediumLow = 2,
    /// Low priority (stale branches).
    Low = 3,
}

impl Priority {
    /// Number of priority levels.
    pub const COUNT: usize = 4;

    /// Returns the index for this priority level.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Creates from index.
    #[must_use]
    pub const fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::High),
            1 => Some(Self::MediumHigh),
            2 => Some(Self::MediumLow),
            3 => Some(Self::Low),
            _ => None,
        }
    }
}

/// A node in the lock-free stack.
struct Node<T> {
    data: T,
    next: AtomicPtr<Node<T>>,
}

/// Lock-free stack for a single priority level.
struct LockFreeStack<T> {
    head: AtomicPtr<Node<T>>,
    len: AtomicUsize,
}

impl<T> LockFreeStack<T> {
    const fn new() -> Self {
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
            len: AtomicUsize::new(0),
        }
    }

    fn push(&self, data: T) {
        let node = Box::into_raw(Box::new(Node {
            data,
            next: AtomicPtr::new(ptr::null_mut()),
        }));

        loop {
            let head = self.head.load(Ordering::Acquire);
            unsafe { (*node).next.store(head, Ordering::Relaxed) };

            if self
                .head
                .compare_exchange_weak(head, node, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                self.len.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
    }

    fn pop(&self) -> Option<T> {
        loop {
            let head = self.head.load(Ordering::Acquire);
            if head.is_null() {
                return None;
            }

            let next = unsafe { (*head).next.load(Ordering::Relaxed) };

            if self
                .head
                .compare_exchange_weak(head, next, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                self.len.fetch_sub(1, Ordering::Relaxed);
                let node = unsafe { Box::from_raw(head) };
                return Some(node.data);
            }
        }
    }

    fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Drop for LockFreeStack<T> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}

/// Multi-producer multi-consumer priority queue.
///
/// Uses weighted random selection for queue access to prevent starvation.
pub struct MpmcQueue<T> {
    queues: [LockFreeStack<T>; Priority::COUNT],
    /// Weights for priority selection (higher = more likely to be selected).
    weights: [f64; Priority::COUNT],
}

impl<T> MpmcQueue<T> {
    /// Creates a new MPMC queue with default weights.
    #[must_use]
    pub fn new() -> Self {
        Self::with_weights([0.5, 0.3, 0.15, 0.05])
    }

    /// Creates a new MPMC queue with custom weights.
    ///
    /// Weights should sum to 1.0 for predictable behavior.
    #[must_use]
    pub fn with_weights(weights: [f64; Priority::COUNT]) -> Self {
        Self {
            queues: [
                LockFreeStack::new(),
                LockFreeStack::new(),
                LockFreeStack::new(),
                LockFreeStack::new(),
            ],
            weights,
        }
    }

    /// Pushes an item with the given priority.
    pub fn push(&self, priority: Priority, item: T) {
        self.queues[priority.index()].push(item);
    }

    /// Pops an item using weighted random selection.
    ///
    /// Falls back to lower priority queues if higher ones are empty.
    pub fn pop(&self, rand_value: f64) -> Option<T> {
        // Weighted random selection
        let selected = self.select_queue(rand_value);

        // Try selected queue first
        if let Some(item) = self.queues[selected].pop() {
            return Some(item);
        }

        // Fall back to other queues in priority order
        for i in 0..Priority::COUNT {
            if i != selected {
                if let Some(item) = self.queues[i].pop() {
                    return Some(item);
                }
            }
        }

        None
    }

    /// Pops from a specific priority queue.
    pub fn pop_from(&self, priority: Priority) -> Option<T> {
        self.queues[priority.index()].pop()
    }

    fn select_queue(&self, rand: f64) -> usize {
        let mut cumulative = 0.0;
        for (i, &weight) in self.weights.iter().enumerate() {
            cumulative += weight;
            if rand < cumulative {
                return i;
            }
        }
        Priority::COUNT - 1
    }

    /// Returns the total number of items across all queues.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queues.iter().map(LockFreeStack::len).sum()
    }

    /// Returns whether the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queues.iter().all(LockFreeStack::is_empty)
    }

    /// Returns the number of items in each priority queue.
    #[must_use]
    pub fn len_by_priority(&self) -> [usize; Priority::COUNT] {
        [
            self.queues[0].len(),
            self.queues[1].len(),
            self.queues[2].len(),
            self.queues[3].len(),
        ]
    }
}

impl<T> Default for MpmcQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: MpmcQueue is safe to share across threads
unsafe impl<T: Send> Send for MpmcQueue<T> {}
unsafe impl<T: Send> Sync for MpmcQueue<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_basic_push_pop() {
        let queue = MpmcQueue::new();
        queue.push(Priority::High, 1);
        queue.push(Priority::Low, 2);

        assert_eq!(queue.len(), 2);
        assert!(queue.pop(0.0).is_some());
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_concurrent_access() {
        let queue = std::sync::Arc::new(MpmcQueue::new());
        let mut handles = vec![];

        // Spawn producers
        for i in 0..4 {
            let q = queue.clone();
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    q.push(Priority::High, i * 100 + j);
                }
            }));
        }

        // Wait for producers
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(queue.len(), 400);
    }
}
