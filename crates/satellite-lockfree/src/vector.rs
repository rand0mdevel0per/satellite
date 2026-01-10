//! Lock-free growable vector for clause storage.

use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::ptr;

/// Chunk size for allocation (number of elements per chunk).
const CHUNK_SIZE: usize = 1024;

/// A chunk of contiguous elements.
struct Chunk<T> {
    data: Box<[Option<T>; CHUNK_SIZE]>,
    next: AtomicPtr<Chunk<T>>,
}

impl<T> Chunk<T> {
    fn new() -> *mut Self {
        // Initialize with None values
        let data: Box<[Option<T>; CHUNK_SIZE]> = Box::new(std::array::from_fn(|_| None));
        Box::into_raw(Box::new(Self {
            data,
            next: AtomicPtr::new(ptr::null_mut()),
        }))
    }
}

/// A lock-free growable vector.
///
/// Grows in chunks to avoid reallocation.
/// Supports atomic append and parallel read.
pub struct LockFreeVec<T> {
    head: AtomicPtr<Chunk<T>>,
    len: AtomicUsize,
}

impl<T> LockFreeVec<T> {
    /// Creates a new empty vector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            head: AtomicPtr::new(Chunk::new()),
            len: AtomicUsize::new(0),
        }
    }

    /// Returns the number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Returns whether the vector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Appends an element and returns its index.
    pub fn push(&self, value: T) -> usize {
        let index = self.len.fetch_add(1, Ordering::SeqCst);
        let (chunk_idx, slot_idx) = (index / CHUNK_SIZE, index % CHUNK_SIZE);

        // Navigate to the correct chunk, allocating if needed
        let mut current = self.head.load(Ordering::Acquire);
        for _ in 0..chunk_idx {
            let chunk = unsafe { &*current };
            let mut next = chunk.next.load(Ordering::Acquire);

            if next.is_null() {
                let new_chunk = Chunk::new();
                match chunk.next.compare_exchange(
                    ptr::null_mut(),
                    new_chunk,
                    Ordering::Release,
                    Ordering::Acquire,
                ) {
                    Ok(_) => next = new_chunk,
                    Err(existing) => {
                        // Someone else allocated, free ours
                        unsafe { drop(Box::from_raw(new_chunk)) };
                        next = existing;
                    }
                }
            }
            current = next;
        }

        // Store the value
        let chunk = unsafe { &mut *current };
        chunk.data[slot_idx] = Some(value);

        index
    }

    /// Gets an element by index.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len() {
            return None;
        }

        let (chunk_idx, slot_idx) = (index / CHUNK_SIZE, index % CHUNK_SIZE);

        let mut current = self.head.load(Ordering::Acquire);
        for _ in 0..chunk_idx {
            if current.is_null() {
                return None;
            }
            current = unsafe { (*current).next.load(Ordering::Acquire) };
        }

        if current.is_null() {
            return None;
        }

        unsafe { (*current).data[slot_idx].as_ref() }
    }
}

impl<T> Default for LockFreeVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for LockFreeVec<T> {
    fn drop(&mut self) {
        let mut current = self.head.load(Ordering::Relaxed);
        while !current.is_null() {
            let chunk = unsafe { Box::from_raw(current) };
            current = chunk.next.load(Ordering::Relaxed);
        }
    }
}

// Safety: LockFreeVec is safe to share across threads
unsafe impl<T: Send> Send for LockFreeVec<T> {}
unsafe impl<T: Send + Sync> Sync for LockFreeVec<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_get() {
        let vec = LockFreeVec::new();
        let idx = vec.push(42);
        assert_eq!(vec.get(idx), Some(&42));
    }

    #[test]
    fn test_multiple_chunks() {
        let vec = LockFreeVec::new();
        for i in 0..CHUNK_SIZE * 2 + 1 {
            vec.push(i);
        }
        assert_eq!(vec.len(), CHUNK_SIZE * 2 + 1);
        assert_eq!(vec.get(0), Some(&0));
        assert_eq!(vec.get(CHUNK_SIZE), Some(&CHUNK_SIZE));
        assert_eq!(vec.get(CHUNK_SIZE * 2), Some(&(CHUNK_SIZE * 2)));
    }
}
