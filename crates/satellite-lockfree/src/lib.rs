//! # satellite-lockfree
//!
//! Lock-free data structures for high-performance concurrent access.
//!
//! - **MPMC Queue**: Multi-producer multi-consumer priority queue
//! - **Skiplist**: Concurrent skiplist for branch state tracking
//! - **Vector**: Lock-free growable vector for clause storage
//! - **MVCC**: Multi-version concurrency control for hot path data

pub mod mpmc;
pub mod mvcc;
pub mod skiplist;
pub mod vector;

pub use mpmc::{MpmcQueue, Priority};
pub use mvcc::MvccCell;
pub use skiplist::ConcurrentSkiplist;
pub use vector::LockFreeVec;
