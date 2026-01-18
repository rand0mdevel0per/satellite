# satellite-lockfree

Lock-free data structures for parallel SAT solving.

## Data Structures

- **Lock-free Vector** - Append-only growable array
- **Lock-free Skiplist** - Concurrent sorted map
- **MPMC Queue** - Multi-producer multi-consumer queue with priority levels

## Usage

```rust
use satellite_lockfree::{LockFreeVec, MpmcQueue};

let vec = LockFreeVec::new();
vec.push(42);

let queue = MpmcQueue::new(4); // 4 priority levels
queue.push(job, Priority::High);
```

## License

MIT
