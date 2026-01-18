# satellite-worker

Worker pool implementation for parallel SAT solving.

## Features

- **CPU worker pool** - Thread pool with NUMA awareness
- **Job scheduling** - Priority-based job execution
- **Work stealing** - Steal work from idle threads

## Usage

```rust
use satellite_worker::{WorkerPool, Job};

let pool = WorkerPool::new(num_cpus::get());
pool.submit(Job::Bcp { clause_range: 0..32 });
```

## License

MIT
