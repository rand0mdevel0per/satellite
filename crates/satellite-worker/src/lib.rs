//! # satellite-worker
//!
//! CPU worker pool and GPU bridge for heterogeneous computing.
//!
//! - CPU workers run on a fixed-size thread pool
//! - GPU workers are accessed via FFI to cpp_src/gpu_worker

pub mod gpu_bridge;
pub mod job;
pub mod pool;
pub mod priority;

pub use job::{Job, JobResult};
pub use pool::{WorkerPool, WorkerPoolConfig};
