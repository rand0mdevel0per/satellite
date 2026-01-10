//! # satellite-worker
//!
//! CPU worker pool and GPU bridge for heterogeneous computing.
//!
//! - CPU workers run on a fixed-size thread pool
//! - GPU workers are accessed via FFI to cpp_src/gpu_worker

pub mod pool;
pub mod job;
pub mod priority;
pub mod gpu_bridge;

pub use pool::{WorkerPool, WorkerPoolConfig};
pub use job::{Job, JobResult};
