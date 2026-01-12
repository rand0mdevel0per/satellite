//! # satellite-gpu
//!
//! GPU acceleration bindings for the Satellite SAT solver.
//!
//! This crate provides Rust bindings to the C GPU worker API,
//! enabling GPU-accelerated Boolean Constraint Propagation (BCP).
//!
//! ## Features
//!
//! - `cuda` - Enable NVIDIA CUDA support
//! - `hip` - Enable AMD HIP support
//!
//! ## Example
//!
//! ```no_run
//! use satellite_gpu::{GpuWorker, GpuStatus};
//!
//! let mut worker = GpuWorker::new()?;
//! if worker.is_available() {
//!     worker.submit_bcp(&clauses, &assignments)?;
//!     worker.sync();
//!     if let Some(result) = worker.poll_result()? {
//!         println!("Conflict: {}", result.has_conflict);
//!     }
//! }
//! ```

mod ffi;
mod worker;
mod status;
mod abi_op;

pub use worker::GpuWorker;
pub use status::{GpuStatus, GpuMemoryInfo, BcpResult};
pub use abi_op::{CompiledOp, TruthTable, AbiOpRegistry, GpuAbiOpExecutor};

/// GPU error types
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("GPU not available")]
    NotAvailable,
    
    #[error("GPU initialization failed")]
    InitFailed,
    
    #[error("GPU error: {0}")]
    Error(String),
    
    #[error("Invalid parameter")]
    InvalidParameter,
}

pub type Result<T> = std::result::Result<T, GpuError>;
