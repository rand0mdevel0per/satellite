//! Unified error types for Satellite.

use thiserror::Error;

/// The main error type for Satellite operations.
#[derive(Debug, Error)]
pub enum Error {
    /// Variable not found in the solver context.
    #[error("Variable not found: {0}")]
    VariableNotFound(u64),

    /// Type mismatch in constraint.
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch {
        expected: &'static str,
        actual: &'static str,
    },

    /// Invalid batch dimension.
    #[error("Invalid batch dimension: {0}")]
    InvalidDimension(usize),

    /// Constraint compilation failed.
    #[error("Constraint compilation failed: {0}")]
    CompilationError(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization/deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Resource exhaustion.
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// GPU error.
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Internal solver error.
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Convenient Result type alias.
pub type Result<T> = std::result::Result<T, Error>;
