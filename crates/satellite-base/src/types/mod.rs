//! Type system for Satellite variables.
//!
//! Hierarchy:
//! - `bool` → Single boolean variable
//! - `batch[dim]` → Bus-like batch of booleans (similar to Verilog)
//! - `int` → Integer stored as big-endian in batch
//! - `vec[batch[dim1], dim2]` → Vector of batches
//! - `float` → Arbitrary-precision float

mod batch;
mod bool_var;
mod float_var;
mod int_var;
mod vec_var;

pub use batch::Batch;
pub use bool_var::BoolVar;
pub use float_var::FloatVar;
pub use int_var::IntVar;
pub use vec_var::VecVar;

use serde::{Deserialize, Serialize};

/// Unique identifier for a variable in the solver.
pub type VarId = u64;

/// Type descriptor for solver variables.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VarType {
    /// Single boolean.
    Bool,
    /// Batch of booleans with specified dimension.
    Batch { dim: usize },
    /// Integer with specified bit width.
    Int { bits: usize },
    /// Vector of batches.
    Vec { inner_dim: usize, outer_dim: usize },
    /// Arbitrary-precision float.
    Float { precision: usize },
}

impl VarType {
    /// Returns the total number of boolean variables needed to represent this type.
    #[must_use]
    pub fn bool_count(&self) -> usize {
        match self {
            Self::Bool => 1,
            Self::Batch { dim } | Self::Int { bits: dim } => *dim,
            Self::Vec {
                inner_dim,
                outer_dim,
            } => inner_dim * outer_dim,
            Self::Float { precision } => *precision,
        }
    }
}
