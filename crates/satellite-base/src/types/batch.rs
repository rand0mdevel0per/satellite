//! Batch type - a bus-like collection of booleans.

use super::{BoolVar, VarId};
use serde::{Deserialize, Serialize};

/// A batch of boolean variables, similar to Verilog vectors.
///
/// Batches enable efficient bitwise operations on groups of booleans.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Batch {
    /// Starting variable ID (variables are contiguous).
    base_id: VarId,
    /// Number of bits in this batch.
    dim: usize,
}

impl Batch {
    /// Creates a new batch starting at `base_id` with `dim` bits.
    #[must_use]
    pub const fn new(base_id: VarId, dim: usize) -> Self {
        Self { base_id, dim }
    }

    /// Returns the dimension (number of bits).
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the base variable ID.
    #[must_use]
    pub const fn base_id(&self) -> VarId {
        self.base_id
    }

    /// Gets the boolean variable at the specified index.
    ///
    /// # Panics
    /// Panics if `index >= dim`.
    #[must_use]
    pub fn get(&self, index: usize) -> BoolVar {
        assert!(
            index < self.dim,
            "Index {index} out of bounds for batch of dim {}",
            self.dim
        );
        BoolVar::new(self.base_id + index as VarId)
    }

    /// Iterates over all boolean variables in this batch.
    pub fn iter(&self) -> impl Iterator<Item = BoolVar> + '_ {
        (0..self.dim).map(|i| self.get(i))
    }

    /// Slices this batch to create a sub-batch.
    ///
    /// # Panics
    /// Panics if the range is out of bounds.
    #[must_use]
    pub fn slice(&self, start: usize, end: usize) -> Self {
        assert!(start <= end && end <= self.dim);
        Self {
            base_id: self.base_id + start as VarId,
            dim: end - start,
        }
    }

    /// Returns the base ID (alias for `base_id()`).
    #[must_use]
    pub const fn id(&self) -> VarId {
        self.base_id
    }

    /// Returns the literal for bit at index (1-indexed, positive).
    #[must_use]
    pub fn lit(&self, index: usize) -> i64 {
        assert!(index < self.dim, "Bit index out of bounds");
        (self.base_id + index as VarId) as i64 + 1 // 1-indexed literal
    }
}

impl std::ops::Index<usize> for Batch {
    type Output = BoolVar;

    fn index(&self, index: usize) -> &Self::Output {
        // Note: This is a bit of a hack since BoolVar is Copy.
        // In practice, use .get() for cleaner code.
        panic!("Use .get(index) instead of indexing");
    }
}
