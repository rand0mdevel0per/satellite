//! Vector of batches.

use serde::{Deserialize, Serialize};
use super::{VarId, Batch};

/// A vector of batches, enabling 2D boolean arrays.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VecVar {
    /// Starting variable ID.
    base_id: VarId,
    /// Dimension of each inner batch.
    inner_dim: usize,
    /// Number of batches in the vector.
    outer_dim: usize,
}

impl VecVar {
    /// Creates a new vector of batches.
    #[must_use]
    pub const fn new(base_id: VarId, inner_dim: usize, outer_dim: usize) -> Self {
        Self { base_id, inner_dim, outer_dim }
    }

    /// Returns the inner batch dimension.
    #[must_use]
    pub const fn inner_dim(&self) -> usize {
        self.inner_dim
    }

    /// Returns the outer dimension (number of batches).
    #[must_use]
    pub const fn outer_dim(&self) -> usize {
        self.outer_dim
    }

    /// Returns the total number of boolean variables.
    #[must_use]
    pub const fn total_vars(&self) -> usize {
        self.inner_dim * self.outer_dim
    }

    /// Gets the batch at the specified index.
    ///
    /// # Panics
    /// Panics if `index >= outer_dim`.
    #[must_use]
    pub fn get(&self, index: usize) -> Batch {
        assert!(index < self.outer_dim, "Index {index} out of bounds for vec of dim {}", self.outer_dim);
        let offset = (index * self.inner_dim) as VarId;
        Batch::new(self.base_id + offset, self.inner_dim)
    }

    /// Iterates over all batches in this vector.
    pub fn iter(&self) -> impl Iterator<Item = Batch> + '_ {
        (0..self.outer_dim).map(|i| self.get(i))
    }
}
