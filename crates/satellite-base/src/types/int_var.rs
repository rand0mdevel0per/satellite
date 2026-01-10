//! Integer variable stored as big-endian batch.

use serde::{Deserialize, Serialize};
use super::{VarId, Batch};

/// An integer variable represented as a big-endian batch of booleans.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntVar {
    /// The underlying batch storage.
    batch: Batch,
    /// Whether this integer is signed.
    signed: bool,
}

impl IntVar {
    /// Creates a new unsigned integer variable.
    #[must_use]
    pub const fn new(base_id: VarId, bits: usize) -> Self {
        Self {
            batch: Batch::new(base_id, bits),
            signed: false,
        }
    }

    /// Creates a new signed integer variable.
    #[must_use]
    pub const fn new_signed(base_id: VarId, bits: usize) -> Self {
        Self {
            batch: Batch::new(base_id, bits),
            signed: true,
        }
    }

    /// Returns the bit width.
    #[must_use]
    pub const fn bits(&self) -> usize {
        self.batch.dim()
    }

    /// Returns whether this is a signed integer.
    #[must_use]
    pub const fn is_signed(&self) -> bool {
        self.signed
    }

    /// Returns the underlying batch.
    #[must_use]
    pub const fn as_batch(&self) -> &Batch {
        &self.batch
    }

    /// Returns the most significant bit (sign bit for signed integers).
    #[must_use]
    pub fn msb(&self) -> super::BoolVar {
        self.batch.get(0) // Big-endian: MSB is at index 0
    }

    /// Returns the least significant bit.
    #[must_use]
    pub fn lsb(&self) -> super::BoolVar {
        self.batch.get(self.bits() - 1)
    }
}
