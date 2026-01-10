//! Arbitrary-precision floating point.

use serde::{Deserialize, Serialize};
use super::{VarId, VecVar};

/// An arbitrary-precision floating point variable.
///
/// Stored as segmented batches aggregated into a vector.
/// The precision determines how many bits are used for representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FloatVar {
    /// Underlying storage as vector of precision segments.
    segments: VecVar,
    /// Total precision in bits.
    precision: usize,
}

impl FloatVar {
    /// Segment size in bits.
    pub const SEGMENT_SIZE: usize = 64;

    /// Creates a new float variable with the specified precision.
    #[must_use]
    pub fn new(base_id: VarId, precision: usize) -> Self {
        let num_segments = (precision + Self::SEGMENT_SIZE - 1) / Self::SEGMENT_SIZE;
        Self {
            segments: VecVar::new(base_id, Self::SEGMENT_SIZE, num_segments),
            precision,
        }
    }

    /// Returns the precision in bits.
    #[must_use]
    pub const fn precision(&self) -> usize {
        self.precision
    }

    /// Returns the number of segments.
    #[must_use]
    pub const fn num_segments(&self) -> usize {
        self.segments.outer_dim()
    }

    /// Returns the underlying vector storage.
    #[must_use]
    pub const fn as_vec(&self) -> &VecVar {
        &self.segments
    }

    /// Returns the total number of boolean variables used.
    #[must_use]
    pub const fn total_vars(&self) -> usize {
        self.segments.total_vars()
    }
}
