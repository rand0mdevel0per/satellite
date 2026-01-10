//! Common traits for Satellite components.

use crate::error::Result;

/// Trait for components that can be serialized to/from bytes.
pub trait Serializable: Sized {
    /// Serializes this value to bytes.
    fn to_bytes(&self) -> Result<Vec<u8>>;

    /// Deserializes from bytes.
    fn from_bytes(bytes: &[u8]) -> Result<Self>;
}

/// Trait for components that can be cloned efficiently.
pub trait FastClone: Clone {
    /// Performs a fast clone, potentially reusing allocations.
    fn fast_clone(&self) -> Self {
        self.clone()
    }
}

/// Trait for solver variable types.
pub trait Variable {
    /// Returns the number of underlying boolean variables.
    fn bool_count(&self) -> usize;

    /// Returns the base variable ID.
    fn base_id(&self) -> crate::types::VarId;
}
