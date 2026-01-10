//! # satellite-branch
//!
//! Git-style constraint branching model.
//!
//! Branches represent semantic constraint splits (e.g., XOR expansions, conditionals)
//! rather than traditional CDCL decision tree branching.

pub mod branch;
pub mod refcount;
pub mod lifecycle;

pub use branch::{Branch, BranchId, BranchStatus};
pub use lifecycle::BranchManager;
