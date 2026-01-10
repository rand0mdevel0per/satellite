//! # satellite-base
//!
//! Core types and utilities for the Satellite SAT solver.
//!
//! This crate provides the foundational building blocks used across all other
//! Satellite crates, including:
//!
//! - **Type System**: Bool, Batch, Int, Vec, Float variable types
//! - **Error Types**: Unified error handling across the solver
//! - **Traits**: Common interfaces for solver components
//! - **Utilities**: Helper functions and macros

pub mod error;
pub mod traits;
pub mod types;
pub mod utils;

pub use error::{Error, Result};
pub use types::{Batch, BoolVar, FloatVar, IntVar, VecVar};
