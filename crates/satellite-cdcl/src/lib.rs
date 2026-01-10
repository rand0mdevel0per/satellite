//! # satellite-cdcl
//!
//! Conflict-Driven Clause Learning (CDCL) core algorithm.
//!
//! Implements:
//! - Decision phase with multiple heuristics (VSIDS, EVSIDS, LRB)
//! - Boolean Constraint Propagation (BCP)
//! - Conflict analysis with UIP resolution
//! - Clause database management

pub mod bcp;
pub mod clause_db;
pub mod conflict;
pub mod decision;
pub mod heuristics;
pub mod solver;

pub use solver::{CdclConfig, CdclSolver, SatResult};
