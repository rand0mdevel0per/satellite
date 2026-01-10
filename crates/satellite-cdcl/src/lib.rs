//! # satellite-cdcl
//!
//! Conflict-Driven Clause Learning (CDCL) core algorithm.
//!
//! Implements:
//! - Decision phase with multiple heuristics (VSIDS, EVSIDS, LRB)
//! - Boolean Constraint Propagation (BCP)
//! - Conflict analysis with UIP resolution
//! - Clause database management

pub mod solver;
pub mod decision;
pub mod bcp;
pub mod conflict;
pub mod clause_db;
pub mod heuristics;

pub use solver::{CdclSolver, CdclConfig, SatResult};
