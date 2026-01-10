//! # satellite-kit
//!
//! The core solver library and API for Satellite.
//!
//! This crate provides the high-level API for:
//! - Creating and configuring solvers
//! - Adding constraints
//! - Solving and retrieving results
//!
//! # Example
//!
//! ```ignore
//! use satellite_kit::*;
//!
//! let mut solver = Solver::new();
//! let x = solver.bool_var();
//! let y = solver.batch_var(32);
//!
//! solver.add_constraint(x.and(y[0]));
//!
//! match solver.solve() {
//!     SatResult::Sat(model) => println!("Solution: {:?}", model),
//!     SatResult::Unsat => println!("No solution"),
//!     SatResult::Unknown => println!("Timeout"),
//! }
//! ```

pub mod solver;
pub mod constraint;
pub mod result;

// Re-export core types
pub use satellite_base::types::{BoolVar, Batch, IntVar, VecVar, FloatVar};
pub use satellite_base::{Error, Result};
pub use satellite_cdcl::SatResult;

pub use solver::Solver;
pub use constraint::Constraint;
pub use result::Model;
