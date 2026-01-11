//! # satellite-kit
//!
//! The core solver library and API for Satellite.
//!
//! This crate provides the high-level API for:
//! - Creating and configuring solvers
//! - Adding constraints
//! - Solving and retrieving results
//! - Circuit gadgets (XOR, ADD, ITE, etc.)
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
//!     SatResult::Unknown(_) => println!("Timeout"),
//! }
//! ```

pub mod constraint;
pub mod context_manager;
pub mod frontend_manager;
pub mod gadgets;
pub mod result;
pub mod solver;

// Re-export core types
pub use satellite_base::types::{Batch, BitVec, BoolVar, FloatVar, IntVar, MemoryView, VecVar, Word};
pub use satellite_base::{Error, Result};
pub use satellite_cdcl::SatResult;

pub use constraint::Constraint;
pub use gadgets::CircuitBuilder;
pub use result::Model;
pub use solver::Solver;

