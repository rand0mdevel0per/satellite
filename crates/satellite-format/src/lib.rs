//! # satellite-format
//!
//! File format parsing and serialization for Satellite.
//!
//! Supports:
//! - **Advanced-CNF**: Native JSON format with type information
//! - **DIMACS CNF**: Standard SAT competition format
//! - **Snapshot**: Solver state snapshots for reproducibility

pub mod advanced_cnf;
pub mod dimacs;
pub mod snapshot;
pub mod canonical;
pub mod parser;

pub use advanced_cnf::AdvancedCnf;
pub use dimacs::DimacsCnf;
pub use snapshot::Snapshot;
pub use canonical::CanonicalCnf;
pub use parser::{Tokenizer, Expr, Parser};
