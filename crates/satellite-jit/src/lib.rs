//! # satellite-jit
//!
//! LLVM JIT compilation for ABI-OP (user-defined) constraints.
//!
//! Features:
//! - LLVM IR generation and O3 optimization via inkwell
//! - Control flow analysis and segmentation
//! - Truth table optimization for small inputs
//! - Code hash caching
//! - Sandboxed execution

pub mod compiler;
pub mod analysis;
pub mod truth_table;
pub mod sandbox;
pub mod cache;

pub use compiler::JitCompiler;
pub use cache::CodeCache;
