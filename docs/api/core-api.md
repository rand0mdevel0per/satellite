# Core API Reference (Rust)

This document describes the Rust API provided by `satellite-kit`.

## Crate Overview

```toml
[dependencies]
satellite-kit = { path = "crates/satellite-kit" }
```

## Solver

The main entry point for solving SAT problems.

```rust
use satellite_kit::{Solver, SatResult};

let mut solver = Solver::new();

// Create variables
let x = solver.bool_var();
let y = solver.bool_var();
let z = solver.batch_var(32);  // 32-bit bitvector

// Add clauses
solver.add_clause(vec![x.id() as i64 + 1, y.id() as i64 + 1]);
solver.add_clause(vec![-(x.id() as i64 + 1), y.id() as i64 + 1]);

// Solve
match solver.solve() {
    Ok(SatResult::Sat(model)) => println!("SAT: {:?}", model),
    Ok(SatResult::Unsat) => println!("UNSAT"),
    Ok(SatResult::Unknown(reason)) => println!("Unknown: {}", reason),
    Err(e) => eprintln!("Error: {}", e),
}
```

### Methods

#### Variable Creation

```rust
/// Creates a new boolean variable.
pub fn bool_var(&mut self) -> BoolVar;

/// Creates a batch (bitvector) variable of specified dimension.
pub fn batch_var(&mut self, dim: usize) -> Batch;

/// Creates an integer variable with specified bit width.
pub fn int_var(&mut self, bits: usize) -> IntVar;

/// Creates a float variable with specified precision.
pub fn float_var(&mut self, precision: usize) -> FloatVar;
```

#### Clause Management

```rust
/// Adds a single clause to the solver.
pub fn add_clause(&mut self, literals: Vec<i64>);

/// Returns the raw clauses for serialization.
pub fn get_clauses(&self) -> &[Vec<i64>];
```

#### Solving

```rust
/// Solves the problem.
pub fn solve(&mut self) -> Result<SatResult>;

/// Solves with temporary assumptions.
pub fn solve_with_assumptions(&mut self, assumptions: &[i64]) -> Result<SatResult>;

/// Sets the timeout in milliseconds.
pub fn set_timeout(&mut self, timeout_ms: u64);
```

#### Cloning

```rust
impl Clone for Solver {
    /// Deep copy for parallel exploration.
    fn clone(&self) -> Self;
}
```

## Types

### BoolVar

```rust
pub struct BoolVar {
    id: VarId,
}

impl BoolVar {
    pub fn id(&self) -> VarId;
    pub fn positive_lit(&self) -> i64;  // id + 1
    pub fn negative_lit(&self) -> i64;  // -(id + 1)
}
```

### Batch

```rust
pub struct Batch {
    base_id: VarId,
    dim: usize,
}

impl Batch {
    pub fn dim(&self) -> usize;
    pub fn id(&self) -> VarId;
    pub fn lit(&self, index: usize) -> i64;
    pub fn get(&self, index: usize) -> BoolVar;
    pub fn slice(&self, start: usize, end: usize) -> Batch;
}
```

### BitVec

```rust
pub struct BitVec {
    inner: Batch,
    width: usize,
}

impl BitVec {
    pub fn new(batch: Batch) -> Self;
    pub fn with_width(start_id: VarId, width: usize) -> Self;
    pub fn width(&self) -> usize;
    pub fn bit(&self, index: usize) -> i64;
    
    /// Generate XOR constraints: result = self ^ other
    pub fn xor(&self, other: &BitVec, result: &BitVec) -> Vec<Vec<i64>>;
    
    /// Generate AND constraints
    pub fn and(&self, other: &BitVec, result: &BitVec) -> Vec<Vec<i64>>;
    
    /// Generate OR constraints
    pub fn or(&self, other: &BitVec, result: &BitVec) -> Vec<Vec<i64>>;
    
    /// Generate NOT constraints
    pub fn not(&self, result: &BitVec) -> Vec<Vec<i64>>;
    
    /// Generate left shift constraints
    pub fn shl(&self, amount: usize, result: &BitVec, zero_lit: i64) -> Vec<Vec<i64>>;
    
    /// Generate right shift constraints
    pub fn shr(&self, amount: usize, result: &BitVec) -> Vec<Vec<i64>>;
}
```

### SatResult

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum SatResult {
    /// Satisfiable with model
    Sat(Vec<bool>),
    /// Unsatisfiable
    Unsat,
    /// Unknown (timeout or error)
    Unknown(String),
}
```

## CircuitBuilder

High-level gadget construction.

```rust
use satellite_kit::{CircuitBuilder, BitVec};

let mut cb = CircuitBuilder::new(start_var_id);

// Allocate bitvectors
let a = cb.new_bitvec(8);
let b = cb.new_bitvec(8);

// Build circuit
let xor_result = cb.add_xor(&a, &b);
let and_result = cb.add_and(&a, &b);
let or_result = cb.add_or(&a, &b);
let not_result = cb.add_not(&a);
let shifted = cb.add_shl(&a, 2);
let mux = cb.add_ite(cond_lit, &a, &b);

// Get generated clauses
let clauses = cb.into_clauses();
```

### Methods

```rust
impl CircuitBuilder {
    pub fn new(start_var: VarId) -> Self;
    pub fn new_var(&mut self) -> i64;
    pub fn new_bitvec(&mut self, width: usize) -> BitVec;
    pub fn add_clause(&mut self, clause: Vec<i64>);
    pub fn add_clauses(&mut self, clauses: Vec<Vec<i64>>);
    
    // Gates
    pub fn add_xor(&mut self, a: &BitVec, b: &BitVec) -> BitVec;
    pub fn add_and(&mut self, a: &BitVec, b: &BitVec) -> BitVec;
    pub fn add_or(&mut self, a: &BitVec, b: &BitVec) -> BitVec;
    pub fn add_not(&mut self, a: &BitVec) -> BitVec;
    pub fn add_shl(&mut self, a: &BitVec, shift: usize) -> BitVec;
    pub fn add_shr(&mut self, a: &BitVec, shift: usize) -> BitVec;
    pub fn add_ite(&mut self, cond: i64, then_val: &BitVec, else_val: &BitVec) -> BitVec;
    pub fn add_add(&mut self, a: &BitVec, b: &BitVec, carry_in: Option<i64>) -> (BitVec, i64);
    pub fn add_sub(&mut self, a: &BitVec, b: &BitVec) -> BitVec;
    pub fn add_eq(&mut self, a: &BitVec, b: &BitVec) -> i64;
    
    pub fn get_clauses(&self) -> &[Vec<i64>];
    pub fn into_clauses(self) -> Vec<Vec<i64>>;
}
```

## Context Manager

Handle-based session management for FFI.

```rust
use satellite_kit::context_manager::*;

// Initialize worker pool (optional, for async)
init_worker_pool(4);

// Create context
let ctx = create_context();

// Add clauses
add_clause(ctx, vec![1, 2, -3]);
add_clause(ctx, vec![-1, 2]);

// Synchronous solve
let result = solve(ctx);

// Or async solve
let job_id = submit_solve(ctx);
let status = poll_job(job_id);
let finished = fetch_finished_jobs(10);

// Fork for parallel exploration
let clones = fork_context(ctx, 4);

// Cleanup
destroy_context(ctx);
```

### Functions

```rust
/// Create new solver context
pub fn create_context() -> u32;

/// Destroy context
pub fn destroy_context(ctx_id: u32) -> bool;

/// Execute closure with context reference
pub fn with_context<F, R>(ctx_id: u32, f: F) -> Option<R>;

/// Add clause to context
pub fn add_clause(ctx_id: u32, literals: Vec<i64>);

/// Add clauses from raw buffer (zero-copy)
pub fn add_clauses_buffer(ctx_id: u32, buffer: &[u8]) -> usize;

/// Create boolean variable
pub fn new_bool_var(ctx_id: u32) -> Option<i64>;

/// Solve synchronously
pub fn solve(ctx_id: u32) -> Option<SatResult>;

/// Submit async solve job
pub fn submit_solve(ctx_id: u32) -> u64;

/// Poll job status
pub fn poll_job(job_id: u64) -> JobStatus;

/// Fetch completed jobs
pub fn fetch_finished_jobs(max_count: usize) -> Vec<(u64, SatResult)>;

/// Fork context into clones
pub fn fork_context(src_ctx_id: u32, num_clones: u32) -> Vec<u32>;

/// Initialize worker pool
pub fn init_worker_pool(num_workers: usize);

/// Get context count
pub fn context_count() -> usize;

/// Get job count
pub fn job_count() -> usize;
```

### JobStatus

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum JobStatus {
    Pending,
    Running,
    Completed(SatResult),
    NotFound,
}
```

## CDCL Solver (Low-Level)

Direct access to the CDCL solver.

```rust
use satellite_cdcl::{CdclSolver, CdclConfig};
use satellite_format::AdvancedCnf;

// Load problem
let problem = AdvancedCnf::from_json_file("problem.json")?;

// Create solver with config
let config = CdclConfig::default();
let mut solver = CdclSolver::with_config(&problem, config);

// Enable UNSAT core tracking
solver.enable_unsat_core();

// Solve
match solver.solve()? {
    SatResult::Sat(model) => { /* ... */ }
    SatResult::Unsat => {
        // Get UNSAT core
        if let Some(core) = solver.get_unsat_core() {
            println!("UNSAT core clauses: {:?}", core);
        }
        if let Some(clauses) = solver.get_unsat_core_clauses() {
            for clause in clauses {
                println!("  {:?}", clause);
            }
        }
    }
    _ => {}
}

// Get statistics
let stats = solver.stats();
println!("Decisions: {}", stats.decisions);
println!("Conflicts: {}", stats.conflicts);
```

### CdclConfig

```rust
pub struct CdclConfig {
    pub restart_interval: u64,
    pub clause_cleanup_interval: u64,
    pub heuristic_weights: HeuristicWeights,
}
```

### SolverStats

```rust
pub struct SolverStats {
    pub decisions: u64,
    pub conflicts: u64,
    pub propagations: u64,
    pub restarts: u64,
    pub learned_clauses: u64,
}
```
