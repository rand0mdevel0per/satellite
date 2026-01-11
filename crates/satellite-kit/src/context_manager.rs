//! Handle-based context manager for session management.
//!
//! This module provides a global registry for solver contexts,
//! allowing Python and other FFI clients to reference solvers by ID.

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;

use crate::solver::Solver;
use satellite_cdcl::SatResult;

/// A solver context with optional pending job information.
pub struct Context {
    /// The underlying solver.
    pub solver: Solver,
    /// Optional pending assumptions for batch solving.
    pub pending_assumptions: Option<Vec<i64>>,
}

impl Context {
    /// Creates a new context with a fresh solver.
    pub fn new() -> Self {
        Self {
            solver: Solver::new(),
            pending_assumptions: None,
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

/// Job status for async solving.
#[derive(Debug, Clone, PartialEq)]
pub enum JobStatus {
    /// Job is pending execution.
    Pending,
    /// Job is currently running.
    Running,
    /// Job completed with result.
    Completed(SatResult),
    /// Job not found.
    NotFound,
}

/// A pending or completed job.
struct Job {
    ctx_id: u32,
    status: JobStatus,
}

/// Global context registry.
static CONTEXTS: Lazy<Mutex<HashMap<u32, Context>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Global job registry.
static JOBS: Lazy<Mutex<HashMap<u64, Job>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Atomic counter for generating unique context IDs.
static NEXT_CTX_ID: AtomicU32 = AtomicU32::new(1);

/// Atomic counter for generating unique job IDs.
static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

/// Creates a new solver context and returns its ID.
pub fn create_context() -> u32 {
    let id = NEXT_CTX_ID.fetch_add(1, Ordering::SeqCst);
    let ctx = Context::new();
    CONTEXTS.lock().unwrap().insert(id, ctx);
    id
}

/// Destroys a solver context by its ID.
pub fn destroy_context(ctx_id: u32) -> bool {
    CONTEXTS.lock().unwrap().remove(&ctx_id).is_some()
}

/// Executes a closure with a reference to a context.
pub fn with_context<R, F: FnOnce(&mut Context) -> R>(ctx_id: u32, f: F) -> Option<R> {
    let mut guard = CONTEXTS.lock().unwrap();
    guard.get_mut(&ctx_id).map(f)
}

/// Adds a clause to a context.
pub fn add_clause(ctx_id: u32, literals: Vec<i64>) -> bool {
    with_context(ctx_id, |ctx| {
        ctx.solver.add_clause(literals);
    })
    .is_some()
}

/// Adds multiple clauses from a raw buffer (zero-copy friendly).
///
/// # Arguments
/// * `ctx_id` - The context handle.
/// * `buffer` - Raw bytes: each clause is preceded by its length (u32 LE),
///              followed by that many i64 literals (LE).
///
/// # Returns
/// Number of clauses added, or 0 if context not found or buffer invalid.
pub fn add_clauses_buffer(ctx_id: u32, buffer: &[u8]) -> usize {
    with_context(ctx_id, |ctx| {
        let mut count = 0;
        let mut offset = 0;
        
        while offset + 4 <= buffer.len() {
            // Read clause length
            let len_bytes: [u8; 4] = buffer[offset..offset + 4].try_into().unwrap();
            let clause_len = u32::from_le_bytes(len_bytes) as usize;
            offset += 4;
            
            // Read literals
            let lit_bytes_needed = clause_len * 8; // i64 = 8 bytes
            if offset + lit_bytes_needed > buffer.len() {
                break;
            }
            
            let mut literals = Vec::with_capacity(clause_len);
            for _ in 0..clause_len {
                let lit_bytes: [u8; 8] = buffer[offset..offset + 8].try_into().unwrap();
                literals.push(i64::from_le_bytes(lit_bytes));
                offset += 8;
            }
            
            ctx.solver.add_clause(literals);
            count += 1;
        }
        
        count
    })
    .unwrap_or(0)
}

/// Solves the problem in a context (synchronous).
pub fn solve(ctx_id: u32) -> Option<SatResult> {
    with_context(ctx_id, |ctx| ctx.solver.solve().ok())
        .flatten()
}

/// Submits a solve job asynchronously (non-blocking).
///
/// # Returns
/// A job ID for tracking, or 0 if context not found.
pub fn submit_solve(ctx_id: u32) -> u64 {
    // For now, we execute synchronously but return a job ID for API compatibility.
    // TODO: Implement actual async execution with worker threads.
    let result = solve(ctx_id);
    
    match result {
        Some(sat_result) => {
            let job_id = NEXT_JOB_ID.fetch_add(1, Ordering::SeqCst);
            let job = Job {
                ctx_id,
                status: JobStatus::Completed(sat_result),
            };
            JOBS.lock().unwrap().insert(job_id, job);
            job_id
        }
        None => 0,
    }
}

/// Polls the status of a job.
pub fn poll_job(job_id: u64) -> JobStatus {
    JOBS.lock()
        .unwrap()
        .get(&job_id)
        .map(|j| j.status.clone())
        .unwrap_or(JobStatus::NotFound)
}

/// Fetches completed jobs up to max_count.
///
/// # Returns
/// Vector of (job_id, result) pairs for completed jobs.
pub fn fetch_finished_jobs(max_count: usize) -> Vec<(u64, SatResult)> {
    let mut guard = JOBS.lock().unwrap();
    let mut results = Vec::new();
    let mut to_remove = Vec::new();
    
    for (&job_id, job) in guard.iter() {
        if results.len() >= max_count {
            break;
        }
        if let JobStatus::Completed(ref result) = job.status {
            results.push((job_id, result.clone()));
            to_remove.push(job_id);
        }
    }
    
    for job_id in to_remove {
        guard.remove(&job_id);
    }
    
    results
}

/// Solves with multiple assumption sets in batch.
///
/// # Arguments
/// * `ctx_id` - The context handle.
/// * `assumptions_buffer` - Raw bytes: each assumption set is preceded by its length (u32 LE),
///                          followed by that many i64 literals (LE).
///
/// # Returns
/// Vector of results for each assumption set.
pub fn solve_with_assumptions_batch(ctx_id: u32, assumptions_buffer: &[u8]) -> Vec<SatResult> {
    // Parse assumptions from buffer
    let mut assumption_sets = Vec::new();
    let mut offset = 0;
    
    while offset + 4 <= assumptions_buffer.len() {
        let len_bytes: [u8; 4] = assumptions_buffer[offset..offset + 4].try_into().unwrap();
        let set_len = u32::from_le_bytes(len_bytes) as usize;
        offset += 4;
        
        let lit_bytes_needed = set_len * 8;
        if offset + lit_bytes_needed > assumptions_buffer.len() {
            break;
        }
        
        let mut assumptions = Vec::with_capacity(set_len);
        for _ in 0..set_len {
            let lit_bytes: [u8; 8] = assumptions_buffer[offset..offset + 8].try_into().unwrap();
            assumptions.push(i64::from_le_bytes(lit_bytes));
            offset += 8;
        }
        
        assumption_sets.push(assumptions);
    }
    
    // Solve with each assumption set
    // TODO: Parallelize this with a thread pool
    let mut results = Vec::with_capacity(assumption_sets.len());
    
    for _assumptions in assumption_sets {
        // For now, just solve without assumptions (need to add assumptions support to solver)
        let result = solve(ctx_id).unwrap_or(SatResult::Unknown(String::new()));
        results.push(result);
    }
    
    results
}

/// Gets values for specific variables after a SAT result.
///
/// # Arguments
/// * `ctx_id` - The context handle.
/// * `var_ids` - Variable IDs (1-indexed) to retrieve.
///
/// # Returns
/// Vector of (var_id, value) pairs, or empty if context not found.
pub fn get_values(ctx_id: u32, var_ids: &[u32]) -> Vec<(u32, bool)> {
    // TODO: Implement model retrieval from solver
    // For now, return empty - need to add model storage to CdclSolver
    let _ = (ctx_id, var_ids);
    Vec::new()
}

/// Creates a boolean variable in a context.
pub fn new_bool_var(ctx_id: u32) -> Option<i64> {
    with_context(ctx_id, |ctx| {
        let var = ctx.solver.bool_var();
        var.id() as i64 + 1 // Convert to 1-indexed literal
    })
}

/// Forks a context into multiple clones.
pub fn fork_context(src_ctx_id: u32, num_clones: u32) -> Vec<u32> {
    let mut guard = CONTEXTS.lock().unwrap();
    
    let src = match guard.get(&src_ctx_id) {
        Some(ctx) => ctx,
        None => return Vec::new(),
    };
    
    // TODO: In future, support GPU-side cloning for real performance.
    let num_vars = src.solver.num_vars();
    
    let mut new_ids = Vec::with_capacity(num_clones as usize);
    for _ in 0..num_clones {
        let id = NEXT_CTX_ID.fetch_add(1, Ordering::SeqCst);
        let mut ctx = Context::new();
        for _ in 0..num_vars {
            ctx.solver.bool_var();
        }
        guard.insert(id, ctx);
        new_ids.push(id);
    }
    
    new_ids
}

/// Gets the number of currently active contexts.
pub fn context_count() -> usize {
    CONTEXTS.lock().unwrap().len()
}

/// Gets the number of pending/completed jobs.
pub fn job_count() -> usize {
    JOBS.lock().unwrap().len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_destroy_context() {
        let id = create_context();
        assert!(id > 0);
        assert!(destroy_context(id));
        assert!(!destroy_context(id));
    }

    #[test]
    fn test_add_clause_and_solve() {
        let id = create_context();
        
        let x = new_bool_var(id).unwrap();
        let y = new_bool_var(id).unwrap();
        
        add_clause(id, vec![x, y]);
        add_clause(id, vec![-x, y]);
        
        let result = solve(id);
        assert!(result.is_some());
        
        destroy_context(id);
    }

    #[test]
    fn test_add_clauses_buffer() {
        let id = create_context();
        
        let _x = new_bool_var(id).unwrap();
        let _y = new_bool_var(id).unwrap();
        
        // Build buffer: clause 1: [1, 2], clause 2: [-1, 2]
        let mut buffer = Vec::new();
        // Clause 1: length = 2
        buffer.extend_from_slice(&2u32.to_le_bytes());
        buffer.extend_from_slice(&1i64.to_le_bytes());
        buffer.extend_from_slice(&2i64.to_le_bytes());
        // Clause 2: length = 2
        buffer.extend_from_slice(&2u32.to_le_bytes());
        buffer.extend_from_slice(&(-1i64).to_le_bytes());
        buffer.extend_from_slice(&2i64.to_le_bytes());
        
        let count = add_clauses_buffer(id, &buffer);
        assert_eq!(count, 2);
        
        destroy_context(id);
    }

    #[test]
    fn test_async_job() {
        let id = create_context();
        
        let x = new_bool_var(id).unwrap();
        add_clause(id, vec![x]);
        
        let job_id = submit_solve(id);
        assert!(job_id > 0);
        
        let status = poll_job(job_id);
        assert!(matches!(status, JobStatus::Completed(_)));
        
        let finished = fetch_finished_jobs(10);
        assert_eq!(finished.len(), 1);
        
        destroy_context(id);
    }
}

