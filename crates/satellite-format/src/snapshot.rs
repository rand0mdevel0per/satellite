//! Solver state snapshots for reproducibility.

use serde::{Deserialize, Serialize};

/// A snapshot of solver state for reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// SHA256 hash of the problem.
    pub problem_hash: String,
    /// Branch tree structure.
    pub branch_tree: BranchTree,
    /// Learned clauses.
    pub learned_clauses: Vec<Vec<i64>>,
    /// PRNG state.
    pub prng_state: PrngState,
    /// Solver configuration.
    pub solver_config: SolverConfig,
}

/// Branch tree structure in snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchTree {
    /// All branches.
    pub branches: Vec<BranchInfo>,
}

/// Information about a single branch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchInfo {
    /// Branch ID.
    pub id: String,
    /// Parent branch ID (None for root).
    pub parent_id: Option<String>,
    /// Depth in the tree.
    pub depth: u32,
    /// Current status.
    pub status: BranchStatus,
}

/// Branch status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BranchStatus {
    Active,
    Failed,
    Solved,
}

/// PRNG state for reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrngState {
    /// XorShift64 state.
    pub xorshift_state: u64,
}

/// Solver configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// Number of CPU workers.
    pub cpu_workers: usize,
    /// Number of GPU workers (warps).
    pub gpu_workers: usize,
    /// Decision heuristic weights.
    pub heuristic_weights: HeuristicWeights,
}

/// Weights for decision heuristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeuristicWeights {
    /// VSIDS weight.
    pub vsids: f64,
    /// EVSIDS weight.
    pub evsids: f64,
    /// LRB weight.
    pub lrb: f64,
}

impl Default for HeuristicWeights {
    fn default() -> Self {
        Self {
            vsids: 0.4,
            evsids: 0.3,
            lrb: 0.3,
        }
    }
}

impl Snapshot {
    /// Serializes to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserializes from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}
