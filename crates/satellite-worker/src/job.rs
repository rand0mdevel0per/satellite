//! Job definitions.

use satellite_branch::BranchId;
use std::sync::Arc;

/// A job to be executed by a worker.
pub struct Job {
    /// Job type.
    kind: JobKind,
    /// Branch this job belongs to.
    branch_id: BranchId,
    /// Branch failed checker.
    branch_checker: Option<Arc<dyn Fn(BranchId) -> bool + Send + Sync>>,
}

/// Types of jobs.
#[derive(Debug, Clone)]
pub enum JobKind {
    /// BCP check for a range of clauses.
    BcpCheck {
        /// Starting clause index.
        start: usize,
        /// Ending clause index (exclusive).
        end: usize,
    },
    /// Decision heuristic computation.
    DecisionHeuristic {
        /// Which heuristic to compute.
        heuristic: Heuristic,
    },
    /// Conflict analysis.
    ConflictAnalysis {
        /// Conflict clause ID.
        conflict_clause: usize,
    },
    /// Aggregation after all BCP checks complete.
    BcpAggregation,
    /// Shutdown signal.
    Shutdown,
}

/// Decision heuristic types.
#[derive(Debug, Clone, Copy)]
pub enum Heuristic {
    Vsids,
    Evsids,
    Lrb,
}

/// Result of job execution.
#[derive(Debug)]
pub struct JobResult {
    /// Branch ID.
    pub branch_id: BranchId,
    /// Result kind.
    pub kind: JobResultKind,
}

/// Types of job results.
#[derive(Debug)]
pub enum JobResultKind {
    /// BCP completed without conflict.
    BcpOk,
    /// BCP found a conflict.
    BcpConflict { clause_id: usize },
    /// Heuristic score computed.
    HeuristicScore {
        heuristic: Heuristic,
        var: u64,
        score: f64,
    },
    /// Aggregation complete.
    AggregationComplete { all_satisfied: bool },
    /// Learned clause from conflict analysis.
    LearnedClause { literals: Vec<i64>, lbd: u32 },
}

impl Job {
    /// Creates a BCP check job.
    pub fn bcp_check(branch_id: BranchId, start: usize, end: usize) -> Self {
        Self {
            kind: JobKind::BcpCheck { start, end },
            branch_id,
            branch_checker: None,
        }
    }

    /// Creates a decision heuristic job.
    pub fn decision_heuristic(branch_id: BranchId, heuristic: Heuristic) -> Self {
        Self {
            kind: JobKind::DecisionHeuristic { heuristic },
            branch_id,
            branch_checker: None,
        }
    }

    /// Creates a shutdown job.
    pub fn shutdown() -> Self {
        Self {
            kind: JobKind::Shutdown,
            branch_id: 0,
            branch_checker: None,
        }
    }

    /// Sets the branch checker.
    pub fn with_branch_checker<F>(mut self, checker: F) -> Self
    where
        F: Fn(BranchId) -> bool + Send + Sync + 'static,
    {
        self.branch_checker = Some(Arc::new(checker));
        self
    }

    /// Returns whether this is a shutdown job.
    pub fn is_shutdown(&self) -> bool {
        matches!(self.kind, JobKind::Shutdown)
    }

    /// Checks if the branch has failed.
    pub fn is_branch_failed(&self) -> bool {
        self.branch_checker
            .as_ref()
            .map(|f| f(self.branch_id))
            .unwrap_or(false)
    }

    /// Executes the job.
    pub fn execute(self) -> JobResult {
        // TODO: Implement actual job execution
        match self.kind {
            JobKind::BcpCheck { start, end } => JobResult {
                branch_id: self.branch_id,
                kind: JobResultKind::BcpOk,
            },
            JobKind::DecisionHeuristic { heuristic } => JobResult {
                branch_id: self.branch_id,
                kind: JobResultKind::HeuristicScore {
                    heuristic,
                    var: 0,
                    score: 0.0,
                },
            },
            JobKind::ConflictAnalysis { conflict_clause } => JobResult {
                branch_id: self.branch_id,
                kind: JobResultKind::LearnedClause {
                    literals: vec![],
                    lbd: 1,
                },
            },
            JobKind::BcpAggregation => JobResult {
                branch_id: self.branch_id,
                kind: JobResultKind::AggregationComplete {
                    all_satisfied: true,
                },
            },
            JobKind::Shutdown => JobResult {
                branch_id: 0,
                kind: JobResultKind::BcpOk,
            },
        }
    }
}
