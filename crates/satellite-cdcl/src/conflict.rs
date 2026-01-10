//! Conflict analysis with 1-UIP resolution.

use satellite_base::types::VarId;

/// Implication graph for conflict analysis.
pub struct ImplicationGraph {
    /// For each variable, the reason clause that implied it.
    reasons: Vec<Option<usize>>,
    /// For each variable, the decision level it was assigned at.
    levels: Vec<usize>,
    /// Trail of assignments.
    trail: Vec<VarId>,
}

impl ImplicationGraph {
    /// Creates a new implication graph.
    pub fn new(num_vars: usize) -> Self {
        Self {
            reasons: vec![None; num_vars],
            levels: vec![0; num_vars],
            trail: Vec::new(),
        }
    }

    /// Records an assignment.
    pub fn assign(&mut self, var: VarId, level: usize, reason: Option<usize>) {
        self.reasons[var as usize] = reason;
        self.levels[var as usize] = level;
        self.trail.push(var);
    }

    /// Gets the reason for an assignment.
    pub fn reason(&self, var: VarId) -> Option<usize> {
        self.reasons[var as usize]
    }

    /// Gets the level of an assignment.
    pub fn level(&self, var: VarId) -> usize {
        self.levels[var as usize]
    }

    /// Backtracks to the given level.
    pub fn backtrack(&mut self, level: usize) {
        while let Some(&var) = self.trail.last() {
            if self.levels[var as usize] <= level {
                break;
            }
            self.trail.pop();
            self.reasons[var as usize] = None;
        }
    }
}

/// Conflict analyzer.
pub struct ConflictAnalyzer {
    /// Temporary buffer for resolution.
    learnt: Vec<i64>,
    /// Seen flags for variables.
    seen: Vec<bool>,
}

impl ConflictAnalyzer {
    /// Creates a new conflict analyzer.
    pub fn new(num_vars: usize) -> Self {
        Self {
            learnt: Vec::new(),
            seen: vec![false; num_vars],
        }
    }

    /// Analyzes a conflict and returns (learned clause, backtrack level, LBD).
    pub fn analyze(
        &mut self,
        _conflict_clause: &[i64],
        _graph: &ImplicationGraph,
        _current_level: usize,
    ) -> (Vec<i64>, usize, u32) {
        // TODO: Implement 1-UIP analysis
        // 1. Start with conflict clause
        // 2. Resolve with reason clauses until 1 literal at current level
        // 3. Compute LBD (number of distinct decision levels)
        // 4. Return learned clause and backtrack level

        self.learnt.clear();
        (self.learnt.clone(), 0, 1)
    }

    /// Computes the Literal Block Distance (LBD) of a clause.
    pub fn compute_lbd(&self, clause: &[i64], graph: &ImplicationGraph) -> u32 {
        let mut levels = std::collections::HashSet::new();
        for &lit in clause {
            let var = lit.unsigned_abs() as VarId - 1;
            levels.insert(graph.level(var));
        }
        levels.len() as u32
    }
}
