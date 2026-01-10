//! Decision engine for variable selection.

use satellite_base::types::VarId;

/// Manages decision levels and variable selection.
pub struct DecisionEngine {
    /// Decision levels (stack of decision points).
    levels: Vec<DecisionLevel>,
    /// Number of variables.
    num_vars: usize,
}

/// A decision level.
struct DecisionLevel {
    /// The variable decided at this level.
    decision_var: Option<VarId>,
    /// Trail position at the start of this level.
    trail_pos: usize,
}

impl DecisionEngine {
    /// Creates a new decision engine.
    pub fn new(num_vars: usize) -> Self {
        Self {
            levels: vec![DecisionLevel {
                decision_var: None,
                trail_pos: 0,
            }],
            num_vars,
        }
    }

    /// Returns the current decision level.
    pub fn level(&self) -> usize {
        self.levels.len() - 1
    }

    /// Pushes a new decision level.
    pub fn push_level(&mut self) {
        self.levels.push(DecisionLevel {
            decision_var: None,
            trail_pos: 0,
        });
    }

    /// Backtracks to the given level.
    pub fn backtrack_to(&mut self, level: usize) {
        self.levels.truncate(level + 1);
    }

    /// Picks the next variable to branch on.
    ///
    /// Returns None if all variables are assigned.
    pub fn pick_variable(&self, assignments: &[Option<bool>]) -> Option<VarId> {
        // Simple strategy: first unassigned variable
        // TODO: Implement VSIDS/EVSIDS/LRB weighted selection
        for (i, assignment) in assignments.iter().enumerate() {
            if assignment.is_none() {
                return Some(i as VarId);
            }
        }
        None
    }
}
