//! Core CDCL solver implementation.

use satellite_base::{Result, types::VarId};
use satellite_format::AdvancedCnf;
use crate::clause_db::ClauseDatabase;
use crate::decision::DecisionEngine;
use crate::heuristics::HeuristicWeights;

/// The result of a SAT solve.
#[derive(Debug, Clone)]
pub enum SatResult {
    /// Satisfiable with a model (variable assignments).
    Sat(Vec<bool>),
    /// Unsatisfiable.
    Unsat,
    /// Unknown (timeout or resource limit reached).
    Unknown(String),
}

/// Configuration for the CDCL solver.
#[derive(Debug, Clone)]
pub struct CdclConfig {
    /// Heuristic weights for decision making.
    pub heuristic_weights: HeuristicWeights,
    /// Restart interval (conflicts before restart).
    pub restart_interval: u64,
    /// Clause deletion threshold (LBD).
    pub deletion_lbd_threshold: u32,
}

impl Default for CdclConfig {
    fn default() -> Self {
        Self {
            heuristic_weights: HeuristicWeights::default(),
            restart_interval: 100,
            deletion_lbd_threshold: 6,
        }
    }
}

/// The main CDCL solver.
pub struct CdclSolver {
    /// Number of variables.
    num_vars: usize,
    /// Clause database.
    clauses: ClauseDatabase,
    /// Current variable assignments (None = unassigned).
    assignments: Vec<Option<bool>>,
    /// Decision engine.
    decision: DecisionEngine,
    /// Configuration.
    config: CdclConfig,
    /// Statistics.
    stats: SolverStats,
}

/// Solver statistics.
#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    /// Number of decisions made.
    pub decisions: u64,
    /// Number of conflicts encountered.
    pub conflicts: u64,
    /// Number of propagations.
    pub propagations: u64,
    /// Number of restarts.
    pub restarts: u64,
    /// Number of learned clauses.
    pub learned_clauses: u64,
}

impl CdclSolver {
    /// Creates a new solver from an Advanced-CNF problem.
    pub fn new(problem: &AdvancedCnf) -> Self {
        let num_vars = problem.variables.len();

        Self {
            num_vars,
            clauses: ClauseDatabase::new(),
            assignments: vec![None; num_vars],
            decision: DecisionEngine::new(num_vars),
            config: CdclConfig::default(),
            stats: SolverStats::default(),
        }
    }

    /// Creates a new solver with custom configuration.
    pub fn with_config(problem: &AdvancedCnf, config: CdclConfig) -> Self {
        let mut solver = Self::new(problem);
        solver.config = config;
        solver
    }

    /// Solves the problem.
    pub fn solve(&mut self) -> Result<SatResult> {
        // Main CDCL loop
        loop {
            // Boolean Constraint Propagation
            if let Some(conflict_clause) = self.propagate() {
                self.stats.conflicts += 1;

                // At decision level 0, problem is UNSAT
                if self.decision_level() == 0 {
                    return Ok(SatResult::Unsat);
                }

                // Conflict analysis and backtrack
                let (learned_clause, backtrack_level) = self.analyze_conflict(conflict_clause);
                self.backtrack(backtrack_level);
                self.add_learned_clause(learned_clause);

                // Check for restart
                if self.should_restart() {
                    self.restart();
                }
            } else {
                // No conflict - try to make a decision
                if let Some(var) = self.pick_branch_variable() {
                    self.stats.decisions += 1;
                    self.decide(var, true); // Default to true
                } else {
                    // All variables assigned - SAT!
                    let model = self.extract_model();
                    return Ok(SatResult::Sat(model));
                }
            }
        }
    }

    /// Returns the current decision level.
    fn decision_level(&self) -> usize {
        self.decision.level()
    }

    /// Performs BCP and returns a conflict clause if one is found.
    fn propagate(&mut self) -> Option<usize> {
        // TODO: Implement full BCP with watched literals
        self.stats.propagations += 1;
        None
    }

    /// Analyzes a conflict and returns (learned clause, backtrack level).
    fn analyze_conflict(&mut self, _conflict: usize) -> (Vec<i64>, usize) {
        // TODO: Implement 1-UIP conflict analysis
        self.stats.learned_clauses += 1;
        (vec![], 0)
    }

    /// Backtracks to the given level.
    fn backtrack(&mut self, level: usize) {
        self.decision.backtrack_to(level);
        // TODO: Undo assignments
    }

    /// Adds a learned clause.
    fn add_learned_clause(&mut self, clause: Vec<i64>) {
        if !clause.is_empty() {
            self.clauses.add_learned(clause);
        }
    }

    /// Checks if we should restart.
    fn should_restart(&self) -> bool {
        self.stats.conflicts % self.config.restart_interval == 0
    }

    /// Performs a restart.
    fn restart(&mut self) {
        self.stats.restarts += 1;
        self.backtrack(0);
    }

    /// Picks the next unassigned variable to branch on.
    fn pick_branch_variable(&self) -> Option<VarId> {
        self.decision.pick_variable(&self.assignments)
    }

    /// Makes a decision.
    fn decide(&mut self, var: VarId, value: bool) {
        self.decision.push_level();
        self.assignments[var as usize] = Some(value);
    }

    /// Extracts the model (variable assignments).
    fn extract_model(&self) -> Vec<bool> {
        self.assignments
            .iter()
            .map(|a| a.unwrap_or(false))
            .collect()
    }

    /// Returns solver statistics.
    pub fn stats(&self) -> &SolverStats {
        &self.stats
    }
}
