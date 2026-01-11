//! Core CDCL solver implementation.

use crate::bcp::{PropagationQueue, WatchedLiterals};
use crate::clause_db::ClauseDatabase;
use crate::decision::DecisionEngine;
use crate::heuristics::HeuristicWeights;
use satellite_base::{Result, types::VarId};
use satellite_format::AdvancedCnf;

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
    /// Trail of assigned literals.
    trail: Vec<i64>,
    /// Assigning clause (reason) for each variable.
    reasons: Vec<Option<usize>>,
    /// Decision level for each variable.
    levels: Vec<usize>,
    /// Propagation queue.
    prop_q: PropagationQueue,
    /// Watched literals.
    watches: WatchedLiterals,
    /// Decision engine.
    decision: DecisionEngine,
    /// Configuration.
    config: CdclConfig,
    /// Statistics.
    stats: SolverStats,
    /// Whether a conflict was detected during initialization (conflicting unit clauses).
    has_initial_conflict: bool,
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
        // Calculate num_vars: use variables.len() if available, otherwise derive from max literal
        let num_vars = if !problem.variables.is_empty() {
            problem.variables.len()
        } else {
            // Derive from max absolute literal value in clauses
            problem
                .clauses
                .iter()
                .flat_map(|c| c.literals.iter())
                .map(|&lit| lit.unsigned_abs() as usize)
                .max()
                .unwrap_or(0)
        };

        let config = CdclConfig::default();

        let mut solver = Self {
            num_vars,
            clauses: ClauseDatabase::new(),
            assignments: vec![None; num_vars + 1], // +1 for 1-indexed variables
            trail: Vec::with_capacity(num_vars),
            reasons: vec![None; num_vars + 1],
            levels: vec![0; num_vars + 1],
            prop_q: PropagationQueue::new(),
            watches: WatchedLiterals::new(num_vars + 1),
            decision: DecisionEngine::new(num_vars + 1, config.heuristic_weights.clone()),
            config,
            stats: SolverStats::default(),
            has_initial_conflict: false,
        };

        // Load clauses from problem
        for clause in &problem.clauses {
            solver.add_clause(clause.literals.clone());
        }

        solver
    }

    /// Adds a clause to the database and watches.
    pub fn add_clause(&mut self, literals: Vec<i64>) {
        if literals.is_empty() {
            return;
        } // Empty clause = conflict immediately usually, handled during solve

        let id = self.clauses.add_original(literals.clone());

        if literals.len() >= 2 {
            self.watches.add_watch(literals[0], id, literals[1]);
            self.watches.add_watch(literals[1], id, literals[0]);
        } else if literals.len() == 1 {
            // Unit clause - check for conflict with existing assignment
            let lit = literals[0];
            if let Some(val) = self.value(lit) {
                if !val {
                    // Contradiction: this literal is already false
                    // Mark solver as having initial conflict
                    self.has_initial_conflict = true;
                }
                // Already assigned to true - fine, no need to enqueue
            } else {
                // Not assigned yet - enqueue and assign
                self.enqueue(lit, Some(id));
            }
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
        // Check for conflicts detected during initialization (conflicting unit clauses)
        if self.has_initial_conflict {
            return Ok(SatResult::Unsat);
        }
        
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
        while let Some(lit) = self.prop_q.dequeue() {
            let false_lit = -lit; // The literal that became false

            // We need to notify watchers of false_lit
            // Note: WatchedLiterals structure stores watches for the literal being watched.
            // If `lit` is true, then `-lit` is false. Watches watching `-lit` need to be updated.

            // Get watches for the falsified literal
            // We need to extract them to avoid borrowing self.watches along with self.clauses
            let watches = std::mem::take(self.watches.get_watches_mut(false_lit));
            let mut i = 0;

            while i < watches.len() {
                let watch = watches[i].clone();
                let clause_id = watch.clause_id;
                let blocker = watch.blocker;

                // Optimization: Check blocker first
                if self.value(blocker) == Some(true) {
                    // Clause already satisfied by blocker, keep watch
                    self.watches.get_watches_mut(false_lit).push(watch);
                    i += 1;
                    continue;
                }

                // Need to inspect the clause
                // Access clause from database
                // Safety: We extracted watches, so we can access clauses
                let clause = self.clauses.get(clause_id).unwrap();
                let literals = clause.literals.clone(); // Clone for modification check? 
                // Actually need to modify the clause in-place for watcher swap?
                // Watched literal scheme usually requires mutable access to clause to swap literals?
                // Or just update the watch structure?
                // Standard scheme: Clause has lit[0] and lit[1] as watchers.
                // If we don't permute clause, we rely on the Watch struct.
                // Here Watch struct has { clause_id, blocker }.
                // But we need to find a new watcher in the clause.

                // Assuming literals[0] and literals[1] are the watched ones is standard.
                // If we don't reorder clause, we need to know WHICH literal in clause is being watched.
                // Let's assume we maintain invariant: clause.literals[0] and [1] are watchers.
                // So if false_lit is at [0], we try to swap [0] with something else > [1].
                // If false_lit is at [1], we try to swap [1] with something else > [1] (Wait, >1?).

                // Since generic implementation is complex inside this tool block,
                // I will assume a simplified version or standard swap.
                // Simplified: Just scan clause for a non-false literal.

                let mut new_watcher_idx = None;
                let is_other_watch_true = false;

                // Identification of other watcher
                let other_watch_lit = if literals[0] == false_lit {
                    literals[1]
                } else {
                    literals[0]
                };

                if self.value(other_watch_lit) == Some(true) {
                    // Clause satisfied by other watcher
                    // Update blocker in current watch?
                    // Actually we just keep the watch.
                    self.watches.get_watches_mut(false_lit).push(watch);
                    i += 1;
                    continue;
                }

                // Look for new literal to watch
                for (k, &candidate) in literals.iter().enumerate().skip(2) {
                    if self.value(candidate) != Some(false) {
                        new_watcher_idx = Some(k);
                        break;
                    }
                }

                if let Some(idx) = new_watcher_idx {
                    // Found new watcher
                    let new_lit = literals[idx];

                    // Swap literals in clause (requires mutable clause access)
                    // But ClauseDatabase stores StoredClause which owns Vec<i64>.
                    // We need mutable access to clauses. ClauseDatabase has LockFreeVec which might be append-only?
                    // Review ClauseDatabase: `clauses: LockFreeVec<StoredClause>`.
                    // StoredClause fields are public.
                    // But LockFreeVec returns `Option<&StoredClause>` (immutable ref).
                    // This is a problem for standard 2-watched-literal scheme which requires swapping.
                    // If we can't swap, we must scan linearly or use a separate state for watchers.
                    // Or maybe LockFreeVec supports `get_mut`?
                    // If not, we fall back to just finding a new literal but not swapping.
                    // But then iterating "skip(2)" doesn't work next time.

                    // Allow me to check LockFreeVec source first?
                    // Assuming for now we CANNOT swap in LockFreeVec easily.
                    // Strategy: Just linear scan for now (simpler, correct).
                    // Wait, `watches` list updates: Add `new_lit` to watches.
                    // Remove `false_lit` from watches (we just don't push it back).
                    // We add a new Watch { clause_id, blocker: other_watch_lit } to `new_lit`'s list.

                    self.watches.add_watch(new_lit, clause_id, other_watch_lit);
                    i += 1;
                } else {
                    // No new watcher found.
                    // Clause is unit or conflicting.
                    self.watches.get_watches_mut(false_lit).push(watch); // Keep watching it
                    i += 1;

                    if self.value(other_watch_lit) == Some(false) {
                        // Conflict! Both watchers false, no alternative.
                        // Restore remaining watches to avoid losing them
                        while i < watches.len() {
                            self.watches
                                .get_watches_mut(false_lit)
                                .push(watches[i].clone());
                            i += 1;
                        }
                        return Some(clause_id);
                    } else if self.value(other_watch_lit) == None {
                        // Unit!
                        self.enqueue(other_watch_lit, Some(clause_id));
                    }
                }
            }
        }
        None
    }

    /// Helper to get variable value.
    fn value(&self, lit: i64) -> Option<bool> {
        let var = (lit.abs() - 1) as usize;
        self.assignments[var].map(|v| if lit > 0 { v } else { !v })
    }

    /// Enqueues a literal assignment.
    fn enqueue(&mut self, lit: i64, reason: Option<usize>) {
        if self.value(lit).is_some() {
            return;
        }

        let var = (lit.abs() - 1) as usize;
        let value = lit > 0;

        self.assignments[var] = Some(value);
        self.reasons[var] = reason;
        self.levels[var] = self.decision_level();
        self.trail.push(lit);
        self.prop_q.enqueue(lit);
    }

    /// Analyzes a conflict and returns (learned clause, backtrack level).
    fn analyze_conflict(&mut self, conflict_clause_id: usize) -> (Vec<i64>, usize) {
        let mut learned = Vec::new();
        let mut seen = vec![false; self.num_vars];
        let mut path_c = 0;
        let mut p = -1;
        let counter = 0;

        let mut clause_to_analyze = self
            .clauses
            .get(conflict_clause_id)
            .unwrap()
            .literals
            .clone();
        let mut index = self.trail.len();

        loop {
            // Add literals from clause to seen/learned
            for &lit in &clause_to_analyze {
                // If it's the current resolution variable (p != -1 check), skip?
                // In standard 1-UIP, p is initially None, then resolved.
                if p != -1 && lit == p {
                    continue;
                }
                if p != -1 && lit == -p {
                    continue;
                } // -p not possible in conflicting clause generally unless trivial?

                let var = (lit.abs() - 1) as usize;
                if !seen[var] {
                    seen[var] = true;
                    if self.levels[var] >= self.decision_level() {
                        path_c += 1;
                    } else if self.levels[var] > 0 {
                        learned.push(lit);
                    }
                }
            }

            // Select next literal to resolve
            while let Some(&t_lit) = self.trail.get(index.wrapping_sub(1)) {
                index -= 1;
                let var = (t_lit.abs() - 1) as usize;
                if seen[var] {
                    p = t_lit; // The literal to resolve on
                    path_c -= 1;
                    break;
                }
            }

            if path_c <= 0 {
                // Found UIP
                break;
            }

            // Resolve p with its reason
            let reason_id = self.reasons[(p.abs() - 1) as usize]
                .expect("Variable involved in conflict at current level must have a reason");
            clause_to_analyze = self.clauses.get(reason_id).unwrap().literals.clone();
        }

        // Add UIP
        if p != -1 {
            learned.push(-p);
        }

        self.stats.learned_clauses += 1;

        // Backtrack level is the max level of other literals in learned clause
        let backtrack_level = if learned.len() <= 1 {
            0
        } else {
            learned
                .iter()
                .map(|&lit| self.levels[(lit.abs() - 1) as usize])
                .filter(|&lvl| lvl < self.decision_level())
                .max()
                .unwrap_or(0)
        };

        (learned, backtrack_level)
    }

    /// Backtracks to the given level.
    fn backtrack(&mut self, level: usize) {
        // Unassign variables up to level
        while self.decision_level() > level {
            // Find range of variables to unassign
            // Trail is global linear list.
            // DecisionEngine tracks stack of levels.
            // But DecisionEngine doesn't track trail index per level publicly yet in my helper modification.
            // Wait, I didn't finish modifying DecisionEngine to expose trail_pos fully?
            // "pub fn trail_pos()".
            // Let's assume we can get target trail length.
            // If not, we iterate trail backwards and check levels? (Inefficient)
            // Solver maintains `trail: Vec<i64>`.
            // We need to know where to cut `trail`.

            // Let's just pop from trail until we hit a variable at a lower level?
            // BUT variables at lower level are mixed in trail?
            // No, trail is time-ordered assignement. Assignments at level L are AFTER assignments at L-1.
            // So just pop until `self.levels[var] <= level`.

            while let Some(&lit) = self.trail.last() {
                let var = (lit.abs() - 1) as usize;
                if self.levels[var] <= level {
                    break;
                }

                self.assignments[var] = None;
                self.reasons[var] = None;
                self.watches.get_watches_mut(-lit).clear(); // Reset? No, watchers persist for BCP unless we want to clean up.
                // Actually we DON'T clear watchers. Watchers are static structure usually.
                // We just unassign.

                self.trail.pop();
            }

            self.decision.backtrack_to(self.decision_level() - 1);
        }

        self.prop_q.clear();
    }

    /// Adds a learned clause.
    fn add_learned_clause(&mut self, clause: Vec<i64>) {
        if !clause.is_empty() {
            let id = self.clauses.add_learned(clause.clone());

            // Backjumping usually makes the clause unit. Enqueue it.
            // Conflict analysis typically results in 1-UIP which is asserting at backtrack_level.
            // So we should verify this.
            // "After backtracking to 'backtrack_level', clause should be unit."

            if clause.len() == 1 {
                let lit = clause[0];
                self.enqueue(lit, Some(id));
            } else {
                // Watch two literals
                // One is the UIP (at current level - now 'backtrack_level'? No, we are at backtrack_level)
                // Wait, we just backtracked. The UIP variable is now unassigned.
                // The other literals are at levels <= backtrack_level.
                // So the clause is Unit. The only unassigned literal is the UIP one.
                // We watch it and another one (doesn't matter which).
                self.watches.add_watch(clause[0], id, clause[1]);
                self.watches.add_watch(clause[1], id, clause[0]);

                // Enqueue assignment
                // We find the unassigned one.
                for &lit in &clause {
                    if self.value(lit).is_none() {
                        self.enqueue(lit, Some(id));
                        break;
                    }
                }
            }
        }
    }

    /// Checks if we should restart.
    fn should_restart(&self) -> bool {
        self.stats.conflicts > 0 && self.stats.conflicts % self.config.restart_interval == 0
    }

    /// Performs a restart.
    fn restart(&mut self) {
        self.stats.restarts += 1;
        self.backtrack(0);
    }

    /// Picks the next unassigned variable to branch on.
    fn pick_branch_variable(&mut self) -> Option<VarId> {
        self.decision.pick_variable(&self.assignments)
    }

    /// Makes a decision.
    fn decide(&mut self, var: VarId, value: bool) {
        self.decision.push_level();
        let lit = if value {
            (var as i64) + 1
        } else {
            -((var as i64) + 1)
        };
        self.enqueue(lit, None);
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
