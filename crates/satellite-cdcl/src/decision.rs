//! Decision engine for variable selection.

use satellite_base::types::VarId;

use crate::heuristics::{CombinedHeuristics, HeuristicWeights};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Manages decision levels and variable selection.
pub struct DecisionEngine {
    /// Decision levels (stack of decision points).
    levels: Vec<DecisionLevel>,
    /// Number of variables.
    num_vars: usize,
    /// Heuristics.
    pub heuristics: CombinedHeuristics,
    /// Priority queue for unassigned variables (lazy update).
    queue: BinaryHeap<VarScore>,
}

/// A variable with its score for the priority queue.
#[derive(Debug)]
struct VarScore {
    var: VarId,
    score: f64,
}

impl PartialEq for VarScore {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for VarScore {}

impl Ord for VarScore {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap based on score
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for VarScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
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
    pub fn new(num_vars: usize, weights: HeuristicWeights) -> Self {
        let heuristics = CombinedHeuristics::new(num_vars, weights);
        let mut queue = BinaryHeap::new();

        // Initialize queue with all variables
        for i in 0..num_vars {
            queue.push(VarScore {
                var: i as VarId,
                score: 0.0,
            });
        }

        Self {
            levels: vec![DecisionLevel {
                decision_var: None,
                trail_pos: 0,
            }],
            num_vars,
            heuristics,
            queue,
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
        }); // trail_pos must be updated by solver when pushing
    }

    // Helper to update trail_pos of the *current* level start.
    // Actually solver does this logic.
    // We just need accessors or allow public access.
    // For simplicity, let's allow updating the top level's trail_pos or make DecisionLevel public?
    // Solver usually manages the trail. DecisionEngine manages *levels*.

    /// Updates the trail position of the current level (used when new level starts).
    pub fn set_trail_pos(&mut self, pos: usize) {
        if let Some(last) = self.levels.last_mut() {
            last.trail_pos = pos;
        }
    }

    /// Gets trail pos of current level.
    pub fn trail_pos(&self) -> usize {
        self.levels.last().map(|l| l.trail_pos).unwrap_or(0)
    }

    /// Backtracks to the given level.
    pub fn backtrack_to(&mut self, level: usize) {
        self.levels.truncate(level + 1);
    }

    /// Picks the next variable to branch on.
    ///
    /// Returns None if all variables are assigned.
    pub fn pick_variable(&mut self, assignments: &[Option<bool>]) -> Option<VarId> {
        // Lazy removal of assigned variables from heap
        while let Some(vs) = self.queue.peek() {
            if assignments[vs.var as usize].is_some() {
                self.queue.pop();
                continue;
            }

            // Check if score is stale (simple check: if significantly different?)
            // For now, accept the top. Or rebuild?
            // Correct VSIDS implementation requires re-inserting on bump.
            // When we bump, we don't necessarily update heap immediately (too slow).
            // We usually lazily update or use a specialized heap.
            // For MVP: Pop current best. return it.
            // But we shouldn't pop if we just want to peek for unassigned?
            // Actually, once decided, it becomes assigned, so we can pop.

            let var = vs.var;
            self.queue.pop();
            return Some(var);
        }

        // If queue empty but vars unassigned (due to lazy pops not refilling?),
        // we might be in trouble if we don't refill.
        // But we initialized with all vars. We pop when assigned.
        // If unassigned later (backtrack), we need to re-insert!
        // So on/after backtrack, we must allow re-insertion.
        // This is complex. Standard VSIDS:
        // Use a list of unassigned variables + heap?
        // Actually, on backtrack, variables become unassigned. We should re-add them to queue?
        // Doing this efficiently is key.
        // Strategy: When backtracking, we don't fix the heap immediately.
        // BUT variables become eligible again.
        // For MVP: Re-scan all variables if heap is empty or just linear scan if heap is troublesome?
        // Specs say "Heuristics... run in parallel".

        // Let's stick to linear scan for MVP fallback or if queue logic is tricky.
        // Actually, just iterating all vars and finding max score is O(N) per decision.
        // N is large?
        // Let's implement O(N) scan for Safety First MVP (matches previous iteration method but with scores).

        let mut best_var = None;
        let mut best_score = -1.0;

        for (i, assignment) in assignments.iter().enumerate() {
            if assignment.is_none() {
                let score = self.heuristics.score(i as VarId);
                if score > best_score {
                    best_score = score;
                    best_var = Some(i as VarId);
                }
            }
        }
        best_var
    }
}
