//! Decision heuristics (VSIDS, EVSIDS, LRB).

use satellite_base::types::VarId;

/// Weights for combining multiple heuristics.
#[derive(Debug, Clone)]
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

/// VSIDS (Variable State Independent Decaying Sum) scores.
pub struct VsidsScores {
    /// Activity score for each variable.
    scores: Vec<f64>,
    /// Decay factor.
    decay: f64,
    /// Increment value.
    increment: f64,
}

impl VsidsScores {
    /// Creates new VSIDS scores.
    pub fn new(num_vars: usize) -> Self {
        Self {
            scores: vec![0.0; num_vars],
            decay: 0.95,
            increment: 1.0,
        }
    }

    /// Bumps the activity of a variable.
    pub fn bump(&mut self, var: VarId) {
        self.scores[var as usize] += self.increment;

        // Rescale if too large
        if self.scores[var as usize] > 1e100 {
            for score in &mut self.scores {
                *score *= 1e-100;
            }
            self.increment *= 1e-100;
        }
    }

    /// Decays all activities.
    pub fn decay(&mut self) {
        self.increment /= self.decay;
    }

    /// Gets the score for a variable.
    pub fn score(&self, var: VarId) -> f64 {
        self.scores[var as usize]
    }
}

/// EVSIDS (Exponential VSIDS) scores.
pub struct EvsidsScores {
    /// Activity score for each variable.
    scores: Vec<f64>,
    /// Conflict counter.
    conflicts: u64,
}

impl EvsidsScores {
    /// Creates new EVSIDS scores.
    pub fn new(num_vars: usize) -> Self {
        Self {
            scores: vec![0.0; num_vars],
            conflicts: 0,
        }
    }

    /// Bumps the activity of a variable.
    pub fn bump(&mut self, var: VarId) {
        // Exponential scoring based on conflict number
        let exp_factor = (self.conflicts as f64 / 1000.0).exp();
        self.scores[var as usize] += exp_factor;
    }

    /// Records a conflict.
    pub fn on_conflict(&mut self) {
        self.conflicts += 1;
    }

    /// Gets the score for a variable.
    pub fn score(&self, var: VarId) -> f64 {
        self.scores[var as usize]
    }
}

/// LRB (Learning Rate Based) scores.
pub struct LrbScores {
    /// Assigned count for each variable.
    assigned: Vec<u64>,
    /// Participated count for each variable.
    participated: Vec<u64>,
    /// Step size.
    alpha: f64,
}

impl LrbScores {
    /// Creates new LRB scores.
    pub fn new(num_vars: usize) -> Self {
        Self {
            assigned: vec![0; num_vars],
            participated: vec![0; num_vars],
            alpha: 0.4,
        }
    }

    /// Records that a variable was assigned.
    pub fn on_assign(&mut self, var: VarId) {
        self.assigned[var as usize] += 1;
    }

    /// Records that a variable participated in a conflict.
    pub fn on_participate(&mut self, var: VarId) {
        self.participated[var as usize] += 1;
    }

    /// Gets the learning rate for a variable.
    pub fn learning_rate(&self, var: VarId) -> f64 {
        let assigned = self.assigned[var as usize] as f64;
        let participated = self.participated[var as usize] as f64;

        if assigned > 0.0 {
            participated / assigned
        } else {
            0.0
        }
    }
}

/// Combined heuristic scores.
pub struct CombinedHeuristics {
    vsids: VsidsScores,
    evsids: EvsidsScores,
    lrb: LrbScores,
    weights: HeuristicWeights,
}

impl CombinedHeuristics {
    /// Creates new combined heuristics.
    pub fn new(num_vars: usize, weights: HeuristicWeights) -> Self {
        Self {
            vsids: VsidsScores::new(num_vars),
            evsids: EvsidsScores::new(num_vars),
            lrb: LrbScores::new(num_vars),
            weights,
        }
    }

    /// Gets the combined score for a variable.
    pub fn score(&self, var: VarId) -> f64 {
        let v = self.vsids.score(var);
        let e = self.evsids.score(var);
        let l = self.lrb.learning_rate(var);

        self.weights.vsids * v + self.weights.evsids * e + self.weights.lrb * l
    }

    /// Bumps a variable in all heuristics.
    pub fn bump(&mut self, var: VarId) {
        self.vsids.bump(var);
        self.evsids.bump(var);
        self.lrb.on_participate(var);
    }

    /// Called after a conflict.
    pub fn on_conflict(&mut self) {
        self.vsids.decay();
        self.evsids.on_conflict();
    }

    /// Called when a variable is assigned.
    pub fn on_assign(&mut self, var: VarId) {
        self.lrb.on_assign(var);
    }
}
