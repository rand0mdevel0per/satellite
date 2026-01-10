//! Constraint builders.

        // use std::iter::once;

/// A constraint that can be converted to CNF clauses.
pub enum Constraint {
    /// A single clause (disjunction).
    Clause(Vec<i64>),
    /// Multiple clauses (conjunction of disjunctions).
    Clauses(Vec<Vec<i64>>),
    /// At-most-k constraint.
    AtMostK { literals: Vec<i64>, k: usize },
    /// Exactly-k constraint.
    ExactlyK { literals: Vec<i64>, k: usize },
    /// XOR constraint.
    Xor(Vec<i64>),
}

impl Constraint {
    /// Creates a unit clause (single literal).
    pub fn unit(lit: i64) -> Self {
        Self::Clause(vec![lit])
    }

    /// Creates a binary clause.
    pub fn binary(a: i64, b: i64) -> Self {
        Self::Clause(vec![a, b])
    }

    /// Creates a ternary clause.
    pub fn ternary(a: i64, b: i64, c: i64) -> Self {
        Self::Clause(vec![a, b, c])
    }

    /// Creates an implication: a => b (equivalent to -a OR b).
    pub fn implies(a: i64, b: i64) -> Self {
        Self::Clause(vec![-a, b])
    }

    /// Creates a biconditional: a <=> b.
    pub fn iff(a: i64, b: i64) -> Self {
        Self::Clauses(vec![
            vec![-a, b],  // a => b
            vec![a, -b],  // b => a
        ])
    }

    /// Creates an at-most-k constraint.
    pub fn at_most_k(literals: Vec<i64>, k: usize) -> Self {
        Self::AtMostK { literals, k }
    }

    /// Creates an at-least-k constraint (negation of at-most-(k-1) on negated literals).
    pub fn at_least_k(literals: Vec<i64>, k: usize) -> Self {
        // at_least_k(lits, k) = at_most_k(neg(lits), n - k)
        let negated: Vec<i64> = literals.iter().map(|&l| -l).collect();
        let n = literals.len();
        Self::AtMostK { literals: negated, k: n - k }
    }

    /// Creates an exactly-k constraint.
    pub fn exactly_k(literals: Vec<i64>, k: usize) -> Self {
        Self::ExactlyK { literals, k }
    }

    /// Creates an XOR constraint.
    pub fn xor(literals: Vec<i64>) -> Self {
        Self::Xor(literals)
    }

    /// Converts the constraint to CNF clauses.
    pub fn to_clauses(&self) -> Vec<Vec<i64>> {
        match self {
            Self::Clause(c) => vec![c.clone()],
            Self::Clauses(cs) => cs.clone(),
            Self::AtMostK { literals, k } => {
                // Commander encoding for at-most-k
                // For simplicity, use naive encoding for small cases
                if literals.len() <= 10 {
                    self.at_most_k_naive(literals, *k)
                } else {
                    // TODO: Use more efficient encoding for large cases
                    self.at_most_k_naive(literals, *k)
                }
            }
            Self::ExactlyK { literals, k } => {
                let mut clauses = self.at_most_k_naive(literals, *k);
                // Add at-least-k clauses
                let negated: Vec<i64> = literals.iter().map(|&l| -l).collect();
                clauses.extend(self.at_most_k_naive(&negated, literals.len() - k));
                clauses
            }
            Self::Xor(literals) => {
                // XOR encoding: odd number of literals must be true
                self.xor_encoding(literals)
            }
        }
    }

    fn at_most_k_naive(&self, literals: &[i64], k: usize) -> Vec<Vec<i64>> {
        // Naive pairwise encoding: for each subset of size k+1, at least one must be false
        // use std::iter::once;

        let n = literals.len();
        if k >= n {
            return vec![]; // Always satisfied
        }

        let mut clauses = Vec::new();

        // Generate all combinations of size k+1
        fn combinations(arr: &[i64], k: usize, start: usize, current: &mut Vec<i64>, result: &mut Vec<Vec<i64>>) {
            if current.len() == k {
                // At least one must be false => disjunction of negations
                result.push(current.iter().map(|&l| -l).collect());
                return;
            }
            for i in start..arr.len() {
                current.push(arr[i]);
                combinations(arr, k, i + 1, current, result);
                current.pop();
            }
        }

        let mut current = Vec::new();
        combinations(literals, k + 1, 0, &mut current, &mut clauses);
        clauses
    }

    fn xor_encoding(&self, literals: &[i64]) -> Vec<Vec<i64>> {
        // XOR of n literals: encode as CNF using exponential expansion
        // (not practical for large n, should use Tseitin transformation)
        if literals.is_empty() {
            return vec![vec![]]; // Empty XOR is false
        }
        if literals.len() == 1 {
            return vec![vec![literals[0]]]; // Single literal XOR is just the literal
        }

        // For 2 literals: a XOR b = (a OR b) AND (NOT a OR NOT b)
        if literals.len() == 2 {
            return vec![
                vec![literals[0], literals[1]],
                vec![-literals[0], -literals[1]],
            ];
        }

        // For more literals, use recursive Tseitin-like encoding
        // TODO: Implement efficient XOR encoding
        vec![]
    }
}
