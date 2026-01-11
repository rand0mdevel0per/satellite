//! Clause database management.

use satellite_lockfree::LockFreeVec;

/// A clause stored in the database.
#[derive(Debug, Clone)]
pub struct StoredClause {
    /// Literals in the clause.
    pub literals: Vec<i64>,
    /// Whether this is a learned clause.
    pub learned: bool,
    /// Literal Block Distance (for learned clauses).
    pub lbd: Option<u32>,
    /// Activity/filter rate for deletion decisions.
    pub activity: f64,
}

/// Database for storing clauses.
pub struct ClauseDatabase {
    /// All clauses.
    clauses: Vec<StoredClause>,
    /// Number of original clauses.
    num_original: usize,
}

impl ClauseDatabase {
    /// Creates a new empty clause database.
    pub fn new() -> Self {
        Self {
            clauses: Vec::new(),
            num_original: 0,
        }
    }

    /// Adds an original clause.
    pub fn add_original(&mut self, literals: Vec<i64>) -> usize {
        let id = self.clauses.len();
        self.clauses.push(StoredClause {
            literals,
            learned: false,
            lbd: None,
            activity: 1.0,
        });
        id
    }

    /// Adds a learned clause.
    pub fn add_learned(&mut self, literals: Vec<i64>) -> usize {
        self.add_learned_with_lbd(literals, 0)
    }

    /// Adds a learned clause with LBD.
    pub fn add_learned_with_lbd(&mut self, literals: Vec<i64>, lbd: u32) -> usize {
        let id = self.clauses.len();
        self.clauses.push(StoredClause {
            literals,
            learned: true,
            lbd: Some(lbd),
            activity: 1.0,
        });
        id
    }

    /// Gets a clause by ID.
    pub fn get(&self, id: usize) -> Option<&StoredClause> {
        self.clauses.get(id)
    }

    /// Returns the number of clauses.
    pub fn len(&self) -> usize {
        self.clauses.len()
    }

    /// Returns whether the database is empty.
    pub fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }
}

impl Default for ClauseDatabase {
    fn default() -> Self {
        Self::new()
    }
}
