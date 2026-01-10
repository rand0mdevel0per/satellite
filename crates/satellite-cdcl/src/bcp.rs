//! Boolean Constraint Propagation (BCP).
//!
//! Uses watched literals for efficient propagation.


/// Result of BCP.
#[derive(Debug)]
pub enum BcpResult {
    /// No conflict, propagation complete.
    Ok,
    /// Conflict detected in the given clause.
    Conflict(usize),
}

/// A watched literal entry.
#[derive(Debug, Clone)]
pub struct Watch {
    /// The clause being watched.
    pub clause_id: usize,
    /// The other watched literal in the clause.
    pub blocker: i64,
}

/// Watched literal data structure.
pub struct WatchedLiterals {
    /// For each literal, list of clauses watching it.
    watches: Vec<Vec<Watch>>,
}

impl WatchedLiterals {
    /// Creates a new watched literals structure.
    pub fn new(num_vars: usize) -> Self {
        // 2 * num_vars for positive and negative literals
        Self {
            watches: vec![Vec::new(); num_vars * 2],
        }
    }

    /// Converts a literal to an index.
    fn lit_to_index(lit: i64) -> usize {
        if lit > 0 {
            (lit as usize - 1) * 2
        } else {
            ((-lit) as usize - 1) * 2 + 1
        }
    }

    /// Adds a watch for a literal.
    pub fn add_watch(&mut self, lit: i64, clause_id: usize, blocker: i64) {
        let idx = Self::lit_to_index(lit);
        self.watches[idx].push(Watch { clause_id, blocker });
    }

    /// Gets watches for a literal.
    pub fn get_watches(&self, lit: i64) -> &[Watch] {
        let idx = Self::lit_to_index(lit);
        &self.watches[idx]
    }

    /// Gets mutable watches for a literal.
    pub fn get_watches_mut(&mut self, lit: i64) -> &mut Vec<Watch> {
        let idx = Self::lit_to_index(lit);
        &mut self.watches[idx]
    }
}

/// Propagation queue.
pub struct PropagationQueue {
    queue: Vec<i64>,
    head: usize,
}

impl PropagationQueue {
    /// Creates a new propagation queue.
    pub fn new() -> Self {
        Self {
            queue: Vec::new(),
            head: 0,
        }
    }

    /// Enqueues a literal for propagation.
    pub fn enqueue(&mut self, lit: i64) {
        self.queue.push(lit);
    }

    /// Dequeues the next literal.
    pub fn dequeue(&mut self) -> Option<i64> {
        if self.head < self.queue.len() {
            let lit = self.queue[self.head];
            self.head += 1;
            Some(lit)
        } else {
            None
        }
    }

    /// Returns whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.head >= self.queue.len()
    }

    /// Clears the queue.
    pub fn clear(&mut self) {
        self.queue.clear();
        self.head = 0;
    }
}

impl Default for PropagationQueue {
    fn default() -> Self {
        Self::new()
    }
}
