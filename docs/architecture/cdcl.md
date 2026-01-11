# CDCL Solver Implementation

This document describes the Conflict-Driven Clause Learning (CDCL) solver implementation in Satellite.

## Overview

The CDCL solver in `satellite-cdcl` implements the core SAT solving algorithm with the following features:

- **Two-Watched Literals** for efficient unit propagation
- **VSIDS/EVSIDS Heuristics** for variable selection
- **1-UIP Conflict Analysis** (structure present, basic backtracking active)
- **Clause Learning** with LBD-based management
- **UNSAT Core Extraction** for advanced analysis
- **Restart Strategy** with geometric increase

## Core Data Structures

### Clause Database

```rust
pub struct ClauseDatabase {
    clauses: Vec<Clause>,      // All clauses
    num_original: usize,       // Original vs learned separation
}

pub struct Clause {
    literals: Vec<i64>,        // Literals (1-indexed, signed)
    lbd: Option<usize>,        // Literal Block Distance (for learned)
    is_learned: bool,
}
```

### Watched Literals

Each literal maintains a watch list:

```rust
pub struct WatchEntry {
    clause_id: usize,          // Which clause
    blocker: i64,              // Other watched literal (cache)
}

pub struct WatchedLiterals {
    pos_watches: Vec<Vec<WatchEntry>>,  // For positive literals
    neg_watches: Vec<Vec<WatchEntry>>,  // For negative literals
}
```

### Trail and Reasons

```rust
struct CdclSolver {
    trail: Vec<i64>,                    // Assigned literals in order
    reasons: Vec<Option<usize>>,        // Clause that implied each var
    levels: Vec<usize>,                 // Decision level of each var
    assignments: Vec<Option<bool>>,     // Current assignment
}
```

## Algorithm Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         Main Loop                                │
│                                                                  │
│    ┌──────────────┐                                             │
│    │   Propagate  │ ───────────────────────────────┐            │
│    │   (BCP)      │                                │            │
│    └──────┬───────┘                                │            │
│           │                                        │            │
│     Conflict?                                      │            │
│      │     │                                       │            │
│     Yes    No                                      │            │
│      │     │                                       │            │
│      │     ▼                                       │            │
│      │  ┌─────────────────┐                        │            │
│      │  │ All vars        │ ──► SAT! Return model │            │
│      │  │ assigned?       │                        │            │
│      │  └────────┬────────┘                        │            │
│      │           │ No                              │            │
│      │           ▼                                 │            │
│      │  ┌─────────────────┐                        │            │
│      │  │ Pick variable   │                        │            │
│      │  │ (VSIDS/EVSIDS)  │                        │            │
│      │  └────────┬────────┘                        │            │
│      │           │                                 │            │
│      │           ▼                                 │            │
│      │  ┌─────────────────┐                        │            │
│      │  │ Decide (enqueue)│ ───────────────────────┘            │
│      │  └─────────────────┘                                     │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────────────────┐                                       │
│  │ Conflict at level 0? │                                       │
│  └──────┬───────────────┘                                       │
│         │                                                        │
│        Yes ──────► UNSAT! Return                                 │
│         │                                                        │
│        No                                                        │
│         ▼                                                        │
│  ┌──────────────────────┐                                       │
│  │ Analyze Conflict     │                                       │
│  │ (1-UIP)              │                                       │
│  └──────────┬───────────┘                                       │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────┐                                       │
│  │ Learn Clause         │                                       │
│  │ Backtrack            │                                       │
│  └──────────┬───────────┘                                       │
│             │                                                    │
│             └────────────────────────────────────────────────► Loop
└──────────────────────────────────────────────────────────────────┘
```

## Boolean Constraint Propagation (BCP)

The `propagate()` function implements efficient BCP using two-watched literals:

```rust
fn propagate(&mut self) -> Option<usize> {
    while let Some(lit) = self.prop_q.dequeue() {
        let false_lit = -lit;  // This literal is now false
        
        // Get watches for the falsified literal
        let watches = std::mem::take(self.watches.get_watches_mut(false_lit));
        
        for watch in watches {
            let clause = &self.clauses.get(watch.clause_id);
            
            // Check blocker (cache optimization)
            if self.value(watch.blocker) == Some(true) {
                self.watches.get_watches_mut(false_lit).push(watch);
                continue;
            }
            
            // Find new watcher or detect unit/conflict
            if let Some(new_lit) = self.find_new_watch(clause, false_lit) {
                self.watches.add_watch(new_lit, watch.clause_id, other_watch);
            } else {
                // Unit or conflict
                if self.is_conflict(clause) {
                    return Some(watch.clause_id);
                }
                self.enqueue(unit_lit, Some(watch.clause_id));
            }
        }
    }
    None
}
```

## Decision Heuristics

### VSIDS (Variable State Independent Decaying Sum)

```rust
pub struct VsidsScores {
    scores: Vec<f64>,
    bump_amount: f64,
    decay_factor: f64,  // 0.95 typically
}

impl VsidsScores {
    pub fn bump(&mut self, var: VarId) {
        self.scores[var] += self.bump_amount;
    }
    
    pub fn decay(&mut self) {
        self.bump_amount /= self.decay_factor;
    }
}
```

### EVSIDS (Extended VSIDS)

Uses exponential moving average with conflict-based updates.

## UNSAT Core Extraction

When enabled, the solver tracks which clauses contribute to conflicts:

```rust
pub fn enable_unsat_core(&mut self) {
    self.track_unsat_core = true;
    self.used_clauses.clear();
}

// During conflict:
fn record_used_clause(&mut self, clause_id: usize) {
    if self.track_unsat_core {
        self.used_clauses.push(clause_id);
    }
}

pub fn get_unsat_core(&self) -> Option<Vec<usize>> {
    if !self.track_unsat_core || self.used_clauses.is_empty() {
        return None;
    }
    let mut core = self.used_clauses.clone();
    core.sort_unstable();
    core.dedup();
    Some(core)
}
```

## Statistics

The solver tracks various statistics:

```rust
pub struct SolverStats {
    pub decisions: u64,        // Number of decision variable picks
    pub conflicts: u64,        // Number of conflicts encountered  
    pub propagations: u64,     // Number of BCP propagations
    pub restarts: u64,         // Number of restarts
    pub learned_clauses: u64,  // Number of clauses learned
}
```

## Configuration

```rust
pub struct CdclConfig {
    pub restart_interval: u64,       // Conflicts between restarts
    pub clause_cleanup_interval: u64,// When to clean learned clauses
    pub heuristic_weights: HeuristicWeights,
}
```

## GPU Integration

For large problems, BCP can be offloaded to GPU:

1. **Clause data** is copied to GPU memory
2. **Persistent kernel** runs continuously
3. **Job submission** via lock-free MPMC queue
4. **Results** are aggregated and returned to CPU

The GPU handles clause checking while CPU manages:
- Decision making
- Conflict analysis
- Clause learning
- UNSAT core tracking
