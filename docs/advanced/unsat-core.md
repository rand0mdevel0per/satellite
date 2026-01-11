# UNSAT Core Extraction

UNSAT core extraction identifies the minimal subset of clauses that cause unsatisfiability.

## Overview

When a SAT problem is unsatisfiable, the UNSAT core tells you **which clauses are responsible**. This is useful for:

- Debugging constraint systems
- Finding conflicting requirements
- Minimal unsatisfiable subset (MUS) computation
- Optimization through clause relaxation

## Usage

### Python

```python
from satellite_lab import BatchingSolver

solver = BatchingSolver()

# Create variables
x = solver.bool_var().id
y = solver.bool_var().id

# Add conflicting clauses
solver.add_clause([x])      # Clause 0: x must be true
solver.add_clause([-x])     # Clause 1: x must be false
solver.add_clause([y])      # Clause 2: y (not in conflict)

# Note: Python wrapper uses CPU-side UNSAT core
# For low-level access, use the Rust API
result = solver.solve()
print(f"SAT: {result.satisfiable}")  # False - UNSAT
```

### Rust (Low-Level)

```rust
use satellite_cdcl::{CdclSolver, SatResult};
use satellite_format::AdvancedCnf;

// Load problem
let problem = load_problem("conflict.json")?;
let mut solver = CdclSolver::new(&problem);

// Enable UNSAT core tracking
solver.enable_unsat_core();

// Solve
match solver.solve()? {
    SatResult::Unsat => {
        // Get the core clause indices
        if let Some(core_indices) = solver.get_unsat_core() {
            println!("UNSAT core clause indices: {:?}", core_indices);
        }
        
        // Get the actual clauses
        if let Some(core_clauses) = solver.get_unsat_core_clauses() {
            println!("UNSAT core clauses:");
            for (i, clause) in core_clauses.iter().enumerate() {
                println!("  {}: {:?}", i, clause);
            }
        }
    }
    _ => {}
}
```

## How It Works

1. **Enable Tracking**: Call `enable_unsat_core()` before solving
2. **Conflict Recording**: During conflict analysis, the solver records which clauses were used
3. **Core Extraction**: After UNSAT, `get_unsat_core()` returns the recorded clauses

### Implementation Details

```rust
// In CdclSolver
struct CdclSolver {
    // ... other fields ...
    used_clauses: Vec<usize>,    // Clause IDs used in conflicts
    track_unsat_core: bool,      // Whether tracking is enabled
}

fn propagate(&mut self) -> Option<usize> {
    // ... BCP logic ...
    if let Some(conflict_clause) = self.find_conflict() {
        // Record this clause for UNSAT core
        self.record_used_clause(conflict_clause);
        return Some(conflict_clause);
    }
    None
}
```

## GPU Considerations

UNSAT core tracking is **CPU-only** for the following reasons:

1. **Complexity**: Clause recording during conflict analysis is sequential
2. **Resolution Chain**: Building the resolution proof requires CPU-side logic
3. **Memory**: Recording all used clauses would require significant GPU memory

When using GPU acceleration:
- GPU performs BCP (clause checking)
- CPU handles conflict analysis + UNSAT core recording
- Results are aggregated on CPU

## Example: Finding Conflicting Requirements

```python
from satellite_lab import BatchingSolver

def find_conflicts(requirements):
    """
    Given a list of requirement clauses, find which ones conflict.
    """
    solver = BatchingSolver()
    
    # Map requirement index to solver clause index
    for i, req in enumerate(requirements):
        solver.add_clause(req)
    
    result = solver.solve()
    
    if result.satisfiable:
        return None  # No conflicts
    
    # Would need Rust API for actual core extraction
    # Python wrapper returns basic UNSAT result
    return "UNSAT - use Rust API for core extraction"

# Example requirements
reqs = [
    [1, 2],    # Feature A OR Feature B
    [-1],      # NOT Feature A  
    [-2],      # NOT Feature B
    [3],       # Feature C (unrelated)
]

conflicts = find_conflicts(reqs)
```

## Performance

UNSAT core tracking adds overhead:

| Operation | Without Core | With Core |
|-----------|-------------|-----------|
| Conflict recording | N/A | O(1) per conflict |
| Memory | O(clauses) | O(clauses + conflicts) |
| Post-processing | N/A | O(core_size log core_size) |

Typical overhead is 5-10% for most problems.

## Limitations

1. **Approximate Core**: The returned core is not guaranteed to be minimal
2. **CPU Only**: Not available on GPU path
3. **SAT Problems**: Only meaningful when result is UNSAT
4. **Learning Impact**: Core may include learned clauses (filtered by default)

## See Also

- [CDCL Implementation](../architecture/cdcl.md)
- [Core API](../api/core-api.md)
