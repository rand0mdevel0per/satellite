# satellite-cdcl

CDCL (Conflict-Driven Clause Learning) SAT solver implementation.

## Features

- **Two-watched literals** for efficient BCP
- **VSIDS/LRB heuristics** for decision making
- **Clause learning** with 1-UIP conflict analysis
- **Restarts** with Luby/Glucose strategies
- **UNSAT core extraction**

## Usage

```rust
use satellite_cdcl::{CdclSolver, CdclConfig};

let config = CdclConfig::default();
let mut solver = CdclSolver::new(config);

solver.add_clause(vec![1, 2, -3]);
solver.add_clause(vec![-1, 3]);

match solver.solve() {
    SatResult::Sat(model) => println!("SAT: {:?}", model),
    SatResult::Unsat => println!("UNSAT"),
    SatResult::Unknown => println!("Timeout"),
}
```

## License

MIT
