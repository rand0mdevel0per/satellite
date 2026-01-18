# satellite-base

Core types and utilities for the Satellite SAT solver.

## Features

- **Boolean types**: `Lit`, `Var`, `VarAssignment`
- **Clause types**: `Clause`, `ClauseRef`
- **Batch types**: `Batch`, `BitVec` for bitvector operations
- **Serialization**: Serde support for all types

## Usage

```rust
use satellite_base::{Lit, Var, Clause};

let x = Var::new(1);
let lit = Lit::positive(x);
let clause = Clause::from_lits(vec![lit, Lit::negative(Var::new(2))]);
```

## License

MIT
