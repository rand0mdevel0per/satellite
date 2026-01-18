# satellite-kit

High-level API for the Satellite SAT solver.

## Features

- **Solver** - Easy-to-use solver interface
- **Circuit gadgets** - XOR, AND, OR, adders, comparators
- **Context management** - Fork/clone solver states
- **Async solving** - Non-blocking job submission

## Usage

```rust
use satellite_kit::{Solver, SatResult};

let mut solver = Solver::new();
solver.add_clause(vec![1, 2]);
solver.add_clause(vec![-1, -2]);

match solver.solve() {
    SatResult::Sat(model) => println!("Solution: {:?}", model),
    SatResult::Unsat => println!("No solution"),
    _ => {}
}
```

## Circuit Gadgets

```rust
use satellite_kit::{CircuitBuilder, BitVec};

let mut builder = CircuitBuilder::new(&mut solver);
let a = builder.new_bitvec(8);
let b = builder.new_bitvec(8);
let sum = builder.add(&a, &b);
```

## License

MIT
