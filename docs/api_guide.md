# Satellite API Guide

## 1. Defining Problems (Advanced-CNF)

Satellite uses an extended JSON format called **Advanced-CNF**.

```json
{
  "variables": [
    { "id": 1, "type": "bool", "name": "A" },
    { "id": 2, "type": "bool", "name": "B" },
    { "id": 3, "type": "bool", "name": "C" }
  ],
  "clauses": [
    { "literals": [1, -2], "type": "original" },  // A OR NOT B
    { "literals": [-1, 3], "type": "original" }   // NOT A OR C
  ],
  "abi_constraints": [
    {
      "name": "check_xor",
      "inputs": [1, 2, 3],
      "code_hash": "sha256:...",
      "cached": false
    }
  ]
}
```

## 2. CLI Usage

### Basic Solving
```bash
satellite solve problem.json
```

### Distributed Solving
Start the daemon:
```bash
satellite daemon start
```

Submit a job:
```bash
satellite submit problem.json
```

### Batch Mode (High Throughput)
```bash
satellite batch --input-dir problems/ --output-dir results/ --workers 4
```

## 3. Rust API (`satellite-kit`)

### Initialization
```rust
use satellite_kit::Solver;
use satellite_format::AdvancedCnf;

let problem = AdvancedCnf::from_file("problem.json")?;
let mut solver = Solver::new(config);
```

### Adding Constraints Programmatically
```rust
// Add a clause (A OR B)
solver.add_clause(vec![1, 2]);

// Register a custom ABI constraint
solver.register_constraint("my_check", |inputs| {
    // Custom Rust logic
    inputs[0] ^ inputs[1]
});
```

### Solving
```rust
match solver.solve() {
    SatResult::Sat(model) => println!("SAT: {:?}", model),
    SatResult::Unsat => println!("UNSAT"),
    SatResult::Unknown(reason) => println!("Unknown: {}", reason),
}
```
