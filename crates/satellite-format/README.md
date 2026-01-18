# satellite-format

File format parsing and serialization for Satellite SAT solver.

## Supported Formats

- **DIMACS CNF** - Standard SAT competition format (`.cnf`, `.dimacs`)
- **Satellite JSON** - Extended format with type information
- **Advanced-CNF** - Internal representation with metadata

## Usage

```rust
use satellite_format::{DimacsParser, SatelliteJson};

// Parse DIMACS file
let problem = DimacsParser::parse_file("problem.cnf")?;

// Export to JSON
let json = SatelliteJson::export(&problem)?;
```

## License

MIT
