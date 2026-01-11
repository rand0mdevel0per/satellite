# CLI Reference

The Satellite command-line interface provides tools for solving SAT problems.

## Installation

```bash
cargo install --path crates/satellite-cli
```

This installs the `satellite` binary.

## Commands

### solve

Solve a single SAT problem.

```bash
satellite solve <INPUT> [OPTIONS]
```

**Arguments:**
- `<INPUT>`: Path to problem file (.json, .jsonc, .cnf)

**Options:**
- `-o, --output <FILE>`: Output file for result
- `-t, --timeout <SECONDS>`: Timeout in seconds
- `-v, --verbose`: Verbose output

**Examples:**
```bash
# Solve and print result
satellite solve problem.json

# Solve with timeout and save result
satellite solve problem.json -o result.json -t 60
```

### batch

Batch solve multiple problems.

```bash
satellite batch [OPTIONS]
```

**Options:**
- `-i, --input-dir <DIR>`: Directory containing problems (default: current)
- `-o, --output-dir <DIR>`: Directory for results (default: ./results)
- `-w, --workers <N>`: Number of parallel workers (default: 1)
- `-t, --timeout <SECONDS>`: Per-problem timeout
- `--pattern <GLOB>`: File pattern to match (default: *.json)

**Examples:**
```bash
# Solve all JSON files in ./problems
satellite batch -i ./problems -o ./results

# Parallel solving with 4 workers
satellite batch -i ./problems -o ./results -w 4 -t 60
```

### convert

Convert between file formats.

```bash
satellite convert <INPUT> <OUTPUT>
```

**Supported formats:**
- `.cnf` (DIMACS)
- `.json` (Advanced-CNF)
- `.jsonc` (Canonical JSON)

**Examples:**
```bash
# DIMACS to JSON
satellite convert problem.cnf problem.json

# JSON to DIMACS
satellite convert problem.json problem.cnf
```

### install-frontend

Install language frontend adapters.

```bash
satellite install-frontend <LANGUAGE>
```

**Languages:**
- `python` - Python adapter
- `rust` - Rust adapter  
- `cpp` - C++ adapter

### serve

Start the daemon server.

```bash
satellite serve [OPTIONS]
```

**Options:**
- `-p, --port <PORT>`: Port to listen on (default: 8080)
- `--host <HOST>`: Host to bind (default: 127.0.0.1)

## File Formats

### DIMACS CNF

Standard DIMACS format:

```
c Comment line
p cnf 3 2
1 2 0
-1 3 0
```

### Advanced-CNF (JSON)

Extended format with type information:

```json
{
  "variables": [
    {"id": 0, "var_type": "Bool", "name": "x"},
    {"id": 1, "var_type": {"Batch": {"dim": 32}}, "name": "reg"}
  ],
  "clauses": [
    {"literals": [1, 2, -3], "clause_type": "Original"},
    {"literals": [-1, 3], "clause_type": "Original"}
  ]
}
```

### Canonical JSON (.jsonc)

Human-readable format with comments:

```jsonc
// Simple 3-SAT problem
{
  "variables": [
    {"id": 0, "var_type": "Bool", "name": "x"},
    {"id": 1, "var_type": "Bool", "name": "y"}
  ],
  "clauses": [
    {"literals": [1, 2]},  // x OR y
    {"literals": [-1, 2]}  // NOT x OR y
  ]
}
```

## Output Format

Result files contain:

```json
{
  "status": "SAT",
  "model": [true, false, true],
  "time_ms": 42,
  "stats": {
    "decisions": 10,
    "conflicts": 3,
    "propagations": 156
  }
}
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SATELLITE_LOG` | Log level (trace, debug, info, warn, error) |
| `SATELLITE_THREADS` | Default worker count |
| `SATELLITE_TIMEOUT` | Default timeout in seconds |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (SAT) |
| 1 | General error |
| 10 | UNSAT |
| 20 | Unknown/timeout |
