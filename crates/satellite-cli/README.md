# satellite-cli

Command-line interface for Satellite SAT solver.

## Installation

```bash
cargo install satellite-cli
```

## Usage

```bash
# Solve a DIMACS file
satellite solve problem.cnf

# Analyze constraints
satellite analyze problem.cnf

# Batch processing
satellite batch ./problems/ -o ./results/

# Show stats
satellite stats problem.cnf
```

## Commands

| Command | Description |
|---------|-------------|
| `solve` | Solve a SAT problem |
| `analyze` | Analyze constraint structure |
| `batch` | Process multiple files |
| `stats` | Show problem statistics |

## Options

| Option | Description |
|--------|-------------|
| `--timeout` | Solver timeout in seconds |
| `--gpu` | Enable GPU acceleration |
| `--json` | Output in JSON format |

## License

MIT
