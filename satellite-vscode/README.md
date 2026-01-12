# Satellite VSCode Extension

Language support and solver integration for the Satellite SAT solver.

## Features

- **Syntax Highlighting** for `.sat` and `.dimacs`/`.cnf` files
- **Solve Command** (Ctrl+Shift+S) to run the solver
- **GPU Acceleration** support
- **Output Panel** for results

## Installation

1. Build the extension:
   ```bash
   npm install
   npm run compile
   ```

2. Install in VSCode:
   ```bash
   npx vsce package
   code --install-extension satellite-vscode-0.1.0.vsix
   ```

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `satellite.solverPath` | `satellite` | Path to Satellite CLI |
| `satellite.timeout` | `60` | Solver timeout (seconds) |
| `satellite.useGpu` | `true` | Enable GPU acceleration |

## Commands

| Command | Keybinding | Description |
|---------|------------|-------------|
| Satellite: Solve | Ctrl+Shift+S | Solve current file |
| Satellite: Analyze | - | Analyze constraints |
| Satellite: Show Output | - | Show output panel |
