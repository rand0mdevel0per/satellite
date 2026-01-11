//! Tauri commands for the IDE.

use satellite_format::{AdvancedCnf, DimacsCnf};
use satellite_kit::Solver;
use serde::{Deserialize, Serialize};

/// Diagnostic information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub line: usize,
    pub column: usize,
    pub message: String,
    pub severity: DiagnosticSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
    Hint,
}

/// Solve result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveResult {
    pub satisfiable: Option<bool>,
    pub model: Option<Vec<i64>>,
    pub time_ms: u64,
    pub error: Option<String>,
}

/// Parses a file and returns diagnostics.
#[tauri::command]
pub fn parse_file(content: String, format: String) -> Result<Vec<Diagnostic>, String> {
    let mut diagnostics = Vec::new();

    match format.as_str() {
        "dimacs" | "cnf" => {
            if let Err(e) = DimacsCnf::from_str(&content) {
                diagnostics.push(Diagnostic {
                    line: 1,
                    column: 1,
                    message: e.to_string(),
                    severity: DiagnosticSeverity::Error,
                });
            }
        }
        "json" => {
            if let Err(e) = AdvancedCnf::from_json(&content) {
                diagnostics.push(Diagnostic {
                    line: 1,
                    column: 1,
                    message: e.to_string(),
                    severity: DiagnosticSeverity::Error,
                });
            }
        }
        _ => {
            diagnostics.push(Diagnostic {
                line: 1,
                column: 1,
                message: format!("Unknown format: {}", format),
                severity: DiagnosticSeverity::Warning,
            });
        }
    }

    Ok(diagnostics)
}

/// Solves a file.
#[tauri::command]
pub fn solve_file(content: String, format: String) -> SolveResult {
    let start = std::time::Instant::now();

    let problem = match format.as_str() {
        "dimacs" | "cnf" => match DimacsCnf::from_str(&content) {
            Ok(d) => d.to_advanced_cnf(),
            Err(e) => {
                return SolveResult {
                    satisfiable: None,
                    model: None,
                    time_ms: start.elapsed().as_millis() as u64,
                    error: Some(e.to_string()),
                };
            }
        },
        "json" => match AdvancedCnf::from_json(&content) {
            Ok(p) => p,
            Err(e) => {
                return SolveResult {
                    satisfiable: None,
                    model: None,
                    time_ms: start.elapsed().as_millis() as u64,
                    error: Some(e.to_string()),
                };
            }
        },
        _ => {
            return SolveResult {
                satisfiable: None,
                model: None,
                time_ms: start.elapsed().as_millis() as u64,
                error: Some(format!("Unknown format: {}", format)),
            };
        }
    };

    let mut solver = Solver::new();
    for clause in &problem.clauses {
        solver.add_clause(clause.literals.iter().copied());
    }

    match solver.solve() {
        Ok(satellite_kit::SatResult::Sat(model)) => {
            let model_lits: Vec<i64> = model
                .iter()
                .enumerate()
                .map(|(i, &v)| if v { (i + 1) as i64 } else { -((i + 1) as i64) })
                .collect();
            SolveResult {
                satisfiable: Some(true),
                model: Some(model_lits),
                time_ms: start.elapsed().as_millis() as u64,
                error: None,
            }
        }
        Ok(satellite_kit::SatResult::Unsat) => SolveResult {
            satisfiable: Some(false),
            model: None,
            time_ms: start.elapsed().as_millis() as u64,
            error: None,
        },
        Ok(satellite_kit::SatResult::Unknown(reason)) => SolveResult {
            satisfiable: None,
            model: None,
            time_ms: start.elapsed().as_millis() as u64,
            error: Some(format!("Unknown: {}", reason)),
        },
        Err(e) => SolveResult {
            satisfiable: None,
            model: None,
            time_ms: start.elapsed().as_millis() as u64,
            error: Some(e.to_string()),
        },
    }
}

/// Gets diagnostics for a document.
#[tauri::command]
pub fn get_diagnostics(content: String, format: String) -> Vec<Diagnostic> {
    parse_file(content, format).unwrap_or_default()
}

/// Formats a document.
#[tauri::command]
pub fn format_document(content: String, format: String) -> Result<String, String> {
    match format.as_str() {
        "json" => {
            let problem = AdvancedCnf::from_json(&content).map_err(|e| e.to_string())?;
            problem.to_json().map_err(|e| e.to_string())
        }
        _ => Ok(content), // No formatting for other formats
    }
}
// =============================================================================
// Plugin System
// =============================================================================

#[tauri::command]
pub fn load_plugin(_path: String) -> Result<String, String> {
    // In a real app, this would use State<PluginRuntime> to load
    // For now, we just placeholder it to compile
    Ok("Plugin loaded".to_string())
}

// =============================================================================
// LSP Commands
// =============================================================================

use crate::lsp::{LspDispatcher, CompletionItem as LspCompletionItem, HoverResult, Diagnostic as LspDiagnostic};

/// Get completions at a position.
#[tauri::command]
pub fn lsp_complete(
    uri: String,
    line: usize,
    column: usize,
    content: String,
) -> Vec<LspCompletionItem> {
    let dispatcher = LspDispatcher::default();
    dispatcher.complete(&uri, line, column, &content)
}

/// Get hover information at a position.
#[tauri::command]
pub fn lsp_hover(
    uri: String,
    line: usize,
    column: usize,
    content: String,
) -> Option<HoverResult> {
    let dispatcher = LspDispatcher::default();
    dispatcher.hover(&uri, line, column, &content)
}

/// Get diagnostics for a document.
#[tauri::command]
pub fn lsp_diagnostics(uri: String, content: String) -> Vec<LspDiagnostic> {
    let dispatcher = LspDispatcher::default();
    dispatcher.diagnostics(&uri, &content)
}
// =============================================================================
// IDE Visualization & Statistics
// =============================================================================

/// Solver statistics for visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveStats {
    pub num_vars: usize,
    pub num_clauses: usize,
    pub decisions: u64,
    pub conflicts: u64,
    pub propagations: u64,
    pub learned_clauses: u64,
    pub restarts: u64,
    pub solve_time_ms: u64,
}

/// Clause node for visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClauseNode {
    pub id: usize,
    pub literals: Vec<i64>,
    pub is_learned: bool,
}

/// Clause graph for dependency visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClauseGraph {
    pub nodes: Vec<ClauseNode>,
    pub var_count: usize,
}

/// Gets solve statistics without solving.
#[tauri::command]
pub fn get_problem_stats(content: String, format: String) -> Result<SolveStats, String> {
    let problem = match format.as_str() {
        "dimacs" | "cnf" => DimacsCnf::from_str(&content)
            .map_err(|e| e.to_string())?
            .to_advanced_cnf(),
        "json" => AdvancedCnf::from_json(&content).map_err(|e| e.to_string())?,
        _ => return Err(format!("Unknown format: {}", format)),
    };

    let num_vars = problem.variables.len().max(
        problem
            .clauses
            .iter()
            .flat_map(|c| c.literals.iter())
            .map(|&lit| lit.unsigned_abs() as usize)
            .max()
            .unwrap_or(0),
    );

    Ok(SolveStats {
        num_vars,
        num_clauses: problem.clauses.len(),
        decisions: 0,
        conflicts: 0,
        propagations: 0,
        learned_clauses: 0,
        restarts: 0,
        solve_time_ms: 0,
    })
}

/// Gets clause graph for visualization.
#[tauri::command]
pub fn get_clause_graph(content: String, format: String) -> Result<ClauseGraph, String> {
    let problem = match format.as_str() {
        "dimacs" | "cnf" => DimacsCnf::from_str(&content)
            .map_err(|e| e.to_string())?
            .to_advanced_cnf(),
        "json" => AdvancedCnf::from_json(&content).map_err(|e| e.to_string())?,
        _ => return Err(format!("Unknown format: {}", format)),
    };

    let nodes: Vec<ClauseNode> = problem
        .clauses
        .iter()
        .enumerate()
        .map(|(id, c)| ClauseNode {
            id,
            literals: c.literals.clone(),
            is_learned: false,
        })
        .collect();

    let var_count = problem.variables.len().max(
        problem
            .clauses
            .iter()
            .flat_map(|c| c.literals.iter())
            .map(|&lit| lit.unsigned_abs() as usize)
            .max()
            .unwrap_or(0),
    );

    Ok(ClauseGraph { nodes, var_count })
}

/// Exports solve result to different formats.
#[tauri::command]
pub fn export_result(result: SolveResult, format: String) -> Result<String, String> {
    match format.as_str() {
        "json" => serde_json::to_string_pretty(&result).map_err(|e| e.to_string()),
        "csv" => {
            let mut csv = String::from("variable,value\n");
            if let Some(model) = &result.model {
                for (i, &lit) in model.iter().enumerate() {
                    csv.push_str(&format!("{},{}\n", i + 1, if lit > 0 { "true" } else { "false" }));
                }
            }
            Ok(csv)
        }
        "dimacs" => {
            let mut output = String::new();
            if let Some(true) = result.satisfiable {
                output.push_str("s SATISFIABLE\n");
                if let Some(model) = &result.model {
                    output.push_str("v ");
                    for lit in model {
                        output.push_str(&format!("{} ", lit));
                    }
                    output.push_str("0\n");
                }
            } else if let Some(false) = result.satisfiable {
                output.push_str("s UNSATISFIABLE\n");
            } else {
                output.push_str("s UNKNOWN\n");
            }
            Ok(output)
        }
        _ => Err(format!("Unknown export format: {}", format)),
    }
}

