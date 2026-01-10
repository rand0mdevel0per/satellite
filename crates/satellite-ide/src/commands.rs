//! Tauri commands for the IDE.

use satellite_kit::Solver;
use satellite_format::{AdvancedCnf, DimacsCnf};
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
        "dimacs" | "cnf" => {
            match DimacsCnf::from_str(&content) {
                Ok(d) => d.to_advanced_cnf(),
                Err(e) => {
                    return SolveResult {
                        satisfiable: None,
                        model: None,
                        time_ms: start.elapsed().as_millis() as u64,
                        error: Some(e.to_string()),
                    };
                }
            }
        }
        "json" => {
            match AdvancedCnf::from_json(&content) {
                Ok(p) => p,
                Err(e) => {
                    return SolveResult {
                        satisfiable: None,
                        model: None,
                        time_ms: start.elapsed().as_millis() as u64,
                        error: Some(e.to_string()),
                    };
                }
            }
        }
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
