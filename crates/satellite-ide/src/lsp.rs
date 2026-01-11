//! LSP integration for the IDE.
//!
//! Routes LSP requests to appropriate language frontends based on file extension.

use satellite_kit::frontend_manager::FrontendManager;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, RwLock};

// =============================================================================
// LSP Types
// =============================================================================

/// LSP capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspCapabilities {
    pub completion: bool,
    pub hover: bool,
    pub diagnostics: bool,
    pub formatting: bool,
    pub goto_definition: bool,
}

impl Default for LspCapabilities {
    fn default() -> Self {
        Self {
            completion: true,
            hover: true,
            diagnostics: true,
            formatting: true,
            goto_definition: true,
        }
    }
}

/// Completion item for IDE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionItem {
    pub label: String,
    pub kind: String,
    pub detail: Option<String>,
    pub insert_text: Option<String>,
}

/// Hover result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoverResult {
    pub contents: String,
    pub range: Option<(usize, usize, usize, usize)>, // (startLine, startCol, endLine, endCol)
}

/// Diagnostic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub line: usize,
    pub column: usize,
    pub end_line: usize,
    pub end_column: usize,
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

// =============================================================================
// LSP Dispatcher
// =============================================================================

/// Dispatches LSP requests to appropriate language frontends.
pub struct LspDispatcher {
    /// Frontend manager for dynamic language support.
    frontend_manager: Arc<RwLock<FrontendManager>>,
    /// Built-in completions for Satellite language.
    sat_completions: Vec<CompletionItem>,
}

impl LspDispatcher {
    /// Creates a new LSP dispatcher.
    pub fn new(frontend_manager: Arc<RwLock<FrontendManager>>) -> Self {
        Self {
            frontend_manager,
            sat_completions: Self::builtin_sat_completions(),
        }
    }

    /// Returns built-in completions for Satellite language.
    fn builtin_sat_completions() -> Vec<CompletionItem> {
        vec![
            // Keywords
            CompletionItem {
                label: "AND".to_string(),
                kind: "Keyword".to_string(),
                detail: Some("Logical AND operator".to_string()),
                insert_text: Some("AND ".to_string()),
            },
            CompletionItem {
                label: "OR".to_string(),
                kind: "Keyword".to_string(),
                detail: Some("Logical OR operator".to_string()),
                insert_text: Some("OR ".to_string()),
            },
            CompletionItem {
                label: "NOT".to_string(),
                kind: "Keyword".to_string(),
                detail: Some("Logical NOT operator".to_string()),
                insert_text: Some("NOT ".to_string()),
            },
            CompletionItem {
                label: "XOR".to_string(),
                kind: "Keyword".to_string(),
                detail: Some("Logical XOR operator".to_string()),
                insert_text: Some("XOR ".to_string()),
            },
            CompletionItem {
                label: "eq".to_string(),
                kind: "Keyword".to_string(),
                detail: Some("Equality comparison".to_string()),
                insert_text: Some("eq ".to_string()),
            },
            // Variable prefixes
            CompletionItem {
                label: "%".to_string(),
                kind: "Snippet".to_string(),
                detail: Some("Boolean variable".to_string()),
                insert_text: Some("%".to_string()),
            },
            CompletionItem {
                label: "%^".to_string(),
                kind: "Snippet".to_string(),
                detail: Some("Batch variable (integer simulation)".to_string()),
                insert_text: Some("%^".to_string()),
            },
            CompletionItem {
                label: "%*".to_string(),
                kind: "Snippet".to_string(),
                detail: Some("Vector variable".to_string()),
                insert_text: Some("%*".to_string()),
            },
            // Literals
            CompletionItem {
                label: "True".to_string(),
                kind: "Constant".to_string(),
                detail: Some("Boolean true".to_string()),
                insert_text: Some("True".to_string()),
            },
            CompletionItem {
                label: "False".to_string(),
                kind: "Constant".to_string(),
                detail: Some("Boolean false".to_string()),
                insert_text: Some("False".to_string()),
            },
        ]
    }

    /// Provides completions at a given position.
    pub fn complete(
        &self,
        uri: &str,
        _line: usize,
        _column: usize,
        content: &str,
    ) -> Vec<CompletionItem> {
        let extension = Path::new(uri)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match extension {
            "sat" => {
                // Return built-in Satellite completions
                self.sat_completions.clone()
            }
            ext => {
                // Try to dispatch to frontend
                if let Ok(manager) = self.frontend_manager.read() {
                    if let Ok(result) = manager.validate_function(ext, "", "") {
                        // Frontend exists - in real impl we'd call completion
                        tracing::debug!("Frontend for .{} exists, validation: {:?}", ext, result);
                    }
                }
                // Return empty for now - frontends would provide their own
                Vec::new()
            }
        }
    }

    /// Provides hover information at a position.
    pub fn hover(
        &self,
        uri: &str,
        line: usize,
        column: usize,
        content: &str,
    ) -> Option<HoverResult> {
        let extension = Path::new(uri)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match extension {
            "sat" => {
                // Simple hover for Satellite: find word at position
                let lines: Vec<&str> = content.lines().collect();
                if line >= lines.len() {
                    return None;
                }
                
                let line_content = lines[line];
                let word = Self::word_at_column(line_content, column)?;
                
                let description = match word.as_str() {
                    "AND" | "and" | "&&" => Some("Logical AND operator. All operands must be true."),
                    "OR" | "or" | "||" => Some("Logical OR operator. At least one operand must be true."),
                    "NOT" | "not" | "!" => Some("Logical NOT operator. Negates the operand."),
                    "XOR" | "xor" | "^" => Some("Logical XOR operator. Exactly one operand must be true."),
                    "eq" => Some("Equality operator. Compares two values for equality."),
                    "True" | "true" => Some("Boolean constant representing true."),
                    "False" | "false" => Some("Boolean constant representing false."),
                    _ if word.starts_with('%') => Some("Variable reference. `%` = bool, `%^` = batch, `%*` = vec."),
                    _ => None,
                };
                
                description.map(|d| HoverResult {
                    contents: d.to_string(),
                    range: None,
                })
            }
            _ => None,
        }
    }

    /// Extracts the word at a given column position.
    fn word_at_column(line: &str, column: usize) -> Option<String> {
        let chars: Vec<char> = line.chars().collect();
        if column >= chars.len() {
            return None;
        }
        
        // Find word boundaries
        let mut start = column;
        while start > 0 && (chars[start - 1].is_alphanumeric() || chars[start - 1] == '%' || chars[start - 1] == '^' || chars[start - 1] == '*' || chars[start - 1] == '_') {
            start -= 1;
        }
        
        let mut end = column;
        while end < chars.len() && (chars[end].is_alphanumeric() || chars[end] == '_') {
            end += 1;
        }
        
        if start < end {
            Some(chars[start..end].iter().collect())
        } else {
            None
        }
    }

    /// Returns diagnostics for a document.
    pub fn diagnostics(&self, uri: &str, content: &str) -> Vec<Diagnostic> {
        let extension = Path::new(uri)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match extension {
            "sat" => {
                // Basic Satellite validation
                let mut diagnostics = Vec::new();
                
                for (line_num, line) in content.lines().enumerate() {
                    // Check for unmatched parentheses
                    let open = line.matches('(').count();
                    let close = line.matches(')').count();
                    if open != close {
                        diagnostics.push(Diagnostic {
                            line: line_num,
                            column: 0,
                            end_line: line_num,
                            end_column: line.len(),
                            message: format!("Unmatched parentheses: {} open, {} close", open, close),
                            severity: DiagnosticSeverity::Error,
                        });
                    }
                }
                
                diagnostics
            }
            _ => Vec::new(),
        }
    }
}

impl Default for LspDispatcher {
    fn default() -> Self {
        Self {
            frontend_manager: Arc::new(RwLock::new(FrontendManager::new())),
            sat_completions: Self::builtin_sat_completions(),
        }
    }
}
