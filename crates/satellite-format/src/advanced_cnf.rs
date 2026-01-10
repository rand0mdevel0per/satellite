//! Advanced-CNF JSON format.
//!
//! The native format for Satellite, supporting typed variables and ABI constraints.

use satellite_base::types::VarType;
use serde::{Deserialize, Serialize};

/// A variable definition in Advanced-CNF.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableDef {
    /// Unique variable ID.
    pub id: u64,
    /// Variable type.
    #[serde(rename = "type")]
    pub var_type: VarType,
    /// Optional human-readable name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// A clause in Advanced-CNF.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Clause {
    /// Literals in DIMACS format (positive = true, negative = negated).
    pub literals: Vec<i64>,
    /// Whether this is an original or learned clause.
    #[serde(rename = "type")]
    pub clause_type: ClauseType,
    /// Literal Block Distance for learned clauses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lbd: Option<u32>,
}

/// Type of clause.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClauseType {
    Original,
    Learned,
}

/// An ABI constraint reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbiConstraint {
    /// Function name.
    pub name: String,
    /// Input variable IDs.
    pub inputs: Vec<u64>,
    /// SHA256 hash of the compiled code.
    pub code_hash: String,
    /// Whether the compiled code is cached.
    #[serde(default)]
    pub cached: bool,
}

/// The Advanced-CNF document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedCnf {
    /// Variable definitions.
    pub variables: Vec<VariableDef>,
    /// Clauses.
    pub clauses: Vec<Clause>,
    /// ABI constraint references.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub abi_constraints: Vec<AbiConstraint>,
}

impl AdvancedCnf {
    /// Creates an empty Advanced-CNF document.
    #[must_use]
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            clauses: Vec::new(),
            abi_constraints: Vec::new(),
        }
    }

    /// Parses from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serializes to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Parses from JSON reader.
    pub fn from_reader<R: std::io::Read>(reader: R) -> Result<Self, serde_json::Error> {
        serde_json::from_reader(reader)
    }

    /// Writes to JSON writer.
    pub fn to_writer<W: std::io::Write>(&self, writer: W) -> Result<(), serde_json::Error> {
        serde_json::to_writer_pretty(writer, self)
    }
}

impl Default for AdvancedCnf {
    fn default() -> Self {
        Self::new()
    }
}
