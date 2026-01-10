//! DIMACS CNF format parser.
//!
//! Standard format used in SAT competitions.

use satellite_base::{Error, Result};
use std::io::{BufRead, BufReader, Read};

/// A DIMACS CNF formula.
#[derive(Debug, Clone)]
pub struct DimacsCnf {
    /// Number of variables.
    pub num_vars: usize,
    /// Clauses as vectors of literals.
    pub clauses: Vec<Vec<i64>>,
}

impl DimacsCnf {
    /// Parses DIMACS CNF from a reader.
    pub fn from_reader<R: Read>(reader: R) -> Result<Self> {
        let reader = BufReader::new(reader);
        let mut num_vars = 0;
        let mut num_clauses = 0;
        let mut clauses = Vec::new();
        let mut header_found = false;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('c') {
                continue;
            }

            // Parse header
            if line.starts_with('p') {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 && parts[1] == "cnf" {
                    num_vars = parts[2].parse().map_err(|e| {
                        Error::Serialization(format!("Invalid variable count: {e}"))
                    })?;
                    num_clauses = parts[3]
                        .parse()
                        .map_err(|e| Error::Serialization(format!("Invalid clause count: {e}")))?;
                    header_found = true;
                    clauses.reserve(num_clauses);
                }
                continue;
            }

            if !header_found {
                return Err(Error::Serialization(
                    "DIMACS header 'p cnf ...' not found".to_string(),
                ));
            }

            // Parse clause
            let mut clause = Vec::new();
            for token in line.split_whitespace() {
                let lit: i64 = token
                    .parse()
                    .map_err(|e| Error::Serialization(format!("Invalid literal: {e}")))?;
                if lit == 0 {
                    if !clause.is_empty() {
                        clauses.push(clause);
                        clause = Vec::new();
                    }
                } else {
                    clause.push(lit);
                }
            }
            // Handle clauses not terminated by 0
            if !clause.is_empty() {
                clauses.push(clause);
            }
        }

        Ok(Self { num_vars, clauses })
    }

    /// Parses DIMACS CNF from a string.
    pub fn from_str(s: &str) -> Result<Self> {
        Self::from_reader(s.as_bytes())
    }

    /// Converts to DIMACS string.
    #[must_use]
    pub fn to_dimacs(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("p cnf {} {}\n", self.num_vars, self.clauses.len()));
        for clause in &self.clauses {
            for lit in clause {
                out.push_str(&format!("{lit} "));
            }
            out.push_str("0\n");
        }
        out
    }

    /// Converts to Advanced-CNF format.
    #[must_use]
    pub fn to_advanced_cnf(&self) -> super::AdvancedCnf {
        use super::advanced_cnf::{AdvancedCnf, Clause, ClauseType, VariableDef};
        use satellite_base::types::VarType;

        let variables: Vec<VariableDef> = (0..self.num_vars as u64)
            .map(|id| VariableDef {
                id,
                var_type: VarType::Bool,
                name: None,
            })
            .collect();

        let clauses: Vec<Clause> = self
            .clauses
            .iter()
            .map(|lits| Clause {
                literals: lits.clone(),
                clause_type: ClauseType::Original,
                lbd: None,
            })
            .collect();

        AdvancedCnf {
            variables,
            clauses,
            abi_constraints: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dimacs() {
        let input = r"
c This is a comment
p cnf 3 2
1 -2 0
2 3 0
";
        let cnf = DimacsCnf::from_str(input).unwrap();
        assert_eq!(cnf.num_vars, 3);
        assert_eq!(cnf.clauses.len(), 2);
        assert_eq!(cnf.clauses[0], vec![1, -2]);
        assert_eq!(cnf.clauses[1], vec![2, 3]);
    }
}
