//! Solver results and model extraction.

use satellite_base::types::VarId;
use std::collections::HashMap;

/// A satisfying model (variable assignments).
#[derive(Debug, Clone)]
pub struct Model {
    /// Boolean assignments indexed by variable ID.
    assignments: HashMap<VarId, bool>,
}

impl Model {
    /// Creates a new model from assignments.
    pub fn new(assignments: impl IntoIterator<Item = (VarId, bool)>) -> Self {
        Self {
            assignments: assignments.into_iter().collect(),
        }
    }

    /// Creates a model from a DIMACS-style literal list.
    pub fn from_dimacs(literals: &[i64]) -> Self {
        let assignments = literals
            .iter()
            .filter(|&&l| l != 0)
            .map(|&l| {
                if l > 0 {
                    ((l - 1) as VarId, true)
                } else {
                    ((-l - 1) as VarId, false)
                }
            })
            .collect();
        Self { assignments }
    }

    /// Gets the value of a boolean variable.
    pub fn get_bool(&self, var: VarId) -> Option<bool> {
        self.assignments.get(&var).copied()
    }

    /// Gets the value of a batch as a u64.
    pub fn get_batch(&self, base_id: VarId, dim: usize) -> Option<u64> {
        let mut value = 0u64;
        for i in 0..dim.min(64) {
            if let Some(&bit) = self.assignments.get(&(base_id + i as VarId)) {
                if bit {
                    value |= 1 << (dim - 1 - i); // Big-endian
                }
            } else {
                return None;
            }
        }
        Some(value)
    }

    /// Gets the value of an integer.
    pub fn get_int(&self, base_id: VarId, bits: usize, signed: bool) -> Option<i64> {
        let unsigned = self.get_batch(base_id, bits)?;

        if signed && bits > 0 && (unsigned >> (bits - 1)) & 1 == 1 {
            // Sign extend
            let mask = !((1u64 << bits) - 1);
            Some((unsigned | mask) as i64)
        } else {
            Some(unsigned as i64)
        }
    }

    /// Returns all assignments.
    pub fn assignments(&self) -> &HashMap<VarId, bool> {
        &self.assignments
    }

    /// Returns the number of assigned variables.
    pub fn len(&self) -> usize {
        self.assignments.len()
    }

    /// Returns whether the model is empty.
    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }
}

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut vars: Vec<_> = self.assignments.iter().collect();
        vars.sort_by_key(|(k, _)| *k);

        write!(f, "[")?;
        for (i, &(&var, &val)) in vars.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "x{} = {}", var, if val { 1 } else { 0 })?;
        }
        write!(f, "]")
    }
}
