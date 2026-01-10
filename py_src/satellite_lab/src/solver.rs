//! Python Solver wrapper.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use satellite_kit::{Solver as RustSolver, SatResult};
use crate::types::{PyBoolVar, PyBatch, PyIntVar};

/// SAT solving result.
#[pyclass]
#[derive(Clone)]
pub struct PySatResult {
    #[pyo3(get)]
    pub satisfiable: Option<bool>,
    #[pyo3(get)]
    pub model: Option<Vec<i64>>,
    #[pyo3(get)]
    pub time_ms: u64,
}

#[pymethods]
impl PySatResult {
    fn __repr__(&self) -> String {
        match self.satisfiable {
            Some(true) => format!("SatResult(SAT, {} vars)", self.model.as_ref().map(|m| m.len()).unwrap_or(0)),
            Some(false) => "SatResult(UNSAT)".to_string(),
            None => "SatResult(UNKNOWN)".to_string(),
        }
    }
}

/// The main Satellite solver.
#[pyclass]
pub struct PySolver {
    inner: RustSolver,
}

#[pymethods]
impl PySolver {
    /// Creates a new solver.
    #[new]
    fn new() -> Self {
        Self {
            inner: RustSolver::new(),
        }
    }

    /// Creates a new boolean variable.
    fn bool_var(&mut self, name: Option<String>) -> PyBoolVar {
        let var = match name {
            Some(n) => self.inner.bool_var_named(Some(&n)),
            None => self.inner.bool_var(),
        };
        PyBoolVar { id: var.id() }
    }

    /// Creates a new batch variable.
    fn batch_var(&mut self, dim: usize, name: Option<String>) -> PyBatch {
        let batch = match name {
            Some(n) => self.inner.batch_var_named(dim, Some(&n)),
            None => self.inner.batch_var(dim),
        };
        PyBatch {
            base_id: batch.base_id(),
            dim: batch.dim(),
        }
    }

    /// Creates a new integer variable.
    fn int_var(&mut self, bits: usize, name: Option<String>) -> PyIntVar {
        let var = match name {
            Some(n) => self.inner.int_var_named(bits, Some(&n)),
            None => self.inner.int_var(bits),
        };
        PyIntVar {
            base_id: var.as_batch().base_id(),
            bits: var.bits(),
        }
    }

    /// Adds a clause (list of literals).
    fn add_clause(&mut self, literals: Vec<i64>) {
        self.inner.add_clause(literals);
    }

    /// Adds multiple clauses at once (batch API).
    fn add_clauses(&mut self, clauses: Vec<Vec<i64>>) {
        for clause in clauses {
            self.inner.add_clause(clause);
        }
    }

    /// Solves the problem.
    fn solve(&mut self) -> PyResult<PySatResult> {
        let start = std::time::Instant::now();
        
        match self.inner.solve() {
            Ok(SatResult::Sat(model)) => {
                let model_lits: Vec<i64> = model
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| if v { (i + 1) as i64 } else { -((i + 1) as i64) })
                    .collect();
                Ok(PySatResult {
                    satisfiable: Some(true),
                    model: Some(model_lits),
                    time_ms: start.elapsed().as_millis() as u64,
                })
            }
            Ok(SatResult::Unsat) => Ok(PySatResult {
                satisfiable: Some(false),
                model: None,
                time_ms: start.elapsed().as_millis() as u64,
            }),
            Ok(SatResult::Unknown(_)) => Ok(PySatResult {
                satisfiable: None,
                model: None,
                time_ms: start.elapsed().as_millis() as u64,
            }),
            Err(e) => Err(PyValueError::new_err(e.to_string())),
        }
    }

    /// Returns the number of variables.
    #[getter]
    fn num_vars(&self) -> usize {
        self.inner.num_vars()
    }

    /// Returns the number of clauses.
    #[getter]
    fn num_clauses(&self) -> usize {
        self.inner.num_clauses()
    }
}
