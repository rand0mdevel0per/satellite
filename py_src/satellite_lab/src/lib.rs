//! PyO3 bindings for Satellite SAT solver.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

mod solver;
mod types;

use solver::PySolver;
use types::{PyBoolVar, PyBatch, PyIntVar};

/// Satellite SAT Solver Python bindings.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySolver>()?;
    m.add_class::<PyBoolVar>()?;
    m.add_class::<PyBatch>()?;
    m.add_class::<PyIntVar>()?;
    m.add_class::<solver::PySatResult>()?;
    
    m.add("__version__", "0.1.0")?;
    
    Ok(())
}
