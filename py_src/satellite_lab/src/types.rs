//! Python type wrappers.

use pyo3::prelude::*;

/// A boolean variable.
#[pyclass]
#[derive(Clone, Copy)]
pub struct PyBoolVar {
    pub id: u64,
}

#[pymethods]
impl PyBoolVar {
    /// Returns the variable ID.
    #[getter]
    fn id(&self) -> u64 {
        self.id
    }

    /// Returns a positive literal.
    fn pos(&self) -> i64 {
        (self.id + 1) as i64
    }

    /// Returns a negative (negated) literal.
    fn neg(&self) -> i64 {
        -((self.id + 1) as i64)
    }

    fn __repr__(&self) -> String {
        format!("BoolVar({})", self.id)
    }

    fn __pos__(&self) -> i64 {
        self.pos()
    }

    fn __neg__(&self) -> i64 {
        self.neg()
    }
}

/// A batch of boolean variables.
#[pyclass]
#[derive(Clone)]
pub struct PyBatch {
    pub base_id: u64,
    pub dim: usize,
}

#[pymethods]
impl PyBatch {
    /// Returns the batch dimension.
    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    /// Gets a boolean variable at the specified index.
    fn __getitem__(&self, index: usize) -> PyResult<PyBoolVar> {
        if index >= self.dim {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("Index {} out of bounds for batch of dim {}", index, self.dim)
            ));
        }
        Ok(PyBoolVar { id: self.base_id + index as u64 })
    }

    fn __len__(&self) -> usize {
        self.dim
    }

    fn __repr__(&self) -> String {
        format!("Batch(base={}, dim={})", self.base_id, self.dim)
    }
}

/// An integer variable.
#[pyclass]
#[derive(Clone)]
pub struct PyIntVar {
    pub base_id: u64,
    pub bits: usize,
}

#[pymethods]
impl PyIntVar {
    /// Returns the bit width.
    #[getter]
    fn bits(&self) -> usize {
        self.bits
    }

    /// Converts to a batch.
    fn to_batch(&self) -> PyBatch {
        PyBatch {
            base_id: self.base_id,
            dim: self.bits,
        }
    }

    /// Gets the bit at the specified index (0 = MSB).
    fn __getitem__(&self, index: usize) -> PyResult<PyBoolVar> {
        if index >= self.bits {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("Index {} out of bounds for int of {} bits", index, self.bits)
            ));
        }
        Ok(PyBoolVar { id: self.base_id + index as u64 })
    }

    fn __repr__(&self) -> String {
        format!("IntVar(base={}, bits={})", self.base_id, self.bits)
    }
}
