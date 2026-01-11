"""
Satellite Lab - Python wrapper for Satellite SAT solver with constraint batching.

Example:
    >>> from satellite_lab import Solver, BatchingSolver
    >>> # Standard solver (direct FFI)
    >>> solver = Solver()
    >>> x = solver.bool_var(name="x")
    >>> solver.add_clause([+x.id])
    >>> result = solver.solve()
    
    >>> # Batching solver (cached FFI, send all at solve)
    >>> solver = BatchingSolver()
    >>> x = solver.bool_var(name="x")
    >>> solver.add_clause([+x.id])  # Cached locally
    >>> solver.add_clause([-x.id])  # Cached locally
    >>> result = solver.solve()      # All clauses sent at once
"""

from satellite_lab._core import (
    PySolver as Solver,
    PyBoolVar as BoolVar,
    PyBatch as Batch,
    PyIntVar as IntVar,
    PySatResult as SatResult,
    __version__,
)

__all__ = [
    "Solver",
    "BatchingSolver",
    "BoolVar",
    "Batch",
    "IntVar",
    "SatResult",
    "__version__",
]


class BatchingSolver:
    """
    A solver wrapper that batches constraint additions to minimize FFI overhead.
    
    Constraints are cached locally and sent to Rust in a single batch when
    solve() or finalize() is called.
    """
    
    def __init__(self):
        self._solver = Solver()
        self._pending_clauses = []
        self._finalized = False
    
    def bool_var(self, name=None):
        """Creates a new boolean variable."""
        return self._solver.bool_var(name)
    
    def batch_var(self, dim, name=None):
        """Creates a new batch variable."""
        return self._solver.batch_var(dim, name)
    
    def int_var(self, bits, name=None):
        """Creates a new integer variable."""
        return self._solver.int_var(bits, name)
    
    def add_clause(self, literals):
        """
        Cache clause locally (no FFI call).
        
        Args:
            literals: List of integer literals (positive = true, negative = false)
        """
        self._pending_clauses.append(list(literals))
    
    def add_clauses(self, clauses):
        """
        Cache multiple clauses locally (no FFI call).
        
        Args:
            clauses: List of lists of integer literals
        """
        for clause in clauses:
            self._pending_clauses.append(list(clause))
    
    def finalize(self):
        """
        Batch send all cached constraints to Rust (single FFI call).
        
        This is called automatically by solve() if not called explicitly.
        """
        if self._pending_clauses:
            self._solver.add_clauses(self._pending_clauses)
            self._pending_clauses.clear()
        self._finalized = True
    
    def solve(self):
        """
        Finalize constraints if needed, then solve.
        
        Returns:
            SatResult with satisfiable, model, and time_ms attributes
        """
        if not self._finalized:
            self.finalize()
        return self._solver.solve()
    
    @property
    def num_vars(self):
        """Number of variables in the solver."""
        return self._solver.num_vars
    
    @property
    def num_clauses(self):
        """Number of clauses (excluding pending)."""
        return self._solver.num_clauses
    
    @property
    def pending_clauses_count(self):
        """Number of pending clauses waiting to be sent."""
        return len(self._pending_clauses)
    
    def reset(self):
        """Reset the solver state for a new problem."""
        self._solver = Solver()
        self._pending_clauses.clear()
        self._finalized = False


def satellite_constraint(func):
    """
    Decorator to mark a function as a Satellite constraint (ABI-OP).
    
    The function will be compiled/analyzed by the Satellite frontend adapter.
    """
    # Mark the function for inspection
    func._is_satellite_constraint = True
    return func
