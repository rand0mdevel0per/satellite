"""
Satellite Lab - Python wrapper for Satellite SAT solver.

Example:
    >>> from satellite_lab import Solver
    >>> solver = Solver()
    >>> x = solver.bool_var(name="x")
    >>> y = solver.bool_var(name="y")
    >>> solver.add_clause([+x, +y])  # x OR y
    >>> solver.add_clause([-x, -y])  # NOT x OR NOT y
    >>> result = solver.solve()
    >>> print(result)
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
    "BoolVar",
    "Batch",
    "IntVar",
    "SatResult",
    "__version__",
]
