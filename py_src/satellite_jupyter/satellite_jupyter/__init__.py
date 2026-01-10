"""
Satellite Jupyter - Jupyter kernel for Satellite SAT solver.
"""

from .kernel import SatelliteKernel
from .magic import load_ipython_extension

__version__ = "0.1.0"
__all__ = ["SatelliteKernel", "load_ipython_extension"]
