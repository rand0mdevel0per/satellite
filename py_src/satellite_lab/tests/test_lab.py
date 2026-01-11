import sys
import os
import unittest
from unittest.mock import MagicMock

# Add module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

# Mock the native extension BEFORE importing satellite_lab
mock_core = MagicMock()
mock_core.PySolver = MagicMock
mock_core.PyBoolVar = MagicMock
mock_core.PyBatch = MagicMock
mock_core.PyIntVar = MagicMock
mock_core.PySatResult = MagicMock
mock_core.__version__ = "0.1.0"
sys.modules["satellite_lab._core"] = mock_core

from satellite_lab import BatchingSolver, satellite_constraint
import satellite_lab

class MockSolver:
    def __init__(self):
        self.clauses = []
        self.num_vars = 0
        self.num_clauses = 0
    
    def bool_var(self, name=None):
        self.num_vars += 1
        return self.num_vars
    
    def add_clauses(self, clauses):
        self.clauses.extend(clauses)
        self.num_clauses += len(clauses)
        
    def solve(self):
        return "Solved"

class TestBatchingSolver(unittest.TestCase):
    def setUp(self):
        # We need to patch the real Solver used by BatchingSolver if we want isolated tests
        # But BatchingSolver aliases Solver at module level.
        # We can try to mock it or just rely on functionality if built.
        # For this test, let's assume we can substitute the internal _solver if we construct it carefully
        # or just test the logic around the internal _solver if possible.
        pass

    def test_decorator(self):
        @satellite_constraint
        def my_constraint(x):
            return x
        
        self.assertTrue(hasattr(my_constraint, '_is_satellite_constraint'))
        self.assertTrue(my_constraint._is_satellite_constraint)

    def test_batching_logic(self):
        # We can test BatchingSolver's caching logic without needing the real Rust backend
        # by manually replacing self._solver after init, OR just checking _pending_clauses
        
        solver = BatchingSolver()
        # Mock the internal backend to avoid needing compiled extension just for logic test
        solver._solver = MockSolver()
        
        # Test Caching
        solver.add_clause([1, -2])
        self.assertEqual(len(solver._pending_clauses), 1)
        self.assertEqual(solver.pending_clauses_count, 1)
        
        solver.add_clauses([[3, 4], [-5]])
        self.assertEqual(len(solver._pending_clauses), 3)
        self.assertEqual(solver.pending_clauses_count, 3)
        
        # Test Finalize
        solver.finalize()
        self.assertEqual(len(solver._pending_clauses), 0)
        self.assertTrue(solver._finalized)
        self.assertEqual(len(solver._solver.clauses), 3)
        
        # Test Reset
        solver.reset()
        solver._solver = MockSolver() # Re-mock
        self.assertEqual(len(solver._pending_clauses), 0)
        self.assertFalse(solver._finalized)

if __name__ == '__main__':
    unittest.main()
