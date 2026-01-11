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
    "BitVec",
    "CircuitBuilder",
    "create_context",
    "destroy_context",
    "add_clause",
    "solve",
    "solve_with_assumptions",
    "submit_solve",
    "poll_job",
    "fetch_finished_jobs",
    "fork_context",
    "add_clauses_buffer",
    "init_worker_pool",
    "__version__",
]


class BitVec:
    """
    A fixed-width bitvector for symbolic execution.
    
    Supports common bitwise operations that generate SAT clauses.
    """
    
    def __init__(self, solver: "BatchingSolver", width: int, start_id: int = None):
        self.solver = solver
        self.width = width
        if start_id is None:
            # Allocate new variables
            self._bits = [solver.bool_var().id for _ in range(width)]
        else:
            self._bits = [start_id + i for i in range(width)]
    
    def bit(self, index: int) -> int:
        """Get the literal for bit at index (0 = LSB)."""
        return self._bits[index]
    
    def __getitem__(self, index: int) -> int:
        return self._bits[index]
    
    def xor(self, other: "BitVec", result: "BitVec"):
        """Add XOR constraints: result = self ^ other."""
        for i in range(self.width):
            a, b, r = self[i], other[i], result[i]
            # XOR encoding
            self.solver.add_clause([-a, -b, -r])
            self.solver.add_clause([a, b, -r])
            self.solver.add_clause([a, -b, r])
            self.solver.add_clause([-a, b, r])
    
    def and_(self, other: "BitVec", result: "BitVec"):
        """Add AND constraints: result = self & other."""
        for i in range(self.width):
            a, b, r = self[i], other[i], result[i]
            self.solver.add_clause([-a, -b, r])
            self.solver.add_clause([a, -r])
            self.solver.add_clause([b, -r])
    
    def or_(self, other: "BitVec", result: "BitVec"):
        """Add OR constraints: result = self | other."""
        for i in range(self.width):
            a, b, r = self[i], other[i], result[i]
            self.solver.add_clause([a, b, -r])
            self.solver.add_clause([-a, r])
            self.solver.add_clause([-b, r])
    
    def not_(self, result: "BitVec"):
        """Add NOT constraints: result = ~self."""
        for i in range(self.width):
            a, r = self[i], result[i]
            self.solver.add_clause([a, r])
            self.solver.add_clause([-a, -r])


class CircuitBuilder:
    """
    High-level circuit builder for generating SAT constraints.
    
    Provides common gadgets like XOR, AND, adders, muxes.
    """
    
    def __init__(self, solver: "BatchingSolver"):
        self.solver = solver
    
    def new_bitvec(self, width: int) -> BitVec:
        """Allocate a new bitvector with the given width."""
        return BitVec(self.solver, width)
    
    def add_xor(self, a: BitVec, b: BitVec) -> BitVec:
        """XOR two bitvectors, returning result."""
        result = self.new_bitvec(a.width)
        a.xor(b, result)
        return result
    
    def add_and(self, a: BitVec, b: BitVec) -> BitVec:
        """AND two bitvectors, returning result."""
        result = self.new_bitvec(a.width)
        a.and_(b, result)
        return result
    
    def add_or(self, a: BitVec, b: BitVec) -> BitVec:
        """OR two bitvectors, returning result."""
        result = self.new_bitvec(a.width)
        a.or_(b, result)
        return result
    
    def add_not(self, a: BitVec) -> BitVec:
        """NOT a bitvector, returning result."""
        result = self.new_bitvec(a.width)
        a.not_(result)
        return result
    
    def add_ite(self, cond: int, then_val: BitVec, else_val: BitVec) -> BitVec:
        """If-then-else (mux): result = cond ? then_val : else_val."""
        result = self.new_bitvec(then_val.width)
        for i in range(then_val.width):
            t, e, r = then_val[i], else_val[i], result[i]
            self.solver.add_clause([-cond, -t, r])
            self.solver.add_clause([-cond, t, -r])
            self.solver.add_clause([cond, -e, r])
            self.solver.add_clause([cond, e, -r])
        return result


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
        self._timeout_ms = None
    
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
    
    def set_timeout(self, timeout_ms: int):
        """Set solving timeout in milliseconds."""
        self._timeout_ms = timeout_ms
    
    def clone(self) -> "BatchingSolver":
        """Deep copy this solver (for parallel exploration)."""
        new_solver = BatchingSolver()
        new_solver._pending_clauses = [c[:] for c in self._pending_clauses]
        new_solver._finalized = self._finalized
        new_solver._timeout_ms = self._timeout_ms
        # Clone Rust solver state
        new_solver._solver = self._solver.clone()
        return new_solver
    
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
    
    def solve_with_assumptions(self, assumptions: list):
        """
        Solve with temporary assumption literals.
        
        Args:
            assumptions: List of literals to assume true temporarily
            
        Returns:
            SatResult
        """
        if not self._finalized:
            self.finalize()
        return self._solver.solve_with_assumptions(assumptions)
    
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


# =============================================================================
# Handle-based Context API (for advanced use cases)
# =============================================================================

_context_registry = {}
_next_context_id = 1
_jobs = {}
_next_job_id = 1


def create_context() -> int:
    """Create a new solver context and return its handle."""
    global _next_context_id
    ctx_id = _next_context_id
    _next_context_id += 1
    _context_registry[ctx_id] = Solver()
    return ctx_id


def destroy_context(ctx_id: int) -> bool:
    """Destroy a solver context by handle."""
    if ctx_id in _context_registry:
        del _context_registry[ctx_id]
        return True
    return False


def add_clause(ctx_id: int, literals: list) -> bool:
    """Add a clause to a context."""
    if ctx_id in _context_registry:
        _context_registry[ctx_id].add_clause(literals)
        return True
    return False


def solve(ctx_id: int):
    """Solve a context synchronously."""
    if ctx_id in _context_registry:
        return _context_registry[ctx_id].solve()
    return None


def solve_with_assumptions(ctx_id: int, assumptions: list):
    """Solve with assumptions."""
    if ctx_id in _context_registry:
        return _context_registry[ctx_id].solve_with_assumptions(assumptions)
    return None


def submit_solve(ctx_id: int) -> int:
    """Submit a solve job asynchronously (non-blocking). Returns job ID."""
    global _next_job_id
    if ctx_id not in _context_registry:
        return 0
    job_id = _next_job_id
    _next_job_id += 1
    # For now, execute synchronously (true async requires native threads)
    result = _context_registry[ctx_id].solve()
    _jobs[job_id] = {"status": "completed", "result": result}
    return job_id


def poll_job(job_id: int) -> str:
    """Poll job status: 'pending', 'running', 'completed', 'not_found'."""
    if job_id not in _jobs:
        return "not_found"
    return _jobs[job_id]["status"]


def fetch_finished_jobs(max_count: int = 10) -> list:
    """Fetch completed jobs as list of (job_id, result)."""
    finished = []
    to_remove = []
    for job_id, job in _jobs.items():
        if len(finished) >= max_count:
            break
        if job["status"] == "completed":
            finished.append((job_id, job["result"]))
            to_remove.append(job_id)
    for job_id in to_remove:
        del _jobs[job_id]
    return finished


def fork_context(src_ctx_id: int, num_clones: int) -> list:
    """Fork a context into multiple clones."""
    if src_ctx_id not in _context_registry:
        return []
    global _next_context_id
    new_ids = []
    for _ in range(num_clones):
        new_id = _next_context_id
        _next_context_id += 1
        _context_registry[new_id] = _context_registry[src_ctx_id].clone()
        new_ids.append(new_id)
    return new_ids


def add_clauses_buffer(ctx_id: int, buffer: bytes) -> int:
    """Add clauses from raw buffer. Returns number of clauses added."""
    if ctx_id not in _context_registry:
        return 0
    # Parse buffer (format: [u32 len][i64 lit]*len repeated)
    offset = 0
    count = 0
    while offset + 4 <= len(buffer):
        clause_len = int.from_bytes(buffer[offset:offset+4], 'little')
        offset += 4
        if offset + clause_len * 8 > len(buffer):
            break
        literals = []
        for _ in range(clause_len):
            lit = int.from_bytes(buffer[offset:offset+8], 'little', signed=True)
            literals.append(lit)
            offset += 8
        _context_registry[ctx_id].add_clause(literals)
        count += 1
    return count


def init_worker_pool(num_workers: int):
    """Initialize worker pool for async solving. (Python stub - uses native Rust pool)."""
    pass  # In pure Python, we can use concurrent.futures, but this is a stub


def satellite_constraint(func):
    """
    Decorator to mark a function as a Satellite constraint (ABI-OP).
    
    The function will be compiled/analyzed by the Satellite frontend adapter.
    """
    # Mark the function for inspection
    func._is_satellite_constraint = True
    return func

