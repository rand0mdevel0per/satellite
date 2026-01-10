//! High-level solver API.

use crate::constraint::Constraint;
use crate::result::Model;
use satellite_base::Result;
use satellite_base::types::{Batch, BoolVar, FloatVar, IntVar, VarId, VarType, VecVar};
use satellite_cdcl::{CdclConfig, CdclSolver, SatResult};
use satellite_format::AdvancedCnf;
use satellite_worker::{WorkerPool, WorkerPoolConfig};

/// Solver configuration.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// CDCL configuration.
    pub cdcl: CdclConfig,
    /// Worker pool configuration.
    pub worker: WorkerPoolConfig,
    /// Timeout in seconds (None = no timeout).
    pub timeout_secs: Option<u64>,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            cdcl: CdclConfig::default(),
            worker: WorkerPoolConfig::default(),
            timeout_secs: None,
        }
    }
}

/// The main Satellite solver.
pub struct Solver {
    /// Next variable ID to allocate.
    next_var_id: VarId,
    /// Variable definitions.
    variables: Vec<(VarId, VarType, Option<String>)>,
    /// Clauses (in DIMACS format).
    clauses: Vec<Vec<i64>>,
    /// Configuration.
    config: SolverConfig,
}

impl Solver {
    /// Creates a new solver with default configuration.
    pub fn new() -> Self {
        Self::with_config(SolverConfig::default())
    }

    /// Creates a new solver with custom configuration.
    pub fn with_config(config: SolverConfig) -> Self {
        Self {
            next_var_id: 0,
            variables: Vec::new(),
            clauses: Vec::new(),
            config,
        }
    }

    /// Creates a new boolean variable.
    pub fn bool_var(&mut self) -> BoolVar {
        self.bool_var_named(None)
    }

    /// Creates a new named boolean variable.
    pub fn bool_var_named(&mut self, name: Option<&str>) -> BoolVar {
        let id = self.alloc_var(VarType::Bool, name);
        BoolVar::new(id)
    }

    /// Creates a new batch variable.
    pub fn batch_var(&mut self, dim: usize) -> Batch {
        self.batch_var_named(dim, None)
    }

    /// Creates a new named batch variable.
    pub fn batch_var_named(&mut self, dim: usize, name: Option<&str>) -> Batch {
        let id = self.alloc_var(VarType::Batch { dim }, name);
        // Allocate additional IDs for the batch elements
        for _ in 1..dim {
            self.next_var_id += 1;
        }
        Batch::new(id, dim)
    }

    /// Creates a new integer variable.
    pub fn int_var(&mut self, bits: usize) -> IntVar {
        self.int_var_named(bits, None)
    }

    /// Creates a new named integer variable.
    pub fn int_var_named(&mut self, bits: usize, name: Option<&str>) -> IntVar {
        let id = self.alloc_var(VarType::Int { bits }, name);
        for _ in 1..bits {
            self.next_var_id += 1;
        }
        IntVar::new(id, bits)
    }

    /// Creates a new vector variable.
    pub fn vec_var(&mut self, inner_dim: usize, outer_dim: usize) -> VecVar {
        let id = self.alloc_var(
            VarType::Vec {
                inner_dim,
                outer_dim,
            },
            None,
        );
        let total = inner_dim * outer_dim;
        for _ in 1..total {
            self.next_var_id += 1;
        }
        VecVar::new(id, inner_dim, outer_dim)
    }

    /// Creates a new float variable.
    pub fn float_var(&mut self, precision: usize) -> FloatVar {
        let id = self.alloc_var(VarType::Float { precision }, None);
        let var = FloatVar::new(id, precision);
        for _ in 1..var.total_vars() {
            self.next_var_id += 1;
        }
        var
    }

    fn alloc_var(&mut self, var_type: VarType, name: Option<&str>) -> VarId {
        let id = self.next_var_id;
        self.next_var_id += 1;
        self.variables.push((id, var_type, name.map(String::from)));
        id
    }

    /// Adds a clause (disjunction of literals).
    pub fn add_clause(&mut self, literals: impl IntoIterator<Item = i64>) {
        self.clauses.push(literals.into_iter().collect());
    }

    /// Adds a constraint.
    pub fn add_constraint(&mut self, constraint: Constraint) {
        for clause in constraint.to_clauses() {
            self.clauses.push(clause);
        }
    }

    /// Solves the problem.
    pub fn solve(&mut self) -> Result<SatResult> {
        let problem = self.build_problem();
        let mut cdcl = CdclSolver::with_config(&problem, self.config.cdcl.clone());
        cdcl.solve()
    }

    fn build_problem(&self) -> AdvancedCnf {
        use satellite_format::advanced_cnf::{Clause, ClauseType, VariableDef};

        let variables: Vec<VariableDef> = self
            .variables
            .iter()
            .map(|(id, var_type, name)| VariableDef {
                id: *id,
                var_type: var_type.clone(),
                name: name.clone(),
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

    /// Returns the number of variables.
    pub fn num_vars(&self) -> usize {
        self.next_var_id as usize
    }

    /// Returns the number of clauses.
    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}
