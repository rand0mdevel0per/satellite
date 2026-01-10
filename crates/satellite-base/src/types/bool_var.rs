//! Single boolean variable.

use serde::{Deserialize, Serialize};
use super::VarId;

/// A single boolean variable in the solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BoolVar {
    /// Unique identifier.
    id: VarId,
}

impl BoolVar {
    /// Creates a new boolean variable with the given ID.
    #[must_use]
    pub const fn new(id: VarId) -> Self {
        Self { id }
    }

    /// Returns the variable ID.
    #[must_use]
    pub const fn id(&self) -> VarId {
        self.id
    }

    /// Creates a negated literal.
    #[must_use]
    pub const fn not(&self) -> Literal {
        Literal { var: *self, negated: true }
    }

    /// Creates a positive literal.
    #[must_use]
    pub const fn pos(&self) -> Literal {
        Literal { var: *self, negated: false }
    }
}

/// A literal is a variable with optional negation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Literal {
    /// The underlying variable.
    pub var: BoolVar,
    /// Whether this literal is negated.
    pub negated: bool,
}

impl Literal {
    /// Negates this literal.
    #[must_use]
    pub const fn negate(&self) -> Self {
        Self {
            var: self.var,
            negated: !self.negated,
        }
    }

    /// Converts to DIMACS format (positive = var+1, negative = -(var+1)).
    #[must_use]
    pub fn to_dimacs(&self) -> i64 {
        let base = (self.var.id() + 1) as i64;
        if self.negated { -base } else { base }
    }
}

impl std::ops::Not for BoolVar {
    type Output = Literal;

    fn not(self) -> Self::Output {
        Literal { var: self, negated: true }
    }
}

impl std::ops::Not for Literal {
    type Output = Literal;

    fn not(self) -> Self::Output {
        self.negate()
    }
}
