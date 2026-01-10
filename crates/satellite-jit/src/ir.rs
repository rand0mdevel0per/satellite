//! Intermediate Representation for JIT compilation.

/// A constraint expression that can be compiled to LLVM IR.
#[derive(Debug, Clone)]
pub enum JitConstraint {
    /// A generic boolean variable (0-based index).
    Var(usize),
    /// A literal (1-based index with sign).
    Lit(i64),
    /// Logical AND.
    And(Vec<JitConstraint>),
    /// Logical OR.
    Or(Vec<JitConstraint>),
    /// Logical NOT.
    Not(Box<JitConstraint>),
    /// Logical XOR.
    Xor(Vec<JitConstraint>),
    /// Arithmetic Sum == K.
    SumEq { args: Vec<JitConstraint>, k: i64 },
    /// AtMostK (Cardinality <= K).
    AtMostK { args: Vec<JitConstraint>, k: usize },
    /// AtLeastK (Cardinality >= K).
    AtLeastK { args: Vec<JitConstraint>, k: usize },
}
