//! BitVec type for register-like operations.
//!
//! BitVec represents a fixed-width bitvector for symbolic execution,
//! supporting standard bitwise and arithmetic operations.

use super::{VarId, Batch};

/// A fixed-width bitvector variable (like a CPU register).
#[derive(Debug, Clone)]
pub struct BitVec {
    /// The underlying batch storing the bits.
    inner: Batch,
    /// The width in bits.
    width: usize,
}

impl BitVec {
    /// Creates a new BitVec from a batch.
    pub fn new(batch: Batch) -> Self {
        let width = batch.dim();
        Self { inner: batch, width }
    }

    /// Creates a BitVec with a specific width.
    pub fn with_width(start_id: VarId, width: usize) -> Self {
        Self {
            inner: Batch::new(start_id, width),
            width,
        }
    }

    /// Returns the width in bits.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the starting variable ID.
    pub fn id(&self) -> VarId {
        self.inner.id()
    }

    /// Gets the literal for bit at index (0 = LSB).
    pub fn bit(&self, index: usize) -> i64 {
        assert!(index < self.width, "Bit index out of bounds");
        self.inner.lit(index)
    }

    /// Returns the underlying batch.
    pub fn as_batch(&self) -> &Batch {
        &self.inner
    }

    /// Creates XOR constraints: result = self ^ other.
    /// Returns clauses for CNF encoding.
    pub fn xor(&self, other: &BitVec, result: &BitVec) -> Vec<Vec<i64>> {
        assert_eq!(self.width, other.width);
        assert_eq!(self.width, result.width);

        let mut clauses = Vec::new();
        for i in 0..self.width {
            let a = self.bit(i);
            let b = other.bit(i);
            let r = result.bit(i);

            // XOR encoding: r = a XOR b
            // Clauses: (¬a ∨ ¬b ∨ ¬r), (a ∨ b ∨ ¬r), (a ∨ ¬b ∨ r), (¬a ∨ b ∨ r)
            clauses.push(vec![-a, -b, -r]);
            clauses.push(vec![a, b, -r]);
            clauses.push(vec![a, -b, r]);
            clauses.push(vec![-a, b, r]);
        }
        clauses
    }

    /// Creates AND constraints: result = self & other.
    pub fn and(&self, other: &BitVec, result: &BitVec) -> Vec<Vec<i64>> {
        assert_eq!(self.width, other.width);
        assert_eq!(self.width, result.width);

        let mut clauses = Vec::new();
        for i in 0..self.width {
            let a = self.bit(i);
            let b = other.bit(i);
            let r = result.bit(i);

            // AND encoding: r = a AND b
            // Clauses: (¬a ∨ ¬b ∨ r), (a ∨ ¬r), (b ∨ ¬r)
            clauses.push(vec![-a, -b, r]);
            clauses.push(vec![a, -r]);
            clauses.push(vec![b, -r]);
        }
        clauses
    }

    /// Creates OR constraints: result = self | other.
    pub fn or(&self, other: &BitVec, result: &BitVec) -> Vec<Vec<i64>> {
        assert_eq!(self.width, other.width);
        assert_eq!(self.width, result.width);

        let mut clauses = Vec::new();
        for i in 0..self.width {
            let a = self.bit(i);
            let b = other.bit(i);
            let r = result.bit(i);

            // OR encoding: r = a OR b
            // Clauses: (a ∨ b ∨ ¬r), (¬a ∨ r), (¬b ∨ r)
            clauses.push(vec![a, b, -r]);
            clauses.push(vec![-a, r]);
            clauses.push(vec![-b, r]);
        }
        clauses
    }

    /// Creates NOT constraints: result = ~self.
    pub fn not(&self, result: &BitVec) -> Vec<Vec<i64>> {
        assert_eq!(self.width, result.width);

        let mut clauses = Vec::new();
        for i in 0..self.width {
            let a = self.bit(i);
            let r = result.bit(i);

            // NOT encoding: r = NOT a
            // Clauses: (a ∨ r), (¬a ∨ ¬r)
            clauses.push(vec![a, r]);
            clauses.push(vec![-a, -r]);
        }
        clauses
    }

    /// Creates left shift constraints: result = self << shift_amount.
    /// Lower bits are filled with zeros.
    pub fn shl(&self, shift_amount: usize, result: &BitVec, zero_lit: i64) -> Vec<Vec<i64>> {
        assert_eq!(self.width, result.width);

        let mut clauses = Vec::new();
        for i in 0..self.width {
            let r = result.bit(i);
            if i < shift_amount {
                // Low bits become 0
                clauses.push(vec![-r]); // Force result bit to false
            } else {
                let a = self.bit(i - shift_amount);
                // r = a (equivalence)
                clauses.push(vec![-a, r]);
                clauses.push(vec![a, -r]);
            }
        }
        let _ = zero_lit; // Reserved for future use
        clauses
    }

    /// Creates right shift constraints: result = self >> shift_amount.
    /// Upper bits are filled with zeros (logical shift).
    pub fn shr(&self, shift_amount: usize, result: &BitVec) -> Vec<Vec<i64>> {
        assert_eq!(self.width, result.width);

        let mut clauses = Vec::new();
        for i in 0..self.width {
            let r = result.bit(i);
            if i + shift_amount >= self.width {
                // High bits become 0
                clauses.push(vec![-r]);
            } else {
                let a = self.bit(i + shift_amount);
                clauses.push(vec![-a, r]);
                clauses.push(vec![a, -r]);
            }
        }
        clauses
    }
}

/// A Word represents a memory cell (wrapper around BitVec with address semantics).
#[derive(Debug, Clone)]
pub struct Word {
    /// The underlying bitvector.
    pub data: BitVec,
    /// Optional symbolic address (if memory-mapped).
    pub address: Option<BitVec>,
}

impl Word {
    /// Creates a new Word with just data.
    pub fn new(data: BitVec) -> Self {
        Self { data, address: None }
    }

    /// Creates a Word with address.
    pub fn with_address(data: BitVec, address: BitVec) -> Self {
        Self { data, address: Some(address) }
    }

    /// Returns the data width.
    pub fn width(&self) -> usize {
        self.data.width()
    }
}

/// A MemoryView represents a slice of symbolic memory.
#[derive(Debug, Clone)]
pub struct MemoryView {
    /// Words in this memory view.
    words: Vec<Word>,
    /// Word width in bits.
    word_width: usize,
}

impl MemoryView {
    /// Creates an empty memory view.
    pub fn new(word_width: usize) -> Self {
        Self {
            words: Vec::new(),
            word_width,
        }
    }

    /// Adds a word to the memory view.
    pub fn push(&mut self, word: Word) {
        assert_eq!(word.width(), self.word_width);
        self.words.push(word);
    }

    /// Returns the number of words.
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Gets a word by index.
    pub fn get(&self, index: usize) -> Option<&Word> {
        self.words.get(index)
    }
}
