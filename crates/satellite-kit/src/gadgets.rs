//! Circuit gadgets for common operations.
//!
//! This module provides high-level circuit gadgets that generate
//! CNF clauses for common operations like arithmetic and multiplexers.

use satellite_base::types::{BitVec, VarId};

/// A circuit builder that allocates variables and generates clauses.
pub struct CircuitBuilder {
    /// Next variable ID to allocate.
    next_var: VarId,
    /// Generated clauses.
    clauses: Vec<Vec<i64>>,
}

impl CircuitBuilder {
    /// Creates a new circuit builder.
    pub fn new(start_var: VarId) -> Self {
        Self {
            next_var: start_var,
            clauses: Vec::new(),
        }
    }

    /// Allocates a new boolean variable.
    pub fn new_var(&mut self) -> i64 {
        let var = self.next_var;
        self.next_var += 1;
        var as i64 + 1 // 1-indexed
    }

    /// Allocates a new BitVec with the given width.
    pub fn new_bitvec(&mut self, width: usize) -> BitVec {
        let start = self.next_var;
        self.next_var += width as u64;
        BitVec::with_width(start, width)
    }

    /// Returns all generated clauses.
    pub fn get_clauses(&self) -> &[Vec<i64>] {
        &self.clauses
    }

    /// Consumes the builder and returns clauses.
    pub fn into_clauses(self) -> Vec<Vec<i64>> {
        self.clauses
    }

    /// Adds a clause.
    pub fn add_clause(&mut self, clause: Vec<i64>) {
        self.clauses.push(clause);
    }

    /// Adds multiple clauses.
    pub fn add_clauses(&mut self, clauses: Vec<Vec<i64>>) {
        self.clauses.extend(clauses);
    }

    // =========================================================================
    // Circuit Gadgets
    // =========================================================================

    /// Creates an XOR gate: result = a XOR b.
    pub fn add_xor(&mut self, a: &BitVec, b: &BitVec) -> BitVec {
        let result = self.new_bitvec(a.width());
        let clauses = a.xor(b, &result);
        self.add_clauses(clauses);
        result
    }

    /// Creates an AND gate: result = a AND b.
    pub fn add_and(&mut self, a: &BitVec, b: &BitVec) -> BitVec {
        let result = self.new_bitvec(a.width());
        let clauses = a.and(b, &result);
        self.add_clauses(clauses);
        result
    }

    /// Creates an OR gate: result = a OR b.
    pub fn add_or(&mut self, a: &BitVec, b: &BitVec) -> BitVec {
        let result = self.new_bitvec(a.width());
        let clauses = a.or(b, &result);
        self.add_clauses(clauses);
        result
    }

    /// Creates a NOT gate: result = NOT a.
    pub fn add_not(&mut self, a: &BitVec) -> BitVec {
        let result = self.new_bitvec(a.width());
        let clauses = a.not(&result);
        self.add_clauses(clauses);
        result
    }

    /// Creates a left shift: result = a << shift_amount.
    pub fn add_shl(&mut self, a: &BitVec, shift_amount: usize) -> BitVec {
        let result = self.new_bitvec(a.width());
        let zero_lit = self.new_var();
        self.add_clause(vec![-zero_lit]); // Force zero to false
        let clauses = a.shl(shift_amount, &result, zero_lit);
        self.add_clauses(clauses);
        result
    }

    /// Creates a right shift: result = a >> shift_amount.
    pub fn add_shr(&mut self, a: &BitVec, shift_amount: usize) -> BitVec {
        let result = self.new_bitvec(a.width());
        let clauses = a.shr(shift_amount, &result);
        self.add_clauses(clauses);
        result
    }

    /// Creates an ITE (if-then-else / mux) gate: result = cond ? then_val : else_val.
    ///
    /// For each bit: result[i] = (cond AND then[i]) OR (NOT cond AND else[i])
    pub fn add_ite(&mut self, cond: i64, then_val: &BitVec, else_val: &BitVec) -> BitVec {
        assert_eq!(then_val.width(), else_val.width());
        let width = then_val.width();
        let result = self.new_bitvec(width);

        for i in 0..width {
            let t = then_val.bit(i);
            let e = else_val.bit(i);
            let r = result.bit(i);

            // ITE encoding: r = (cond AND t) OR (NOT cond AND e)
            // Equivalent to: (cond → (r ↔ t)) AND (NOT cond → (r ↔ e))
            // Clauses:
            // (¬cond ∨ ¬t ∨ r)   - if cond and t, then r
            // (¬cond ∨ t ∨ ¬r)   - if cond and not t, then not r
            // (cond ∨ ¬e ∨ r)    - if not cond and e, then r
            // (cond ∨ e ∨ ¬r)    - if not cond and not e, then not r
            self.add_clause(vec![-cond, -t, r]);
            self.add_clause(vec![-cond, t, -r]);
            self.add_clause(vec![cond, -e, r]);
            self.add_clause(vec![cond, e, -r]);
        }

        result
    }

    /// Creates a ripple-carry adder: (result, carry_out) = a + b + carry_in.
    pub fn add_add(&mut self, a: &BitVec, b: &BitVec, carry_in: Option<i64>) -> (BitVec, i64) {
        assert_eq!(a.width(), b.width());
        let width = a.width();
        let result = self.new_bitvec(width);

        let mut carry = match carry_in {
            Some(c) => c,
            None => {
                let c = self.new_var();
                self.add_clause(vec![-c]); // No carry in = 0
                c
            }
        };

        for i in 0..width {
            let ai = a.bit(i);
            let bi = b.bit(i);
            let ri = result.bit(i);
            let new_carry = self.new_var();

            // Full adder: sum = a XOR b XOR carry, new_carry = (a AND b) OR (carry AND (a XOR b))
            // Sum encoding (XOR of 3 bits):
            // Using auxiliary variable for a XOR b
            let ab_xor = self.new_var();
            // ab_xor = a XOR b
            self.add_clause(vec![-ai, -bi, -ab_xor]);
            self.add_clause(vec![ai, bi, -ab_xor]);
            self.add_clause(vec![ai, -bi, ab_xor]);
            self.add_clause(vec![-ai, bi, ab_xor]);

            // ri = ab_xor XOR carry
            self.add_clause(vec![-ab_xor, -carry, -ri]);
            self.add_clause(vec![ab_xor, carry, -ri]);
            self.add_clause(vec![ab_xor, -carry, ri]);
            self.add_clause(vec![-ab_xor, carry, ri]);

            // Carry out: new_carry = (a AND b) OR (carry AND ab_xor)
            // = MAJ(a, b, carry) when considering the structure
            // Simplified: at least 2 of {a, b, carry} are true
            self.add_clause(vec![-ai, -bi, new_carry]);       // a AND b → carry
            self.add_clause(vec![-ai, -carry, new_carry]);    // a AND c → carry
            self.add_clause(vec![-bi, -carry, new_carry]);    // b AND c → carry
            self.add_clause(vec![ai, bi, -new_carry]);        // ¬a ∧ ¬b → ¬carry
            self.add_clause(vec![ai, carry, -new_carry]);     // ¬a ∧ ¬c → ¬carry
            self.add_clause(vec![bi, carry, -new_carry]);     // ¬b ∧ ¬c → ¬carry

            carry = new_carry;
        }

        (result, carry)
    }

    /// Creates a subtractor: result = a - b (wrapping).
    pub fn add_sub(&mut self, a: &BitVec, b: &BitVec) -> BitVec {
        // a - b = a + (~b + 1) = a + ~b with carry_in = 1
        let not_b = self.add_not(b);
        
        // Carry in = 1 (for two's complement)
        let carry_in = self.new_var();
        self.add_clause(vec![carry_in]); // Force carry to true
        
        let (result, _) = self.add_add(a, &not_b, Some(carry_in));
        result
    }

    /// Creates an equality comparison: result = (a == b).
    pub fn add_eq(&mut self, a: &BitVec, b: &BitVec) -> i64 {
        assert_eq!(a.width(), b.width());
        let width = a.width();

        // result = AND of (a[i] XNOR b[i]) for all i
        // XNOR = NOT XOR = (a ∧ b) ∨ (¬a ∧ ¬b)
        
        let mut bit_eqs = Vec::with_capacity(width);
        for i in 0..width {
            let ai = a.bit(i);
            let bi = b.bit(i);
            let eq_i = self.new_var();

            // eq_i = ai XNOR bi
            self.add_clause(vec![-ai, -bi, eq_i]);
            self.add_clause(vec![ai, bi, eq_i]);
            self.add_clause(vec![ai, -bi, -eq_i]);
            self.add_clause(vec![-ai, bi, -eq_i]);

            bit_eqs.push(eq_i);
        }

        // Result is AND of all bit equalities
        let result = self.new_var();
        
        // result → all bit_eqs (result implies each eq)
        for &eq in &bit_eqs {
            self.add_clause(vec![-result, eq]);
        }
        
        // all bit_eqs → result
        let mut clause: Vec<i64> = bit_eqs.iter().map(|&e| -e).collect();
        clause.push(result);
        self.add_clause(clause);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_builder_xor() {
        let mut cb = CircuitBuilder::new(0);
        let a = cb.new_bitvec(4);
        let b = cb.new_bitvec(4);
        let _result = cb.add_xor(&a, &b);
        
        // Should have 4 bits * 4 clauses per XOR = 16 clauses
        assert_eq!(cb.get_clauses().len(), 16);
    }

    #[test]
    fn test_circuit_builder_adder() {
        let mut cb = CircuitBuilder::new(0);
        let a = cb.new_bitvec(4);
        let b = cb.new_bitvec(4);
        let (_result, _carry) = cb.add_add(&a, &b, None);
        
        // Adder generates many clauses
        assert!(cb.get_clauses().len() > 0);
    }
}
