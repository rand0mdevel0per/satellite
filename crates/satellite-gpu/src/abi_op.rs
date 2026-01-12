//! ABI-OP GPU Execution Module
//!
//! Provides GPU execution for user-defined constraint operations compiled via LLVM.
//! 
//! # Pipeline
//! ```text
//! User Code (Rust/C++) → LLVM IR → GPU PTX → Execute on GPU
//! ```

use crate::{GpuError, Result};
use std::collections::HashMap;

/// Compiled ABI-OP bytecode for GPU execution
#[derive(Debug, Clone)]
pub struct CompiledOp {
    /// Unique identifier (hash of source)
    pub id: u64,
    /// Operation name
    pub name: String,
    /// Input variable count
    pub input_count: usize,
    /// Output variable count
    pub output_count: usize,
    /// GPU kernel bytecode (PTX for CUDA, HSACO for HIP)
    pub kernel_bytecode: Vec<u8>,
    /// Truth table (if small enough for lookup optimization)
    pub truth_table: Option<TruthTable>,
}

/// Pre-computed truth table for small operations
#[derive(Debug, Clone)]
pub struct TruthTable {
    /// Maximum input dimension
    pub input_bits: usize,
    /// Lookup table: input_bits → output value
    pub table: Vec<bool>,
}

impl TruthTable {
    /// Create truth table for operation with given input bits
    pub fn new(input_bits: usize) -> Self {
        let size = 1 << input_bits;
        Self {
            input_bits,
            table: vec![false; size],
        }
    }

    /// Lookup output for given input
    pub fn lookup(&self, input: usize) -> bool {
        self.table.get(input).copied().unwrap_or(false)
    }

    /// Set output for given input
    pub fn set(&mut self, input: usize, output: bool) {
        if input < self.table.len() {
            self.table[input] = output;
        }
    }

    /// Check if truth table can be used (input dimension small enough)
    pub fn is_small_enough(input_bits: usize) -> bool {
        // Limit to 20 bits = 1M entries (1MB for bool table)
        input_bits <= 20
    }
}

/// ABI-OP registry for GPU execution
pub struct AbiOpRegistry {
    /// Compiled operations by ID
    ops: HashMap<u64, CompiledOp>,
    /// Operations by name
    by_name: HashMap<String, u64>,
}

impl AbiOpRegistry {
    pub fn new() -> Self {
        Self {
            ops: HashMap::new(),
            by_name: HashMap::new(),
        }
    }

    /// Register a compiled operation
    pub fn register(&mut self, op: CompiledOp) {
        let id = op.id;
        let name = op.name.clone();
        self.ops.insert(id, op);
        self.by_name.insert(name, id);
    }

    /// Get operation by ID
    pub fn get(&self, id: u64) -> Option<&CompiledOp> {
        self.ops.get(&id)
    }

    /// Get operation by name
    pub fn get_by_name(&self, name: &str) -> Option<&CompiledOp> {
        self.by_name.get(name).and_then(|id| self.ops.get(id))
    }

    /// Remove operation
    pub fn remove(&mut self, id: u64) -> Option<CompiledOp> {
        if let Some(op) = self.ops.remove(&id) {
            self.by_name.remove(&op.name);
            Some(op)
        } else {
            None
        }
    }

    /// List all registered operations
    pub fn list(&self) -> impl Iterator<Item = &CompiledOp> {
        self.ops.values()
    }
}

impl Default for AbiOpRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU ABI-OP executor
pub struct GpuAbiOpExecutor {
    registry: AbiOpRegistry,
    #[allow(dead_code)]
    gpu_available: bool,
}

impl GpuAbiOpExecutor {
    pub fn new() -> Self {
        Self {
            registry: AbiOpRegistry::new(),
            gpu_available: cfg!(any(feature = "cuda", feature = "hip")),
        }
    }

    /// Register an operation
    pub fn register(&mut self, op: CompiledOp) {
        self.registry.register(op);
    }

    /// Execute operation on GPU
    /// 
    /// # Arguments
    /// * `op_id` - Operation ID
    /// * `inputs` - Input variable IDs
    /// * `assignments` - Current variable assignments
    /// 
    /// # Returns
    /// * `Ok(Vec<bool>)` - Output values
    pub fn execute(
        &self,
        op_id: u64,
        inputs: &[i64],
        assignments: &[i8],
    ) -> Result<Vec<bool>> {
        let op = self.registry.get(op_id)
            .ok_or_else(|| GpuError::Error(format!("Operation {} not found", op_id)))?;

        // Try truth table lookup first
        if let Some(ref table) = op.truth_table {
            return self.execute_truth_table(table, inputs, assignments);
        }

        // GPU kernel execution
        self.execute_gpu_kernel(op, inputs, assignments)
    }

    fn execute_truth_table(
        &self,
        table: &TruthTable,
        inputs: &[i64],
        assignments: &[i8],
    ) -> Result<Vec<bool>> {
        // Convert inputs to index
        let mut index: usize = 0;
        for (i, &lit) in inputs.iter().enumerate() {
            let var = lit.unsigned_abs() as usize - 1;
            let val = assignments.get(var).copied().unwrap_or(0);
            
            // Handle unassigned variables (return error or default)
            if val == 0 {
                return Err(GpuError::Error("Unassigned variable in truth table lookup".into()));
            }
            
            let bit_val = if (lit > 0 && val == 1) || (lit < 0 && val == -1) {
                1
            } else {
                0
            };
            
            index |= bit_val << i;
        }

        Ok(vec![table.lookup(index)])
    }

    fn execute_gpu_kernel(
        &self,
        _op: &CompiledOp,
        _inputs: &[i64],
        _assignments: &[i8],
    ) -> Result<Vec<bool>> {
        // TODO: Actual GPU kernel execution
        // This would:
        // 1. Load PTX kernel if not already loaded
        // 2. Copy inputs to GPU
        // 3. Launch kernel
        // 4. Copy outputs back
        
        #[cfg(any(feature = "cuda", feature = "hip"))]
        {
            // Real GPU execution would go here
            Err(GpuError::Error("GPU kernel execution not yet implemented".into()))
        }
        
        #[cfg(not(any(feature = "cuda", feature = "hip")))]
        {
            Err(GpuError::NotAvailable)
        }
    }
}

impl Default for GpuAbiOpExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truth_table() {
        let mut table = TruthTable::new(3);
        
        // Set XOR gate: output = a ^ b ^ c
        for i in 0..8 {
            let a = (i >> 0) & 1;
            let b = (i >> 1) & 1;
            let c = (i >> 2) & 1;
            table.set(i, (a ^ b ^ c) == 1);
        }
        
        assert!(!table.lookup(0b000)); // 0^0^0 = 0
        assert!(table.lookup(0b001));  // 1^0^0 = 1
        assert!(table.lookup(0b010));  // 0^1^0 = 1
        assert!(!table.lookup(0b011)); // 1^1^0 = 0
        assert!(table.lookup(0b100));  // 0^0^1 = 1
        assert!(!table.lookup(0b101)); // 1^0^1 = 0
    }

    #[test]
    fn test_registry() {
        let mut registry = AbiOpRegistry::new();
        
        let op = CompiledOp {
            id: 42,
            name: "test_xor".to_string(),
            input_count: 2,
            output_count: 1,
            kernel_bytecode: vec![],
            truth_table: None,
        };
        
        registry.register(op);
        
        assert!(registry.get(42).is_some());
        assert!(registry.get_by_name("test_xor").is_some());
        assert!(registry.get(999).is_none());
    }
}
