//! LLVM JIT compiler for ABI-OP constraints.

use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::Module;
use satellite_base::{Error, Result};
use std::path::Path;

/// Configuration for JIT compilation.
#[derive(Debug, Clone)]
pub struct JitConfig {
    /// Optimization level.
    pub opt_level: OptLevel,
    /// Whether to enable truth table optimization.
    pub truth_table_opt: bool,
    /// Maximum input dimension for truth table optimization.
    pub max_truth_table_dim: usize,
}

/// Optimization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    None,
    Less,
    Default,
    Aggressive,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            opt_level: OptLevel::Aggressive,
            truth_table_opt: true,
            max_truth_table_dim: 16, // 2^16 = 65536 entries
        }
    }
}

impl From<OptLevel> for OptimizationLevel {
    fn from(level: OptLevel) -> Self {
        match level {
            OptLevel::None => OptimizationLevel::None,
            OptLevel::Less => OptimizationLevel::Less,
            OptLevel::Default => OptimizationLevel::Default,
            OptLevel::Aggressive => OptimizationLevel::Aggressive,
        }
    }
}

/// JIT compiler for ABI-OP constraints.
pub struct JitCompiler {
    context: Context,
    config: JitConfig,
}

impl JitCompiler {
    /// Creates a new JIT compiler.
    pub fn new() -> Self {
        Self::with_config(JitConfig::default())
    }

    /// Creates a new JIT compiler with custom config.
    pub fn with_config(config: JitConfig) -> Self {
        Self {
            context: Context::create(),
            config,
        }
    }

    /// Compiles a constraint to LLVM IR.
    pub fn compile_constraint(&self, constraint: &crate::ir::JitConstraint, name: &str) -> Result<CompiledModule> {
        let module = self.context.create_module(name);
        let builder = self.context.create_builder();

        // fn check(assignments: *const i8) -> bool (i1)
        let i8_type = self.context.i8_type();
        let i8_ptr_type = i8_type.ptr_type(inkwell::AddressSpace::default());
        let bool_type = self.context.bool_type();
        
        let fn_type = bool_type.fn_type(&[i8_ptr_type.into()], false);
        let function = module.add_function("check", fn_type, None);
        let entry_bb = self.context.append_basic_block(function, "entry");
        
        builder.position_at_end(entry_bb);
        
        let assignments_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        
        let result = self.build_expr(&builder, constraint, assignments_ptr)?;
        
        builder.build_return(Some(&result)).map_err(|e| Error::CompilationError(format!("Build return failed: {:?}", e)))?;
        
        if function.verify(true) {
             // Create execution engine
             let _execution_engine = module
                 .create_jit_execution_engine(self.config.opt_level.into())
                 .map_err(|e| Error::CompilationError(e.to_string()))?;

             Ok(CompiledModule {
                 name: name.to_string(),
             })
        } else {
             // module.print_to_stderr();
             Err(Error::CompilationError("Invalid LLVM IR generated".to_string()))
        }
    }
    
    fn build_expr<'a>(
        &'a self, 
        builder: &inkwell::builder::Builder<'a>, 
        expr: &crate::ir::JitConstraint, 
        assignments: inkwell::values::PointerValue<'a>
    ) -> Result<inkwell::values::IntValue<'a>> {
        use crate::ir::JitConstraint::*;
        let bool_type = self.context.bool_type();
        let i8_type = self.context.i8_type();
        let i64_type = self.context.i64_type();
        
        match expr {
             Var(idx) => {
                 let idx_val = i64_type.const_int(*idx as u64, false);
                 let ptr = unsafe { builder.build_gep(i8_type, assignments, &[idx_val], "var_ptr").map_err(|e| Error::CompilationError(format!("GEP failed: {:?}", e)))? };
                 let val = builder.build_load(i8_type, ptr, "val").map_err(|e| Error::CompilationError(format!("Load failed: {:?}", e)))?.into_int_value();
                 let one = i8_type.const_int(1, false);
                 Ok(builder.build_int_compare(inkwell::IntPredicate::EQ, val, one, "is_true").map_err(|e| Error::CompilationError(format!("Cmp failed: {:?}", e)))?)
             },
             Lit(lit) => {
                 let var_idx = (lit.abs() - 1) as u64;
                 let idx_val = i64_type.const_int(var_idx, false);
                 let ptr = unsafe { builder.build_gep(i8_type, assignments, &[idx_val], "lit_ptr").map_err(|e| Error::CompilationError(format!("GEP failed: {:?}", e)))? };
                 let val = builder.build_load(i8_type, ptr, "val").map_err(|e| Error::CompilationError(format!("Load failed: {:?}", e)))?.into_int_value();
                 
                 let target = if *lit > 0 { 1 } else { -1 };
                 let target_val = i8_type.const_int(target as u64, true); 
                 
                 Ok(builder.build_int_compare(inkwell::IntPredicate::EQ, val, target_val, "is_lit_true").map_err(|e| Error::CompilationError(format!("Cmp failed: {:?}", e)))?)
             },
             And(args) => {
                 let mut current = bool_type.const_int(1, false);
                 for arg in args {
                     let val = self.build_expr(builder, arg, assignments)?;
                     current = builder.build_and(current, val, "and_acc").map_err(|e| Error::CompilationError(format!("And failed: {:?}", e)))?;
                 }
                 Ok(current)
             },
             Or(args) => {
                 let mut current = bool_type.const_int(0, false);
                 for arg in args {
                     let val = self.build_expr(builder, arg, assignments)?;
                     current = builder.build_or(current, val, "or_acc").map_err(|e| Error::CompilationError(format!("Or failed: {:?}", e)))?;
                 }
                 Ok(current)
             },
             Not(arg) => {
                 let val = self.build_expr(builder, arg, assignments)?;
                 Ok(builder.build_not(val, "not_res").map_err(|e| Error::CompilationError(format!("Not failed: {:?}", e)))?)
             },
             Xor(args) => {
                 let mut current = bool_type.const_int(0, false);
                 for arg in args {
                     let val = self.build_expr(builder, arg, assignments)?;
                     current = builder.build_xor(current, val, "xor_acc").map_err(|e| Error::CompilationError(format!("Xor failed: {:?}", e)))?;
                 }
                 Ok(current)
             },
             AtMostK { args, k } => {
                 let mut sum = self.context.i32_type().const_int(0, false);
                 for arg in args {
                     let val = self.build_expr(builder, arg, assignments)?;
                     let val_i32 = builder.build_int_z_extend(val, self.context.i32_type(), "zext").map_err(|e| Error::CompilationError(format!("Zext failed: {:?}", e)))?;
                     sum = builder.build_int_add(sum, val_i32, "sum_acc").map_err(|e| Error::CompilationError(format!("Add failed: {:?}", e)))?;
                 }
                 let k_val = self.context.i32_type().const_int(*k as u64, false);
                 Ok(builder.build_int_compare(inkwell::IntPredicate::ULE, sum, k_val, "at_most_k").map_err(|e| Error::CompilationError(format!("Cmp failed: {:?}", e)))?)
             },
             _ => Err(Error::CompilationError("Unsupported expression type".to_string())),
        }
    }

    /// Returns the LLVM version.
    pub fn llvm_version() -> &'static str {
        "21.0.0"
    }
}

impl Default for JitCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// A compiled module.
pub struct CompiledModule {
    name: String,
}

impl CompiledModule {
    /// Returns the module name.
    pub fn name(&self) -> &str {
        &self.name
    }
}
