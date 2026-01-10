//! LLVM JIT compiler for ABI-OP constraints.

use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::OptimizationLevel;
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

    /// Compiles LLVM IR from a string.
    pub fn compile_ir(&self, ir: &str, name: &str) -> Result<CompiledModule> {
        let module = self
            .context
            .create_module_from_ir(inkwell::memory_buffer::MemoryBuffer::create_from_memory_range(
                ir.as_bytes(),
                name,
            ))
            .map_err(|e| Error::CompilationError(e.to_string()))?;

        self.compile_module(module, name)
    }

    /// Compiles a bitcode file.
    pub fn compile_bitcode(&self, path: &Path, name: &str) -> Result<CompiledModule> {
        let module = Module::parse_bitcode_from_path(path, &self.context)
            .map_err(|e| Error::CompilationError(e.to_string()))?;

        self.compile_module(module, name)
    }

    fn compile_module(&self, module: Module<'_>, name: &str) -> Result<CompiledModule> {
        // Run optimization passes
        self.optimize(&module);

        // Create execution engine
        let _execution_engine = module
            .create_jit_execution_engine(self.config.opt_level.into())
            .map_err(|e| Error::CompilationError(e.to_string()))?;

        Ok(CompiledModule {
            name: name.to_string(),
            // Note: ExecutionEngine owns the module
        })
    }

    fn optimize(&self, _module: &Module<'_>) {
        // TODO: Add optimization passes using PassManager
        // For now, rely on JIT's built-in optimization
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
