//! C ABI type definitions for language frontends.
//!
//! Frontends are dynamically loaded libraries that provide:
//! - Source → LLVM IR compilation
//! - Type conversion wrappers for CDCL integration
//! - LSP integration for function validation

use std::ffi::c_char;
use std::os::raw::c_int;

/// Metadata about a frontend.
#[repr(C)]
pub struct FrontendMeta {
    /// Frontend name (e.g., "rust", "python", "cpp").
    pub name: *const c_char,
    /// Frontend version.
    pub version: *const c_char,
    /// Comma-separated file extensions (e.g., ".rs,.rlib").
    pub file_extensions: *const c_char,
}

/// Result of compiling source to LLVM IR.
#[repr(C)]
pub struct CompileResult {
    /// LLVM IR content as a string.
    pub ir_content: *const c_char,
    /// Length of the IR content.
    pub ir_len: usize,
    /// 0 = success, non-zero = error code.
    pub success: c_int,
    /// Error message if success != 0, NULL otherwise.
    pub error_msg: *const c_char,
}

/// Wrapper function info for type conversion.
#[repr(C)]
pub struct WrapperInfo {
    /// Function pointer: fn(*const SatInput, *mut SatOutput) -> bool.
    pub wrapper_fn_ptr: *const (),
    /// Function name for debugging/profiling.
    pub fn_name: *const c_char,
}

/// Result of LSP function validation.
#[repr(C)]
pub struct ValidationResult {
    /// 1 = valid, 0 = invalid.
    pub valid: c_int,
    /// Error message if invalid, NULL otherwise.
    pub error_msg: *const c_char,
    /// Detected function signature.
    pub signature: *const c_char,
}

/// Input data passed to constraint functions.
#[repr(C)]
pub struct SatInput {
    /// Boolean assignments array.
    pub assignments: *const i8,
    /// Number of variables.
    pub num_vars: usize,
    /// Batch data pointer (for batch variables).
    pub batch_data: *const u8,
    /// Batch data length.
    pub batch_len: usize,
    /// Vec data pointer (for vec variables).
    pub vec_data: *const u8,
    /// Vec data length.
    pub vec_len: usize,
}

/// Output data from constraint functions.
#[repr(C)]
pub struct SatOutput {
    /// Constraint satisfied (1=true, 0=false, -1=unknown).
    pub satisfied: c_int,
    /// Propagated literals (array of literal IDs).
    pub propagated: *mut i64,
    /// Number of propagated literals.
    pub propagated_count: usize,
    /// Conflict clause (if any).
    pub conflict: *mut i64,
    /// Conflict clause length.
    pub conflict_len: usize,
}

impl Default for SatOutput {
    fn default() -> Self {
        Self {
            satisfied: -1,
            propagated: std::ptr::null_mut(),
            propagated_count: 0,
            conflict: std::ptr::null_mut(),
            conflict_len: 0,
        }
    }
}

// =============================================================================
// Function pointer types for FFI
// =============================================================================

/// Type for `frontend_get_meta` function.
pub type FrontendGetMetaFn = unsafe extern "C" fn() -> *const FrontendMeta;

/// Type for `frontend_init` function.
pub type FrontendInitFn = unsafe extern "C" fn(config: *const ()) -> c_int;

/// Type for `frontend_shutdown` function.
pub type FrontendShutdownFn = unsafe extern "C" fn();

/// Type for `frontend_compile` function.
pub type FrontendCompileFn = unsafe extern "C" fn(source_path: *const c_char) -> CompileResult;

/// Type for `frontend_free_result` function.
pub type FrontendFreeResultFn = unsafe extern "C" fn(result: *mut CompileResult);

/// Type for `frontend_get_wrapper` function.
pub type FrontendGetWrapperFn = unsafe extern "C" fn(constraint_name: *const c_char) -> WrapperInfo;

/// Type for `frontend_lsp_validate_function` function.
pub type FrontendLspValidateFn = unsafe extern "C" fn(
    module_name: *const c_char,
    function_name: *const c_char,
) -> ValidationResult;

/// Type for `frontend_lsp_hover` function.
pub type FrontendLspHoverFn =
    unsafe extern "C" fn(uri: *const c_char, line: c_int, col: c_int) -> *mut c_char;

/// Type for `frontend_lsp_complete` function.
pub type FrontendLspCompleteFn =
    unsafe extern "C" fn(uri: *const c_char, line: c_int, col: c_int) -> *mut c_char;

/// Type for `frontend_lsp_diagnostics` function.
pub type FrontendLspDiagnosticsFn =
    unsafe extern "C" fn(uri: *const c_char, content: *const c_char) -> *mut c_char;

/// Type for `frontend_lsp_free_string` function.
pub type FrontendLspFreeStringFn = unsafe extern "C" fn(s: *mut c_char);
