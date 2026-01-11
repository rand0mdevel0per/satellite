//! Frontend manager for loading and managing language frontends.
//!
//! Frontends are dynamically loaded libraries that compile user code to LLVM IR
//! and provide wrappers for CDCL integration.

use crate::Result;
use satellite_base::ffi::*;
use libloading::{Library, Symbol};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::path::Path;

/// A loaded frontend handle.
pub struct FrontendHandle {
    /// The dynamically loaded library.
    _library: Library,
    /// Frontend metadata.
    pub meta: FrontendInfo,
    /// Compile function.
    compile_fn: FrontendCompileFn,
    /// Free result function.
    free_result_fn: FrontendFreeResultFn,
    /// Get wrapper function.
    get_wrapper_fn: FrontendGetWrapperFn,
    /// LSP validate function.
    lsp_validate_fn: Option<FrontendLspValidateFn>,
}

/// Frontend metadata (Rust-friendly).
#[derive(Debug, Clone)]
pub struct FrontendInfo {
    /// Frontend name.
    pub name: String,
    /// Frontend version.
    pub version: String,
    /// Supported file extensions.
    pub extensions: Vec<String>,
}

/// Manages loaded frontends.
pub struct FrontendManager {
    /// Loaded frontends by extension.
    frontends: HashMap<String, FrontendHandle>,
    /// Cache of compiled constraints by hash.
    constraint_cache: HashMap<[u8; 32], PackagedConstraint>,
}

/// A compiled and wrapped constraint ready for CDCL execution.
pub struct PackagedConstraint {
    /// The optimized function pointer from JIT compilation.
    pub jit_fn: *const (),
    /// The wrapper function from frontend.
    pub wrapper_fn: *const (),
    /// Cache key for this constraint.
    pub cache_hash: [u8; 32],
    /// Constraint name for debugging.
    pub name: String,
}

impl FrontendManager {
    /// Creates a new frontend manager.
    pub fn new() -> Self {
        Self {
            frontends: HashMap::new(),
            constraint_cache: HashMap::new(),
        }
    }

    /// Loads a frontend from a dynamic library.
    ///
    /// # Safety
    /// The library must export valid frontend symbols.
    pub unsafe fn load_frontend(&mut self, path: &Path) -> Result<()> {
        // SAFETY: Library::new is unsafe in Rust 2024
        let lib = unsafe {
            Library::new(path).map_err(|e| {
                crate::Error::Internal(format!("Failed to load frontend library: {}", e))
            })?
        };

        // Get metadata and function pointers, storing raw values to avoid borrow issues
        let (
            name,
            version,
            extensions,
            compile_fn,
            free_result_fn,
            get_wrapper_fn,
            lsp_validate_fn,
        ) = unsafe {
            // SAFETY: lib.get is unsafe
            let get_meta: Symbol<FrontendGetMetaFn> = lib
                .get(b"frontend_get_meta")
                .map_err(|e| crate::Error::Internal(format!("Missing frontend_get_meta: {}", e)))?;

            let meta_ptr = get_meta();
            if meta_ptr.is_null() {
                return Err(crate::Error::Internal(
                    "frontend_get_meta returned null".to_string(),
                ));
            }

            // SAFETY: meta_ptr is validated non-null
            let meta = &*meta_ptr;
            let name = CStr::from_ptr(meta.name).to_string_lossy().to_string();
            let version = CStr::from_ptr(meta.version).to_string_lossy().to_string();
            let extensions_str = CStr::from_ptr(meta.file_extensions)
                .to_string_lossy()
                .to_string();
            let extensions: Vec<String> =
                extensions_str.split(',').map(|s| s.to_string()).collect();

            // Get required functions
            let compile_fn: FrontendCompileFn = *lib
                .get(b"frontend_compile")
                .map_err(|e| crate::Error::Internal(format!("Missing frontend_compile: {}", e)))?;

            let free_result_fn: FrontendFreeResultFn =
                *lib.get(b"frontend_free_result").map_err(|e| {
                    crate::Error::Internal(format!("Missing frontend_free_result: {}", e))
                })?;

            let get_wrapper_fn: FrontendGetWrapperFn =
                *lib.get(b"frontend_get_wrapper").map_err(|e| {
                    crate::Error::Internal(format!("Missing frontend_get_wrapper: {}", e))
                })?;

            // Optional LSP functions
            let lsp_validate_fn: Option<FrontendLspValidateFn> =
                lib.get(b"frontend_lsp_validate_function").ok().map(|s| *s);

            (
                name,
                version,
                extensions,
                compile_fn,
                free_result_fn,
                get_wrapper_fn,
                lsp_validate_fn,
            )
        };

        let info = FrontendInfo {
            name: name.clone(),
            version,
            extensions: extensions.clone(),
        };

        let handle = FrontendHandle {
            _library: lib,
            meta: info,
            compile_fn,
            free_result_fn,
            get_wrapper_fn,
            lsp_validate_fn,
        };

        // Register for first extension only (since we move handle)
        if let Some(ext) = extensions.into_iter().next() {
            self.frontends.insert(ext, handle);
        }

        tracing::info!("Loaded frontend '{}' for extensions", name);
        Ok(())
    }

    /// Gets a frontend for a file extension.
    pub fn get_frontend(&self, extension: &str) -> Option<&FrontendHandle> {
        self.frontends.get(extension)
    }

    /// Compiles a source file to a packaged constraint.
    ///
    /// # Safety
    /// The source file must exist and the frontend must be valid.
    pub unsafe fn compile_constraint(&mut self, source_path: &Path) -> Result<PackagedConstraint> {
        let extension = source_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| format!(".{}", e))
            .ok_or_else(|| crate::Error::Internal("No file extension".to_string()))?;

        let frontend = self.frontends.get(&extension).ok_or_else(|| {
            crate::Error::Internal(format!("No frontend for extension: {}", extension))
        })?;

        let path_cstr = CString::new(source_path.to_string_lossy().as_bytes())
            .map_err(|_| crate::Error::Internal("Invalid path".to_string()))?;

        // SAFETY: Frontend compile function is validated at load time
        let result = unsafe { (frontend.compile_fn)(path_cstr.as_ptr()) };

        if result.success != 0 {
            let error_msg = if result.error_msg.is_null() {
                "Unknown compilation error".to_string()
            } else {
                // SAFETY: error_msg is guaranteed non-null here
                unsafe {
                    CStr::from_ptr(result.error_msg)
                        .to_string_lossy()
                        .to_string()
                }
            };
            return Err(crate::Error::CompilationError(error_msg));
        }

        // Get LLVM IR
        let ir_content = if result.ir_content.is_null() {
            return Err(crate::Error::Internal("No IR content".to_string()));
        } else {
            // SAFETY: ir_content and ir_len are validated by the frontend
            unsafe { std::slice::from_raw_parts(result.ir_content as *const u8, result.ir_len) }
        };

        // Compute cache hash
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        ir_content.hash(&mut hasher);
        let hash_value = hasher.finish();
        let mut cache_hash = [0u8; 32];
        cache_hash[..8].copy_from_slice(&hash_value.to_le_bytes());

        // TODO: JIT compile the IR and get function pointer
        // For now, return placeholder
        let packaged = PackagedConstraint {
            jit_fn: std::ptr::null(),
            wrapper_fn: std::ptr::null(),
            cache_hash,
            name: source_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
        };

        // Free the compile result
        // SAFETY: result was allocated by the frontend
        unsafe { (frontend.free_result_fn)(&result as *const _ as *mut _) };

        Ok(packaged)
    }

    /// Validates a function reference via frontend LSP.
    pub fn validate_function(
        &self,
        extension: &str,
        module: &str,
        fn_name: &str,
    ) -> Result<LspValidationResult> {
        let frontend = self.frontends.get(extension).ok_or_else(|| {
            crate::Error::Internal(format!("No frontend for extension: {}", extension))
        })?;

        let lsp_validate = frontend.lsp_validate_fn.ok_or_else(|| {
            crate::Error::Internal("Frontend does not support LSP validation".to_string())
        })?;

        let module_cstr = CString::new(module)
            .map_err(|_| crate::Error::Internal("Invalid module name".to_string()))?;
        let fn_cstr = CString::new(fn_name)
            .map_err(|_| crate::Error::Internal("Invalid function name".to_string()))?;

        let result = unsafe { lsp_validate(module_cstr.as_ptr(), fn_cstr.as_ptr()) };

        Ok(LspValidationResult {
            valid: result.valid != 0,
            error_msg: if result.error_msg.is_null() {
                None
            } else {
                Some(unsafe {
                    CStr::from_ptr(result.error_msg)
                        .to_string_lossy()
                        .to_string()
                })
            },
            signature: if result.signature.is_null() {
                None
            } else {
                Some(unsafe {
                    CStr::from_ptr(result.signature)
                        .to_string_lossy()
                        .to_string()
                })
            },
        })
    }

    /// Lists all loaded frontends.
    pub fn list_frontends(&self) -> Vec<&FrontendInfo> {
        self.frontends.values().map(|h| &h.meta).collect()
    }

    /// Discovers and loads frontends from a directory.
    pub fn discover_frontends(&mut self, dir: &Path) -> Result<()> {
        if !dir.exists() {
            return Ok(());
        }

        tracing::info!("Discovering frontends in {:?}", dir);

        for entry in std::fs::read_dir(dir).map_err(|e| crate::Error::Internal(format!("Failed to read directory: {}", e)))? {
            let entry = entry.map_err(|e| crate::Error::Internal(format!("Failed to read entry: {}", e)))?;
            let path = entry.path();
            
            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                    #[cfg(target_os = "windows")]
                    let is_lib = ext.eq_ignore_ascii_case("dll");
                    #[cfg(not(target_os = "windows"))]
                    let is_lib = ext == "so" || ext == "dylib";

                    if is_lib {
                        tracing::debug!("Attempting to load frontend from {:?}", path);
                        // We suppress errors for individual files to avoid stopping discovery
                        // But log them as warnings
                         match unsafe { self.load_frontend(&path) } {
                             Ok(_) => {},
                             Err(e) => {
                                 tracing::warn!("Failed to load frontend {:?}: {}", path, e);
                             }
                         }
                    }
                }
            }
        }
        Ok(())
    }
}

impl Default for FrontendManager {
    fn default() -> Self {
        Self::new()
    }
}

/// LSP validation result (Rust-friendly).
#[derive(Debug, Clone)]
pub struct LspValidationResult {
    /// Whether the function is valid.
    pub valid: bool,
    /// Error message if invalid.
    pub error_msg: Option<String>,
    /// Detected function signature.
    pub signature: Option<String>,
}
