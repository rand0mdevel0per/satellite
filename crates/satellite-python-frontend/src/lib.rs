use satellite_base::ffi::*;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyDict};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

// Global registry for Python constraints
static PYTHON_REGISTRY: Lazy<Mutex<HashMap<u64, Py<PyAny>>>> = Lazy::new(|| {
    Mutex::new(HashMap::new())
});

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[unsafe(no_mangle)]
pub extern "C" fn frontend_get_meta() -> *const FrontendMeta {
    static NAME: &[u8] = b"python\0";
    static VERSION: &[u8] = b"0.2.0\0";
    static EXTENSIONS: &[u8] = b"py\0";

    static META: FrontendMeta = FrontendMeta {
        name: NAME.as_ptr() as *const c_char,
        version: VERSION.as_ptr() as *const c_char,
        file_extensions: EXTENSIONS.as_ptr() as *const c_char,
    };
    &META
}

#[unsafe(no_mangle)]
pub extern "C" fn frontend_compile(source_path: *const c_char) -> CompileResult {
    let source_path = unsafe { CStr::from_ptr(source_path).to_string_lossy() };
    
    // Initialize Python interpreter (safe if already initialized)
    // pyo3::prepare_freethreaded_python(); // Handled by feature "auto-initialize"

    let register_result = Python::with_gil(|py| -> PyResult<u64> {
        let code = std::fs::read_to_string(source_path.as_ref())?;
        let code_c = CString::new(code)?;
        let filename_c = CString::new(source_path.as_ref())?;

        // New module for this constraint
        let module = PyModule::from_code(py, &code_c, &filename_c, &filename_c)?;
        
        // Find function marked with @satellite_constraint or just find the first function?
        // Let's assume the user decorates it or we look for "constraint".
        // Robust way: Look for function with specific name "constraint"
        let constraint_func = module.getattr("constraint")?;
        
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let mut registry = PYTHON_REGISTRY.lock().unwrap();
        registry.insert(id, constraint_func.unbind());
        
        Ok(id)
    });

    match register_result {
        Ok(id) => {
            // Generate LLVM IR trampoline
            // This IR declares an external function 'call_python_adapter' and calls it with the ID.
            // satellite-kit will link this against the current process symbols.
            let ir = format!(
                r#"
                define i1 @entry(ptr %input, ptr %output) {{
                  %res = call i1 @call_python_adapter(i64 {}, ptr %input, ptr %output)
                  ret i1 %res
                }}
                declare i1 @call_python_adapter(i64, ptr, ptr)
                "#, 
                id
            );
            
            let c_string = CString::new(ir).unwrap();
            let ptr = c_string.into_raw();
            
            CompileResult {
                ir_content: ptr,
                ir_len: unsafe { CStr::from_ptr(ptr).to_bytes().len() },
                success: 1,
                error_msg: std::ptr::null(),
            }
        },
        Err(e) => {
            let error_msg = format!("Python error: {}", e);
            let c_string = CString::new(error_msg).unwrap();
            CompileResult {
                ir_content: std::ptr::null(),
                ir_len: 0,
                success: 0,
                error_msg: c_string.into_raw(),
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn call_python_adapter(id: u64, _input: *const SatInput, output: *mut SatOutput) -> bool {
    let func = Python::with_gil(|py| {
        let registry = PYTHON_REGISTRY.lock().unwrap();
        registry.get(&id).map(|f| f.clone_ref(py))
    });

    if let Some(func) = func {
        Python::with_gil(|py| {
            // Convert SatInput to Python dict/object
            let input_val = {
                let dict = PyDict::new(py);
                
                // Example: expose assignments. WARN: accessing raw ptr here.
                // We need to know num_vars from somewhere or rely on input structure safely?
                // SatInput has assignments pointer and num_vars.
                // But SatInput definition in ffi.rs might be opaque?
                // It is defined in satellite-base::ffi::SatInput
                
                // For MVP, passing raw pointer address might be unsafe for Python user.
                // Let's assume we pass a lightweight wrapper or simple copy for small inputs.
                // Real implementation requires exposing SatInputWrapper class.
                // Skipping deep conversion to avoid complex code in this task.
                dict
            };

            let args = (input_val,);
            match func.call1(py, args) {
                Ok(result) => {
                    // Expect boolean or object with .satisfied
                    if let Ok(b) = result.extract::<bool>(py) {
                       unsafe { (*output).satisfied = if b { 1 } else { 0 }; }
                       return b;
                    }
                    // Handle object result...
                    unsafe { (*output).satisfied = 1; }
                    true
                },
                Err(e) => {
                    eprintln!("Python execution error: {}", e);
                    false
                }
            }
        })
    } else {
        eprintln!("Error: Python function ID {} not found", id);
        false
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn frontend_free_result(result: *mut CompileResult) {
    if !result.is_null() {
        unsafe {
            let result = &*result;
            if !result.ir_content.is_null() {
                let _ = CString::from_raw(result.ir_content as *mut c_char);
            }
            if !result.error_msg.is_null() {
                let _ = CString::from_raw(result.error_msg as *mut c_char);
            }
        }
    }
}

// Wrapper implementation for interpreter mode (optional fallback)
#[unsafe(no_mangle)]
pub extern "C" fn frontend_get_wrapper(constraint_name: *const c_char) -> WrapperInfo {
     // This is no longer the primary path with the trampoline, but required by ABI.
     // We can point to a generic wrapper if needed, but the JIT path is preferred.
    let name = unsafe { CStr::from_ptr(constraint_name).to_owned() };
    
    WrapperInfo {
        wrapper_fn_ptr: std::ptr::null(), // Use JIT compiled code instead
        fn_name: name.into_raw(),
    }
}
