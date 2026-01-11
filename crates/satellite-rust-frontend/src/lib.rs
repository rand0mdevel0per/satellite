use satellite_base::ffi::*;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::process::Command;
use std::ptr;
use std::fs;
use std::path::Path;

#[unsafe(no_mangle)]
pub extern "C" fn frontend_get_meta() -> *const FrontendMeta {
    static NAME: &[u8] = b"rust\0";
    static VERSION: &[u8] = b"0.1.0\0";
    static EXTENSIONS: &[u8] = b"rs\0";

    static META: FrontendMeta = FrontendMeta {
        name: NAME.as_ptr() as *const c_char,
        version: VERSION.as_ptr() as *const c_char,
        file_extensions: EXTENSIONS.as_ptr() as *const c_char,
    };
    &META
}

#[unsafe(no_mangle)]
pub extern "C" fn frontend_compile(source_path: *const c_char) -> CompileResult {
    let source_path_str = unsafe { CStr::from_ptr(source_path).to_string_lossy() };
    
    // Generate temp file path for LLVM IR
    let mut temp_path = std::env::temp_dir();
    let unique_id = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    temp_path.push(format!("satellite_rust_{}.ll", unique_id));
    
    // Invoke rustc
    // command: rustc --emit=llvm-ir <source> -o <temp_path> --crate-type=lib
    let output = Command::new("rustc")
        .arg("--emit=llvm-ir")
        .arg(source_path_str.as_ref())
        .arg("-o")
        .arg(&temp_path)
        .arg("--crate-type=lib")
        .output();

    match output {
        Ok(out) if out.status.success() => {
            // Read the generated .ll file
            // rustc might output <temp_path>.ll if we didn't specify extension correctly or it appends it?
            // If we say -o path.ll, it usually uses it.
            // But sometimes it creates path.bc or similar.
            // Check if file exists.
            
            let content = match fs::read_to_string(&temp_path) {
                Ok(c) => c,
                Err(_) => {
                    // Try with implicit extension?
                     return error_result("Failed to read output file");
                }
            };
            
            // Cleanup
            let _ = fs::remove_file(&temp_path);
            
            let c_string = CString::new(content).unwrap_or_default();
            let ptr = c_string.into_raw();

            CompileResult {
                ir_content: ptr,
                ir_len: unsafe { CStr::from_ptr(ptr).to_bytes().len() },
                success: 0,
                error_msg: ptr::null(),
            }
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            error_result(&format!("rustc failed: {}", stderr))
        }
        Err(e) => {
             error_result(&format!("Failed to execute rustc: {}", e))
        }
    }
}

fn error_result(msg: &str) -> CompileResult {
    let c_string = CString::new(msg).unwrap_or_default();
    CompileResult {
        ir_content: ptr::null(),
        ir_len: 0,
        success: 1,
        error_msg: c_string.into_raw(),
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

// Wrapper implementation
unsafe extern "C" fn rust_wrapper(
    input: *const SatInput,
    output: *mut SatOutput,
    jit_fn: *const (),
) -> bool {
    if jit_fn.is_null() {
        return false;
    }
    // Cast generic jit_fn to expected signature
    // Since we compile Rust code that SHOULD utilize SatInput/SatOutput directly (C ABI compatible),
    // the signature should be: extern "C" fn(*const SatInput, *mut SatOutput) -> bool
    
    let func: unsafe extern "C" fn(*const SatInput, *mut SatOutput) -> bool = unsafe { std::mem::transmute(jit_fn) };
    unsafe { func(input, output) }
}

#[unsafe(no_mangle)]
pub extern "C" fn frontend_get_wrapper(constraint_name: *const c_char) -> WrapperInfo {
    let name = unsafe { CStr::from_ptr(constraint_name).to_owned() };
    
    WrapperInfo {
        wrapper_fn_ptr: rust_wrapper as *const (),
        fn_name: name.into_raw(),
    }
}
