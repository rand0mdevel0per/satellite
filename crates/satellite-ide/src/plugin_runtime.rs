//! Plugin runtime for Satellite IDE using Wasmtime.

use anyhow::Result;
use std::sync::{Arc, Mutex};
use wasmtime::*;

/// Context for a running plugin.
pub struct PluginContext {
    /// Plugin name/ID.
    pub id: String,
    /// Memory export from the WASM module.
    pub memory: Option<Memory>,
    /// WASI context (if we want to support WASI later).
    pub wasi: Option<()>,
}

/// The plugin runtime environment.
pub struct PluginRuntime {
    engine: Engine,
    linker: Linker<PluginContext>,
    store: Arc<Mutex<Store<PluginContext>>>,
}

impl PluginRuntime {
    /// Creates a new plugin runtime.
    pub fn new() -> Result<Self> {
        let engine = Engine::default();
        let mut linker = Linker::new(&engine);

        // =========================================================================
        // Host Functions (Syscalls)
        // =========================================================================

        // satellite_log(level: i32, msg_ptr: i32, msg_len: i32)
        linker.func_wrap(
            "env",
            "satellite_log",
            |caller: Caller<'_, PluginContext>, level: i32, ptr: i32, len: i32| {
                let mem = match caller.data().memory {
                    Some(m) => m,
                    None => return,
                };
                
                let data = match mem.data(&caller).get(ptr as usize..(ptr + len) as usize) {
                    Some(d) => d,
                    None => return,
                };
                
                let msg = String::from_utf8_lossy(data);
                match level {
                    0 => tracing::debug!("[Plugin {}] {}", caller.data().id, msg),
                    1 => tracing::info!("[Plugin {}] {}", caller.data().id, msg),
                    2 => tracing::warn!("[Plugin {}] {}", caller.data().id, msg),
                    _ => tracing::error!("[Plugin {}] {}", caller.data().id, msg),
                }
            },
        )?;

        // satellite_solve_sat(cnf_ptr: i32, cnf_len: i32) -> i32
        // Returns 1 for SAT, 0 for UNSAT, -1 for Error
        linker.func_wrap(
            "env",
            "satellite_solve_sat",
            |caller: Caller<'_, PluginContext>, ptr: i32, len: i32| -> i32 {
                 let mem = match caller.data().memory {
                    Some(m) => m,
                    None => return -1,
                };
                
                let data = match mem.data(&caller).get(ptr as usize..(ptr + len) as usize) {
                    Some(d) => d,
                    None => return -1,
                };
                
                // Parse CNF from string
                let content = String::from_utf8_lossy(data);
                
                // TODO: Invoke actual solver here. For now, mock it.
                // In real impl, we would decode DIMACS/JSON and run solver.
                if content.contains("UNSAT") { 0 } else { 1 }
            },
        )?;

        let store = Store::new(&engine, PluginContext {
            id: "system".to_string(),
            memory: None,
            wasi: None,
        });

        Ok(Self {
            engine,
            linker,
            store: Arc::new(Mutex::new(store)),
        })
    }

    /// Loads a WASM plugin.
    pub fn load_plugin(&self, id: &str, wasm_bytes: &[u8]) -> Result<()> {
        let module = Module::new(&self.engine, wasm_bytes)?;
        
        // Create a fresh store for this plugin
        let mut store = Store::new(&self.engine, PluginContext {
            id: id.to_string(),
            memory: None,
            wasi: None,
        });

        let instance = self.linker.instantiate(&mut store, &module)?;
        
        // Grab exported memory if available
        let memory = instance.get_memory(&mut store, "memory");
        store.data_mut().memory = memory;

        // Call _start or init if present
        if let Ok(func) = instance.get_typed_func::<(), ()>(&mut store, "_start") {
            let _ = func.call(&mut store, ());
        }

        Ok(())
    }
}
