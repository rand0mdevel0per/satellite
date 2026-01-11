//! Satellite IDE - Modern IDE with LSP support.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod lsp;
mod plugin_runtime;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            commands::solve_file,
            commands::parse_file,
            commands::get_diagnostics,
            commands::format_document,
            commands::get_problem_stats,
            commands::get_clause_graph,
            commands::export_result,
            commands::load_plugin,
            commands::lsp_complete,
            commands::lsp_hover,
            commands::lsp_diagnostics,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
