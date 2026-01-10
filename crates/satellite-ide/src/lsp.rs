//! LSP integration for the IDE.
//!
//! Provides Language Server Protocol support for .sat files.

/// LSP capabilities.
pub struct LspCapabilities {
    pub completion: bool,
    pub hover: bool,
    pub diagnostics: bool,
    pub formatting: bool,
    pub goto_definition: bool,
}

impl Default for LspCapabilities {
    fn default() -> Self {
        Self {
            completion: true,
            hover: true,
            diagnostics: true,
            formatting: true,
            goto_definition: true,
        }
    }
}

/// LSP server state.
pub struct LspServer {
    capabilities: LspCapabilities,
}

impl LspServer {
    /// Creates a new LSP server.
    pub fn new() -> Self {
        Self {
            capabilities: LspCapabilities::default(),
        }
    }

    /// Returns server capabilities.
    pub fn capabilities(&self) -> &LspCapabilities {
        &self.capabilities
    }
}

impl Default for LspServer {
    fn default() -> Self {
        Self::new()
    }
}
