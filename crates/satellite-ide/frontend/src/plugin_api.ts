/**
 * Satellite IDE Plugin API
 * 
 * This module exposes a powerful API to plugins, allowing them to:
 * - Register custom views and sidebar items
 * - Integrate with Monaco Editor (completion, hover, definition)
 * - Invoke backend Tauri commands
 * - Subscribe to solver events
 */

import type * as Monaco from 'monaco-editor';
import { invoke } from '@tauri-apps/api/core';
import React from 'react';

// =============================================================================
// Types
// =============================================================================

export interface SolveResult {
    satisfiable: boolean | null;
    model: number[] | null;
    time_ms: number;
    error: string | null;
}

export interface PluginManifest {
    id: string;
    name: string;
    version: string;
    description?: string;
    entry: string; // Path to main JS file
    wasm?: string; // Optional path to WASM backend
}

export interface RegisteredView {
    id: string;
    title: string;
    icon: string;
    component: React.ComponentType<any>;
}

export type SolveEventHandler = (result: SolveResult) => void;
export type EditorEventHandler = (editor: Monaco.editor.IStandaloneCodeEditor) => void;

// =============================================================================
// Plugin API Interface
// =============================================================================

export interface SatelliteAPI {
    // -------------------------------------------------------------------------
    // UI Registration
    // -------------------------------------------------------------------------

    /** Register a custom view panel */
    registerView(id: string, title: string, icon: string, component: React.ComponentType<any>): void;

    /** Register a sidebar item that opens a view */
    registerSidebarItem(id: string, icon: string, viewId: string): void;

    /** Get all registered views */
    getRegisteredViews(): RegisteredView[];

    // -------------------------------------------------------------------------
    // Monaco Editor Integration
    // -------------------------------------------------------------------------

    /** Direct access to Monaco namespace */
    monaco: typeof Monaco | null;

    /** Register a completion provider for a language */
    registerCompletionProvider(
        languageId: string,
        provider: Monaco.languages.CompletionItemProvider
    ): Monaco.IDisposable | null;

    /** Register a hover provider for a language */
    registerHoverProvider(
        languageId: string,
        provider: Monaco.languages.HoverProvider
    ): Monaco.IDisposable | null;

    /** Register a definition provider */
    registerDefinitionProvider(
        languageId: string,
        provider: Monaco.languages.DefinitionProvider
    ): Monaco.IDisposable | null;

    /** Register a custom language */
    registerLanguage(languageId: string, config: Monaco.languages.LanguageConfiguration): void;

    /** Set monarch tokenizer for syntax highlighting */
    setMonarchTokensProvider(languageId: string, languageDef: Monaco.languages.IMonarchLanguage): void;

    /** Get current editor instance */
    getActiveEditor(): Monaco.editor.IStandaloneCodeEditor | null;

    /** Subscribe to editor creation events */
    onEditorCreated(handler: EditorEventHandler): () => void;

    // -------------------------------------------------------------------------
    // Backend Communication
    // -------------------------------------------------------------------------

    /** Invoke a Tauri command */
    invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T>;

    /** Subscribe to solve finished events */
    onSolveFinished(handler: SolveEventHandler): () => void;

    // -------------------------------------------------------------------------
    // Plugin Utilities
    // -------------------------------------------------------------------------

    /** Log a message (appears in IDE console) */
    log(level: 'debug' | 'info' | 'warn' | 'error', message: string): void;

    /** Show a notification toast */
    showNotification(message: string, type?: 'info' | 'success' | 'warning' | 'error'): void;
}

// =============================================================================
// Plugin Manager (Host-side Implementation)
// =============================================================================

class PluginManager implements SatelliteAPI {
    private views: Map<string, RegisteredView> = new Map();
    private sidebarItems: Map<string, { icon: string; viewId: string }> = new Map();
    private solveHandlers: Set<SolveEventHandler> = new Set();
    private editorHandlers: Set<EditorEventHandler> = new Set();
    private activeEditor: Monaco.editor.IStandaloneCodeEditor | null = null;

    public monaco: typeof Monaco | null = null;

    // -------------------------------------------------------------------------
    // UI Registration
    // -------------------------------------------------------------------------

    registerView(id: string, title: string, icon: string, component: React.ComponentType<any>): void {
        this.views.set(id, { id, title, icon, component });
        console.log(`[PluginAPI] Registered view: ${id}`);
    }

    registerSidebarItem(id: string, icon: string, viewId: string): void {
        this.sidebarItems.set(id, { icon, viewId });
        console.log(`[PluginAPI] Registered sidebar item: ${id} -> ${viewId}`);
    }

    getRegisteredViews(): RegisteredView[] {
        return Array.from(this.views.values());
    }

    // -------------------------------------------------------------------------
    // Monaco Editor Integration
    // -------------------------------------------------------------------------

    setMonaco(monaco: typeof Monaco): void {
        this.monaco = monaco;
        console.log('[PluginAPI] Monaco editor instance set');
    }

    setActiveEditor(editor: Monaco.editor.IStandaloneCodeEditor): void {
        this.activeEditor = editor;
        this.editorHandlers.forEach(handler => handler(editor));
    }

    registerCompletionProvider(
        languageId: string,
        provider: Monaco.languages.CompletionItemProvider
    ): Monaco.IDisposable | null {
        if (!this.monaco) {
            console.warn('[PluginAPI] Monaco not initialized');
            return null;
        }
        return this.monaco.languages.registerCompletionItemProvider(languageId, provider);
    }

    registerHoverProvider(
        languageId: string,
        provider: Monaco.languages.HoverProvider
    ): Monaco.IDisposable | null {
        if (!this.monaco) {
            console.warn('[PluginAPI] Monaco not initialized');
            return null;
        }
        return this.monaco.languages.registerHoverProvider(languageId, provider);
    }

    registerDefinitionProvider(
        languageId: string,
        provider: Monaco.languages.DefinitionProvider
    ): Monaco.IDisposable | null {
        if (!this.monaco) {
            console.warn('[PluginAPI] Monaco not initialized');
            return null;
        }
        return this.monaco.languages.registerDefinitionProvider(languageId, provider);
    }

    registerLanguage(languageId: string, config: Monaco.languages.LanguageConfiguration): void {
        if (!this.monaco) {
            console.warn('[PluginAPI] Monaco not initialized');
            return;
        }
        this.monaco.languages.register({ id: languageId });
        this.monaco.languages.setLanguageConfiguration(languageId, config);
    }

    setMonarchTokensProvider(languageId: string, languageDef: Monaco.languages.IMonarchLanguage): void {
        if (!this.monaco) {
            console.warn('[PluginAPI] Monaco not initialized');
            return;
        }
        this.monaco.languages.setMonarchTokensProvider(languageId, languageDef);
    }

    getActiveEditor(): Monaco.editor.IStandaloneCodeEditor | null {
        return this.activeEditor;
    }

    onEditorCreated(handler: EditorEventHandler): () => void {
        this.editorHandlers.add(handler);
        return () => this.editorHandlers.delete(handler);
    }

    // -------------------------------------------------------------------------
    // Backend Communication
    // -------------------------------------------------------------------------

    async invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
        return invoke<T>(cmd, args);
    }

    onSolveFinished(handler: SolveEventHandler): () => void {
        this.solveHandlers.add(handler);
        return () => this.solveHandlers.delete(handler);
    }

    // Internal: called by App when solve completes
    notifySolveFinished(result: SolveResult): void {
        this.solveHandlers.forEach(handler => handler(result));
    }

    // -------------------------------------------------------------------------
    // Plugin Utilities
    // -------------------------------------------------------------------------

    log(level: 'debug' | 'info' | 'warn' | 'error', message: string): void {
        const prefix = `[Plugin]`;
        switch (level) {
            case 'debug': console.debug(prefix, message); break;
            case 'info': console.info(prefix, message); break;
            case 'warn': console.warn(prefix, message); break;
            case 'error': console.error(prefix, message); break;
        }
    }

    showNotification(message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info'): void {
        // TODO: Implement toast notifications
        console.log(`[Notification:${type}] ${message}`);
    }

    // -------------------------------------------------------------------------
    // Plugin Loading
    // -------------------------------------------------------------------------

    async loadPlugin(manifest: PluginManifest): Promise<void> {
        console.log(`[PluginAPI] Loading plugin: ${manifest.name} v${manifest.version}`);

        // Load frontend script
        const script = document.createElement('script');
        script.src = manifest.entry;
        script.onload = () => {
            console.log(`[PluginAPI] Plugin ${manifest.id} loaded successfully`);
        };
        script.onerror = (e) => {
            console.error(`[PluginAPI] Failed to load plugin ${manifest.id}:`, e);
        };
        document.body.appendChild(script);

        // Load WASM backend if present
        if (manifest.wasm) {
            try {
                await invoke('load_plugin', { path: manifest.wasm });
                console.log(`[PluginAPI] WASM backend for ${manifest.id} loaded`);
            } catch (e) {
                console.error(`[PluginAPI] Failed to load WASM for ${manifest.id}:`, e);
            }
        }
    }
}

// =============================================================================
// Global Instance
// =============================================================================

export const pluginManager = new PluginManager();

// Expose to window for plugins
declare global {
    interface Window {
        satellite: SatelliteAPI;
    }
}

if (typeof window !== 'undefined') {
    window.satellite = pluginManager;
}
