import { useState, useCallback } from 'react';
import Editor, { OnMount } from '@monaco-editor/react';
import { invoke } from '@tauri-apps/api/core';
import { pluginManager } from './plugin_api';
import './App.css';

// =============================================================================
// Types
// =============================================================================

interface SolveResult {
    satisfiable: boolean | null;
    model: number[] | null;
    time_ms: number;
    error: string | null;
}

interface SolveStats {
    num_vars: number;
    num_clauses: number;
    decisions: number;
    conflicts: number;
    propagations: number;
    learned_clauses: number;
    restarts: number;
    solve_time_ms: number;
}

interface ClauseNode {
    id: number;
    literals: number[];
    is_learned: boolean;
}

interface ClauseGraph {
    nodes: ClauseNode[];
    var_count: number;
}

type FileFormat = 'dimacs' | 'json' | 'sat';
type ViewMode = 'editor' | 'stats' | 'graph';

// =============================================================================
// Sidebar Component
// =============================================================================

function Sidebar({
    activeView,
    onViewChange
}: {
    activeView: ViewMode;
    onViewChange: (v: ViewMode) => void;
}) {
    return (
        <div className="sidebar">
            <div className="sidebar-icons">
                <button
                    className={`sidebar-icon ${activeView === 'editor' ? 'active' : ''}`}
                    onClick={() => onViewChange('editor')}
                    title="Editor"
                >
                    📝
                </button>
                <button
                    className={`sidebar-icon ${activeView === 'stats' ? 'active' : ''}`}
                    onClick={() => onViewChange('stats')}
                    title="Statistics"
                >
                    📊
                </button>
                <button
                    className={`sidebar-icon ${activeView === 'graph' ? 'active' : ''}`}
                    onClick={() => onViewChange('graph')}
                    title="Clause Graph"
                >
                    🕸️
                </button>
            </div>
            <div className="sidebar-bottom">
                <button className="sidebar-icon" title="Settings">⚙️</button>
            </div>
        </div>
    );
}

// =============================================================================
// Stats Panel Component
// =============================================================================

function StatsPanel({ stats }: { stats: SolveStats | null }) {
    if (!stats) {
        return <div className="stats-panel empty">No statistics available. Parse a file first.</div>;
    }

    return (
        <div className="stats-panel">
            <h3>Problem Statistics</h3>
            <div className="stats-grid">
                <div className="stat-item">
                    <span className="stat-label">Variables</span>
                    <span className="stat-value">{stats.num_vars}</span>
                </div>
                <div className="stat-item">
                    <span className="stat-label">Clauses</span>
                    <span className="stat-value">{stats.num_clauses}</span>
                </div>
                <div className="stat-item">
                    <span className="stat-label">Decisions</span>
                    <span className="stat-value">{stats.decisions}</span>
                </div>
                <div className="stat-item">
                    <span className="stat-label">Conflicts</span>
                    <span className="stat-value">{stats.conflicts}</span>
                </div>
                <div className="stat-item">
                    <span className="stat-label">Propagations</span>
                    <span className="stat-value">{stats.propagations}</span>
                </div>
                <div className="stat-item">
                    <span className="stat-label">Learned</span>
                    <span className="stat-value">{stats.learned_clauses}</span>
                </div>
            </div>
        </div>
    );
}

// =============================================================================
// Clause Graph Component (Simple Table View)
// =============================================================================

function ClauseGraphPanel({ graph }: { graph: ClauseGraph | null }) {
    if (!graph) {
        return <div className="graph-panel empty">No graph available. Parse a file first.</div>;
    }

    return (
        <div className="graph-panel">
            <h3>Clause Graph ({graph.var_count} vars, {graph.nodes.length} clauses)</h3>
            <div className="clause-list">
                {graph.nodes.slice(0, 100).map((node) => (
                    <div key={node.id} className={`clause-item ${node.is_learned ? 'learned' : ''}`}>
                        <span className="clause-id">C{node.id}:</span>
                        <span className="clause-lits">
                            {node.literals.map((lit, i) => (
                                <span key={i} className={`literal ${lit > 0 ? 'pos' : 'neg'}`}>
                                    {lit > 0 ? `x${lit}` : `¬x${Math.abs(lit)}`}
                                    {i < node.literals.length - 1 && ' ∨ '}
                                </span>
                            ))}
                        </span>
                    </div>
                ))}
                {graph.nodes.length > 100 && (
                    <div className="clause-item more">...and {graph.nodes.length - 100} more</div>
                )}
            </div>
        </div>
    );
}

// =============================================================================
// Result Panel Component
// =============================================================================

function ResultPanel({ result }: { result: SolveResult | null }) {
    if (!result) {
        return <div className="result-panel empty">Click Solve to see results</div>;
    }

    return (
        <div className="result-panel">
            {result.error ? (
                <div className="error">Error: {result.error}</div>
            ) : (
                <>
                    <div className={`status ${result.satisfiable ? 'sat' : 'unsat'}`}>
                        {result.satisfiable === null
                            ? 'UNKNOWN'
                            : result.satisfiable
                                ? '✅ SATISFIABLE'
                                : '❌ UNSATISFIABLE'}
                    </div>
                    <div className="time">Time: {result.time_ms}ms</div>
                    {result.model && (
                        <div className="model">
                            <h4>Model</h4>
                            <div className="model-vars">
                                {result.model.map((lit, i) => (
                                    <span key={i} className={`var ${lit > 0 ? 'true' : 'false'}`}>
                                        x{Math.abs(lit)}={lit > 0 ? '⊤' : '⊥'}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}
                </>
            )}
        </div>
    );
}

// =============================================================================
// Main App Component
// =============================================================================

function App() {
    const [content, setContent] = useState(`c Example SAT problem
p cnf 3 2
1 -2 0
2 3 0
`);
    const [format, setFormat] = useState<FileFormat>('dimacs');
    const [result, setResult] = useState<SolveResult | null>(null);
    const [stats, setStats] = useState<SolveStats | null>(null);
    const [graph, setGraph] = useState<ClauseGraph | null>(null);
    const [solving, setSolving] = useState(false);
    const [activeView, setActiveView] = useState<ViewMode>('editor');

    const handleSolve = useCallback(async () => {
        setSolving(true);
        try {
            const res = await invoke<SolveResult>('solve_file', { content, format });
            setResult(res);
            // Notify plugins of solve result
            pluginManager.notifySolveFinished(res);
        } catch (err) {
            const errResult = {
                satisfiable: null,
                model: null,
                time_ms: 0,
                error: String(err),
            };
            setResult(errResult);
            pluginManager.notifySolveFinished(errResult);
        } finally {
            setSolving(false);
        }
    }, [content, format]);

    const handleParse = useCallback(async () => {
        try {
            const statsRes = await invoke<SolveStats>('get_problem_stats', { content, format });
            setStats(statsRes);
            const graphRes = await invoke<ClauseGraph>('get_clause_graph', { content, format });
            setGraph(graphRes);
        } catch (err) {
            console.error('Parse error:', err);
        }
    }, [content, format]);

    const getLanguage = () => {
        if (format === 'json') return 'json';
        if (format === 'sat') return 'satellite';
        return 'plaintext';
    };

    // Monaco editor mount handler: expose to plugin API
    const handleEditorMount: OnMount = (editor, monaco) => {
        pluginManager.setMonaco(monaco);
        pluginManager.setActiveEditor(editor);

        // Register Satellite language with basic tokenization
        monaco.languages.register({ id: 'satellite' });
        monaco.languages.setMonarchTokensProvider('satellite', {
            tokenizer: {
                root: [
                    [/\/\/.*/, 'comment'],
                    [/\/\*/, 'comment', '@comment'],
                    [/%\^?\*?[a-zA-Z_][a-zA-Z0-9_]*/, 'variable'],
                    [/\b(AND|OR|NOT|XOR|and|or|not|xor|eq|True|False|true|false)\b/, 'keyword'],
                    [/\b(i|f)\d+(\.\d+)?\b/, 'number'],
                    [/[+\-*\/%\^&|~<>=!]+/, 'operator'],
                ],
                comment: [
                    [/[^\/*]+/, 'comment'],
                    [/\*\//, 'comment', '@pop'],
                    [/[\/*]/, 'comment']
                ]
            }
        } as any);
    };

    return (
        <div className="app">
            <Sidebar activeView={activeView} onViewChange={setActiveView} />

            <div className="main-content">
                {/* Toolbar */}
                <header className="toolbar">
                    <div className="toolbar-left">
                        <span className="logo">🛰️ Satellite</span>
                        <select
                            value={format}
                            onChange={(e) => setFormat(e.target.value as FileFormat)}
                            className="format-select"
                        >
                            <option value="dimacs">DIMACS CNF</option>
                            <option value="json">Advanced JSON</option>
                            <option value="sat">Satellite (.sat)</option>
                        </select>
                    </div>
                    <div className="toolbar-actions">
                        <button onClick={handleParse} className="btn secondary">
                            📊 Analyze
                        </button>
                        <button onClick={handleSolve} disabled={solving} className="btn primary">
                            {solving ? '⏳ Solving...' : '▶️ Solve'}
                        </button>
                    </div>
                </header>

                {/* Main Area */}
                <div className="workspace">
                    {/* Editor Panel */}
                    <div className={`panel editor-panel ${activeView !== 'editor' ? 'hidden' : ''}`}>
                        <Editor
                            height="100%"
                            language={getLanguage()}
                            theme="vs-dark"
                            value={content}
                            onChange={(value) => setContent(value || '')}
                            onMount={handleEditorMount}
                            options={{
                                fontSize: 14,
                                minimap: { enabled: true },
                                lineNumbers: 'on',
                                wordWrap: 'on',
                                automaticLayout: true,
                            }}
                        />
                    </div>

                    {/* Stats Panel */}
                    <div className={`panel ${activeView !== 'stats' ? 'hidden' : ''}`}>
                        <StatsPanel stats={stats} />
                    </div>

                    {/* Graph Panel */}
                    <div className={`panel ${activeView !== 'graph' ? 'hidden' : ''}`}>
                        <ClauseGraphPanel graph={graph} />
                    </div>

                    {/* Result Panel (Always visible at bottom) */}
                    <div className="result-container">
                        <ResultPanel result={result} />
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;
