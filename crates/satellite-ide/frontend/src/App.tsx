import { useState, useCallback } from 'react';
import Editor, { OnMount } from '@monaco-editor/react';
import { invoke } from '@tauri-apps/api/core';
import { pluginManager } from './plugin_api';
import {
    ActivityBar,
    defaultActivityItems,
    FileExplorer,
    TerminalPanel,
    StatusBar,
    Toolbar,
    Sidebar,
} from './components';
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

type BottomPanelId = 'terminal' | 'build' | 'problems' | null;
type LeftPanelId = 'files' | 'git' | 'search' | null;

// =============================================================================
// Main App Component
// =============================================================================

function App() {
    // Editor state
    const [content, setContent] = useState(`c Example SAT problem
c Pigeonhole principle: 2 pigeons, 1 hole
p cnf 2 3
1 2 0
-1 0
-2 0
`);
    const [cursorPosition, setCursorPosition] = useState({ line: 1, column: 1 });
    const [solving, setSolving] = useState(false);
    const [result, setResult] = useState<SolveResult | null>(null);
    const [statusMessage, setStatusMessage] = useState('Ready');

    // Panel visibility
    const [leftPanel, setLeftPanel] = useState<LeftPanelId>('files');
    const [bottomPanelVisible, setBottomPanelVisible] = useState(true);
    const [bottomPanel, setBottomPanel] = useState<BottomPanelId>('terminal');

    // Activity bar handler
    const handleActivityClick = (id: string) => {
        if (['files', 'git', 'search'].includes(id)) {
            setLeftPanel(leftPanel === id ? null : id as LeftPanelId);
        } else if (['terminal', 'tasks'].includes(id)) {
            if (id === 'terminal') {
                setBottomPanel('terminal');
                setBottomPanelVisible(true);
            }
        } else if (id === 'settings') {
            // TODO: Open settings
        }
    };

    // Solve handler
    const handleSolve = useCallback(async () => {
        setSolving(true);
        setStatusMessage('Solving...');
        setResult(null);

        try {
            const res = await invoke<SolveResult>('solve_content', { content });
            setResult(res);
            setStatusMessage(
                res.satisfiable === true ? 'SAT' :
                    res.satisfiable === false ? 'UNSAT' :
                        'Unknown'
            );
        } catch (err) {
            setResult({
                satisfiable: null,
                model: null,
                time_ms: 0,
                error: String(err),
            });
            setStatusMessage('Error');
        } finally {
            setSolving(false);
        }
    }, [content]);

    // Analyze handler
    const handleAnalyze = useCallback(async () => {
        setStatusMessage('Analyzing...');
        try {
            await invoke('analyze_content', { content });
            setStatusMessage('Analysis complete');
        } catch (err) {
            setStatusMessage('Analysis failed');
        }
    }, [content]);

    // Monaco editor mount
    const handleEditorMount: OnMount = (editor, monaco) => {
        pluginManager.setMonaco(monaco);
        pluginManager.setActiveEditor(editor);

        // Track cursor position
        editor.onDidChangeCursorPosition((e) => {
            setCursorPosition({
                line: e.position.lineNumber,
                column: e.position.column,
            });
        });

        // Register Satellite language
        monaco.languages.register({ id: 'satellite' });
        monaco.languages.setMonarchTokensProvider('satellite', {
            tokenizer: {
                root: [
                    [/\/\/.*/, 'comment'],
                    [/\/\*/, 'comment', '@comment'],
                    [/^c\s.*/, 'comment'],
                    [/^p\s.*/, 'keyword'],
                    [/%\^?\*?[a-zA-Z_][a-zA-Z0-9_]*/, 'variable'],
                    [/\b(AND|OR|NOT|XOR|and|or|not|xor|eq|True|False|true|false)\b/, 'keyword'],
                    [/\b-?\d+\b/, 'number'],
                    [/[+\-*\/%\^&|~<>=!]+/, 'operator'],
                ],
                comment: [
                    [/[^\/*]+/, 'comment'],
                    [/\*\//, 'comment', '@pop'],
                    [/[\/*]/, 'comment'],
                ],
            },
        } as any);
    };

    // Handle file selection
    const handleFileSelect = async (path: string) => {
        try {
            const fileContent = await invoke<string>('read_file', { path });
            setContent(fileContent);
            setStatusMessage(`Opened: ${path}`);
        } catch (err) {
            setStatusMessage(`Failed to open: ${path}`);
        }
    };

    // Render left panel content
    const renderLeftPanel = () => {
        switch (leftPanel) {
            case 'files':
                return <FileExplorer onFileSelect={handleFileSelect} />;
            case 'git':
                return (
                    <div className="panel-placeholder">
                        <span>Source Control</span>
                        <small>Coming soon</small>
                    </div>
                );
            case 'search':
                return (
                    <div className="panel-placeholder">
                        <span>Search</span>
                        <small>Coming soon</small>
                    </div>
                );
            default:
                return null;
        }
    };

    return (
        <div className="app">
            {/* Toolbar */}
            <Toolbar
                onSolve={handleSolve}
                onAnalyze={handleAnalyze}
                solving={solving}
            />

            {/* Main content area */}
            <div className="main-area">
                {/* Left Activity Bar */}
                <ActivityBar
                    items={defaultActivityItems}
                    activeId={leftPanel}
                    onItemClick={handleActivityClick}
                />

                {/* Left Sidebar */}
                <Sidebar position="left" visible={leftPanel !== null}>
                    {renderLeftPanel()}
                </Sidebar>

                {/* Center area: Editor + Bottom Panel */}
                <div className="center-area">
                    {/* Editor */}
                    <div className="editor-container">
                        <Editor
                            height="100%"
                            language="satellite"
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
                                scrollBeyondLastLine: false,
                            }}
                        />
                    </div>

                    {/* Bottom Panel */}
                    {bottomPanelVisible && (
                        <div className="bottom-panel">
                            <div className="bottom-panel-tabs">
                                <button
                                    className={`tab ${bottomPanel === 'terminal' ? 'active' : ''}`}
                                    onClick={() => setBottomPanel('terminal')}
                                >
                                    Terminal
                                </button>
                                <button
                                    className={`tab ${bottomPanel === 'build' ? 'active' : ''}`}
                                    onClick={() => setBottomPanel('build')}
                                >
                                    Build
                                </button>
                                <button
                                    className={`tab ${bottomPanel === 'problems' ? 'active' : ''}`}
                                    onClick={() => setBottomPanel('problems')}
                                >
                                    Problems
                                </button>
                            </div>
                            <div className="bottom-panel-content">
                                <TerminalPanel visible={bottomPanel === 'terminal'} />
                                {bottomPanel === 'build' && (
                                    <div className="panel-placeholder">Build output</div>
                                )}
                                {bottomPanel === 'problems' && (
                                    <div className="panel-placeholder">No problems</div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Result overlay when solving completes */}
                    {result && (
                        <div className="result-toast">
                            <span className={result.satisfiable ? 'sat' : 'unsat'}>
                                {result.satisfiable === true ? '✓ SAT' :
                                    result.satisfiable === false ? '✗ UNSAT' :
                                        '? Unknown'}
                            </span>
                            <span className="time">{result.time_ms.toFixed(1)}ms</span>
                            <button onClick={() => setResult(null)}>×</button>
                        </div>
                    )}
                </div>

                {/* Right Activity Bar (empty for now, plugin-extensible) */}
                <div className="activity-bar-right">
                    {/* Plugins can add icons here */}
                </div>
            </div>

            {/* Status Bar */}
            <StatusBar
                line={cursorPosition.line}
                column={cursorPosition.column}
                message={statusMessage}
                language="satellite"
                gpuStatus="unavailable"
            />
        </div>
    );
}

export default App;
