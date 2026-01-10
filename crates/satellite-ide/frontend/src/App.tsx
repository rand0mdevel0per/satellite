import { useState, useCallback } from 'react';
import Editor from '@monaco-editor/react';
import { invoke } from '@tauri-apps/api/core';
import './App.css';

interface SolveResult {
    satisfiable: boolean | null;
    model: number[] | null;
    time_ms: number;
    error: string | null;
}

function App() {
    const [content, setContent] = useState(`c Example SAT problem
p cnf 3 2
1 -2 0
2 3 0
`);
    const [result, setResult] = useState<SolveResult | null>(null);
    const [solving, setSolving] = useState(false);

    const handleSolve = useCallback(async () => {
        setSolving(true);
        try {
            const res = await invoke<SolveResult>('solve_file', {
                content,
                format: 'dimacs',
            });
            setResult(res);
        } catch (err) {
            setResult({
                satisfiable: null,
                model: null,
                time_ms: 0,
                error: String(err),
            });
        } finally {
            setSolving(false);
        }
    }, [content]);

    return (
        <div className="app">
            <header className="header">
                <h1>🛰️ Satellite IDE</h1>
                <button onClick={handleSolve} disabled={solving}>
                    {solving ? 'Solving...' : 'Solve'}
                </button>
            </header>

            <main className="main">
                <div className="editor-container">
                    <Editor
                        height="100%"
                        defaultLanguage="plaintext"
                        theme="vs-dark"
                        value={content}
                        onChange={(value) => setContent(value || '')}
                        options={{
                            fontSize: 14,
                            minimap: { enabled: false },
                            lineNumbers: 'on',
                            wordWrap: 'on',
                        }}
                    />
                </div>

                <div className="output-panel">
                    <h2>Result</h2>
                    {result && (
                        <div className="result">
                            {result.error ? (
                                <div className="error">Error: {result.error}</div>
                            ) : (
                                <>
                                    <div className={`status ${result.satisfiable ? 'sat' : 'unsat'}`}>
                                        {result.satisfiable === null
                                            ? 'Unknown'
                                            : result.satisfiable
                                                ? 'SATISFIABLE'
                                                : 'UNSATISFIABLE'}
                                    </div>
                                    <div className="time">Time: {result.time_ms}ms</div>
                                    {result.model && (
                                        <div className="model">
                                            <h3>Model</h3>
                                            <code>{result.model.join(' ')}</code>
                                        </div>
                                    )}
                                </>
                            )}
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}

export default App;
