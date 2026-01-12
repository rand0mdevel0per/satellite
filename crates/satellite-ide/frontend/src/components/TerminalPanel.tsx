import { useEffect, useRef, useState } from 'react';
import { Terminal as XTerm } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import { invoke } from '@tauri-apps/api/core';
import '@xterm/xterm/css/xterm.css';
import './TerminalPanel.css';

interface TerminalPanelProps {
    visible: boolean;
}

export function TerminalPanel({ visible }: TerminalPanelProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const terminalRef = useRef<XTerm | null>(null);
    const fitAddonRef = useRef<FitAddon | null>(null);
    const [initialized, setInitialized] = useState(false);

    useEffect(() => {
        if (!containerRef.current || initialized) return;

        const term = new XTerm({
            fontFamily: 'Consolas, "Courier New", monospace',
            fontSize: 13,
            theme: {
                background: '#1e1e1e',
                foreground: '#cccccc',
                cursor: '#ffffff',
                selectionBackground: '#264f78',
            },
            cursorBlink: true,
            allowProposedApi: true,
        });

        const fitAddon = new FitAddon();
        term.loadAddon(fitAddon);

        term.open(containerRef.current);
        fitAddon.fit();

        terminalRef.current = term;
        fitAddonRef.current = fitAddon;
        setInitialized(true);

        // Welcome message
        term.writeln('\x1b[1;34mSatellite IDE Terminal\x1b[0m');
        term.writeln('Type commands to interact with the solver.\n');
        term.write('\x1b[32mPS>\x1b[0m ');

        // Handle input
        let currentLine = '';
        term.onData(async (data) => {
            const code = data.charCodeAt(0);

            if (code === 13) {
                // Enter
                term.writeln('');
                if (currentLine.trim()) {
                    await executeCommand(term, currentLine.trim());
                }
                currentLine = '';
                term.write('\x1b[32mPS>\x1b[0m ');
            } else if (code === 127 || code === 8) {
                // Backspace
                if (currentLine.length > 0) {
                    currentLine = currentLine.slice(0, -1);
                    term.write('\b \b');
                }
            } else if (code >= 32) {
                // Printable characters
                currentLine += data;
                term.write(data);
            }
        });

        // Resize observer
        const resizeObserver = new ResizeObserver(() => {
            if (visible && fitAddon) {
                fitAddon.fit();
            }
        });
        resizeObserver.observe(containerRef.current);

        return () => {
            resizeObserver.disconnect();
            term.dispose();
        };
    }, []);

    useEffect(() => {
        if (visible && fitAddonRef.current) {
            setTimeout(() => fitAddonRef.current?.fit(), 50);
        }
    }, [visible]);

    return (
        <div className={`terminal-panel ${visible ? '' : 'hidden'}`}>
            <div className="terminal-container" ref={containerRef} />
        </div>
    );
}

async function executeCommand(term: XTerm, command: string) {
    const parts = command.split(/\s+/);
    const cmd = parts[0].toLowerCase();

    try {
        switch (cmd) {
            case 'help':
                term.writeln('Available commands:');
                term.writeln('  help     - Show this help');
                term.writeln('  clear    - Clear terminal');
                term.writeln('  solve    - Run solver on current file');
                term.writeln('  stats    - Show solver statistics');
                term.writeln('  gpu      - Show GPU status');
                break;

            case 'clear':
                term.clear();
                break;

            case 'solve':
                term.writeln('\x1b[33mRunning solver...\x1b[0m');
                try {
                    const result = await invoke('solve_current');
                    term.writeln(`\x1b[32mResult:\x1b[0m ${JSON.stringify(result)}`);
                } catch (err) {
                    term.writeln(`\x1b[31mError:\x1b[0m ${err}`);
                }
                break;

            case 'stats':
                try {
                    const stats = await invoke('get_stats');
                    term.writeln(JSON.stringify(stats, null, 2));
                } catch (err) {
                    term.writeln(`\x1b[31mError:\x1b[0m ${err}`);
                }
                break;

            case 'gpu':
                try {
                    const gpuStatus = await invoke('get_gpu_status');
                    term.writeln(JSON.stringify(gpuStatus, null, 2));
                } catch {
                    term.writeln('\x1b[33mGPU status unavailable\x1b[0m');
                }
                break;

            default:
                term.writeln(`\x1b[31mUnknown command:\x1b[0m ${cmd}`);
                term.writeln('Type "help" for available commands.');
        }
    } catch (err) {
        term.writeln(`\x1b[31mError:\x1b[0m ${err}`);
    }
}

export default TerminalPanel;
