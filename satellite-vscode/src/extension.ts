import * as vscode from 'vscode';
import { SolverClient } from './solver/solverClient';

let outputChannel: vscode.OutputChannel;
let statusBarItem: vscode.StatusBarItem;
let solverClient: SolverClient;

export function activate(context: vscode.ExtensionContext) {
    console.log('Satellite extension activated');

    // Create output channel
    outputChannel = vscode.window.createOutputChannel('Satellite');

    // Create status bar item
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.text = '$(circuit-board) Satellite';
    statusBarItem.tooltip = 'Satellite SAT Solver';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    // Initialize solver client
    const config = vscode.workspace.getConfiguration('satellite');
    solverClient = new SolverClient(
        config.get('solverPath', 'satellite'),
        config.get('timeout', 60)
    );

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('satellite.solve', () => solveCurrentFile()),
        vscode.commands.registerCommand('satellite.analyze', () => analyzeCurrentFile()),
        vscode.commands.registerCommand('satellite.showOutput', () => outputChannel.show())
    );

    // Update status on config change
    vscode.workspace.onDidChangeConfiguration(e => {
        if (e.affectsConfiguration('satellite')) {
            const config = vscode.workspace.getConfiguration('satellite');
            solverClient = new SolverClient(
                config.get('solverPath', 'satellite'),
                config.get('timeout', 60)
            );
        }
    });
}

async function solveCurrentFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active file to solve');
        return;
    }

    const filePath = editor.document.uri.fsPath;
    const config = vscode.workspace.getConfiguration('satellite');

    outputChannel.clear();
    outputChannel.show();
    outputChannel.appendLine(`Solving: ${filePath}`);
    outputChannel.appendLine('---');

    statusBarItem.text = '$(sync~spin) Solving...';

    try {
        const result = await solverClient.solve(filePath, {
            useGpu: config.get('useGpu', true),
            timeout: config.get('timeout', 60)
        });

        if (result.satisfiable === true) {
            statusBarItem.text = '$(check) SAT';
            outputChannel.appendLine('Result: SATISFIABLE');
            if (result.model) {
                outputChannel.appendLine(`Model: ${result.model.join(' ')}`);
            }
        } else if (result.satisfiable === false) {
            statusBarItem.text = '$(x) UNSAT';
            outputChannel.appendLine('Result: UNSATISFIABLE');
        } else {
            statusBarItem.text = '$(question) Unknown';
            outputChannel.appendLine('Result: UNKNOWN');
        }

        outputChannel.appendLine(`Time: ${result.timeMs}ms`);

        // Show diagnostics for UNSAT core if available
        if (result.unsatCore && result.unsatCore.length > 0) {
            outputChannel.appendLine(`UNSAT Core: ${result.unsatCore.length} clauses`);
        }

    } catch (error) {
        statusBarItem.text = '$(error) Error';
        outputChannel.appendLine(`Error: ${error}`);
        vscode.window.showErrorMessage(`Solver error: ${error}`);
    }

    // Reset status after 5 seconds
    setTimeout(() => {
        statusBarItem.text = '$(circuit-board) Satellite';
    }, 5000);
}

async function analyzeCurrentFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active file to analyze');
        return;
    }

    outputChannel.clear();
    outputChannel.show();
    outputChannel.appendLine(`Analyzing: ${editor.document.uri.fsPath}`);
    outputChannel.appendLine('---');

    try {
        const result = await solverClient.analyze(editor.document.uri.fsPath);
        outputChannel.appendLine(`Variables: ${result.numVars}`);
        outputChannel.appendLine(`Clauses: ${result.numClauses}`);
        outputChannel.appendLine(`Max clause size: ${result.maxClauseSize}`);
    } catch (error) {
        outputChannel.appendLine(`Error: ${error}`);
    }
}

export function deactivate() {
    if (outputChannel) {
        outputChannel.dispose();
    }
}
