"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const solverClient_1 = require("./solver/solverClient");
let outputChannel;
let statusBarItem;
let solverClient;
function activate(context) {
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
    solverClient = new solverClient_1.SolverClient(config.get('solverPath', 'satellite'), config.get('timeout', 60));
    // Register commands
    context.subscriptions.push(vscode.commands.registerCommand('satellite.solve', () => solveCurrentFile()), vscode.commands.registerCommand('satellite.analyze', () => analyzeCurrentFile()), vscode.commands.registerCommand('satellite.showOutput', () => outputChannel.show()));
    // Update status on config change
    vscode.workspace.onDidChangeConfiguration(e => {
        if (e.affectsConfiguration('satellite')) {
            const config = vscode.workspace.getConfiguration('satellite');
            solverClient = new solverClient_1.SolverClient(config.get('solverPath', 'satellite'), config.get('timeout', 60));
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
        }
        else if (result.satisfiable === false) {
            statusBarItem.text = '$(x) UNSAT';
            outputChannel.appendLine('Result: UNSATISFIABLE');
        }
        else {
            statusBarItem.text = '$(question) Unknown';
            outputChannel.appendLine('Result: UNKNOWN');
        }
        outputChannel.appendLine(`Time: ${result.timeMs}ms`);
        // Show diagnostics for UNSAT core if available
        if (result.unsatCore && result.unsatCore.length > 0) {
            outputChannel.appendLine(`UNSAT Core: ${result.unsatCore.length} clauses`);
        }
    }
    catch (error) {
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
    }
    catch (error) {
        outputChannel.appendLine(`Error: ${error}`);
    }
}
function deactivate() {
    if (outputChannel) {
        outputChannel.dispose();
    }
}
//# sourceMappingURL=extension.js.map