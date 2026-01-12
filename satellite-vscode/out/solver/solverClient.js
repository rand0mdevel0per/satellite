"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.SolverClient = void 0;
const child_process_1 = require("child_process");
class SolverClient {
    solverPath;
    defaultTimeout;
    constructor(solverPath, defaultTimeout) {
        this.solverPath = solverPath;
        this.defaultTimeout = defaultTimeout;
    }
    async solve(filePath, options = {}) {
        const args = ['solve', filePath, '--json'];
        if (options.useGpu) {
            args.push('--gpu');
        }
        const timeout = options.timeout ?? this.defaultTimeout;
        args.push('--timeout', timeout.toString());
        return new Promise((resolve, reject) => {
            let stdout = '';
            let stderr = '';
            const proc = (0, child_process_1.spawn)(this.solverPath, args, {
                timeout: (timeout + 5) * 1000
            });
            proc.stdout.on('data', (data) => {
                stdout += data.toString();
            });
            proc.stderr.on('data', (data) => {
                stderr += data.toString();
            });
            proc.on('close', (code) => {
                if (code === 0) {
                    try {
                        const result = JSON.parse(stdout);
                        resolve({
                            satisfiable: result.satisfiable,
                            model: result.model,
                            timeMs: result.time_ms || 0,
                            unsatCore: result.unsat_core
                        });
                    }
                    catch {
                        // Parse plain text output
                        resolve(this.parsePlainOutput(stdout));
                    }
                }
                else {
                    reject(new Error(stderr || `Solver exited with code ${code}`));
                }
            });
            proc.on('error', (err) => {
                reject(new Error(`Failed to start solver: ${err.message}`));
            });
        });
    }
    async analyze(filePath) {
        const args = ['analyze', filePath, '--json'];
        return new Promise((resolve, reject) => {
            let stdout = '';
            const proc = (0, child_process_1.spawn)(this.solverPath, args);
            proc.stdout.on('data', (data) => {
                stdout += data.toString();
            });
            proc.on('close', (code) => {
                if (code === 0) {
                    try {
                        const result = JSON.parse(stdout);
                        resolve(result);
                    }
                    catch {
                        // Default values if parsing fails
                        resolve({
                            numVars: 0,
                            numClauses: 0,
                            maxClauseSize: 0
                        });
                    }
                }
                else {
                    reject(new Error(`Analysis failed with code ${code}`));
                }
            });
            proc.on('error', (err) => {
                reject(new Error(`Failed to start solver: ${err.message}`));
            });
        });
    }
    parsePlainOutput(output) {
        const lines = output.toLowerCase().split('\n');
        let satisfiable = null;
        let model;
        let timeMs = 0;
        for (const line of lines) {
            if (line.includes('satisfiable') || line.includes('sat')) {
                if (line.includes('unsatisfiable') || line.includes('unsat')) {
                    satisfiable = false;
                }
                else {
                    satisfiable = true;
                }
            }
            if (line.startsWith('v ') || line.startsWith('model:')) {
                const nums = line.match(/-?\d+/g);
                if (nums) {
                    model = nums.map(n => parseInt(n)).filter(n => n !== 0);
                }
            }
            const timeMatch = line.match(/(\d+(?:\.\d+)?)\s*(?:ms|milliseconds)/);
            if (timeMatch) {
                timeMs = parseFloat(timeMatch[1]);
            }
        }
        return { satisfiable, model, timeMs };
    }
}
exports.SolverClient = SolverClient;
//# sourceMappingURL=solverClient.js.map