"""
Satellite Jupyter kernel implementation.
"""

from ipykernel.kernelbase import Kernel
from satellite_lab import Solver
import json


class SatelliteKernel(Kernel):
    """Jupyter kernel for Satellite SAT solver."""
    
    implementation = "Satellite"
    implementation_version = "0.1.0"
    language = "satellite"
    language_version = "0.1"
    language_info = {
        "name": "satellite",
        "mimetype": "text/x-satellite",
        "file_extension": ".sat",
    }
    banner = "Satellite SAT Solver Kernel"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solver = Solver()

    def do_execute(
        self,
        code,
        silent,
        store_history=True,
        user_expressions=None,
        allow_stdin=False,
    ):
        """Execute satellite code."""
        if not silent:
            try:
                # Parse and execute code
                result = self._execute_satellite(code)
                
                stream_content = {
                    "name": "stdout",
                    "text": result,
                }
                self.send_response(self.iopub_socket, "stream", stream_content)
                
            except Exception as e:
                stream_content = {
                    "name": "stderr",
                    "text": f"Error: {e}",
                }
                self.send_response(self.iopub_socket, "stream", stream_content)
                return {
                    "status": "error",
                    "execution_count": self.execution_count,
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "traceback": [],
                }

        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }

    def _execute_satellite(self, code: str) -> str:
        """Execute satellite DSL code."""
        lines = code.strip().split("\n")
        output = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
                
            if line.startswith("var "):
                # Variable declaration
                name = line[4:].strip()
                var = self.solver.bool_var(name=name)
                output.append(f"Created variable {name} (id={var.id})")
                
            elif line.startswith("clause "):
                # Clause addition
                lits = [int(l) for l in line[7:].split()]
                self.solver.add_clause(lits)
                output.append(f"Added clause {lits}")
                
            elif line == "solve":
                # Solve
                result = self.solver.solve()
                if result.satisfiable is True:
                    output.append(f"SAT (time: {result.time_ms}ms)")
                    output.append(f"Model: {result.model}")
                elif result.satisfiable is False:
                    output.append(f"UNSAT (time: {result.time_ms}ms)")
                else:
                    output.append(f"UNKNOWN (time: {result.time_ms}ms)")
                    
            elif line == "reset":
                # Reset solver
                self.solver = Solver()
                output.append("Solver reset")
                
        return "\n".join(output)

    def do_complete(self, code, cursor_pos):
        """Provide code completion."""
        keywords = ["var", "clause", "solve", "reset", "constraint"]
        
        # Simple prefix matching
        prefix = code[:cursor_pos].split()[-1] if code[:cursor_pos].split() else ""
        matches = [k for k in keywords if k.startswith(prefix)]
        
        return {
            "matches": matches,
            "cursor_start": cursor_pos - len(prefix),
            "cursor_end": cursor_pos,
            "status": "ok",
        }


if __name__ == "__main__":
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=SatelliteKernel)
