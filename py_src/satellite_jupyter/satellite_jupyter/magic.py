"""
IPython magic for Satellite solver.

Usage in Jupyter notebook:

    %load_ext satellite_jupyter
    
    %%satellite
    var x
    var y
    clause 1 2
    clause -1 -2
    solve
"""

from IPython.core.magic import Magics, magics_class, cell_magic
from satellite_lab import Solver


@magics_class
class SatelliteMagics(Magics):
    """IPython magics for Satellite solver."""
    
    def __init__(self, shell):
        super().__init__(shell)
        self.solver = Solver()
        self.variables = {}
    
    @cell_magic
    def satellite(self, line, cell):
        """Execute Satellite DSL code.
        
        Syntax:
            var <name>        - Create a boolean variable
            clause <lits...>  - Add a clause (space-separated literals)
            solve             - Solve the problem
            reset             - Reset the solver
        """
        from tqdm.auto import tqdm
        
        lines = cell.strip().split("\n")
        
        for line in tqdm(lines, desc="Processing"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
                
            if line.startswith("var "):
                name = line[4:].strip()
                var = self.solver.bool_var(name=name)
                self.variables[name] = var
                print(f"Variable {name} = {var.id}")
                
            elif line.startswith("clause "):
                lits = [int(l) for l in line[7:].split()]
                self.solver.add_clause(lits)
                
            elif line == "solve":
                print("Solving...")
                result = self.solver.solve()
                if result.satisfiable is True:
                    print(f"✅ SAT (time: {result.time_ms}ms)")
                    print(f"Model: {result.model}")
                elif result.satisfiable is False:
                    print(f"❌ UNSAT (time: {result.time_ms}ms)")
                else:
                    print(f"❓ UNKNOWN (time: {result.time_ms}ms)")
                    
            elif line == "reset":
                self.solver = Solver()
                self.variables = {}
                print("Solver reset")


def load_ipython_extension(ipython):
    """Load the Satellite magic extension."""
    ipython.register_magics(SatelliteMagics)


def unload_ipython_extension(ipython):
    """Unload the extension."""
    pass
