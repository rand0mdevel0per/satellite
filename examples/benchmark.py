import os
import json
import subprocess
import random
import time
import argparse

def generate_random_3sat(num_vars, num_clauses):
    variables = [{"id": i+1, "type": "Bool", "name": f"x{i+1}"} for i in range(num_vars)]
    clauses = []
    for _ in range(num_clauses):
        literals = []
        for _ in range(3):
            var = random.randint(1, num_vars)
            sign = random.choice([1, -1])
            literals.append(var * sign)
        clauses.append({
            "literals": literals,
            "type": "original"
        })
    
    return {
        "variables": variables,
        "clauses": clauses,
        "abi_constraints": []
    }

def main():
    parser = argparse.ArgumentParser(description="Satellite Benchmark Generator")
    parser.add_argument("--count", type=int, default=10, help="Number of problems to generate")
    parser.add_argument("--vars", type=int, default=50, help="Number of variables per problem")
    parser.add_argument("--clauses", type=int, default=215, help="Number of clauses per problem (approx 4.3 ratio)")
    parser.add_argument("--output-dir", default="bench_data", help="Directory to save problems")
    parser.add_argument("--run", action="store_true", help="Run satellite batch after generation")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Generating {args.count} problems in {args.output_dir}...")
    
    for i in range(args.count):
        problem = generate_random_3sat(args.vars, args.clauses)
        filename = os.path.join(args.output_dir, f"problem_{i:03d}.json")
        with open(filename, "w") as f:
            json.dump(problem, f, indent=2)
            
    print("Generation complete.")

    if args.run:
        print("Running Satellite Batch Solver...")
        cli_path = os.path.join("..", "target", "release", "satellite.exe")
        if not os.path.exists(cli_path):
             cli_path = os.path.join("..", "target", "debug", "satellite.exe")
        
        results_dir = os.path.join(args.output_dir, "results")
        
        cmd = [
            cli_path, 
            "batch", 
            "--input-dir", args.output_dir, 
            "--output-dir", results_dir,
            "--workers", "4"
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        start_time = time.time()
        try:
            subprocess.run(cmd, check=True)
            elapsed = time.time() - start_time
            print(f"Benchmark complete in {elapsed:.2f}s")
        except subprocess.CalledProcessError as e:
            print(f"Error running solver: {e}")
        except FileNotFoundError:
             print("satellite.exe not found! Build it first.")

if __name__ == "__main__":
    main()
