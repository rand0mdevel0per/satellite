"""
Install the Satellite Jupyter kernel.
"""

import json
import os
import sys


def install_kernel():
    """Install the Satellite kernel spec."""
    from jupyter_client.kernelspec import KernelSpecManager
    
    kernel_spec = {
        "argv": [
            sys.executable,
            "-m",
            "satellite_jupyter.kernel",
            "-f",
            "{connection_file}",
        ],
        "display_name": "Satellite",
        "language": "satellite",
    }
    
    # Create kernel spec directory
    ksm = KernelSpecManager()
    kernel_dir = os.path.join(ksm.user_kernel_dir, "satellite")
    os.makedirs(kernel_dir, exist_ok=True)
    
    # Write kernel.json
    with open(os.path.join(kernel_dir, "kernel.json"), "w") as f:
        json.dump(kernel_spec, f, indent=2)
    
    print(f"Installed kernel spec to {kernel_dir}")


def main():
    """CLI entry point."""
    install_kernel()


if __name__ == "__main__":
    main()
