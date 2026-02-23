import subprocess
from pathlib import Path

def run_main_py(base_dir: Path):
    # Recursively iterate through all main.py files
    for path in base_dir.rglob("main.py"):
        print(f"Running {path}")
        log_file = path.parent / "log.txt"
        # Run main.py from its own directory
        with log_file.open("w") as f:
            subprocess.run(
                ["python", str(path.name)],  # only filename, run in its folder
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=path.parent  # run from the directory containing main.py
            )

if __name__ == "__main__":
    cwd = Path.cwd()
    run_main_py(cwd)