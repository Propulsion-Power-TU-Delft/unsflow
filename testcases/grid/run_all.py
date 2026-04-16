import subprocess
from pathlib import Path

def run_main_py(base_dir: Path):
    failed_scripts = []

    for path in base_dir.rglob("main.py"):
        print(f"Running {path}")
        log_file = path.parent / "log.txt"
        try:
            # Run main.py from its own directory
            result = subprocess.run(
                ["python", str(path.name)],
                stdout=open(log_file, "w"),
                stderr=subprocess.STDOUT,
                cwd=path.parent
            )
            # Check if the script exited with a non-zero code
            if result.returncode != 0:
                print(f"⚠ Warning: {path} did not finish successfully (exit code {result.returncode})")
                failed_scripts.append(path)
            else:
                print(f"\tSuccessful run!\n")
        except Exception as e:
            print(f"⚠ Error running {path}: {e}")
            failed_scripts.append(path)

    # Summary at the end
    print("\n=== Summary ===")
    print(f"Total scripts run: {len(list(base_dir.rglob('main.py')))}")
    if failed_scripts:
        print(f"Scripts failed: {len(failed_scripts)}")
        for f in failed_scripts:
            print(f" - {f}")
    else:
        print("All scripts ran successfully!")

if __name__ == "__main__":
    run_main_py(Path.cwd())