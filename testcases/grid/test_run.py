import subprocess
from pathlib import Path
import pytest


import sys

def find_main_scripts(base_dir: Path):
    return list(base_dir.rglob("main.py"))


# Collect all scripts once
BASE_DIR = Path(__file__).parent
MAIN_SCRIPTS = find_main_scripts(BASE_DIR)


@pytest.mark.parametrize("script_path", MAIN_SCRIPTS)
def test_run_main(script_path: Path):
    log_file = script_path.parent / "log.txt"

    with open(log_file, "w") as f:
        result = subprocess.run(
            [sys.executable, script_path.name],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=script_path.parent
        )

    assert result.returncode == 0, (
        f"{script_path} failed with exit code {result.returncode}. "
        f"Check log: {log_file}"
    )