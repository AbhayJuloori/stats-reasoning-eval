"""Runs a Python snippet in a separate process with a time limit.

Model written code and reference solutions both go through here, so they see the same
library versions and the same working directory (data/sheets).
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHEETS = ROOT / "data" / "sheets"

CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text):
    """Return the longest fenced code block, or the raw text if there is none."""
    blocks = CODE_BLOCK.findall(text or "")
    if blocks:
        return max(blocks, key=len)
    return text or ""


def run_code(code, timeout=90):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("import warnings\nwarnings.filterwarnings('ignore')\n" + code)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, "-I", path],
            cwd=SHEETS, capture_output=True, text=True, timeout=timeout,
        )
        return {"ok": proc.returncode == 0, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": f"timed out after {timeout}s"}
    finally:
        Path(path).unlink(missing_ok=True)


def last_line(stdout):
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""
