from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import modal


app = modal.App("cursor-agent-tools")


def _build_repo_image(python_version: str | None = None) -> modal.Image:
    py = python_version or "3.11"
    return (
        modal.Image.debian_slim(python_version=py)
        .apt_install("git")
        .pip_install("pytest")
    )


@dataclass
class RepoCIResult:
    exit_code: int
    stdout: str
    stderr: str


@app.function(image=_build_repo_image(None), timeout=60 * 60)
async def run(repo: str, ref: str, cmd: str, python: Optional[str] = None) -> Dict[str, str | int]:
    workdir = Path("/tmp/repo")
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()

    def _run(command: str, cwd: Path | None = None) -> RepoCIResult:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
        )
        return RepoCIResult(proc.returncode, proc.stdout, proc.stderr)

    steps: list[tuple[str, RepoCIResult]] = []

    steps.append(("git_clone", _run(f"git clone --depth 1 {repo} .", cwd=workdir)))
    if steps[-1][1].exit_code != 0:
        return {
            "exit_code": steps[-1][1].exit_code,
            "stdout": steps[-1][1].stdout[-20000:],
            "stderr": steps[-1][1].stderr[-20000:],
        }

    steps.append(("git_fetch", _run(f"git fetch --depth 1 origin {ref}", cwd=workdir)))
    steps.append(("git_checkout", _run(f"git checkout {ref}", cwd=workdir)))

    req = workdir / "requirements.txt"
    pyproject = workdir / "pyproject.toml"

    if req.exists():
        steps.append(("pip_install", _run(f"python -m pip install -r requirements.txt", cwd=workdir)))
    elif pyproject.exists():
        steps.append(("pip_install", _run(f"python -m pip install .", cwd=workdir)))

    steps.append(("run_cmd", _run(cmd, cwd=workdir)))
    final = steps[-1][1]

    return {
        "exit_code": final.exit_code,
        "stdout": final.stdout[-20000:],
        "stderr": final.stderr[-20000:],
    }
