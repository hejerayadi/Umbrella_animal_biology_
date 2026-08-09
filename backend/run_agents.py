"""Start every agent service at once, each in its OWN virtual environment.

The orchestrator calls agents over HTTP, so all nine have to be running before
a query can get past the first worker. Starting nine uvicorn processes by hand
is tedious; this does it in one command.

Crucially, each agent is launched with the interpreter from *its own*
`.venv`, not with the one running this script. That is what keeps the
environments isolated: the Reconstruction agent can pin `torch` and the
Biodiversity agent can pin `plotly` without ever seeing each other's
packages. An agent with no `.venv` yet falls back to the current interpreter
and is flagged loudly in the output.

Run from the repository root:

    python -m backend.run_agents --setup    # create all nine venvs + install
    python -m backend.run_agents            # start all nine services

`--setup` is a one-time (slow) step. Ctrl+C stops every service. Logs are
interleaved on this terminal, prefixed with the agent name.

This is a convenience for local development only - in a real deployment each
agent is its own container, which is why each declares its own dependencies.

Two declaration styles are supported, and `--setup` picks per agent:
`pyproject.toml` + `uv.lock`, installed with `uv sync`, or the older
`requirements.txt`, installed with pip into a `python -m venv` environment.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from urllib.parse import urlparse

from .registry import AGENT_ENDPOINTS, _AGENT_FOLDERS

_AGENTS_DIR = Path(__file__).parent / "agents"
_REPO_ROOT = Path(__file__).parent.parent

# Where a venv keeps its interpreter differs by platform.
_VENV_PYTHON = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")


def _venv_dir(folder: str) -> Path:
    return _AGENTS_DIR / folder / ".venv"


def _agent_python(folder: str) -> tuple[Path, bool]:
    """The interpreter to launch one agent with.

    Returns the path plus whether it is the agent's own venv. Falling back to
    the current interpreter keeps the launcher usable before anyone has run
    `--setup`, but it means that agent is NOT isolated - so the caller warns.
    """
    candidate = _venv_dir(folder) / _VENV_PYTHON
    if candidate.exists():
        return candidate, True
    return Path(sys.executable), False


def _stream(prefix: str, process: subprocess.Popen) -> None:
    """Echo one child's output with its agent name in front of every line."""
    for line in process.stdout:
        print(f"[{prefix}] {line.rstrip()}", flush=True)


def _explain(pip_output: str) -> str | None:
    """Turn a wall of pip output into one actionable line, when we recognise it."""

    if "Failed building wheel for" in pip_output or "Microsoft Visual C++" in pip_output:
        packages = sorted(
            {
                line.split("Failed building wheel for ")[1].strip()
                for line in pip_output.splitlines()
                if "Failed building wheel for " in line
            }
        )
        named = ", ".join(packages) if packages else "a pinned package"
        return (
            f"{named} has no prebuilt wheel for Python {sys.version_info.major}."
            f"{sys.version_info.minor} on this platform, so pip tried to compile it "
            f"from source and failed. Bump the pin in that agent's requirements.txt "
            f"to a version that ships wheels for your Python."
        )
    if "No matching distribution found" in pip_output:
        return "A pinned version does not exist for this Python. Check the pins."
    return None


def _manifest(folder: str) -> Path:
    """The dependency file one agent declares, whichever style it uses."""
    pyproject = _AGENTS_DIR / folder / "pyproject.toml"
    return pyproject if pyproject.exists() else _AGENTS_DIR / folder / "requirements.txt"


def _setup_with_uv(folder: str) -> tuple[bool, str]:
    """Build one agent's environment from its pyproject.toml + uv.lock.

    `uv sync` creates the `.venv` itself and installs the exact versions the
    lock file records, so a developer's environment matches the one CI resolved
    rather than whatever the ranges happen to allow today.
    """
    if shutil.which("uv") is None:
        return False, "uv is not on PATH. Install it from https://docs.astral.sh/uv/"
    result = subprocess.run(
        ["uv", "sync", "--project", str(_AGENTS_DIR / folder)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout + result.stderr


def _setup_with_pip(folder: str) -> tuple[bool, str]:
    """Build one agent's environment from its requirements.txt."""
    venv_dir = _venv_dir(folder)
    if not venv_dir.exists():
        created = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            capture_output=True,
            text=True,
        )
        if created.returncode != 0:
            return False, created.stdout + created.stderr

    python = venv_dir / _VENV_PYTHON
    installed = subprocess.run(
        [str(python), "-m", "pip", "install", "-q", "-r", str(_AGENTS_DIR / folder / "requirements.txt")],
        capture_output=True,
        text=True,
    )
    return installed.returncode == 0, installed.stdout + installed.stderr


def setup() -> None:
    """Create a virtual environment per agent and install its dependencies.

    One agent failing must not stop the other eight: each is independent, so
    every failure is collected and reported at the end instead of aborting.
    """

    print(f"Setting up {len(_AGENT_FOLDERS)} agent environments on Python "
          f"{sys.version_info.major}.{sys.version_info.minor}.")
    print("This takes a while - some agents pin large packages (torch).\n")

    failures: list[tuple[str, str, str]] = []

    for agent_name, folder in _AGENT_FOLDERS.items():
        manifest = _manifest(folder)

        print(f"--- {agent_name} ({folder})")
        if manifest.name == "pyproject.toml":
            ok, output = _setup_with_uv(folder)
            tool = "uv sync"
        else:
            ok, output = _setup_with_pip(folder)
            tool = "pip install"

        if not ok:
            print(f"    FAILED ({tool})\n")
            failures.append((agent_name, folder, output))
            continue

        print(f"    {tool} -> {manifest.relative_to(_REPO_ROOT)}\n")

    succeeded = len(_AGENT_FOLDERS) - len(failures)
    print(f"\n{'=' * 72}")
    print(f"{succeeded}/{len(_AGENT_FOLDERS)} agents ready.")

    if not failures:
        print("\nNow run:  python -m backend.run_agents")
        return

    print(f"\n{len(failures)} failed:\n")
    for agent_name, folder, output in failures:
        print(f"  {agent_name}  ({_manifest(folder).relative_to(_REPO_ROOT)})")
        hint = _explain(output)
        if hint:
            print(f"    -> {hint}")
        else:
            tail = [line for line in output.strip().splitlines() if line.strip()][-3:]
            for line in tail:
                print(f"    | {line[:100]}")
        print()

    print("The other agents were set up fine and can be started now:")
    print("  python -m backend.run_agents")
    print("Agents that failed will run on the shared interpreter and be flagged.")
    sys.exit(1)


def serve() -> None:
    """Launch every agent service, each with its own interpreter."""

    processes: list[subprocess.Popen] = []
    unisolated: list[str] = []

    for agent_name, folder in _AGENT_FOLDERS.items():
        port = urlparse(AGENT_ENDPOINTS[agent_name]).port
        module = f"backend.agents.{folder}.api:app"
        python, isolated = _agent_python(folder)

        if not isolated:
            unisolated.append(agent_name)

        # cwd is the repository root so that `-m` puts it on sys.path and the
        # `backend.agents...` package path resolves, whichever venv is used.
        process = subprocess.Popen(
            [str(python), "-m", "uvicorn", module, "--port", str(port)],
            cwd=str(_REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes.append(process)
        threading.Thread(target=_stream, args=(agent_name, process), daemon=True).start()

        marker = "own venv" if isolated else "SHARED venv"
        print(f"starting {agent_name:16} :{port}  [{marker}]")

    if unisolated:
        print(
            f"\nWARNING: {', '.join(unisolated)} have no .venv and are running on "
            f"\n{' ' * 9}the current interpreter - their dependencies are NOT isolated."
            f"\n{' ' * 9}Run `python -m backend.run_agents --setup` to fix."
        )

    print(f"\n{len(processes)} agents starting. Ctrl+C to stop them all.\n")

    try:
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        print("\nstopping agents...")
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--setup",
        action="store_true",
        help="create a venv per agent and install its requirements, then exit",
    )
    args = parser.parse_args()

    if args.setup:
        setup()
    else:
        serve()


if __name__ == "__main__":
    main()
