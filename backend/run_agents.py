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

uv is not a hard requirement. Without it, a `pyproject.toml` agent is set up
from the dependency ranges it declares, using the same pip path as the others
- the environment works, it is just resolved fresh instead of replayed from
the lock file. Install uv (`pip install uv`) to get the locked versions.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import tomllib
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from .registry import AGENT_ENDPOINTS, _AGENT_FOLDERS

_AGENTS_DIR = Path(__file__).parent / "agents"
_REPO_ROOT = Path(__file__).parent.parent

# Where a venv keeps its interpreter differs by platform.
_VENV_PYTHON = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")

# Extra environment variables for specific agents, applied when launching them.
#
# The Trait Discovery Agent picks its implementation from TRAIT_AGENT_IMPL and
# defaults to the canned stub in its `mock.py`. That default is deliberate (see
# its api.py) so the port always opens even without a NIM key - but it means the
# orchestrator silently gets fake traits, which downstream agents like
# ImageGeneration then render as if they were real findings. Running the real
# LangGraph workflow is the point of the system, so ask for it here.
#
# It has to be a real environment variable: that agent's api.py reads it with
# os.getenv at import time, and nothing calls load_dotenv before then, so
# putting it in its .env would silently do nothing.
#
# The Evolution Agent is the same story with a different name: its api.py
# defaults to the stub in `mock.py`, which only ever escalates to Genome and
# Literature and then answers "Evolutionary relationship completed." The real
# LangGraph pipeline (species resolver -> molecular comparison -> phylogenetic
# reconstruction) is behind EVOLUTION_AGENT_IMPL=orchestrator. It needs that
# agent's own .env for intent classification; without one it logs the reason
# and falls back to the mock rather than failing to start.
#
# Anything already exported in the shell wins, so `set TRAIT_AGENT_IMPL=mock`
# still gets you the stub for an offline demo.
#
# The Trait Discovery Agent also reaches the Literature Agent directly, rather
# than through the orchestrator, for the evidence behind a trait-gene link. It
# defaults that URL to `http://literature-agent:8000/execute` - a compose
# service name, because the client was written for a deployment where the two
# are containers on one network. Nothing resolves that hostname when the agents
# are nine local uvicorn processes, so it is pointed at the port this launcher
# actually starts the Literature Agent on.
_AGENT_ENV: dict[str, dict[str, str]] = {
    "Trait": {
        "TRAIT_AGENT_IMPL": "workflow",
        "LITERATURE_AGENT_URL": f"{AGENT_ENDPOINTS['Literature']}/execute",
    },
    "Evolution": {"EVOLUTION_AGENT_IMPL": "orchestrator"},
    "Biodiversity": {"BIODIVERSITY_AGENT_IMPL": "orchestrator"},
}


def _agent_dotenv(agent_name: str) -> dict[str, str]:
    """The variables in one agent's own `.env`, or an empty dict if it has none.

    Most agents load their `.env` themselves, somewhere in the import chain that
    builds their LLM client. Not all do: the Multimodal Recognition Agent reads
    `os.environ` and nothing else on purpose - `config.py` says so, so that no
    credential is ever read or cached by that module - and it leaves populating
    the environment to whoever starts the service. That is this launcher, and
    until this function existed nobody did it: the agent's `remote`/`real`/
    `azure` settings were never seen, every provider fell back to its mock
    default, and the UI labelled real requests "Demonstration data".

    Deliberately NOT fixed inside that agent's `api.py`: four of its test modules
    import `api.py`, and its suite asserts that no offline test ever loads a
    `.env`. Loading one there would put live Azure credentials into the pytest
    process and turn a credential-leak test green for the wrong reason.

    `dotenv_values` parses the file without touching this process's environment,
    so the launcher itself stays clean and only the child sees these values.
    """
    env_file = _AGENTS_DIR / _AGENT_FOLDERS[agent_name] / ".env"
    if not env_file.is_file():
        return {}
    try:
        from dotenv import dotenv_values
    except ImportError:
        # Loudly, not silently: an agent configured for real providers that
        # starts on its mocks looks like it is working, which is the worse
        # failure. Nothing here prints a value - only the path.
        print(
            f"  ! {agent_name}: {env_file} exists but python-dotenv is not "
            f"installed in {sys.executable}, so it will start on its defaults. "
            f"Install python-dotenv to have it read.",
            file=sys.stderr,
        )
        return {}
    return {key: value for key, value in dotenv_values(env_file).items() if value is not None}


def _agent_environment(agent_name: str) -> dict[str, str]:
    """The environment for one agent: this process's, its `.env`, its defaults.

    Precedence, strongest first: anything exported in the shell, then the
    agent's own `.env`, then the launcher's per-agent defaults below. The shell
    keeping the last word is the contract `_AGENT_ENV` already stated; an
    agent's `.env` outranking a launcher default is the same idea one level
    down - explicit per-agent configuration beats a blanket nudge.
    """
    environment = dict(os.environ)
    for key, value in _agent_dotenv(agent_name).items():
        environment.setdefault(key, value)
    for key, value in _AGENT_ENV.get(agent_name, {}).items():
        environment.setdefault(key, value)
    return environment


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
    """Turn a wall of pip or uv output into one actionable line, when we recognise it."""

    # Windows locks the .pyd/.dll files a running process has loaded, so
    # rewriting an environment underneath a live agent fails on the first
    # compiled package it tries to replace - numpy, usually. Neither pip nor uv
    # says why, they just report the rename that was denied, which reads like a
    # permissions problem and sends you looking for an admin shell.
    if (
        "Access is denied" in pip_output
        or "os error 5" in pip_output
        or "WinError 5" in pip_output
    ):
        return (
            "A file in that agent's .venv is in use, so it could not be replaced. "
            "This is almost always an agent that is still running - Windows locks "
            "the libraries a live process has loaded. Stop `python -m "
            "backend.run_agents` (and any stray uvicorn) and run --setup again."
        )

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
            f"from source and failed. Bump the pin in the dependency file named "
            f"above to a version that ships wheels for your Python."
        )
    if "No matching distribution found" in pip_output:
        return "A pinned version does not exist for this Python. Check the pins."
    return None


def _manifest(folder: str) -> Path:
    """The dependency file one agent declares, whichever style it uses."""
    pyproject = _AGENTS_DIR / folder / "pyproject.toml"
    return pyproject if pyproject.exists() else _AGENTS_DIR / folder / "requirements.txt"


@lru_cache(maxsize=1)
def _uv_command() -> list[str] | None:
    """How to invoke uv here, or None if it is not installed at all.

    Two ways it can be present. Installed standalone it is a `uv` executable on
    PATH; installed with `pip install uv` into the environment running this
    script it is a module, and on Windows the shim only lands on PATH while
    that venv is activated. Checking both means `python -m backend.run_agents
    --setup` works either way.

    Cached because answering it costs a subprocess, and `setup()` asks once per
    pyproject agent plus once more inside `_setup_with_uv`. uv does not appear
    or vanish part-way through one run.
    """
    on_path = shutil.which("uv")
    if on_path:
        return [on_path]

    found = subprocess.run(
        [sys.executable, "-m", "uv", "--version"],
        capture_output=True,
        text=True,
    )
    if found.returncode == 0:
        return [sys.executable, "-m", "uv"]

    return None


def _setup_with_uv(folder: str) -> tuple[bool, str]:
    """Build one agent's environment from its pyproject.toml + uv.lock.

    `uv sync` creates the `.venv` itself and installs the exact versions the
    lock file records, so a developer's environment matches the one CI resolved
    rather than whatever the ranges happen to allow today.
    """
    uv = _uv_command()
    if uv is None:
        return False, "uv is not installed"

    result = subprocess.run(
        [*uv, "sync", "--project", str(_AGENTS_DIR / folder)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout + result.stderr


def _pyproject_dependencies(folder: str) -> list[str]:
    """The runtime requirements one pyproject declares, as pip would read them.

    Only `[project].dependencies` - not the optional extras, and not the dev
    group. Those are opt-in under uv too (`uv sync --extra embeddings`), and
    the extras here are the expensive ones: the Protein agent's `embeddings`
    extra pulls in torch at around 2 GB for a feature it falls back out of.
    """
    with (_AGENTS_DIR / folder / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    return list(pyproject.get("project", {}).get("dependencies", []))


def _setup_with_pyproject_pip(folder: str) -> tuple[bool, str]:
    """Build a pyproject agent's environment WITHOUT uv, using pip.

    The fallback for a machine that has no uv. It reads the same dependency
    list uv would, but resolves it fresh from the version ranges instead of
    replaying `uv.lock`, so the result is a working environment rather than a
    reproducible one. That is the right trade for local development - the
    alternative is one agent that cannot start at all - but it is why the
    caller labels this run differently in the output.

    Nothing is built or installed as a package: these projects set
    `package = false` because the importable path is
    `backend.agents.<agent>.*` from the repository root, not this directory.
    """
    venv_dir = _venv_dir(folder)
    if not venv_dir.exists():
        created = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            capture_output=True,
            text=True,
        )
        if created.returncode != 0:
            return False, created.stdout + created.stderr

    dependencies = _pyproject_dependencies(folder)
    if not dependencies:
        return False, f"{folder}/pyproject.toml declares no [project].dependencies"

    python = venv_dir / _VENV_PYTHON

    # A `.venv` left behind by an earlier `uv sync` has no pip in it - uv
    # installs packages itself and does not seed one. Reusing that directory
    # would fail with "No module named pip", which reads like a broken Python
    # rather than a missing bootstrap, so put pip there first.
    has_pip = subprocess.run(
        [str(python), "-m", "pip", "--version"], capture_output=True, text=True
    )
    if has_pip.returncode != 0:
        seeded = subprocess.run(
            [str(python), "-m", "ensurepip", "--default-pip"],
            capture_output=True,
            text=True,
        )
        if seeded.returncode != 0:
            return False, seeded.stdout + seeded.stderr

    installed = subprocess.run(
        [str(python), "-m", "pip", "install", "-q", *dependencies],
        capture_output=True,
        text=True,
    )
    return installed.returncode == 0, installed.stdout + installed.stderr


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
    print("This takes a while - some agents pin large packages (torch).")
    print("Stop any running agents first: this rewrites the environments they "
          "are using, and\n         on Windows a live process locks its own "
          "libraries against replacement.\n")

    failures: list[tuple[str, str, str]] = []

    for agent_name, folder in _AGENT_FOLDERS.items():
        manifest = _manifest(folder)

        print(f"--- {agent_name} ({folder})")
        if manifest.name != "pyproject.toml":
            ok, output = _setup_with_pip(folder)
            tool = "pip install"
        elif _uv_command() is not None:
            ok, output = _setup_with_uv(folder)
            tool = "uv sync"
        else:
            # No uv anywhere. Install the declared ranges with pip rather than
            # leaving this agent with no environment - it still runs, it just
            # is not pinned to the lock file.
            ok, output = _setup_with_pyproject_pip(folder)
            tool = "pip install (no uv - lock file NOT used)"

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
            env=_agent_environment(agent_name),
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
