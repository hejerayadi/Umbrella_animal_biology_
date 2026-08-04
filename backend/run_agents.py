"""Start every agent service at once, for local development.

The orchestrator now calls agents over HTTP, so all nine have to be running
before a query can get past the first worker. Starting nine uvicorn processes
by hand is tedious; this does it in one command.

Run from the repository root:

    python -m backend.run_agents

Each agent is served on the port `backend/registry.py` expects. Ctrl+C stops
all of them. Their logs are interleaved on this terminal, prefixed with the
agent name.

This is a convenience for local runs only - in a real deployment each agent
is its own container with its own virtual environment, which is exactly why
each has its own requirements.txt.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from urllib.parse import urlparse

from .registry import AGENT_ENDPOINTS, _AGENT_FOLDERS


def _stream(prefix: str, process: subprocess.Popen) -> None:
    """Echo one child's output with its agent name in front of every line."""
    for line in process.stdout:
        print(f"[{prefix}] {line.rstrip()}", flush=True)


def main() -> None:
    processes: list[subprocess.Popen] = []

    for agent_name, folder in _AGENT_FOLDERS.items():
        port = urlparse(AGENT_ENDPOINTS[agent_name]).port
        module = f"backend.agents.{folder}.api:app"

        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", module, "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes.append(process)
        threading.Thread(target=_stream, args=(agent_name, process), daemon=True).start()
        print(f"starting {agent_name:16} -> http://localhost:{port}  ({module})")

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


if __name__ == "__main__":
    main()
