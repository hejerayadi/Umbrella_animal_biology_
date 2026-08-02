import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for candidate in [ROOT, os.path.join(ROOT, "backend", "agents", "trait_discovery_agent")]:
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)
