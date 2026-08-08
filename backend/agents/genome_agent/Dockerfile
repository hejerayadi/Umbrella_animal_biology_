# Dockerfile
#
# Build context = the genome_agent/ directory itself, e.g.:
#   docker build -t genome-agent .        (run from inside genome_agent/)
#
# The package is copied to /app/genome_agent (not flattened into /app)
# because orchestrator.py and its modules use relative imports
# (`from .subagents...`, `from .workflows...`), so genome_agent must exist
# on disk as an actual subpackage with a real parent directory (/app).
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./genome_agent/requirements.txt
RUN pip install --no-cache-dir -r genome_agent/requirements.txt

COPY . ./genome_agent

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Real end-to-end execution of the LangGraph orchestrator across scenarios
# (happy path, unknown species, NEEDS_AGENT escalation, hard failure,
# partial results) — not a pytest run.
CMD ["python", "-m", "genome_agent.scripts.run_orchestrator_scenarios"]
