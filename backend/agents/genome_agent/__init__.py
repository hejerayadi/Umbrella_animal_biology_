# Package marker for genome_agent.
#
# This file was missing from the repo (never committed — compare to
# trait_discovery_agent/__init__.py, which exists). Without it, genome_agent
# is only an implicit Python 3 namespace package, which is fine for plain
# `import genome_agent...` but breaks pytest's rootdir-walk: pytest decides
# how far up to build a test's dotted module path by walking up through
# directories that have __init__.py, and a namespace package (no
# __init__.py) stops that walk one level too early. That's what was making
# `python -m pytest genome_agent/tests -v` — the exact command
# docker-compose.yml's `test` service runs — fail to even collect most
# test files with "attempted relative import beyond top-level package".
