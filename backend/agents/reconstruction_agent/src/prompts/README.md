# Prompts

Every prompt this agent sends to an LLM lives here as markdown. No prompt text
is hardcoded in Python — `agent/prompts/loader.py` reads these files, and
`agent/prompts/__init__.py` fills in their placeholders.

A prompt change alters agent behaviour as much as a code change does. Keeping
them here means a wording change is a visible diff on its own file, reviewable
by someone who does not read Python.

## Naming

Files are `<role>.<kind>.md` and loaded by that stem:

| File | Loaded as | Used by |
| --- | --- | --- |
| `planner.system.md` | `planner.system` | `agent/planning/planner.py` |
| `planner.user.md` | `planner.user` | `agent/planning/planner.py` |
| `critic.system.md` | `critic.system` | `agent/reasoning/critic.py` |
| `critic.user.md` | `critic.user` | `agent/reasoning/critic.py` |
| `explanation.system.md` | `explanation.system` | `agent/reasoning/reasoner.py` |
| `explanation.user.md` | `explanation.user` | `agent/reasoning/reasoner.py` |

`tests/unit/test_prompts.py` asserts every one of these exists and renders, so
renaming a file without updating the code fails in CI rather than at the first
LLM call in production.

## The one editing rule that matters

**`.system.md` files are loaded verbatim. `.user.md` files are rendered
through `str.format`.**

That means a `.user.md` file may only contain braces that are real
placeholders. If you need a literal `{` or `}` in a *user* prompt — a JSON
example, say — double it: `{{` and `}}`.

System prompts have no such restriction, which is why the JSON output schemas
live in the `.system.md` files. Keep it that way and you can write JSON
examples freely.

The placeholders each user prompt expects:

- `planner.user` — `instruction`, `organism`, `gaps`, `tools`, `critiques_section`
- `critic.user` — `gap`, `candidate`, `evidence`
- `explanation.user` — `gap`, `candidate`, `evidence`

A missing value raises `PromptRenderError` naming the prompt and the key, and
an unescaped literal brace raises one telling you to double it.

## Caching

Prompts are cached after first read, so editing a file does not take effect in
a running process. Restart the agent, or call
`agent.prompts.clear_cache()`.
