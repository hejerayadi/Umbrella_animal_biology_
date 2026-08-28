# Reasoner policy
The reasoner does not call an LLM. It deterministically turns the next approved tool action into typed arguments, so it cannot invent biological evidence, database scopes, scores, or a DNA sequence.

The planner and critic are the only model-guided nodes and load their prompts from Markdown at runtime.
