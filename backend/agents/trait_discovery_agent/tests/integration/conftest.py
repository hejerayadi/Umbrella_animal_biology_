
import pytest


@pytest.fixture
def patch_llm_boundary(monkeypatch):

    import workflows.trait_discovery_graph as td_graph_module
    import workflows.nodes.escalation_nodes as esc_module
    from workflows.capability_resolver import CapabilityResolution

    calls = {"resolve": [], "explain": []}

    async def fake_resolve(*, waiting_agent, need_description, known_context):
        calls["resolve"].append((waiting_agent, need_description))
        
        target = "Literature Agent" if waiting_agent == "Literature Support" else "Genome Agent"
        return CapabilityResolution(
            target_agent=target,
            prompt_to_target_agent=f"[integration-test escalation] {need_description}",
            reasoning="stubbed LLM boundary for integration test",
        )

    async def fake_explain(**kwargs):
        calls["explain"].append(kwargs)
        return f"Integration-test explanation for {kwargs['trait_name']}."

    monkeypatch.setattr(esc_module, "resolve_capability", fake_resolve)
    monkeypatch.setattr(td_graph_module, "write_explanation", fake_explain)

    return calls


@pytest.fixture
def embed_calls(monkeypatch):
  
    import kb.qdrant_store as qdrant_store

    calls: list[str] = []
    real_embed_text = qdrant_store.embed_text

    async def counting_embed_text(text: str):
        calls.append(text)
        return await real_embed_text(text)

    monkeypatch.setattr(qdrant_store, "embed_text", counting_embed_text)
    return calls