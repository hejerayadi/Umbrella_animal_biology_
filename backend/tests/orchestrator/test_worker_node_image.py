"""Worker node: who gets the attached image, and who must never see it.

`image_store.py` explains why this matters: the shared `context` is broadcast.
The worker node POSTs it in full to every agent, the Responder renders every
key into an LLM prompt, and the chat endpoint returns it to the browser. So an
image travels as a short id, and `_context_for` swaps that id back for the real
bytes for the ONE agent that can look at it - stripping it for everyone else.

That swap had no tests. These pin it, because the failure it prevents is silent:
nothing crashes when an image leaks into eight extra agents, it just costs
hundreds of thousands of prompt tokens and ships the user's photo to services
that never needed it.

No network and no LLM - the HTTP client is a stub and the store is a fresh
in-memory one, so these run in milliseconds.
"""

from __future__ import annotations

import base64

import pytest

from backend.image_store import ImageStore
from backend.orchestrator.langgraph.nodes import worker_node
from backend.orchestrator.state import WorkflowState
from backend.registry import AGENT_ENDPOINTS

# The agent allowed to receive the bytes, and the key it reads them from. Both
# are duplicated from `worker_node` on purpose: if either name changes, the
# Recognition agent stops seeing images, and a test that imported the constant
# would happily change with it and prove nothing.
IMAGE_CONSUMER = "Multimodal"
IMAGE_KEY = "recognition_image"
IMAGE_ID_KEY = "recognition_image_id"

# Enough to satisfy the store's magic-byte sniff. `ImageStore` identifies the
# type from the leading bytes and never decodes the pixels, so a real encoder
# is not needed here - and the root environment has no Pillow to provide one.
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake pixel payload"

# Every agent that must be given nothing. Derived from the registry rather than
# hard-coded, so an agent added later is covered by these tests automatically.
OTHER_AGENTS = sorted(name for name in AGENT_ENDPOINTS if name != IMAGE_CONSUMER)


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Stands in for the pooled httpx client, recording what was sent."""

    def __init__(self, payload: dict):
        self._payload = payload
        self.sent: list[dict] = []

    def post(self, url, json=None, **kwargs):
        self.sent.append(json)
        return _FakeResponse(self._payload)


@pytest.fixture
def store(monkeypatch) -> ImageStore:
    """A fresh store in place of the process-wide one, so tests cannot bleed."""
    fresh = ImageStore()
    monkeypatch.setattr(worker_node, "IMAGE_STORE", fresh)
    return fresh


@pytest.fixture
def stored_image(store):
    return store.add(PNG_BYTES, "observation.png")


# --- the agent that may look at the image ----------------------------------

def test_the_recognition_agent_receives_the_image_bytes(stored_image):
    context = _context({IMAGE_ID_KEY: stored_image.image_id})

    assert context[IMAGE_KEY] == {
        "data_url": stored_image.as_data_url(),
        "filename": "observation.png",
    }


def test_the_data_url_carries_the_real_uploaded_bytes(stored_image):
    """A placeholder that merely looks like a data URL would pass the test
    above. Decode it and check the pixels are the ones that were uploaded."""

    context = _context({IMAGE_ID_KEY: stored_image.image_id})

    _, _, payload = context[IMAGE_KEY]["data_url"].partition(",")
    assert base64.b64decode(payload) == PNG_BYTES


def test_the_id_is_swapped_out_not_sent_alongside_the_bytes(stored_image):
    """The agent reads `recognition_image` and nothing else. Forwarding the id
    as well would leak an internal handle for no benefit."""

    context = _context({IMAGE_ID_KEY: stored_image.image_id})

    assert IMAGE_ID_KEY not in context


# --- every other agent ------------------------------------------------------

@pytest.mark.parametrize("agent_name", OTHER_AGENTS)
def test_no_other_agent_is_given_the_image(agent_name, stored_image):
    context = _context({IMAGE_ID_KEY: stored_image.image_id}, agent=agent_name)

    assert IMAGE_KEY not in context
    assert IMAGE_ID_KEY not in context


@pytest.mark.parametrize("agent_name", OTHER_AGENTS)
def test_no_other_agent_is_given_the_id_either(agent_name, stored_image):
    """The id alone is harmless to an agent that cannot resolve it, but it is
    still a key the Responder would render into a prompt."""

    context = _context({IMAGE_ID_KEY: stored_image.image_id}, agent=agent_name)

    assert stored_image.image_id not in str(context)


# --- everything else in the context ----------------------------------------

@pytest.mark.parametrize("agent_name", [IMAGE_CONSUMER, *OTHER_AGENTS])
def test_the_rest_of_the_context_is_passed_through_untouched(
    agent_name, stored_image
):
    context = _context(
        {
            IMAGE_ID_KEY: stored_image.image_id,
            "species": "Panthera leo",
            "gbif_id": 5219404,
        },
        agent=agent_name,
    )

    assert context["species"] == "Panthera leo"
    assert context["gbif_id"] == 5219404


@pytest.mark.parametrize("agent_name", [IMAGE_CONSUMER, *OTHER_AGENTS])
def test_a_context_with_no_image_is_left_alone(agent_name, store):
    original = {"species": "Panthera leo"}

    assert _context(original, agent=agent_name) == original


# --- the id that no longer resolves ----------------------------------------

def test_an_expired_id_sends_no_image_rather_than_failing(store):
    """Evicted, or a stale id from an old browser tab. The agent answers
    MISSING_IMAGE, which is a clearer result than a transport error here."""

    context = _context({IMAGE_ID_KEY: "an-id-that-was-never-stored"})

    assert IMAGE_KEY not in context
    assert IMAGE_ID_KEY not in context


def test_an_evicted_id_sends_no_image(store):
    """The store is bounded and evicts oldest-first, so a long session really
    does produce ids that no longer resolve."""

    first = store.add(PNG_BYTES, "first.png")
    for index in range(40):  # comfortably past _MAX_IMAGES
        store.add(PNG_BYTES, f"later-{index}.png")

    assert store.get(first.image_id) is None
    assert IMAGE_KEY not in _context({IMAGE_ID_KEY: first.image_id})


# --- the guarantee at the boundary that actually sends bytes ---------------

def test_the_posted_body_carries_the_image_only_for_the_recognition_agent(
    monkeypatch, stored_image
):
    """The helper is where the decision is made, but the POST is where a leak
    would actually happen. Assert on the request body itself."""

    state = WorkflowState(
        user_query="What animal is this?",
        context={IMAGE_ID_KEY: stored_image.image_id},
    )

    consumer = _FakeClient({"status": "completed", "output": {}})
    worker_node.make_worker_node(
        IMAGE_CONSUMER, "http://x", client=consumer
    )(state)

    other = _FakeClient({"status": "completed", "output": {}})
    worker_node.make_worker_node("Genome", "http://x", client=other)(state)

    assert IMAGE_KEY in consumer.sent[0]["context"]
    assert IMAGE_KEY not in other.sent[0]["context"]
    # The bytes themselves must not appear anywhere in the other agent's body.
    assert base64.b64encode(PNG_BYTES).decode() not in str(other.sent[0])


def _context(context: dict, agent: str = IMAGE_CONSUMER) -> dict:
    return worker_node._context_for(agent, context)
