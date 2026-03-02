"""Live checks against the real Portkey gateway: the assumptions the agent depends on.

Run with:  RUN_LIVE=1 uv run pytest -m live -v     (reads PORTKEY_API_KEY etc. from .env)
"""

import json
import os

import pytest

from app.config import Settings
from app.llm.embeddings import PortkeyEmbedder
from app.llm.portkey import OpenAIChatLLM
from app.llm.types import LLMResult, TextDelta

# Captured at import, before the autouse fixture clears settings variables for offline tests.
_ENV = {k: v for k, v in os.environ.items() if k.startswith(("PORTKEY_", "EMBEDDING_", "CHAT_MODEL", "LLM_"))}

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to call the real gateway"),
]

RECORD_CITY = {
    "type": "function",
    "function": {
        "name": "record_city",
        "description": "Record the city the user mentions.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}


@pytest.fixture
def live_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    for key, value in _ENV.items():
        monkeypatch.setenv(key, value)
    try:
        return Settings()
    except Exception as exc:
        pytest.skip(f"live settings unavailable: {exc}")


async def collect(llm: OpenAIChatLLM, messages, tools=None, tool_choice=None):
    items = [
        item async for item in llm.stream(messages, tools, trace_id="live-test", tool_choice=tool_choice)
    ]
    result = items[-1]
    assert isinstance(result, LLMResult)
    return [i for i in items if isinstance(i, TextDelta)], result


async def test_streaming_chat(live_settings):
    deltas, result = await collect(
        OpenAIChatLLM(live_settings), [{"role": "user", "content": "Reply with the single word: ready"}]
    )
    assert deltas, "expected streamed text deltas"
    assert "ready" in result.content.lower()


async def test_streaming_tool_call(live_settings):
    _, result = await collect(
        OpenAIChatLLM(live_settings),
        [
            {"role": "system", "content": "Always call record_city when the user mentions a city."},
            {"role": "user", "content": "I live in Paris."},
        ],
        tools=[RECORD_CITY],
    )
    assert result.tool_calls, "the gateway/model did not return a tool call"
    call = result.tool_calls[0]
    assert call.name == "record_city"
    assert call.id
    assert "paris" in json.loads(call.arguments)["city"].lower()


async def test_tool_choice_none_returns_text(live_settings):
    _, result = await collect(
        OpenAIChatLLM(live_settings),
        [{"role": "user", "content": "I live in Paris. Say hello."}],
        tools=[RECORD_CITY],
        tool_choice="none",
    )
    assert not result.tool_calls
    assert result.content


async def test_usage_is_reported(live_settings):
    if not live_settings.llm_stream_usage:
        pytest.skip("LLM_STREAM_USAGE=false")
    _, result = await collect(OpenAIChatLLM(live_settings), [{"role": "user", "content": "Say hi."}])
    assert result.prompt_tokens > 0


async def test_embeddings(live_settings):
    embedder = PortkeyEmbedder.from_settings(live_settings)
    try:
        vectors = await embedder.embed(["protein folding", "galaxy formation"])
    finally:
        await embedder.aclose()
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) > 0
