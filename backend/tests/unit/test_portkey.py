import json

import httpx2
import pytest

from app.config import Settings
from app.llm.portkey import OpenAIChatLLM, create_chat_client, portkey_headers
from app.llm.types import LLMAuthError, LLMBadRequestError, LLMResult, LLMUnavailableError, TextDelta


def sse_body(chunks: list[dict]) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


def chunk(delta: dict, finish_reason: str | None = None, usage: dict | None = None, choices=True) -> dict:
    body = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": "m"}
    body["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish_reason}] if choices else []
    if usage:
        body["usage"] = usage
    return body


class Recorder:
    def __init__(self, status: int = 200, body: bytes = b"", json_body: dict | None = None):
        self.requests: list[httpx2.Request] = []
        self.status = status
        self.body = body
        self.json_body = json_body

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.json_body is not None:
            return httpx2.Response(self.status, json=self.json_body)
        return httpx2.Response(self.status, content=self.body, headers={"content-type": "text/event-stream"})

    @property
    def last_json(self) -> dict:
        return json.loads(self.requests[-1].content)


def make_llm(settings: Settings, recorder: Recorder) -> OpenAIChatLLM:
    client = create_chat_client(
        settings, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(recorder)), max_retries=0
    )
    return OpenAIChatLLM(settings, client=client)


async def collect(llm: OpenAIChatLLM, tools=None, trace_id="run-1"):
    return [item async for item in llm.stream([{"role": "user", "content": "hi"}], tools, trace_id)]


def test_portkey_headers_without_virtual_key():
    assert portkey_headers("pk") == {"x-portkey-api-key": "pk"}


def test_portkey_headers_with_virtual_key():
    assert portkey_headers("pk", "vk") == {"x-portkey-api-key": "pk", "x-portkey-virtual-key": "vk"}


async def test_request_uses_portkey_headers_and_no_authorization(settings):
    settings.portkey_virtual_key = None
    rec = Recorder(body=sse_body([chunk({"content": "ok"}, "stop")]))
    await collect(make_llm(settings, rec), trace_id="run-42")
    headers = rec.requests[0].headers
    assert headers["x-portkey-api-key"] == "test-portkey-key"
    assert headers["x-portkey-trace-id"] == "run-42"
    assert "authorization" not in headers
    assert "x-portkey-virtual-key" not in headers
    assert str(rec.requests[0].url) == f"{settings.portkey_base_url}/chat/completions"


async def test_virtual_key_header_sent_when_configured(tmp_path):
    settings = Settings(_env_file=None, portkey_api_key="pk", portkey_virtual_key="openai-vk")
    rec = Recorder(body=sse_body([chunk({"content": "ok"}, "stop")]))
    await collect(make_llm(settings, rec))
    assert rec.requests[0].headers["x-portkey-virtual-key"] == "openai-vk"


async def test_request_body_has_model_tools_and_stream_usage(settings):
    rec = Recorder(body=sse_body([chunk({"content": "ok"}, "stop")]))
    tools = [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {}}}]
    await collect(make_llm(settings, rec), tools=tools)
    body = rec.last_json
    assert body["model"] == settings.chat_model
    assert body["stream"] is True
    assert body["tools"] == tools
    assert body["stream_options"] == {"include_usage": True}


async def test_tools_omitted_when_none(settings):
    rec = Recorder(body=sse_body([chunk({"content": "ok"}, "stop")]))
    await collect(make_llm(settings, rec), tools=None)
    assert "tools" not in rec.last_json


async def test_stream_usage_can_be_disabled(settings):
    settings.llm_stream_usage = False
    rec = Recorder(body=sse_body([chunk({"content": "ok"}, "stop")]))
    await collect(make_llm(settings, rec))
    assert "stream_options" not in rec.last_json


async def test_text_deltas_then_result_with_usage(settings):
    rec = Recorder(
        body=sse_body(
            [
                chunk({}, choices=False),  # Azure-style prompt-filter chunk with no choices
                chunk({"role": "assistant", "content": "Hel"}),
                chunk({"content": "lo"}, "stop"),
                chunk(
                    {}, choices=False, usage={"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13}
                ),
            ]
        )
    )
    items = await collect(make_llm(settings, rec))
    assert items[:2] == [TextDelta("Hel"), TextDelta("lo")]
    result = items[-1]
    assert isinstance(result, LLMResult)
    assert result.content == "Hello"
    assert result.tool_calls == []
    assert (result.prompt_tokens, result.completion_tokens) == (11, 2)
    assert result.finish_reason == "stop"


async def test_streamed_tool_call_fragments_are_reassembled(settings):
    rec = Recorder(
        body=sse_body(
            [
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "get_publication", "arguments": ""},
                            }
                        ]
                    }
                ),
                chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"publication_'}}]}),
                chunk(
                    {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": 'id": 7}'}},
                            {
                                "index": 1,
                                "id": "call_b",
                                "type": "function",
                                "function": {"name": "resolve_author", "arguments": '{"name"'},
                            },
                        ]
                    }
                ),
                chunk(
                    {"tool_calls": [{"index": 1, "function": {"arguments": ': "Smith J"}'}}]}, "tool_calls"
                ),
            ]
        )
    )
    result = (await collect(make_llm(settings, rec)))[-1]
    assert [(c.id, c.name, c.arguments) for c in result.tool_calls] == [
        ("call_a", "get_publication", '{"publication_id": 7}'),
        ("call_b", "resolve_author", '{"name": "Smith J"}'),
    ]
    assert result.finish_reason == "tool_calls"


async def test_missing_tool_call_id_is_generated(settings):
    rec = Recorder(
        body=sse_body(
            [
                chunk(
                    {"tool_calls": [{"index": 0, "function": {"name": "list_documents", "arguments": "{}"}}]},
                    "tool_calls",
                )
            ]
        )
    )
    result = (await collect(make_llm(settings, rec)))[-1]
    assert result.tool_calls[0].id.startswith("call_")
    assert len(result.tool_calls[0].id) > len("call_")


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (401, LLMAuthError),
        (403, LLMAuthError),
        (503, LLMUnavailableError),
        (429, LLMUnavailableError),
        (400, LLMBadRequestError),
    ],
)
async def test_http_errors_map_to_llm_errors(settings, status, error_type):
    rec = Recorder(status=status, json_body={"error": {"message": "nope", "type": "x"}})
    with pytest.raises(error_type) as exc:
        await collect(make_llm(settings, rec))
    assert exc.value.code in {"llm_auth_failed", "llm_unavailable", "llm_bad_request"}


async def test_auth_error_message_mentions_the_setting(settings):
    rec = Recorder(status=401, json_body={"error": {"message": "invalid key"}})
    with pytest.raises(LLMAuthError) as exc:
        await collect(make_llm(settings, rec))
    assert "PORTKEY_API_KEY" in exc.value.message


async def test_connection_error_maps_to_unavailable(settings):
    def boom(request):
        raise httpx2.ConnectError("refused", request=request)

    client = create_chat_client(
        settings, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(boom)), max_retries=0
    )
    with pytest.raises(LLMUnavailableError):
        await collect(OpenAIChatLLM(settings, client=client))
