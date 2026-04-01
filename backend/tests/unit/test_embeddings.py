import json

import httpx
import pytest
import respx

from app.llm.embeddings import EmbeddingAuthError, EmbeddingError, PortkeyEmbedder

URL = "https://gw.example/v1/embeddings"


class SleepRecorder:
    def __init__(self):
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def ok_response(request: httpx.Request) -> httpx.Response:
    inputs = json.loads(request.content)["input"]
    # Return out of order to prove results are sorted by index.
    data = [
        {"object": "embedding", "index": i, "embedding": [float(len(t)), float(i)]}
        for i, t in enumerate(inputs)
    ]
    return httpx.Response(200, json={"object": "list", "data": list(reversed(data))})


def make(batch_size=2, sleep=None, virtual_key=None) -> PortkeyEmbedder:
    return PortkeyEmbedder(
        base_url="https://gw.example/v1",
        api_key="emb-key",
        virtual_key=virtual_key,
        model="text-embedding-3-small",
        batch_size=batch_size,
        sleep=sleep or SleepRecorder(),
    )


@respx.mock
async def test_batches_requests_and_preserves_order():
    route = respx.post(URL).mock(side_effect=ok_response)
    vectors = await make(batch_size=2).embed(["a", "bb", "ccc", "dddd", "eeeee"])
    assert route.call_count == 3
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]


@respx.mock
async def test_sends_portkey_headers_and_payload():
    route = respx.post(URL).mock(side_effect=ok_response)
    await make(virtual_key="emb-vk").embed(["hello"])
    request = route.calls[0].request
    assert request.headers["x-portkey-api-key"] == "emb-key"
    assert request.headers["x-portkey-virtual-key"] == "emb-vk"
    assert "authorization" not in request.headers
    assert json.loads(request.content) == {
        "model": "text-embedding-3-small",
        "input": ["hello"],
        "encoding_format": "float",
    }


@respx.mock
async def test_empty_input_makes_no_request():
    route = respx.post(URL).mock(side_effect=ok_response)
    assert await make().embed([]) == []
    assert route.call_count == 0


async def test_blank_text_is_rejected():
    with pytest.raises(ValueError):
        await make().embed(["ok", "   "])


@respx.mock
async def test_retries_429_honouring_retry_after():
    sleep = SleepRecorder()
    route = respx.post(URL).mock(side_effect=[httpx.Response(429, headers={"Retry-After": "3"}), ok_response])
    await make(sleep=sleep).embed(["a"])
    assert route.call_count == 2
    assert sleep.calls == [3.0]


@respx.mock
async def test_retries_5xx_with_exponential_backoff():
    sleep = SleepRecorder()
    respx.post(URL).mock(side_effect=[httpx.Response(503), httpx.Response(502), ok_response])
    await make(sleep=sleep).embed(["a"])
    assert sleep.calls == [1.0, 2.0]


@respx.mock
async def test_retries_transport_errors():
    sleep = SleepRecorder()
    respx.post(URL).mock(side_effect=[httpx.ConnectError("down"), ok_response])
    await make(sleep=sleep).embed(["a"])
    assert sleep.calls == [1.0]


@respx.mock
async def test_gives_up_after_five_attempts():
    sleep = SleepRecorder()
    route = respx.post(URL).mock(return_value=httpx.Response(503, text="busy"))
    with pytest.raises(EmbeddingError) as exc:
        await make(sleep=sleep).embed(["a"])
    assert route.call_count == 5
    assert len(sleep.calls) == 4
    assert "503" in str(exc.value)


@respx.mock
async def test_bad_request_is_not_retried():
    route = respx.post(URL).mock(return_value=httpx.Response(400, json={"error": {"message": "bad model"}}))
    with pytest.raises(EmbeddingError) as exc:
        await make().embed(["a"])
    assert route.call_count == 1
    assert "bad model" in str(exc.value)


@respx.mock
async def test_auth_failure_raises_auth_error():
    respx.post(URL).mock(return_value=httpx.Response(401, json={"error": {"message": "invalid"}}))
    with pytest.raises(EmbeddingAuthError) as exc:
        await make().embed(["a"])
    assert exc.value.code == "llm_auth_failed"


@respx.mock
async def test_vector_count_mismatch_raises():
    respx.post(URL).mock(return_value=httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]}))
    with pytest.raises(EmbeddingError):
        await make(batch_size=10).embed(["a", "b"])


def test_from_settings_uses_embedding_credentials(settings):
    settings.embedding_portkey_api_key = None
    embedder = PortkeyEmbedder.from_settings(settings)
    assert embedder.model == settings.embedding_model
    assert embedder.batch_size == settings.embedding_batch_size
    assert embedder.headers["x-portkey-api-key"] == "test-portkey-key"


@respx.mock
async def test_unrouted_embedding_model_error_explains_the_fix():
    respx.post(URL).mock(
        return_value=httpx.Response(
            400,
            json={"error": {"message": "Either x-portkey-config or x-portkey-provider header is required"}},
        )
    )
    with pytest.raises(EmbeddingError) as exc:
        await make().embed(["a"])
    assert "EMBEDDING_MODEL" in str(exc.value)
    assert "@provider/model" in str(exc.value)
