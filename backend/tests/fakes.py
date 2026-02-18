"""Test doubles: a deterministic embedder and a scripted chat model."""

import asyncio
import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, field

from app.llm.types import LLMResult, LLMToolCall, TextDelta

DIM = 64


def fake_vector(text: str, dim: int = DIM) -> list[float]:
    """Hashed bag-of-words: texts that share words get similar vectors, deterministically."""
    vec = [0.0] * dim
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % dim
        vec[bucket] += 1.0
    if not any(vec):
        vec[0] = 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


class FakeEmbedder:
    def __init__(self, dim: int = DIM, model: str = "fake-embedding") -> None:
        self.dim = dim
        self.model = model
        self.calls: list[list[str]] = []
        self.fail_with: Exception | None = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail_with:
            raise self.fail_with
        self.calls.append(list(texts))
        return [fake_vector(t, self.dim) for t in texts]


# ── scripted chat model ──────────────────────────────────────────────────────


@dataclass
class FakeTurn:
    """One model response: optional text and/or tool calls given as (id, name, arguments)."""

    text: str = ""
    tool_calls: list[tuple[str, str, dict | str]] = field(default_factory=list)
    error: Exception | None = None
    prompt_tokens: int = 10
    completion_tokens: int = 5
    delay: float = 0.0


def call(call_id: str, name: str, arguments: dict | str) -> tuple[str, str, dict | str]:
    return (call_id, name, arguments)


class FakeLLM:
    model = "fake-model"

    def __init__(self, turns: list[FakeTurn] | None = None) -> None:
        self.turns = list(turns or [])
        self.requests: list[dict] = []

    def add(self, *turns: FakeTurn) -> None:
        self.turns.extend(turns)

    async def stream(self, messages, tools, trace_id, tool_choice=None):
        self.requests.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": tools,
                "trace_id": trace_id,
                "tool_choice": tool_choice,
            }
        )
        if not self.turns:
            raise AssertionError(
                "FakeLLM script exhausted: the agent made more model calls than the test expected"
            )
        turn = self.turns.pop(0)
        if turn.delay:
            await asyncio.sleep(turn.delay)
        if turn.error:
            raise turn.error
        if turn.text:
            middle = len(turn.text) // 2
            for piece in (turn.text[:middle], turn.text[middle:]):
                if piece:
                    yield TextDelta(piece)
        yield LLMResult(
            content=turn.text,
            tool_calls=[
                LLMToolCall(id=cid, name=name, arguments=args if isinstance(args, str) else json.dumps(args))
                for cid, name, args in turn.tool_calls
            ],
            prompt_tokens=turn.prompt_tokens,
            completion_tokens=turn.completion_tokens,
        )
