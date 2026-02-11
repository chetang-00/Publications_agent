import asyncio
import json

import pytest
from pydantic import BaseModel, Field

from app.tools.base import (
    MAX_TOOL_CONTENT_CHARS,
    InvalidToolArguments,
    Tool,
    ToolArgs,
    ToolContext,
    ToolError,
    ToolRegistry,
    invalid_arguments_content,
    unknown_tool_content,
)


class EchoArgs(ToolArgs):
    """Arguments for echo."""

    text: str = Field(min_length=1, max_length=50, description="Text to echo back")
    times: int = Field(1, ge=1, le=5)
    title_contains: str | None = None


class EchoResult(BaseModel):
    text: str


async def echo(args: EchoArgs, ctx: ToolContext) -> EchoResult:
    return EchoResult(text=args.text * args.times)


def make_tool(handler=echo, **kwargs) -> Tool:
    return Tool(
        name=kwargs.pop("name", "echo"),
        description="Echo text back.",
        args_model=EchoArgs,
        result_model=EchoResult,
        handler=handler,
        **kwargs,
    )


@pytest.fixture
def ctx(settings, db, store, embedder) -> ToolContext:
    return ToolContext(db=db, store=store, embedder=embedder, settings=settings)


def test_openai_schema_shape():
    schema = make_tool().openai_schema()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "echo"
    assert fn["description"] == "Echo text back."
    params = fn["parameters"]
    assert params["type"] == "object"
    assert params["additionalProperties"] is False
    assert params["required"] == ["text"]
    assert params["properties"]["times"]["maximum"] == 5
    assert params["properties"]["text"]["description"] == "Text to echo back"


def test_schema_drops_title_noise_but_keeps_fields_named_like_title():
    params = make_tool().openai_schema()["function"]["parameters"]
    assert "title" not in params
    assert "title" not in params["properties"]["text"]
    assert "title_contains" in params["properties"]


def test_registry_rejects_duplicate_names():
    registry = ToolRegistry([make_tool()])
    with pytest.raises(ValueError):
        registry.register(make_tool())


def test_registry_lookup_and_schemas():
    registry = ToolRegistry([make_tool(), make_tool(name="echo2")])
    assert registry.names() == ["echo", "echo2"]
    assert registry.get("echo2").name == "echo2"
    assert registry.get("missing") is None
    assert [s["function"]["name"] for s in registry.schemas()] == ["echo", "echo2"]


def test_parse_args_valid():
    args = ToolRegistry().parse_args(make_tool(), '{"text": "hi", "times": 2}')
    assert args == EchoArgs(text="hi", times=2)


def test_parse_args_empty_string_means_no_arguments():
    with pytest.raises(InvalidToolArguments) as exc:
        ToolRegistry().parse_args(make_tool(), "")
    assert exc.value.details[0]["loc"] == "text"


@pytest.mark.parametrize(
    "raw",
    [
        '{"text": "hi", "times": 9}',  # out of range
        '{"text": "hi", "colour": "red"}',  # unknown field
        '{"text": ""}',  # too short
        '{"text": "hi"',  # truncated JSON
        "not json at all",
    ],
)
def test_parse_args_rejects_invalid(raw):
    with pytest.raises(InvalidToolArguments) as exc:
        ToolRegistry().parse_args(make_tool(), raw)
    assert exc.value.details
    assert all({"loc", "msg", "type"} <= set(d) for d in exc.value.details)


def test_invalid_arguments_content_is_actionable():
    content = json.loads(
        invalid_arguments_content([{"loc": "times", "msg": "too big", "type": "less_than_equal"}])
    )
    assert content["error"] == "invalid_arguments"
    assert content["details"][0]["loc"] == "times"
    assert "schema" in content["hint"]


def test_unknown_tool_content_lists_available_tools():
    content = json.loads(unknown_tool_content("nope", ["a", "b"]))
    assert content == {"error": "unknown_tool", "tool": "nope", "available_tools": ["a", "b"]}


async def test_execute_ok(ctx):
    tool = make_tool()
    outcome = await ToolRegistry([tool]).execute(tool, EchoArgs(text="ab", times=2), ctx)
    assert outcome.status == "ok"
    assert outcome.result == {"text": "abab"}
    assert outcome.error is None
    assert outcome.duration_ms >= 0
    assert json.loads(outcome.content()) == {"text": "abab"}


async def test_execute_tool_error_message_reaches_llm(ctx):
    async def failing(args, ctx):
        raise ToolError("Publication 999 does not exist.")

    tool = make_tool(handler=failing)
    outcome = await ToolRegistry([tool]).execute(tool, EchoArgs(text="x"), ctx)
    assert outcome.status == "error"
    assert json.loads(outcome.content()) == {"status": "error", "error": "Publication 999 does not exist."}


async def test_execute_unexpected_exception_hides_internals(ctx):
    async def crashing(args, ctx):
        raise RuntimeError("secret internal path /etc/passwd")

    tool = make_tool(handler=crashing)
    outcome = await ToolRegistry([tool]).execute(tool, EchoArgs(text="x"), ctx)
    assert outcome.status == "error"
    assert "secret" not in outcome.content()
    assert "echo" in outcome.error


async def test_execute_validates_result_shape(ctx):
    async def wrong_shape(args, ctx):
        return {"unexpected": 1}

    tool = make_tool(handler=wrong_shape)
    outcome = await ToolRegistry([tool]).execute(tool, EchoArgs(text="x"), ctx)
    assert outcome.status == "error"
    assert "invalid result" in outcome.error


async def test_execute_times_out(ctx):
    async def slow(args, ctx):
        await asyncio.sleep(5)

    tool = make_tool(handler=slow, timeout_seconds=0.05)
    outcome = await ToolRegistry([tool]).execute(tool, EchoArgs(text="x"), ctx)
    assert outcome.status == "timeout"
    assert "timed out" in outcome.error


async def test_execute_uses_settings_timeout_by_default(ctx):
    ctx.settings.tool_timeout_seconds = 0.05

    async def slow(args, ctx):
        await asyncio.sleep(5)

    tool = make_tool(handler=slow)
    outcome = await ToolRegistry([tool]).execute(tool, EchoArgs(text="x"), ctx)
    assert outcome.status == "timeout"


async def test_large_results_are_truncated_with_marker(ctx):
    class Big(BaseModel):
        text: str

    async def big(args, ctx):
        return Big(text='"quoted" ' * 5000)

    tool = Tool(name="big", description="d", args_model=EchoArgs, result_model=Big, handler=big)
    outcome = await ToolRegistry([tool]).execute(tool, EchoArgs(text="x"), ctx)
    content = outcome.content()
    assert len(content) <= MAX_TOOL_CONTENT_CHARS
    data = json.loads(content)
    assert data["truncated"] is True
    assert data["partial"].startswith('{"text"')


async def test_preview_is_short(ctx):
    tool = make_tool()
    outcome = await ToolRegistry([tool]).execute(tool, EchoArgs(text="y" * 50, times=5), ctx)
    assert len(outcome.preview()) <= 500
