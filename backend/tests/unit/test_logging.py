import json
import logging

from app.logging import JsonFormatter, request_id_var, run_id_var


def _record(msg: str, **extra) -> logging.LogRecord:
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, msg, None, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_includes_context_ids():
    req_token = request_id_var.set("req-1")
    run_token = run_id_var.set("run-1")
    try:
        line = JsonFormatter().format(_record("hello"))
    finally:
        request_id_var.reset(req_token)
        run_id_var.reset(run_token)
    data = json.loads(line)
    assert data["msg"] == "hello"
    assert data["level"] == "INFO"
    assert data["logger"] == "app.test"
    assert data["request_id"] == "req-1"
    assert data["run_id"] == "run-1"


def test_json_formatter_includes_extra_fields_and_omits_empty_ids():
    data = json.loads(JsonFormatter().format(_record("tool done", tool="get_publication", duration_ms=12)))
    assert data["tool"] == "get_publication"
    assert data["duration_ms"] == 12
    assert "request_id" not in data


def test_json_formatter_renders_exceptions():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", None, sys.exc_info())
    data = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in data["exc"]
