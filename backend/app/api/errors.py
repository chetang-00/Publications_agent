"""Uniform error responses ({"error": {"code", "message", "details"?}}) and request-id handling."""

import logging
import time
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.logging import request_id_var

log = logging.getLogger("app.http")


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def not_found(what: str) -> ApiError:
    return ApiError(404, "not_found", f"{what} not found.")


def error_body(code: str, message: str, details: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    return {"error": error}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(error_body(exc.code, exc.message), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"loc": ".".join(str(p) for p in err["loc"]), "msg": err["msg"], "type": err["type"]}
            for err in exc.errors()
        ]
        return JSONResponse(
            error_body("validation_error", "The request is invalid.", details), status_code=422
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return JSONResponse(error_body(code, str(exc.detail)), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error")
        return JSONResponse(error_body("internal_error", "An unexpected error occurred."), status_code=500)


class RequestContextMiddleware:
    """Pure ASGI (safe with streaming responses): request id in, request id out, one log line per request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1").strip()
        request_id = incoming[:64] if incoming else uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status = 500

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", [])
                message["headers"] = [*message["headers"], (b"x-request-id", request_id.encode("latin-1"))]
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            if scope.get("path") not in ("/api/health",):
                log.info(
                    "request",
                    extra={
                        "method": scope.get("method"),
                        "path": scope.get("path"),
                        "status": status,
                        "duration_ms": int((time.perf_counter() - started) * 1000),
                    },
                )
            request_id_var.reset(token)


class CrossSiteWriteGuard:
    """Refuse state-changing requests that a browser sends on behalf of another site.

    The API has no login, and multipart uploads are "simple" CORS requests that skip the preflight,
    so without this any web page could add documents (a prompt-injection channel) to the local app.
    Non-browser clients (curl, scripts) send neither header and are unaffected.
    """

    UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

    def __init__(self, app: ASGIApp, allowed_origins: list[str] | None = None) -> None:
        self.app = app
        self.allowed = {o.rstrip("/").lower() for o in allowed_origins or []}

    def _refused(self, scope: Scope) -> bool:
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
        if headers.get("sec-fetch-site", "").lower() == "cross-site":
            return True
        origin = headers.get("origin")
        if origin is None:
            return False
        origin = origin.strip().rstrip("/").lower()
        if origin == "null":
            return True
        same_host = urlsplit(origin).netloc == headers.get("host", "").lower()
        return not (same_host or origin in self.allowed)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("method") in self.UNSAFE_METHODS and self._refused(scope):
            response = JSONResponse(
                error_body("cross_site_request", "Requests from other websites are not allowed."),
                status_code=403,
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
