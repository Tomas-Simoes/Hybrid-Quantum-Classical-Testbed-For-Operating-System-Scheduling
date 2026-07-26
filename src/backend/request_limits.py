from __future__ import annotations

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import settings


BODY_METHODS = {"POST", "PUT", "PATCH"}
REQUEST_TOO_LARGE = {"detail": "Request body is too large."}
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": "frame-ancestors 'none'; base-uri 'none'",
}


class RequestBodyLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method", "").upper() not in BODY_METHODS:
            await self.app(scope, receive, send)
            return

        limit = settings.public_max_request_bytes
        content_length = _content_length(scope)
        if content_length is None or content_length < 0 or content_length > limit:
            await _reject_request(scope, receive, send)
            return

        messages: list[Message] = []
        total_bytes = 0

        while True:
            message = await receive()
            messages.append(message)

            if message["type"] == "http.disconnect":
                break
            if message["type"] != "http.request":
                continue

            total_bytes += len(message.get("body", b""))
            if total_bytes > limit:
                await _reject_request(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        async def replay_receive() -> Message:
            if messages:
                return messages.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay_receive, send)


def _content_length(scope: Scope) -> int | None:
    raw_value = _header(scope, b"content-length")
    if raw_value is None:
        return 0
    try:
        return int(raw_value.decode("latin-1").strip())
    except ValueError:
        return None


def _header(scope: Scope, name: bytes) -> bytes | None:
    for header_name, value in scope.get("headers", []):
        if header_name.lower() == name:
            return value
    return None


async def _reject_request(scope: Scope, receive: Receive, send: Send) -> None:
    response = JSONResponse(
        status_code=413,
        content=REQUEST_TOO_LARGE,
        headers=SECURITY_HEADERS,
    )
    await response(scope, receive, send)
