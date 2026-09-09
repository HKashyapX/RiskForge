"""Transport-level request bounds and response deadlines."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

from starlette.types import Message, Receive, Scope, Send

AsgiApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class RequestControlsMiddleware:
    """Bound HTTP request bodies and cap response wait time.

    RiskForge accepts small JSON requests rather than streaming uploads, so the
    middleware reads at most ``max_request_bytes`` before invoking FastAPI.  It
    also prevents a slow application call from holding an HTTP response open
    beyond the configured deadline.
    """

    def __init__(
        self,
        app: AsgiApp,
        *,
        max_request_bytes: int,
        request_timeout_seconds: float,
    ) -> None:
        if max_request_bytes < 1:
            raise ValueError("max_request_bytes must be positive")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self.app = app
        self.max_request_bytes = max_request_bytes
        self.request_timeout_seconds = request_timeout_seconds

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        correlation_id = _correlation_id(scope)
        messages: list[Message] = []
        total = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            total += len(message.get("body", b""))
            if total > self.max_request_bytes:
                await _send_error(
                    send,
                    status=413,
                    correlation_id=correlation_id,
                    code="request_too_large",
                    message="request body exceeds configured limit",
                    retryable=False,
                )
                return
            if not message.get("more_body", False):
                break

        index = 0

        async def replay_receive() -> Message:
            nonlocal index
            if index < len(messages):
                message = messages[index]
                index += 1
                return message
            return {"type": "http.disconnect"}

        response_started = False

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await asyncio.wait_for(
                self.app(scope, replay_receive, tracked_send),
                timeout=self.request_timeout_seconds,
            )
        except TimeoutError:
            if not response_started:
                await _send_error(
                    send,
                    status=504,
                    correlation_id=correlation_id,
                    code="request_timeout",
                    message="request exceeded configured timeout",
                    retryable=True,
                )


def _correlation_id(scope: Scope) -> str:
    for raw_name, raw_value in scope.get("headers", ()):
        if raw_name.lower() == b"x-correlation-id":
            try:
                value = raw_value.decode("ascii")
            except UnicodeDecodeError:
                return "unavailable"
            if value:
                return value[:128]
    return "unavailable"


async def _send_error(
    send: Send,
    *,
    status: int,
    correlation_id: str,
    code: str,
    message: str,
    retryable: bool,
) -> None:
    body = json.dumps(
        {
            "api_version": "v1",
            "correlation_id": correlation_id,
            "error": {"code": code, "message": message, "retryable": retryable},
        },
        separators=(",", ":"),
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"cache-control", b"no-store"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
