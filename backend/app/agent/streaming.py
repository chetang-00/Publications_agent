"""Decouple agent runs from HTTP connections.

A run executes in its own background task and publishes events to a queue; the SSE response
only reads from that queue. If the browser disconnects, the reader stops but the run finishes
and is persisted. One run per conversation at a time.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from app.agent.events import RunError, to_sse
from app.agent.loop import Emit, NothingPending, RunInProgress
from app.tasks import TaskSet

log = logging.getLogger(__name__)

_DONE = object()


class RunManager:
    def __init__(self, tasks: TaskSet, keepalive_seconds: float = 15.0) -> None:
        self.tasks = tasks
        self.keepalive_seconds = keepalive_seconds
        self._busy: set[str] = set()

    def is_busy(self, conversation_id: str) -> bool:
        return conversation_id in self._busy

    def launch(self, conversation_id: str, work: Callable[[Emit], Awaitable[Any]]) -> AsyncIterator[str]:
        """Start `work(emit)` in the background and return the SSE stream of its events.

        Raises RunInProgress if the conversation already has a live run. The check and the claim
        happen without an await in between, so two simultaneous requests cannot both pass.
        """
        if conversation_id in self._busy:
            raise RunInProgress("This conversation is still answering the previous message.")
        self._busy.add(conversation_id)
        queue: asyncio.Queue[BaseModel | object] = asyncio.Queue()

        async def run() -> None:
            try:
                await work(queue.put)
            except (RunInProgress, NothingPending) as exc:
                await queue.put(RunError(code="conflict", message=str(exc)))
            except Exception:
                log.exception("Run task failed", extra={"conversation_id": conversation_id})
                await queue.put(
                    RunError(code="internal_error", message="Something went wrong. Please try again.")
                )
            finally:
                self._busy.discard(conversation_id)
                await queue.put(_DONE)

        self.tasks.spawn(run(), name=f"agent-run:{conversation_id}")
        return self._events(queue)

    async def _events(self, queue: asyncio.Queue[BaseModel | object]) -> AsyncIterator[str]:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=self.keepalive_seconds)
            except TimeoutError:
                yield ": ping\n\n"  # keeps proxies from closing an idle stream
                continue
            if item is _DONE:
                return
            yield to_sse(item)  # type: ignore[arg-type]
