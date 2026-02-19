"""Background tasks owned by the app (agent runs, document processing).

asyncio only keeps weak references to tasks, so un-referenced tasks can be garbage collected
mid-flight; this set holds them until they finish and logs anything that crashes.
"""

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

log = logging.getLogger(__name__)


class TaskSet:
    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()

    def spawn(self, coro: Coroutine[Any, Any, Any], name: str | None = None) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._finished)
        return task

    def _finished(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("Background task crashed", exc_info=task.exception(), extra={"task": task.get_name()})

    async def wait_all(self) -> None:
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def shutdown(self, grace_seconds: float = 10.0) -> None:
        if not self._tasks:
            return
        _, pending = await asyncio.wait(list(self._tasks), timeout=grace_seconds)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
