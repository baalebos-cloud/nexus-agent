"""
Background TTL-reaper for sandbox sessions.

Provider-side timeouts (E2B's own `timeout=` on the sandbox) are a backstop,
not the primary cleanup mechanism — see PROJECT_CONTEXT.md session lifecycle
notes. This task is the primary mechanism: it polls the session registry on
an interval and closes anything past its app-level TTL, so idle sandboxes
don't sit around accruing billable hours if the app crashes or a task
abandons a session without calling `close_session` explicitly.

Wire this up as a FastAPI lifespan task in `src/api/server.py`:

    from src.sandbox.reaper import SandboxReaper

    reaper = SandboxReaper(manager=sandbox_manager, interval_seconds=60)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await reaper.start()
        yield
        await reaper.stop()
"""

from __future__ import annotations

import asyncio
import logging

from src.sandbox.manager import SandboxManager

logger = logging.getLogger(__name__)


class SandboxReaper:
    """Polls a SandboxManager for expired sessions and closes them."""

    def __init__(
        self,
        manager: SandboxManager,
        interval_seconds: int = 60,
        on_reap_error_backoff_seconds: int = 10,
    ) -> None:
        self._manager = manager
        self._interval_seconds = interval_seconds
        self._backoff_seconds = on_reap_error_backoff_seconds
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None:
            logger.warning("SandboxReaper.start() called while already running; ignoring.")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop(), name="sandbox-ttl-reaper")
        logger.info(
            "SandboxReaper started, interval_seconds=%s", self._interval_seconds
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=self._interval_seconds + 5)
        except asyncio.TimeoutError:
            logger.warning("SandboxReaper did not stop cleanly within timeout; cancelling.")
            self._task.cancel()
        self._task = None
        logger.info("SandboxReaper stopped.")

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                reaped = await self._manager.reap_expired_sessions()
                if reaped:
                    logger.info("SandboxReaper closed %d expired session(s): %s", len(reaped), reaped)
            except Exception as exc:  # noqa: BLE001 - reaper must never crash the app
                logger.error("SandboxReaper iteration failed: %s", exc)
                await self._sleep_or_stop(self._backoff_seconds)
                continue

            await self._sleep_or_stop(self._interval_seconds)

    async def _sleep_or_stop(self, seconds: int) -> None:
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass  # normal case: interval elapsed, loop again
