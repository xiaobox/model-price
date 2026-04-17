"""Background scheduler for periodic v2 entity refresh."""

import asyncio
import logging
from contextlib import suppress

from .entity_store import get_store

logger = logging.getLogger(__name__)


class RefreshScheduler:
    """Runs periodic v2 pipeline refresh in the background."""

    def __init__(self, interval_seconds: int):
        self.interval_seconds = max(interval_seconds, 60)
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    def start(self) -> None:
        if self._task and not self._task.done():
            return

        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="v2-refresh-scheduler")
        logger.info("Auto refresh scheduler started (interval=%ss)", self.interval_seconds)

    async def stop(self) -> None:
        if not self._task:
            return

        self._stop_event.set()
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task

        self._task = None
        logger.info("Auto refresh scheduler stopped")

    async def _run(self) -> None:
        # Kick an initial refresh immediately. Waiting one full interval
        # before the first run means Render free-tier users see stale
        # snapshot data for up to `interval_seconds` after every cold
        # boot — the very window in which freshness matters most.
        await self._refresh_once()

        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.interval_seconds,
                )
                break
            except TimeoutError:
                pass

            if self._stop_event.is_set():
                break

            await self._refresh_once()

    async def _refresh_once(self) -> None:
        try:
            report = await get_store().refresh_from_pipeline(force_network=True)
            logger.info(
                "Scheduled v2 refresh completed: %s entities, %s offerings",
                report.counts.entities_total,
                report.counts.offerings_total,
            )
        except Exception:
            logger.exception("Scheduled v2 refresh failed")
