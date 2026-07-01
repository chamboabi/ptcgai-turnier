"""Best-effort producer-side client for viewer/server.py.

Wire this into a match runner to stream a game live to the Flutter viewer.
If the relay server isn't running, calls are silently dropped — a bot run
must never crash, block, or slow down because nobody is watching.
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time

from websockets.asyncio.client import connect

log = logging.getLogger("viewer.client")


class ViewerStream:
    """Streams one game's steps to the relay server over a background thread.

    Usage::

        viewer = ViewerStream()
        viewer.start_game(game_id, deck0, deck1)
        ...
        viewer.push(rec.new_vis_steps())   # after each rec.select()
        ...
        viewer.finish(result)
        viewer.close()  # give the background thread a moment to flush "done"
    """

    def __init__(self, url: str = "ws://127.0.0.1:8765", maxsize: int = 1000):
        self._url = url
        self._game_id: str | None = None
        self._queue: "queue.Queue[str]" = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _send(self, msg: dict) -> None:
        try:
            self._queue.put_nowait(json.dumps(msg))
        except queue.Full:
            pass  # relay unreachable / backed up — drop, this is watch-only

    def start_game(self, game_id: str, deck0: list[int], deck1: list[int]) -> None:
        self._game_id = game_id
        self._send({"type": "hello", "role": "producer", "game_id": game_id})
        self._send({"type": "start", "game_id": game_id, "deck0": deck0, "deck1": deck1})

    def push(self, steps: list[dict]) -> None:
        if steps:
            self._send({"type": "step", "game_id": self._game_id, "data": steps})

    def finish(self, result) -> None:
        self._send({"type": "done", "game_id": self._game_id, "result": result})

    def close(self, timeout: float = 2.0) -> None:
        """Block briefly so queued messages (notably the final "done") get a
        chance to actually reach the relay before the process exits — the
        background thread is a daemon and would otherwise be killed with
        anything still unsent. Best-effort: never raises on timeout."""
        deadline = time.monotonic() + timeout
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.05)

    # ------------------------------------------------------------------

    def _run(self) -> None:
        asyncio.run(self._main())

    async def _main(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            try:
                async with connect(self._url, open_timeout=2) as ws:
                    while True:
                        msg = await loop.run_in_executor(None, self._queue.get)
                        await ws.send(msg)
            except Exception:
                log.debug("viewer relay unreachable, retrying", exc_info=True)
                await asyncio.sleep(1)
