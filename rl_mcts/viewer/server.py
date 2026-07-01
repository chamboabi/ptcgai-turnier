#!/usr/bin/env python3
"""WebSocket relay between a running match (producer) and viewer apps.

Keeps a single in-memory buffer for the current game so a viewer that
connects mid-match (or reconnects) gets caught up via a "snapshot" message,
then receives live "step"/"done" messages as the match continues.

Run standalone:  env/bin/python viewer/server.py
"""
import asyncio
import json
import logging

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("viewer.server")

HOST = "0.0.0.0"
PORT = 8765

current_game: dict = {
    "game_id": None,
    "deck0": [],
    "deck1": [],
    "steps": [],
    "done": False,
    "result": None,
}
viewers: set = set()


def snapshot_message() -> str:
    return json.dumps({"type": "snapshot", **current_game})


async def broadcast(message: str) -> None:
    if not viewers:
        return
    dead = set()
    for ws in viewers:
        try:
            await ws.send(message)
        except ConnectionClosed:
            dead.add(ws)
    viewers.difference_update(dead)


async def handle_producer(websocket, hello: dict) -> None:
    log.info("producer connected: game_id=%s", hello.get("game_id"))
    async for raw in websocket:
        msg = json.loads(raw)
        mtype = msg.get("type")
        if mtype == "start":
            current_game.update(
                game_id=msg.get("game_id"),
                deck0=msg.get("deck0", []),
                deck1=msg.get("deck1", []),
                steps=[],
                done=False,
                result=None,
            )
        elif mtype == "step":
            if msg.get("game_id") == current_game["game_id"]:
                current_game["steps"].extend(msg.get("data", []))
        elif mtype == "done":
            if msg.get("game_id") == current_game["game_id"]:
                current_game["done"] = True
                current_game["result"] = msg.get("result")
        await broadcast(raw)
    log.info("producer disconnected")


async def handle_viewer(websocket) -> None:
    log.info("viewer connected")
    viewers.add(websocket)
    try:
        await websocket.send(snapshot_message())
        await websocket.wait_closed()
    finally:
        viewers.discard(websocket)
        log.info("viewer disconnected")


async def handler(websocket) -> None:
    try:
        raw = await websocket.recv()
    except ConnectionClosed:
        return
    hello = json.loads(raw)
    role = hello.get("role")
    if role == "producer":
        await handle_producer(websocket, hello)
    elif role == "viewer":
        await handle_viewer(websocket)
    else:
        await websocket.close(1002, "expected hello with role=producer|viewer")


async def main() -> None:
    # max_size=None: the full-backlog snapshot for a long game can exceed the
    # default 1MB frame limit (every step embeds each player's full 60-card
    # deck). Trusted, local-only traffic — no adversarial-size concern here.
    async with serve(handler, HOST, PORT, max_size=None) as server:
        log.info("viewer relay listening on ws://%s:%d", HOST, PORT)
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
