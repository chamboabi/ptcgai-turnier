"""Drive a recorded match to completion and save its outputs."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import torch
from cg.recorder import GameRecorder

from .baselines import set_base_baselines
from .paths import REPLAYS
from .tracer import TurnPathTracer
from mcts import mcts_agent
from viewer.client import ViewerStream

OnStep = Callable[[dict, dict, int], list[int]]


class DeckError(Exception):
    def __init__(self, player: int, error_type: int):
        super().__init__(f"Deck error (player {player}, type {error_type}).")
        self.player = player
        self.error_type = error_type


def start_recorded_game(deck0: list[int], deck1: list[int]) -> tuple[GameRecorder, dict]:
    """Seed, start, and validate a recorded game. Raises DeckError if invalid."""
    rec = GameRecorder(seed=random.randint(0, 2**31 - 1))
    obs, start = rec.start(deck0, deck1)
    if start.errorPlayer >= 0:
        raise DeckError(start.errorPlayer, start.errorType)
    return rec, obs


def play_match(
    obs: dict,
    rec: GameRecorder,
    agents: list,
    tracer: TurnPathTracer,
    on_step: OnStep | None = None,
    viewer: ViewerStream | None = None,
) -> tuple[dict, int]:
    """Drive `obs` to game end, feeding `tracer` and printing progress.

    `on_step(obs, cur, step)` picks the move for `cur["yourIndex"]`; defaults
    to plain `mcts_agent(obs, agents[cur["yourIndex"]])`. Returns (obs, steps).

    If `viewer` is given (already started via `viewer.start_game(...)`),
    each new step is streamed to it live for the Flutter viewer app.
    """
    step = 0
    with torch.inference_mode():
        while obs["current"]["result"] < 0:
            cur = obs["current"]
            set_base_baselines(cur)
            if on_step is not None:
                selected = on_step(obs, cur, step)
            else:
                selected, _ = mcts_agent(obs, agents[cur["yourIndex"]])
            obs = rec.select(selected)
            tracer.feed(obs.get("logs"), obs["current"]["turn"])
            if viewer is not None:
                viewer.push(rec.new_vis_steps())
            step += 1
            if step % 20 == 0:
                print(f"  ...{step} actions, turn {obs['current']['turn']}")
    rec.finish()
    if viewer is not None:
        viewer.finish(obs["current"]["result"])
    return obs, step


def save_replay(rec: GameRecorder, base: Path) -> None:
    REPLAYS.mkdir(parents=True, exist_ok=True)
    rec.save(f"{base}.json")
    rec.save_visualizer(f"{base}_vis.json")
    print(f"Saved replay:     {base}.json")
    print(f"Saved visualizer: {base}_vis.json")
