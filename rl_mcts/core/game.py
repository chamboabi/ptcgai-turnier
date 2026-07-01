"""Start a game between two Players, optionally with a fixed seed."""

from __future__ import annotations

import random

from cg.recorder import GameRecorder

from .player import Player


def start_game(player0: Player, player1: Player, seed: int | None = None):
    """Seed and start a battle for the given players.

    Returns (rec, obs, start). Drive the loop with:
        while obs["current"]["result"] < 0:
            idx = obs["current"]["yourIndex"]
            players = (player0, player1)
            obs = rec.select(players[idx].act(obs))
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    rec = GameRecorder(seed=seed)
    obs, start = rec.start(player0.deck, player1.deck)
    return rec, obs, start
