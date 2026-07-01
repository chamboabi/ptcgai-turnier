"""Bench measured agents against opponents over N games each.

Both measured agents and opponents are `core.Player` (deck + act(obs)->selection).
Wrap an MCTS `Agent` with `agent_to_player` to bench it the same way as a
heuristic player.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from agent import Agent
from cg.game import battle_finish, battle_select, battle_start
from cg.sim import set_seed
from core import DeckError, Player, set_base_baselines
from mcts import mcts_agent


def agent_to_player(agent: Agent, name: str = "") -> Player:
    """Wrap an MCTS `Agent` as a `Player` so it can be benched like any other."""

    def act(obs: dict) -> list[int]:
        selected, _ = mcts_agent(obs, agent)
        return selected

    return Player(deck=agent.deck, act=act, name=name or "agent")


def run_game(p0: Player, p1: Player, seed: int | None = None) -> tuple[int, int]:
    """Play one unrecorded game between p0 and p1.

    Returns (result, steps); result is 0 (p0 wins), 1 (p1 wins), or 2 (draw).
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    set_seed(seed)
    obs, start = battle_start(p0.deck, p1.deck)
    if start.errorPlayer >= 0:
        raise DeckError(start.errorPlayer, start.errorType)

    players = (p0, p1)
    step = 0
    with torch.inference_mode():
        while obs["current"]["result"] < 0:
            cur = obs["current"]
            set_base_baselines(cur)
            selected = players[cur["yourIndex"]].act(obs)
            obs = battle_select(selected)
            step += 1
    battle_finish()
    return obs["current"]["result"], step


_FLIP_RESULT = {0: 1, 1: 0, 2: 2}


@dataclass
class BenchResult:
    measured: str
    opponent: str
    games: int
    wins: int
    losses: int
    draws: int
    avg_steps: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0


def bench_pair(
    measured: Player,
    opponent: Player,
    games: int,
    alternate_sides: bool = True,
    measured_name: str | None = None,
    opponent_name: str | None = None,
) -> BenchResult:
    """Play `games` games between measured and opponent, scored from measured's side.

    `alternate_sides` swaps who plays P0/P1 each game to cancel first-player edge.
    """
    wins = losses = draws = 0
    total_steps = 0
    for i in range(games):
        measured_is_p0 = not (alternate_sides and i % 2 == 1)
        p0, p1 = (measured, opponent) if measured_is_p0 else (opponent, measured)
        result, steps = run_game(p0, p1)
        total_steps += steps
        measured_result = result if measured_is_p0 else _FLIP_RESULT[result]
        if measured_result == 0:
            wins += 1
        elif measured_result == 1:
            losses += 1
        else:
            draws += 1
    return BenchResult(
        measured=measured_name or measured.name,
        opponent=opponent_name or opponent.name,
        games=games,
        wins=wins,
        losses=losses,
        draws=draws,
        avg_steps=total_steps / games,
    )


def bench(
    measured: dict[str, Player],
    opponents: dict[str, Player],
    games_per_matchup: int,
    alternate_sides: bool = True,
) -> list[BenchResult]:
    """Bench every measured agent against every opponent, `games_per_matchup` games each."""
    results = []
    for m_name, m_player in measured.items():
        for o_name, o_player in opponents.items():
            results.append(
                bench_pair(
                    m_player,
                    o_player,
                    games_per_matchup,
                    alternate_sides=alternate_sides,
                    measured_name=m_name,
                    opponent_name=o_name,
                )
            )
    return results
