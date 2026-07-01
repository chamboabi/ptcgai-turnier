"""Bench measured agents against opponents over N games each.

Both measured agents and opponents are `core.Player` (deck + act(obs)->selection).
Wrap an MCTS `Agent` with `agent_to_player` to bench it the same way as a
heuristic player.

Each game gets its own seed. `bench()` generates fresh seeds; `bench.io`
saves the seed/opponent/side info per game so a later run can replay the
exact same games against a different measured player via `bench_from_saved`.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

import torch

from agent import Agent
from cg.game import battle_finish, battle_select, battle_start, visualize_data
from cg.sim import set_seed
from core import DeckError, Player, load_heuristic_player, set_base_baselines
from mcts import mcts_agent
from viewer.client import ViewerStream


def agent_to_player(agent: Agent, name: str = "") -> Player:
    """Wrap an MCTS `Agent` as a `Player` so it can be benched like any other."""

    def act(obs: dict) -> list[int]:
        selected, _ = mcts_agent(obs, agent)
        return selected

    return Player(deck=agent.deck, act=act, name=name or "agent")


def run_game(
    p0: Player,
    p1: Player,
    seed: int | None = None,
    viewer: ViewerStream | None = None,
    game_id: str | None = None,
) -> tuple[int, int]:
    """Play one unrecorded game between p0 and p1.

    Returns (result, steps); result is 0 (p0 wins), 1 (p1 wins), or 2 (draw).

    If `viewer` is given, each new engine step is streamed to it live — bench
    has no GameRecorder, so this reads visualize_data() directly with a
    local cursor instead of GameRecorder.new_vis_steps().
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    set_seed(seed)
    obs, start = battle_start(p0.deck, p1.deck)
    if start.errorPlayer >= 0:
        raise DeckError(start.errorPlayer, start.errorType)
    if viewer is not None:
        viewer.start_game(game_id or f"bench_{seed}", p0.deck, p1.deck)

    players = (p0, p1)
    step = 0
    vis_cursor = 0
    with torch.inference_mode():
        while obs["current"]["result"] < 0:
            cur = obs["current"]
            set_base_baselines(cur)
            selected = players[cur["yourIndex"]].act(obs)
            obs = battle_select(selected)
            step += 1
            if viewer is not None:
                vis = json.loads(visualize_data())
                if len(vis) > vis_cursor:
                    viewer.push(vis[vis_cursor:])
                    vis_cursor = len(vis)
    battle_finish()
    result = obs["current"]["result"]
    if viewer is not None:
        viewer.finish(result)
    return result, step


_FLIP_RESULT = {0: 1, 1: 0, 2: 2}


@dataclass
class GameRecord:
    """One scheduled (or played) game: its seed, which side the measured player
    takes, and the outcome once run (None before/without a run)."""

    seed: int
    measured_is_p0: bool
    result: int | None = None
    steps: int | None = None


def make_game_records(games: int, alternate_sides: bool = True) -> list[GameRecord]:
    """Schedule `games` games with fresh random seeds.

    `alternate_sides` swaps who plays P0/P1 each game to cancel first-player edge.
    """
    return [
        GameRecord(
            seed=random.randint(0, 2**31 - 1),
            measured_is_p0=not (alternate_sides and i % 2 == 1),
        )
        for i in range(games)
    ]


@dataclass
class BenchResult:
    measured: str
    opponent: str
    games: list[GameRecord] = field(default_factory=list)

    @property
    def wins(self) -> int:
        return sum(1 for g in self.games if g.result == 0)

    @property
    def losses(self) -> int:
        return sum(1 for g in self.games if g.result == 1)

    @property
    def draws(self) -> int:
        return sum(1 for g in self.games if g.result == 2)

    @property
    def win_rate(self) -> float:
        return self.wins / len(self.games) if self.games else 0.0

    @property
    def avg_steps(self) -> float:
        steps = [g.steps for g in self.games if g.steps is not None]
        return sum(steps) / len(steps) if steps else 0.0


def run_pair(
    measured: Player,
    opponent: Player,
    games: list[GameRecord],
    measured_name: str | None = None,
    opponent_name: str | None = None,
    viewer: ViewerStream | None = None,
) -> BenchResult:
    """Run each scheduled `GameRecord` (seed + side) between measured and opponent,
    filling in its result/steps in place. Scored from measured's side."""
    m_name = measured_name or measured.name
    o_name = opponent_name or opponent.name
    for i, record in enumerate(games):
        p0, p1 = (measured, opponent) if record.measured_is_p0 else (opponent, measured)
        game_id = f"{m_name}_vs_{o_name}_{i}"
        result, steps = run_game(p0, p1, seed=record.seed, viewer=viewer, game_id=game_id)
        record.result = result if record.measured_is_p0 else _FLIP_RESULT[result]
        record.steps = steps
    return BenchResult(measured=m_name, opponent=o_name, games=games)


def bench(
    measured: Player,
    opponent_names: list[str],
    games_per_matchup: int,
    alternate_sides: bool = True,
    measured_name: str | None = None,
    viewer: ViewerStream | None = None,
) -> list[BenchResult]:
    """Bench `measured` against every named opponent in data/heuristic_player/,
    `games_per_matchup` freshly-seeded games each."""
    results = []
    for name in opponent_names:
        opponent = load_heuristic_player(name)
        games = make_game_records(games_per_matchup, alternate_sides=alternate_sides)
        results.append(run_pair(measured, opponent, games, measured_name=measured_name, opponent_name=name, viewer=viewer))
    return results


def bench_from_saved(
    measured: Player,
    saved: dict[str, list[GameRecord]],
    measured_name: str | None = None,
    viewer: ViewerStream | None = None,
) -> list[BenchResult]:
    """Replay a saved bench (see bench.io.load_bench_games) against `measured`,
    reusing the exact same seeds/opponents/sides."""
    results = []
    for name, games in saved.items():
        opponent = load_heuristic_player(name)
        results.append(run_pair(measured, opponent, games, measured_name=measured_name, opponent_name=name, viewer=viewer))
    return results
