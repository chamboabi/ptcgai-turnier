"""Save/load bench results to JSON.

Saving after a run captures the seeds, opponents, sides, and outcomes.
Loading strips the outcomes back out, leaving just the seeds/opponents/sides
so the exact same games can be replayed against a different measured player
(see `bench.core.bench_from_saved`).
"""

from __future__ import annotations

import json
from pathlib import Path

from .core import BenchResult, GameRecord


def save_bench_results(results: list[BenchResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "measured": r.measured,
            "opponent": r.opponent,
            "games": [
                {
                    "seed": g.seed,
                    "measured_is_p0": g.measured_is_p0,
                    "result": g.result,
                    "steps": g.steps,
                }
                for g in r.games
            ],
        }
        for r in results
    ]
    path.write_text(json.dumps(data, indent=2))


def load_bench_games(path: str | Path) -> dict[str, list[GameRecord]]:
    """Load a saved bench, keyed by opponent name, with results/steps cleared
    so it's ready to run again against a (possibly different) measured player."""
    data = json.loads(Path(path).read_text())
    return {
        entry["opponent"]: [
            GameRecord(seed=g["seed"], measured_is_p0=g["measured_is_p0"])
            for g in entry["games"]
        ]
        for entry in data
    }
