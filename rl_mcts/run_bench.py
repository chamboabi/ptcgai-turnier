"""Bench a player against heuristic_player opponents, or replay a saved bench.

Each game gets its own seed. Results (seeds + opponents + sides + outcomes)
are saved to JSON so the exact same games can be replayed later against a
different player with --load.

Run:
    env/bin/python run_bench.py --player evee
    env/bin/python run_bench.py --player evee --opponents alkazam starmie --games 50
    env/bin/python run_bench.py --player evee_v2 --load out/bench_results/evee_20260701-101500.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from bench import bench, bench_from_saved, load_bench_games, save_bench_results
from bench.core import agent_to_player
from core import ROOT, Player, load_heuristic_player
from core.player_loader import load_rl_player
from viewer.client import ViewerStream

PLAYER_DIR = ROOT / "data" / "player"
HEURISTIC_PLAYER_DIR = ROOT / "data" / "heuristic_player"
BENCH_RESULTS_DIR = ROOT / "out" / "bench_results"


def resolve_player(name: str) -> Player:
    """Load `name` as an RL player package (data/player/) if one exists,
    else as a heuristic player (data/heuristic_player/)."""
    if (PLAYER_DIR / name).is_dir():
        return agent_to_player(load_rl_player(name), name=name)
    return load_heuristic_player(name, name=name)


def default_opponents() -> list[str]:
    return sorted(p.name for p in HEURISTIC_PLAYER_DIR.iterdir() if p.is_dir())


def print_table(results) -> None:
    header = f"{'opponent':<12} {'games':>6} {'W':>4} {'L':>4} {'D':>4} {'win%':>7} {'avg_steps':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.opponent:<12} {len(r.games):>6} {r.wins:>4} {r.losses:>4} {r.draws:>4} "
            f"{r.win_rate * 100:>6.1f}% {r.avg_steps:>10.1f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--player", required=True, help="name under data/player/ or data/heuristic_player/")
    parser.add_argument("--opponents", nargs="*", default=None, help="default: all folders in data/heuristic_player")
    parser.add_argument("--games", type=int, default=20, help="games per matchup (ignored with --load)")
    parser.add_argument("--load", type=Path, default=None, help="replay a saved bench's seeds/opponents/sides")
    parser.add_argument("--save", type=Path, default=None, help="where to save results (default: auto-named)")
    args = parser.parse_args()

    measured = resolve_player(args.player)
    viewer = ViewerStream()

    if args.load is not None:
        saved = load_bench_games(args.load)
        results = bench_from_saved(measured, saved, measured_name=args.player, viewer=viewer)
    else:
        opponent_names = args.opponents or default_opponents()
        results = bench(measured, opponent_names, args.games, measured_name=args.player, viewer=viewer)

    viewer.close()
    print_table(results)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = args.save or BENCH_RESULTS_DIR / f"{args.player}_{stamp}.json"
    save_bench_results(results, save_path)
    print(f"\nSaved to {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
