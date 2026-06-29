"""
Test whether the GOT-patched seed hook makes games reproducible.

Build the hook first:
    bash cg/build_seed_hook.sh

Run from rl_mcts/:
    python debug/seed_test.py [seed] [runs]

Strategy:
  1. Call set_seed(seed) before each game.
  2. Play with a fully deterministic agent (always picks option index 0).
  3. Record the full log event sequence (draws, moves, etc.).
  4. Repeat N times with the same seed.
  5. If all runs are identical → hook works, games are seeded.
     If they differ → hook failed (check stderr output).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cg.recorder import GameRecorder
from cg.api import LogType

SAMPLE_DECK = [
    721, 721, 722, 722, 722, 722, 723, 723, 723, 723,
    1092, 1121, 1121, 1145, 1145, 1163, 1163, 1219, 1219, 1219,
    1219, 1227, 1227, 1227, 1227, 1262, 1262,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3,
]

assert len(SAMPLE_DECK) == 60


def extract_events(obs: dict) -> list[tuple]:
    events = []
    for log in obs.get("logs", []):
        t = log.get("type")
        if t in (LogType.DRAW, LogType.MOVE_CARD, LogType.SHUFFLE, LogType.COIN):
            events.append((t, log.get("cardId"), log.get("playerIndex"), log.get("head")))
    return events


def play_game(seed: int, vis_path: str | None = None) -> tuple[list[tuple], int]:
    rec = GameRecorder(seed=seed)
    obs, start_data = rec.start(SAMPLE_DECK, SAMPLE_DECK)
    if start_data.errorPlayer >= 0:
        raise ValueError(f"Deck error: player={start_data.errorPlayer} type={start_data.errorType}")

    events = extract_events(obs)
    moves = 0
    while obs["current"]["result"] < 0:
        selection = list(range(obs["select"]["maxCount"]))
        obs = rec.select(selection)
        events.extend(extract_events(obs))
        moves += 1
        if moves > 2000:
            print("  [warn] game exceeded 2000 moves, aborting")
            break

    result = obs["current"]["result"]
    rec.finish()  # captures visualize_data() internally

    if vis_path:
        rec.save_visualizer(vis_path)
        print(f"  Visualizer: {vis_path}")

    return events, result


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_test(seed: int = 12345, runs: int = 3, out_dir: str | None = None):
    if out_dir is None:
        out_dir = os.path.join(_SCRIPT_DIR, "games")
    print(f"Seed: {seed}  |  Runs: {runs}  |  Saving to: {out_dir}/\n")

    results = []
    for i in range(runs):
        save_path = f"{out_dir}/seed{seed}_run{i+1}_vis.json"
        events, result = play_game(seed, vis_path=save_path)
        results.append(events)
        print(f"  Run {i+1}: {len(events)} log events, winner={result}")

    all_same = all(r == results[0] for r in results[1:])
    print()
    if all_same:
        print("PASS: All runs identical — seeding works.")
    else:
        for i in range(1, len(results)):
            if results[i] != results[0]:
                for j, (a, b) in enumerate(zip(results[0], results[i])):
                    if a != b:
                        print(f"  First diff at event {j}: run1={a}  run{i+1}={b}")
                        break
                else:
                    print(f"  Run {i+1} has different length: {len(results[0])} vs {len(results[i])}")
        print("FAIL: Runs differ — hook did not intercept the RNG.")
        print("      Check stderr above for error messages from seed_hook.")


if __name__ == "__main__":
    seed    = int(sys.argv[1]) if len(sys.argv) > 1 else 12345
    runs    = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    out_dir = sys.argv[3]      if len(sys.argv) > 3 else None
    run_test(seed, runs, out_dir)
