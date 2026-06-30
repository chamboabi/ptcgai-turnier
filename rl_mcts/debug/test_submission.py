"""Test a BUNDLED submission end-to-end by playing a real game with it.

The bundled cg/ (inside a dist/ folder) is an older library with no recorder and
no set_seed, so we can't drive a game with it directly. Instead this harness:

  * imports the submission's `agent()` from its main.py (assets — deck.csv,
    archetypes.json, model.pth — load relative to that main.py, so the real
    bundled deck/weights/predictor are exercised);
  * runs the game on the PARENT cg engine (rl_mcts/cg), which has a matching
    recorder + set_seed.

Both pieces import `cg`, and parent rl_mcts is put FIRST on sys.path, so `cg`
always resolves to the parent (full) library; the dist dir is appended only so
the submission's `import deck_predict_lite` resolves.

Run from rl_mcts/:
    python debug/test_submission.py                          # self-play default dist
    python debug/test_submission.py path/to/dist             # self-play a given dist
    python debug/test_submission.py path/to/dist --opp random
    python debug/test_submission.py path/to/dist --search 80 --seed 7 --vis

Options:
    --opp {self,random}  opponent for seat 1 (default self = bundled agent both seats)
    --opp-deck PATH      deck for the random opponent (default: random decklist)
    --search N           override SEARCH_COUNT (lower = faster smoke test)
    --seed N             game seed (default random)
    --max-plies N        safety cap (default 4000)
    --vis                save a visualizer JSON under debug/games/
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # rl_mcts/
DEFAULT_DIST = ROOT / "submission_builder" / "dist" / "abomasnow"
DECKLISTS = ROOT / "data" / "decks" / "decklists"


def _load_submission(dist_dir: Path):
    """Import the bundled main.py as a module, wiring sys.path so cg = parent."""
    main_py = dist_dir / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"No main.py in {dist_dir}")
    # parent first -> `import cg` / `import rewards` resolve to the full parent libs
    sys.path.insert(0, str(ROOT))
    # dist appended -> the bundled `import deck_predict_lite` resolves
    sys.path.append(str(dist_dir))
    spec = importlib.util.spec_from_file_location("submission_main", main_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_deck(path: Path) -> list[int]:
    text = path.read_text().replace(",", "\n")
    return [int(tok) for tok in text.split() if tok.strip()][:60]


def main() -> int:
    ap = argparse.ArgumentParser(description="Play a game with a bundled submission.")
    ap.add_argument("dist", nargs="?", default=str(DEFAULT_DIST), help="Path to a built submission dir.")
    ap.add_argument("--opp", choices=["self", "random"], default="self")
    ap.add_argument("--opp-deck", default=None, help="Deck CSV for the random opponent.")
    ap.add_argument("--search", type=int, default=None, help="Override SEARCH_COUNT.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-plies", type=int, default=4000)
    ap.add_argument("--vis", action="store_true", help="Save a visualizer JSON.")
    args = ap.parse_args()

    dist_dir = Path(args.dist).resolve()
    sub = _load_submission(dist_dir)
    if args.search is not None:
        sub.SEARCH_COUNT = args.search

    # parent engine (matching recorder + set_seed)
    from cg.recorder import GameRecorder

    deck = sub._DECK  # seat 0 = bundled submission deck
    print(f"Submission: {dist_dir}")
    print(f"  deck cards: {len(deck)}  | archetype model: {sub._ARCH is not None}  | reward shape: {sub._REWARD is not None}")
    print(f"  SEARCH_COUNT={sub.SEARCH_COUNT}  opp={args.opp}")

    if args.opp == "self":
        opp_deck = deck
    else:
        if args.opp_deck:
            opp_deck = _load_deck(Path(args.opp_deck))
        else:
            files = sorted(DECKLISTS.glob("*.csv"))
            if not files:
                print("No decklists for a random opponent.", file=sys.stderr)
                return 1
            choice = random.choice(files)
            opp_deck = _load_deck(choice)
            print(f"  random opp deck: {choice.name}")

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    rec = GameRecorder(seed=seed)
    obs, start = rec.start(deck, opp_deck)
    if start.errorPlayer >= 0:
        print(f"Deck error: player={start.errorPlayer} type={start.errorType}", file=sys.stderr)
        return 1

    plies = 0
    move_times: list[float] = []
    while obs["current"]["result"] < 0 and plies < args.max_plies:
        yi = obs["current"]["yourIndex"]
        t0 = time.perf_counter()
        if args.opp == "self" or yi == 0:
            sel = sub.agent(obs)              # bundled agent decides
            move_times.append(time.perf_counter() - t0)
        else:
            sel = list(range(obs["select"]["maxCount"]))  # random opp: take first legal combo
        obs = rec.select(sel)
        plies += 1
        if plies % 20 == 0:
            print(f"  ...{plies} plies, turn {obs['current']['turn']}")

    rec.finish()
    result = obs["current"]["result"]
    label = {0: "submission (P0) wins", 1: "opponent (P1) wins", 2: "draw"}.get(result, f"unfinished ({result})")
    avg = sum(move_times) / len(move_times) if move_times else 0.0
    print(f"\nResult: {label} after {plies} plies, turn {obs['current']['turn']}.")
    print(f"Bundled-agent decisions: {len(move_times)}  | avg {avg:.2f}s/decision")

    if args.vis:
        out = HERE / "games"
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = out / f"submission_{dist_dir.name}_seed{seed}_vis.json"
        rec.save_visualizer(str(path))
        print(f"Visualizer: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
