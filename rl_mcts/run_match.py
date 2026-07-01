"""Play ONE recorded match: Abomasnow deck (abomasnow shape) vs a random
decklist (base shape). Saves a replay + visualizer JSON under replays/.

Run:  env/bin/python run_match.py
"""

import random
import sys
import time

import rewards
from agent import Agent
from cg.api import all_card_data
from core import (
    CUSTOMDECKS,
    DECKLISTS,
    DeckError,
    REPLAYS,
    TurnPathTracer,
    build_model,
    load_deck,
    make_mcts_cfg,
    play_match,
    save_replay,
    start_recorded_game,
)
from viewer.client import ViewerStream

CUSTOM = CUSTOMDECKS / "abamasnow.csv"


def main() -> int:
    abomasnow_deck = load_deck(CUSTOM)

    decklist_files = sorted(DECKLISTS.glob("*.csv"))
    if not decklist_files:
        print("No decklists found.", file=sys.stderr)
        return 1
    opp_file = random.choice(decklist_files)
    opp_deck = load_deck(opp_file)
    print(f"P0 Abomasnow ({len(abomasnow_deck)} cards) [abomasnow shape]")
    print(f"P1 {opp_file.name} ({len(opp_deck)} cards) [base shape]")

    model = build_model()
    mcfg = make_mcts_cfg()

    # P0 = Abomasnow deck + abomasnow shape; P1 = random deck + base shape.
    agents = [
        Agent(deck=abomasnow_deck, model=model, mcts_cfg=mcfg, reward_fn=rewards.abomasnow),
        Agent(deck=opp_deck, model=model, mcts_cfg=mcfg, reward_fn=rewards.base),
    ]

    try:
        rec, obs = start_recorded_game(abomasnow_deck, opp_deck)
    except DeckError as e:
        print(e, file=sys.stderr)
        return 1

    tracer = TurnPathTracer({c.cardId: c.name for c in all_card_data()}, "Abomasnow", "base")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    game_id = f"abomasnow_vs_{opp_file.stem}_{stamp}"
    viewer = ViewerStream()
    viewer.start_game(game_id, abomasnow_deck, opp_deck)

    obs, step = play_match(obs, rec, agents, tracer, viewer=viewer)
    viewer.close()

    result = obs["current"]["result"]
    outcome = {0: "P0 (Abomasnow) wins", 1: "P1 (base) wins", 2: "draw"}.get(result, "?")
    print(f"Result: {outcome} after {step} actions, turn {obs['current']['turn']}.")

    base = REPLAYS / game_id
    save_replay(rec, base)
    print("Open debug/visualizer.html in a browser and load the *_vis.json file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
