"""Play ONE recorded match: Evee deck (evee shape) vs a random decklist (base
shape). Saves a replay + visualizer JSON under replays/.

Run:  env/bin/python run_evee.py
"""

import random
import sys
import time
from pathlib import Path

import torch

import rewards
from agent import Agent
from cg.api import all_card_data
from cg.recorder import GameRecorder
from decision_log import write_decision
from deck_predict import load_model
from mcts import mcts_agent
from run_match import (
    DECKLISTS,
    REPLAYS,
    TurnPathTracer,
    build_model,
    load_deck,
    make_mcts_cfg,
    set_base_baselines,
)

HERE = Path(__file__).parent
EVEE = HERE / "data" / "decks" / "customdecks" / "evee.csv"


def main() -> int:
    evee_deck = load_deck(EVEE)

    decklist_files = sorted(DECKLISTS.glob("*.csv"))
    if not decklist_files:
        print("No decklists found.", file=sys.stderr)
        return 1
    opp_file = random.choice(decklist_files)
    opp_deck = load_deck(opp_file)
    print(f"P0 Evee ({len(evee_deck)} cards) [evee shape]")
    print(f"P1 {opp_file.name} ({len(opp_deck)} cards) [base shape]")

    model = build_model()
    mcfg = make_mcts_cfg()

    # Attach the archetype model to Evee so its opponent belief (sampled deck /
    # hand + per-card predictions) is real, not all-UNKNOWN. Needed for the
    # decision log's "what he thinks the opponent has".
    arch_path = HERE / "data" / "archetypes.json"
    archetype_model = load_model(str(arch_path)) if arch_path.exists() else None
    if archetype_model is None:
        print("No data/archetypes.json — opponent belief will be UNKNOWN cards.")

    # P0 = Evee deck + evee shape; P1 = random deck + base shape.
    agents = [
        Agent(deck=evee_deck, model=model, mcts_cfg=mcfg, reward_fn=rewards.evee,
              archetype_model=archetype_model),
        Agent(deck=opp_deck, model=model, mcts_cfg=mcfg, reward_fn=rewards.base),
    ]

    rec = GameRecorder(seed=random.randint(0, 2**31 - 1))
    obs, start = rec.start(evee_deck, opp_deck)
    if start.errorPlayer >= 0:
        print(f"Deck error (player {start.errorPlayer}, type {start.errorType}).", file=sys.stderr)
        return 1

    names = {c.cardId: c.name for c in all_card_data()}
    tracer = TurnPathTracer(names, p0_label="evee", p1_label="base")

    # One folder per run holding a JSON + txt dump of every Evee (P0) decision.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    decisions_dir = REPLAYS / f"evee_vs_{opp_file.stem}_{stamp}_decisions"

    step = 0
    decision_idx = 0
    with torch.inference_mode():
        while obs["current"]["result"] < 0:
            cur = obs["current"]
            set_base_baselines(cur)
            # Only dump Evee's own decisions (P0); opponent runs plain.
            dbg = {} if cur["yourIndex"] == 0 else None
            selected, _ = mcts_agent(obs, agents[cur["yourIndex"]], debug_out=dbg)
            if dbg:
                write_decision(decisions_dir, decision_idx, dbg, names)
                decision_idx += 1
            obs = rec.select(selected)
            tracer.feed(obs.get("logs"), obs["current"]["turn"])
            step += 1
            if step % 20 == 0:
                print(f"  ...{step} actions, turn {obs['current']['turn']}")

    rec.finish()

    result = obs["current"]["result"]
    outcome = {0: "P0 (Evee) wins", 1: "P1 (base) wins", 2: "draw"}.get(result, "?")
    print(f"Result: {outcome} after {step} actions, turn {obs['current']['turn']}.")

    REPLAYS.mkdir(parents=True, exist_ok=True)
    base = REPLAYS / f"evee_vs_{opp_file.stem}_{stamp}"
    rec.save(f"{base}.json")
    rec.save_visualizer(f"{base}_vis.json")
    print(f"Saved replay:     {base}.json")
    print(f"Saved visualizer: {base}_vis.json")
    print(f"Saved decisions:  {decisions_dir}/ ({decision_idx} Evee decisions)")
    print("Open debug/visualizer.html in a browser and load the *_vis.json file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
