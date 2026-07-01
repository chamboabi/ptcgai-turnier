"""Play ONE recorded match: Evee deck (evee shape) vs a random decklist (base
shape). Saves a replay + visualizer JSON under replays/.

Run:  env/bin/python run_evee.py
"""

import random
import subprocess
import sys
import time

import rewards
from agent import Agent
from cg.api import all_card_data
from core import (
    DECKLISTS,
    DeckError,
    REPLAYS,
    ROOT,
    TurnPathTracer,
    load_deck,
    load_rl_player,
    play_match,
    save_replay,
    start_recorded_game,
)
from decision_log import write_decision
from deck_predict import load_model
from mcts import mcts_agent


def main() -> int:
    decklist_files = sorted(DECKLISTS.glob("*.csv"))
    if not decklist_files:
        print("No decklists found.", file=sys.stderr)
        return 1
    opp_file = random.choice(decklist_files)
    opp_deck = load_deck(opp_file)

    # Attach the archetype model to Evee so its opponent belief (sampled deck /
    # hand + per-card predictions) is real, not all-UNKNOWN. Needed for the
    # decision log's "what he thinks the opponent has".
    arch_path = ROOT / "data" / "archetypes.json"
    archetype_model = load_model(str(arch_path)) if arch_path.exists() else None
    if archetype_model is None:
        print("No data/archetypes.json — opponent belief will be UNKNOWN cards.")

    # P0 = data/player/evee package (deck + evee shape + its own weights/config).
    evee_agent = load_rl_player("evee", archetype_model=archetype_model)
    evee_deck = evee_agent.deck
    print(f"P0 Evee ({len(evee_deck)} cards) [evee shape]")
    print(f"P1 {opp_file.name} ({len(opp_deck)} cards) [base shape]")

    # P1 = random deck + base shape, sharing evee's model/mcts config.
    agents = [
        evee_agent,
        Agent(deck=opp_deck, model=evee_agent.model, mcts_cfg=evee_agent.mcts_cfg, reward_fn=rewards.base),
    ]

    try:
        rec, obs = start_recorded_game(evee_deck, opp_deck)
    except DeckError as e:
        print(e, file=sys.stderr)
        return 1

    names = {c.cardId: c.name for c in all_card_data()}
    tracer = TurnPathTracer(names, p0_label="evee", p1_label="base")

    # One folder per run holding a JSON + txt dump of every Evee (P0) decision.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    decisions_dir = REPLAYS / f"evee_vs_{opp_file.stem}_{stamp}_decisions"
    decision_idx = 0

    def on_step(obs: dict, cur: dict, step: int) -> list[int]:
        nonlocal decision_idx
        # Only dump Evee's own decisions (P0); opponent runs plain.
        dbg = {} if cur["yourIndex"] == 0 else None
        selected, _ = mcts_agent(obs, agents[cur["yourIndex"]], debug_out=dbg)
        if dbg:
            write_decision(decisions_dir, decision_idx, dbg, names)
            decision_idx += 1
        return selected

    obs, step = play_match(obs, rec, agents, tracer, on_step=on_step)

    result = obs["current"]["result"]
    outcome = {0: "P0 (Evee) wins", 1: "P1 (base) wins", 2: "draw"}.get(result, "?")
    print(f"Result: {outcome} after {step} actions, turn {obs['current']['turn']}.")

    base = REPLAYS / f"evee_vs_{opp_file.stem}_{stamp}"
    save_replay(rec, base)
    print(f"Saved decisions:  {decisions_dir}/ ({decision_idx} Evee decisions)")
    subprocess.run([sys.executable, str(ROOT / "visualizer.py"), f"{base}_vis.json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
