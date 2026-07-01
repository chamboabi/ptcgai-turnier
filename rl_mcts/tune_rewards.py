"""CMA-ES tuning for base reward shape weights.

Usage:
    python tune_rewards.py [--games N] [--model PATH]

Loads a pre-trained model checkpoint and searches for the reward weight vector
that maximises win rate against a random opponent. Results are saved to
tune_result.json.

NOTE: the base-shape reward this tuner optimised was removed in the reward
restructure; to_vec/from_vec are stubs until a new reward is wired up.
"""

import argparse
import json
import random
from pathlib import Path

import cma
import torch

from agent import Agent
from cg.api import to_observation_class
from cg.game import battle_finish, battle_select, battle_start
from core import CUSTOMDECKS, build_model, load_deck, make_mcts_cfg
from mcts import mcts_agent
from rewards import RewardFn, win_loss_terminal

ABOMASNOW_DECK_CSV = CUSTOMDECKS / "abamasnow.csv"

PARAM_NAMES = [
    "attached_energy_weight",
    "opp_discarded_energy_weight",
    "damage_weight",
    "hand_disruption_weight",
    "hand_build_weight",
    "prize_pressure_weight",
]


# The base-shape machinery this tuner optimised was removed in the reward
# restructure. Rewire `to_vec` / `from_vec` to a new compound reward built from
# rewards.core primitives (PARAM_NAMES are the weight knobs to expose).
def to_vec(_cfg) -> list[float]:
    raise NotImplementedError("Reward restructure pending — wire to_vec to the new reward weights.")


def from_vec(_v: list[float]) -> RewardFn:
    raise NotImplementedError("Reward restructure pending — build a reward from rewards.core primitives.")


def _random_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)


def eval_winrate(reward_fn: RewardFn, agent: Agent, n_games: int) -> float:
    agent.reward_fn = reward_fn

    wins = draws = 0
    for i in range(n_games):
        obs, start_data = battle_start(agent.deck, agent.deck)
        if start_data.errorPlayer >= 0:
            raise ValueError("Deck error during eval.")
        your_index = i % 2
        while True:
            if obs["current"]["result"] >= 0:
                break
            if obs["current"]["yourIndex"] == your_index:
                selected, _ = mcts_agent(obs, agent)
            else:
                selected = _random_agent(obs)
            obs = battle_select(selected)
        battle_finish()
        result = obs["current"]["result"]
        if result == 2:
            draws += 1
        elif result == your_index:
            wins += 1

    decisive = n_games - draws
    return wins / max(1, decisive)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--model", type=str, default="out/model_best.pth")
    args = parser.parse_args()

    model = build_model(checkpoint=Path(args.model))

    deck_path = ABOMASNOW_DECK_CSV
    deck = load_deck(deck_path) if deck_path.exists() else []
    if not deck:
        raise FileNotFoundError(f"Deck not found at {deck_path}")

    agent = Agent(deck=deck, model=model, mcts_cfg=make_mcts_cfg(), reward_fn=from_vec([0.1] * 6))

    x0 = [0.1] * len(PARAM_NAMES)
    es = cma.CMAEvolutionStrategy(x0, sigma0=0.05, inopts={"bounds": [0.0, 0.5], "verbose": -9})

    generation = 0
    with torch.inference_mode():
        while not es.stop():
            solutions = es.ask()
            fitnesses = []
            for v in solutions:
                reward_fn = from_vec(list(v))
                wr = eval_winrate(reward_fn, agent, args.games)
                fitnesses.append(-wr)
            es.tell(solutions, fitnesses)
            best_wr = -min(fitnesses)
            generation += 1
            print(f"Gen {generation:3d}  best_winrate={best_wr:.3f}  sigma={es.sigma:.4f}", flush=True)

    best_v = list(es.result.xbest)
    result = {name: round(val, 6) for name, val in zip(PARAM_NAMES, best_v)}
    result["best_winrate"] = round(-es.result.fbest, 4)

    out_path = Path("tune_result.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nBest config saved to {out_path}:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
