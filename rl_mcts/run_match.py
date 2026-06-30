"""Play ONE recorded match: Abomasnow deck (abomasnow shape) vs a random
decklist (base shape). Saves a replay + visualizer JSON under replays/.

Run:  env/bin/python run_match.py
"""

import random
import sys
import time
from pathlib import Path

import torch

import config as cfg
import rewards
from agent import Agent, MCTSConfig
from cg.api import LogType, all_card_data
from cg.recorder import GameRecorder
from mcts import mcts_agent
from model import MyModel

# Print a per-turn action path (what each agent actually did) at every TURN_END.
DEBUG_TURN_PATH = True

HERE = Path(__file__).parent
CUSTOM = HERE / "data" / "decks" / "customdecks" / "abamasnow.csv"
DECKLISTS = HERE / "data" / "decks" / "decklists"
REPLAYS = HERE / "replays"


def load_deck(path: Path) -> list[int]:
    """Read a deck CSV. Handles both one-ID-per-line and comma-separated."""
    text = path.read_text().replace(",", "\n")
    return [int(tok) for tok in text.split() if tok.strip()]


def build_model() -> MyModel:
    m = MyModel(
        cfg.model["d_model"],
        cfg.model["num_heads"],
        cfg.model["d_feedforward"],
        cfg.model["num_layers_encoder"],
        cfg.model["num_layers_decoder"],
    )
    ckpt = HERE / "out" / "model_best.pth"
    if ckpt.exists():
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"Loaded checkpoint {ckpt}")
    else:
        print("No checkpoint — using randomly initialised model (play is weak).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return m.to(device).eval()


def make_mcts_cfg() -> MCTSConfig:
    return MCTSConfig(
        search_count=cfg.mcts["search_count"],
        max_action_combinations=cfg.mcts["max_action_combinations"],
        ucb_exploration=cfg.mcts["ucb_exploration"],
        policy_temperature=cfg.mcts["policy_temperature"],
        unvisited_penalty=cfg.mcts["unvisited_penalty"],
    )


def set_base_baselines(cur: dict) -> None:
    """Set the shared base-shape hand baselines from the ACTING player's view.

    Both `base` and `abomasnow` shapes read rewards.base_shape_config, and only
    one agent acts per step, so refreshing it from the current perspective each
    step keeps the hand-disruption / hand-build deltas correct for whoever moves.
    """
    yi = cur["yourIndex"]
    players = cur["players"]
    rewards.base_shape_config.disruption.baseline = players[1 - yi]["handCount"]
    rewards.base_shape_config.build.baseline = players[yi]["handCount"]


def _name(names: dict[int, str], cid: int | None) -> str:
    """Card name for a log id, falling back to #id."""
    if cid is None:
        return "?"
    return names.get(cid, f"#{cid}")


def format_log(log: dict, names: dict[int, str]) -> str | None:
    """One readable line for an interesting log event, or None to skip noise."""
    t = log.get("type")
    n = lambda key: _name(names, log.get(key))
    if t == LogType.DRAW:
        return f"draw {n('cardId')}"
    if t == LogType.PLAY:
        return f"play {n('cardId')}"
    if t == LogType.ATTACH:
        return f"attach {n('cardId')} -> {n('cardIdTarget')}"
    if t == LogType.EVOLVE:
        return f"evolve {n('cardIdTarget')} -> {n('cardId')}"
    if t == LogType.DEVOLVE:
        return f"devolve {n('cardIdTarget')} -> {n('cardId')}"
    if t == LogType.SWITCH:
        return f"switch active {n('cardIdActive')} <-> bench {n('cardIdBench')}"
    if t == LogType.CHANGE:
        return f"change active {n('cardIdBefore')} -> {n('cardIdAfter')}"
    if t == LogType.ATTACK:
        return f"ATTACK with {n('cardId')} (attackId {log.get('attackId')})"
    if t == LogType.HP_CHANGE:
        v = log.get("value")
        if v:
            return f"hp {n('cardId')} {v:+d}"
    return None


class TurnPathTracer:
    """Accumulate a player's realized actions over a turn; flush at TURN_END."""

    def __init__(self, names: dict[int, str], p0_label: str = "Abomasnow", p1_label: str = "base"):
        self.names = names
        self.labels = (p0_label, p1_label)
        self.lines: list[str] = []
        self.owner: int | None = None
        self.turn = 0

    def feed(self, logs: list[dict], turn: int) -> None:
        for log in logs or []:
            t = log.get("type")
            if t == LogType.TURN_START:
                self.lines = []
                self.owner = log.get("playerIndex")
                self.turn = turn
            elif t == LogType.TURN_END:
                self._flush()
                self.lines = []
                self.owner = None
            else:
                line = format_log(log, self.names)
                if line is not None:
                    self.lines.append(line)

    def _flush(self) -> None:
        if not DEBUG_TURN_PATH or self.owner is None:
            return
        tag = self.labels[self.owner]
        print(f"\n=== P{self.owner} ({tag}) turn {self.turn} path ===")
        for ln in self.lines:
            print(f"    {ln}")
        if not self.lines:
            print("    (no actions)")
        print("=== end turn ===")


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

    rec = GameRecorder(seed=random.randint(0, 2**31 - 1))
    obs, start = rec.start(abomasnow_deck, opp_deck)
    if start.errorPlayer >= 0:
        print(f"Deck error (player {start.errorPlayer}, type {start.errorType}).", file=sys.stderr)
        return 1

    tracer = TurnPathTracer({c.cardId: c.name for c in all_card_data()})

    step = 0
    with torch.inference_mode():
        while obs["current"]["result"] < 0:
            cur = obs["current"]
            set_base_baselines(cur)
            selected, _ = mcts_agent(obs, agents[cur["yourIndex"]])
            obs = rec.select(selected)
            tracer.feed(obs.get("logs"), obs["current"]["turn"])
            step += 1
            if step % 20 == 0:
                print(f"  ...{step} actions, turn {obs['current']['turn']}")

    rec.finish()

    result = obs["current"]["result"]
    outcome = {0: "P0 (Abomasnow) wins", 1: "P1 (base) wins", 2: "draw"}.get(result, "?")
    print(f"Result: {outcome} after {step} actions, turn {obs['current']['turn']}.")

    REPLAYS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = REPLAYS / f"abomasnow_vs_{opp_file.stem}_{stamp}"
    rec.save(f"{base}.json")
    rec.save_visualizer(f"{base}_vis.json")
    print(f"Saved replay:     {base}.json")
    print(f"Saved visualizer: {base}_vis.json")
    print("Open debug/visualizer.html in a browser and load the *_vis.json file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
