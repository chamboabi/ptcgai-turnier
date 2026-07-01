"""Load a self-contained agent package from data/player/<name>/.

A player package holds everything needed to field one agent:
    deck.csv          60 card IDs (see core.player.load_deck)
    reward.py          defines make_shape(obs0, your_index) -> RewardShapeFn
    config.json         optional; keys here override config.json (root), deep-merged
    weights/
        manifest.json   {"active": "<version>", "versions": {"<version>": {"file": ...}}}
        <version>.pth   checkpoint files named in the manifest

To add a new checkpoint: drop the .pth in weights/, add an entry to
"versions" in manifest.json. To switch which one loads: change "active".
Older versions stay on disk for comparison/rollback.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import config as root_cfg
from agent import Agent, MCTSConfig
from core.model_loader import build_model
from core.player import load_deck
from rewards.core import RewardFn, identity_shape, win_loss_terminal

if TYPE_CHECKING:
    from deck_predict import ArchetypeModel

PLAYER_DIR = Path(__file__).parent.parent / "data" / "player"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override onto base; override wins on conflicts."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _player_path(name: str) -> Path:
    path = PLAYER_DIR / name
    if not path.is_dir():
        raise FileNotFoundError(f"player package not found: {path}")
    return path


def load_player_config(name: str) -> dict[str, Any]:
    """Deep-merge data/player/<name>/config.json over the root config.json."""
    base = {"model": dict(root_cfg.model), "mcts": dict(root_cfg.mcts), "training": dict(root_cfg.training)}
    override_path = _player_path(name) / "config.json"
    if override_path.is_file():
        override = json.loads(override_path.read_text())
        base = _deep_merge(base, override)
    return base


def make_mcts_cfg_from(cfg: dict[str, Any]) -> MCTSConfig:
    mcts = cfg["mcts"]
    return MCTSConfig(
        search_count=mcts["search_count"],
        max_action_combinations=mcts["max_action_combinations"],
        ucb_exploration=mcts["ucb_exploration"],
        policy_temperature=mcts["policy_temperature"],
        unvisited_penalty=mcts["unvisited_penalty"],
        determinizations=mcts.get("determinizations", 1),
    )


def load_player_reward(name: str) -> RewardFn:
    """Import reward.py from the player package and wrap its make_shape as a RewardFn."""
    reward_path = _player_path(name) / "reward.py"
    if not reward_path.is_file():
        raise FileNotFoundError(f"no reward.py in {reward_path.parent}")
    spec = importlib.util.spec_from_file_location(f"player_reward_{name}", reward_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "make_shape"):
        raise AttributeError(f"{reward_path} must define make_shape(obs0, your_index)")
    return RewardFn(terminal=win_loss_terminal, shape=identity_shape, shape_factory=module.make_shape)


def load_player_weights(name: str) -> Path | None:
    """Resolve the active checkpoint from weights/manifest.json, or None if absent."""
    weights_dir = _player_path(name) / "weights"
    manifest_path = weights_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    active = manifest["active"]
    entry = manifest["versions"][active]
    return weights_dir / entry["file"]


def load_rl_player(name: str, archetype_model: "ArchetypeModel | None" = None) -> Agent:
    """Assemble an Agent (deck + model + mcts config + reward) from a player package."""
    path = _player_path(name)
    deck = load_deck(path / "deck.csv")
    cfg = load_player_config(name)
    mcts_cfg = make_mcts_cfg_from(cfg)
    model = build_model(load_player_weights(name))
    reward_fn = load_player_reward(name)
    return Agent(deck=deck, model=model, mcts_cfg=mcts_cfg, reward_fn=reward_fn, archetype_model=archetype_model)
