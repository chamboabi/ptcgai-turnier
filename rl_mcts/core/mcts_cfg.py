"""Build an MCTSConfig from config.json."""

from __future__ import annotations

import config as cfg
from agent import MCTSConfig


def make_mcts_cfg() -> MCTSConfig:
    return MCTSConfig(
        search_count=cfg.mcts["search_count"],
        max_action_combinations=cfg.mcts["max_action_combinations"],
        ucb_exploration=cfg.mcts["ucb_exploration"],
        policy_temperature=cfg.mcts["policy_temperature"],
        unvisited_penalty=cfg.mcts["unvisited_penalty"],
    )
