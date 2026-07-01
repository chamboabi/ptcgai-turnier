from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from model import MyModel
from rewards import RewardFn

if TYPE_CHECKING:
    from deck_predict import ArchetypeModel


@dataclass
class MCTSConfig:
    search_count: int
    max_action_combinations: int
    ucb_exploration: float
    policy_temperature: float
    unvisited_penalty: float


@dataclass
class Agent:
    deck: list[int]
    model: MyModel
    mcts_cfg: MCTSConfig
    reward_fn: RewardFn
    archetype_model: ArchetypeModel | None = None
