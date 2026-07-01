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
    # Number of independent MCTS trees to build per decision, each against its
    # own sampled opponent hand/deck/prize (root-parallel / ensemble
    # determinization). Their root visit stats are pooled to pick the final
    # action, which avoids a single hidden-information sample biasing the
    # whole search ("strategy fusion"). Each tree still runs the full
    # search_count sims -- this is a SEPARATE knob from search_count, so
    # raising it increases total compute (more accuracy) rather than
    # splitting a fixed budget thinner. 1 reproduces the old single-tree
    # behavior exactly.
    determinizations: int = 1


@dataclass
class Agent:
    deck: list[int]
    model: MyModel
    mcts_cfg: MCTSConfig
    reward_fn: RewardFn
    archetype_model: ArchetypeModel | None = None
