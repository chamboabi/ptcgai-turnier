"""
Per-archetype behavior tuning, blended by prediction probabilities.

Usage:
    registry = TacticRegistry()
    registry.register("Starmie", ArchetypeTactic(
        reward_fn=fast_win,
        ucb_exploration=0.6,
    ))
    registry.register("Alakazam ex", ArchetypeTactic(
        reward_fn=prize_pressure,
        unvisited_penalty=0.05,
    ))

    # Each turn, after predicting:
    result = agent.archetype_model.predict(known_opponent_cards)
    tactic = registry.blend(result.archetype_probs)
    adjusted = apply_tactic(agent, tactic)
    action, sample = mcts_agent(obs_dict, adjusted)
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from agent import Agent, MCTSConfig
from rewards import RewardFn

if TYPE_CHECKING:
    from deck_predict import PredictionResult


@dataclass
class ArchetypeTactic:
    """Behavior overrides for one archetype. None = keep agent default."""
    reward_fn: RewardFn | None = None
    ucb_exploration: float | None = None
    policy_temperature: float | None = None
    unvisited_penalty: float | None = None


class TacticRegistry:
    """Maps archetype names to tactics and blends them by probability."""

    def __init__(self) -> None:
        self._tactics: dict[str, ArchetypeTactic] = {}

    def register(self, archetype_name: str, tactic: ArchetypeTactic) -> "TacticRegistry":
        self._tactics[archetype_name] = tactic
        return self

    def blend(self, archetype_probs: dict[str, float]) -> ArchetypeTactic:
        """Probability-weighted blend over registered archetypes.

        Numeric fields: weighted average across archetypes that define them.
        RewardFn: blended by weighting each terminal/shape call by archetype prob.
        Unregistered archetypes are ignored (their weight flows to default).
        """
        entries = [
            (prob, self._tactics[name])
            for name, prob in archetype_probs.items()
            if name in self._tactics
        ]
        if not entries:
            return ArchetypeTactic()

        total_w = sum(p for p, _ in entries)
        if total_w == 0.0:
            return ArchetypeTactic()

        norm = [(p / total_w, t) for p, t in entries]

        def _wavg(field: str) -> float | None:
            s, w = 0.0, 0.0
            for weight, tactic in norm:
                v = getattr(tactic, field)
                if v is not None:
                    s += v * weight
                    w += weight
            return s / w if w > 0.0 else None

        reward_fn: RewardFn | None = None
        reward_entries = [(w, t.reward_fn) for w, t in norm if t.reward_fn is not None]
        if reward_entries:
            def _terminal(obs, your_index, _e=reward_entries):
                return sum(w * fn.terminal(obs, your_index) for w, fn in _e)

            def _shape(obs, your_index, nn_value, _e=reward_entries):
                return sum(w * fn.shape(obs, your_index, nn_value) for w, fn in _e)

            reward_fn = RewardFn(terminal=_terminal, shape=_shape)

        return ArchetypeTactic(
            reward_fn=reward_fn,
            ucb_exploration=_wavg("ucb_exploration"),
            policy_temperature=_wavg("policy_temperature"),
            unvisited_penalty=_wavg("unvisited_penalty"),
        )


def apply_tactic(agent: Agent, tactic: ArchetypeTactic) -> Agent:
    """Return a shallow copy of agent with tactic overrides applied."""
    cfg = agent.mcts_cfg
    new_cfg = replace(
        cfg,
        ucb_exploration=tactic.ucb_exploration if tactic.ucb_exploration is not None else cfg.ucb_exploration,
        policy_temperature=tactic.policy_temperature if tactic.policy_temperature is not None else cfg.policy_temperature,
        unvisited_penalty=tactic.unvisited_penalty if tactic.unvisited_penalty is not None else cfg.unvisited_penalty,
    )
    return replace(
        agent,
        mcts_cfg=new_cfg,
        reward_fn=tactic.reward_fn if tactic.reward_fn is not None else agent.reward_fn,
    )
