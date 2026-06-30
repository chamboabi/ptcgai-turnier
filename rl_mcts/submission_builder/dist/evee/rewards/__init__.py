"""Reward function library for the PTCG AI agent.

Public API — all names importable as `from rewards import X` or `rewards.X`.

Primitives live in rewards/core.py (types, terminals, primitive shapes, helpers).
The old generic shapes (rewards/shapes.py) and compound rewards
(rewards/abomasnow.py) were cleared out during the reward restructure; rebuild
new compound rewards from the core primitives + compose_shapes.

`base`, `abomasnow`, and `base_shape_config` remain as TODO stubs so existing
entrypoints (run_match.py, train.py, submission_builder) import; wiring a real
reward is still pending.
"""

from types import SimpleNamespace

# --- core primitives ---
from rewards.core import (
    RewardFn,
    RewardShapeFn,
    RewardTerminalFn,
    compose_shapes,
    board,
    card_id,
    count_energy,
    energy_type,
    attacks_of,
    find,
    win_loss_terminal,
    fast_win_terminal,
    identity_shape,
    prize_pressure_shape,
    attached_energy_shape,
    attached_energy_uncapped_shape,
    attach_energy_type_shape,
    attach_energy_capped_shape,
    attack_energy_match_shape,
    attack_energy_overload_shape,
    opp_discarded_energy_shape,
    damage_capped_shape,
    damage_uncapped_shape,
    damage_taken_shape,
    damage_taken_uncapped_shape,
    high_deck_energy_shape,
    small_opp_hand_shape,
    big_hand_penalty_shape,
    play_card_penalty_shape,
    search_card_shape,
    opp_hand_discard_shape,
    opp_target_leaves_field_shape,
    opp_target_devolves_shape,
    opp_evolved_devolves_shape,
    race_card_shape,
    race_vs_ex_shape,
    opp_active_is_ex,
    hand_size_range_shape,
    ENERGY_IDS,
    ENERGY_NEEDED,
)
from rewards.shapes import make_base_shape
from rewards.evee import make_evee_shape


# --- TODO stubs (reward restructure pending) ---
def _todo_reward(*_args, **_kwargs):
    raise NotImplementedError(
        "Reward not wired yet — build a new compound reward from rewards.core "
        "primitives and assign it to rewards.base / rewards.abomasnow."
    )


# Bare win/loss fallback so a stripped run still has a defined terminal signal.
win_loss = RewardFn(terminal=win_loss_terminal, shape=identity_shape)

# Real rewards: terminal is win/loss; the per-turn shape is built from the search
# root each turn via shape_factory (mcts_agent rebuilds it), so absolute shapes
# score deltas from a frozen baseline. `shape` is an identity fallback for any
# caller that does not go through mcts_agent.
base = RewardFn(terminal=win_loss_terminal, shape=identity_shape, shape_factory=make_base_shape)
evee = RewardFn(terminal=win_loss_terminal, shape=identity_shape, shape_factory=make_evee_shape)

# Still a stub — rewards/abomasnow.py is empty pending rebuild.
abomasnow = RewardFn(terminal=_todo_reward, shape=_todo_reward)

# Old base-shape config object — entrypoints still set .disruption.baseline /
# .build.baseline each turn; keep settable placeholders so that plumbing no-ops.
base_shape_config = SimpleNamespace(
    disruption=SimpleNamespace(baseline=None),
    build=SimpleNamespace(baseline=None),
)
