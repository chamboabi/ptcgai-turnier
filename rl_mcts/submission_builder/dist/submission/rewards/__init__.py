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
    ENERGY_IDS,
    ENERGY_NEEDED,
    RewardFn,
    RewardShapeFn,
    RewardTerminalFn,
    attach_energy_capped_shape,
    attach_energy_type_shape,
    attached_energy_shape,
    attached_energy_uncapped_shape,
    attack_energy_match_shape,
    attack_energy_overload_shape,
    attacks_of,
    big_hand_penalty_shape,
    board,
    card_id,
    compose_shapes,
    count_energy,
    damage_capped_shape,
    damage_taken_shape,
    damage_taken_uncapped_shape,
    damage_uncapped_shape,
    energy_type,
    fast_win_terminal,
    find,
    hand_size_range_shape,
    high_deck_energy_shape,
    identity_shape,
    opp_active_is_ex,
    opp_discarded_energy_shape,
    opp_evolved_devolves_shape,
    opp_hand_discard_shape,
    opp_target_devolves_shape,
    opp_target_leaves_field_shape,
    play_card_penalty_shape,
    prize_pressure_shape,
    race_card_shape,
    search_card_shape,
    small_opp_hand_shape,
    win_loss_terminal,
)
from rewards.evee import make_evee_shape
from rewards.shapes import make_base_shape

# Bare win/loss fallback so a stripped run still has a defined terminal signal.
win_loss = RewardFn(terminal=win_loss_terminal, shape=identity_shape)

# Real rewards: terminal is win/loss; the per-turn shape is built from the search
# root each turn via shape_factory (mcts_agent rebuilds it), so absolute shapes
# score deltas from a frozen baseline. `shape` is an identity fallback for any
# caller that does not go through mcts_agent.
base = RewardFn(terminal=win_loss_terminal, shape=identity_shape, shape_factory=make_base_shape)
evee = RewardFn(terminal=win_loss_terminal, shape=identity_shape, shape_factory=make_evee_shape)


# Old base-shape config object — entrypoints still set .disruption.baseline /
# .build.baseline each turn; keep settable placeholders so that plumbing no-ops.
base_shape_config = SimpleNamespace(
    disruption=SimpleNamespace(baseline=None),
    build=SimpleNamespace(baseline=None),
)
