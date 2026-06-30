"""Reward function library for the PTCG AI agent.

Public API — all names importable as `from rewards import X` or `rewards.X`.

Compound reward files live alongside this package:
    rewards/abomasnow.py   — Mega Abomasnow ex + Kyogre deck
    rewards/<new_deck>.py  — add a new file here for each new compound reward

Primitives and generic shapes:
    rewards/core.py        — types, terminals, primitive shapes, helpers
    rewards/shapes.py      — generic configs, shape makers, base preset
"""

# --- core primitives ---
from rewards.core import (
    RewardFn,
    RewardShapeFn,
    RewardTerminalFn,
    compose_shapes,
    board,
    card_id,
    count_energy,
    find,
    win_loss_terminal,
    fast_win_terminal,
    identity_shape,
    prize_pressure_shape,
    attached_energy_shape,
    opp_discarded_energy_shape,
    damage_shape,
    high_deck_energy_shape,
    ATTACHED_ENERGY_CAP,
    OPP_DISCARDED_ENERGY_CAP,
    DAMAGE_CAP,
    DECK_ENERGY_TOTAL,
    ENERGY_IDS,
    ENERGY_NEEDED,
)

# --- generic shapes ---
from rewards.shapes import (
    AttackerConfig,
    make_attacker_energy_shape,
    GetAttackerConfig,
    make_get_attacker_shape,
    BenchKeepConfig,
    make_bench_keep_shape,
    TargetCardConfig,
    make_target_card_shape,
    HandDisruptionConfig,
    make_hand_disruption_shape,
    HandBuildConfig,
    make_hand_build_shape,
    BaseShapeConfig,
    make_base_shape,
    hand_build_reward,
    hand_disruption_reward,
    get_attacker_reward,
    target_reward,
    base_shape_config,
    base,
    win_loss,
    prize_pressure,
)

# --- abomasnow compound reward ---
from rewards.abomasnow import (
    AbomasnowConfig,
    make_abomasnow_shape,
    AbomasnowGetConfig,
    make_abomasnow_get_shape,
    abomasnow_get_config as abomasnow_get,  # backward-compat name
    abomasnow,
)
