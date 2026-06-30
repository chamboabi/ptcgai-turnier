"""Generic, deck-agnostic reward shapes and configs.

Compound reward files should import what they need from here and from core.
"""

import functools
from dataclasses import dataclass, field

from cg.api import Observation
from rewards.core import (
    ATTACHED_ENERGY_CAP,
    DAMAGE_CAP,
    ENERGY_NEEDED,
    OPP_DISCARDED_ENERGY_CAP,
    RewardFn,
    RewardShapeFn,
    attached_energy_shape,
    board,
    card_id,
    compose_shapes,
    damage_shape,
    identity_shape,
    opp_discarded_energy_shape,
    prize_pressure_shape,
    win_loss_terminal,
)

# --- attacker energy ---


@dataclass
class AttackerConfig:
    main: dict[int, float] = field(default_factory=dict)
    secondary: dict[int, float] = field(default_factory=dict)
    needed: dict[int, int] = field(default_factory=dict)
    weight: float = 0.1


def attacker_progress(poke, values: dict[int, float], needed_override: dict[int, int]) -> tuple[float, float]:
    if poke is None or poke.id not in values:
        return 0.0, 0.0
    needed = needed_override.get(poke.id, ENERGY_NEEDED.get(poke.id, 0))
    progress = 1.0 if needed <= 0 else min(1.0, len(poke.energyCards) / needed)
    val = values[poke.id]
    return val, val * progress


def make_attacker_energy_shape(config: AttackerConfig) -> RewardShapeFn:
    def shape(obs: Observation, your_index: int, nn_value: float) -> float:
        you = obs.current.players[your_index]
        b = board(you)
        total_value = 0.0
        earned = 0.0
        for values in (config.main, config.secondary):
            if not values:
                continue
            best_val, best_earned = 0.0, 0.0
            for poke in b:
                val, e = attacker_progress(poke, values, config.needed)
                if val > 0 and e >= best_earned:
                    best_val, best_earned = val, e
            total_value += best_val
            earned += best_earned
        if total_value <= 0:
            return nn_value
        return nn_value * (1.0 - config.weight) + (earned / total_value) * config.weight

    return shape


# --- get attacker ---


@dataclass
class GetAttackerConfig:
    targets: dict[int, float] = field(default_factory=dict)
    hand_credit: float = 0.5
    weight: float = 0.1


def make_get_attacker_shape(config: GetAttackerConfig) -> RewardShapeFn:
    def shape(obs: Observation, your_index: int, nn_value: float) -> float:
        if not config.targets:
            return nn_value
        you = obs.current.players[your_index]
        board_ids = {p.id for p in board(you) if p is not None}
        hand_ids = {card_id(c) for c in (you.hand or [])}
        total_value = 0.0
        earned = 0.0
        for cid, val in config.targets.items():
            total_value += val
            if cid in board_ids:
                earned += val
            elif cid in hand_ids:
                earned += val * config.hand_credit
        if total_value <= 0:
            return nn_value
        return nn_value * (1.0 - config.weight) + (earned / total_value) * config.weight

    return shape


# --- bench keep ---


@dataclass
class BenchKeepConfig:
    keep: dict[int, float] = field(default_factory=dict)
    weight: float = 0.1


def make_bench_keep_shape(config: BenchKeepConfig) -> RewardShapeFn:
    def shape(obs: Observation, your_index: int, nn_value: float) -> float:
        if not config.keep:
            return nn_value
        you = obs.current.players[your_index]
        bench_ids = {p.id for p in you.bench if p is not None}
        total = sum(config.keep.values())
        earned = sum(val for cid, val in config.keep.items() if cid in bench_ids)
        return nn_value * (1.0 - config.weight) + (earned / total) * config.weight

    return shape


# --- target card ---


@dataclass
class TargetCardConfig:
    targets: dict[int, float] = field(default_factory=dict)
    weight: float = 0.1


def make_target_card_shape(config: TargetCardConfig) -> RewardShapeFn:
    def shape(obs: Observation, your_index: int, nn_value: float) -> float:
        if not config.targets:
            return nn_value
        opp = obs.current.players[1 - your_index]
        on_board = {p.id: p for p in board(opp) if p is not None}
        discard = {card_id(c) for c in (opp.discard or [])}
        total_value = 0.0
        earned = 0.0
        for cid, val in config.targets.items():
            if cid in on_board:
                poke = on_board[cid]
                progress = max(0.0, 1.0 - poke.hp / poke.maxHp) if poke.maxHp > 0 else 0.0
                total_value += val
                earned += val * progress
            elif cid in discard:
                total_value += val
                earned += val
        if total_value <= 0:
            return nn_value
        return nn_value * (1.0 - config.weight) + (earned / total_value) * config.weight

    return shape


# --- hand disruption ---


@dataclass
class HandDisruptionConfig:
    target: int = 2
    baseline: int | None = None
    weight: float = 0.1


def make_hand_disruption_shape(config: HandDisruptionConfig) -> RewardShapeFn:
    def shape(obs: Observation, your_index: int, nn_value: float) -> float:
        opp = obs.current.players[1 - your_index]
        base = config.baseline if config.baseline is not None else opp.handCount
        target = max(1, config.target)
        removed = base - opp.handCount
        signal = max(-1.0, min(removed, target) / target)
        return nn_value * (1.0 - config.weight) + signal * config.weight

    return shape


# --- hand build ---


@dataclass
class HandBuildConfig:
    target: int = 2
    baseline: int | None = None
    weight: float = 0.1


def make_hand_build_shape(config: HandBuildConfig) -> RewardShapeFn:
    def shape(obs: Observation, your_index: int, nn_value: float) -> float:
        you = obs.current.players[your_index]
        base = config.baseline if config.baseline is not None else you.handCount
        target = max(1, config.target)
        gained = you.handCount - base
        signal = max(-1.0, min(gained, target) / target)
        return nn_value * (1.0 - config.weight) + signal * config.weight

    return shape


# --- base shape (general-purpose) ---


@dataclass
class BaseShapeConfig:
    """Hold this and set baselines each turn before searching:
    cfg.disruption.baseline = obs.current.players[1 - your_index].handCount
    cfg.build.baseline      = obs.current.players[your_index].handCount
    """

    disruption: HandDisruptionConfig = field(default_factory=HandDisruptionConfig)
    build: HandBuildConfig = field(default_factory=HandBuildConfig)
    attached_energy_weight: float = 0.1
    opp_discarded_energy_weight: float = 0.1
    damage_weight: float = 0.1
    prize_pressure_weight: float = 0.1


def make_base_shape(config: BaseShapeConfig | None = None) -> RewardShapeFn:
    cfg = config if config is not None else BaseShapeConfig()
    return compose_shapes(
        functools.partial(attached_energy_shape, weight=cfg.attached_energy_weight),
        functools.partial(opp_discarded_energy_shape, weight=cfg.opp_discarded_energy_weight),
        functools.partial(damage_shape, weight=cfg.damage_weight),
        make_hand_disruption_shape(cfg.disruption),
        make_hand_build_shape(cfg.build),
        functools.partial(prize_pressure_shape, weight=cfg.prize_pressure_weight),
    )


# --- bare RewardFn factories ---


def attach_energy_reward(weight: float = 0.1) -> RewardFn:
    return RewardFn(
        terminal=win_loss_terminal,
        shape=functools.partial(attached_energy_shape, weight=weight),
    )


def hand_build_reward(target: int = 2, weight: float = 0.1) -> RewardFn:
    return RewardFn(
        terminal=win_loss_terminal,
        shape=make_hand_build_shape(HandBuildConfig(target=target, weight=weight)),
    )


def hand_disruption_reward(target: int = 2, weight: float = 0.1) -> RewardFn:
    return RewardFn(
        terminal=win_loss_terminal,
        shape=make_hand_disruption_shape(HandDisruptionConfig(target=target, weight=weight)),
    )


def get_attacker_reward(targets: dict[int, float], hand_credit: float = 0.5, weight: float = 0.1) -> RewardFn:
    return RewardFn(
        terminal=win_loss_terminal,
        shape=make_get_attacker_shape(GetAttackerConfig(targets=targets, hand_credit=hand_credit, weight=weight)),
    )


def target_reward(targets: dict[int, float], weight: float = 0.1) -> RewardFn:
    return RewardFn(
        terminal=win_loss_terminal,
        shape=make_target_card_shape(TargetCardConfig(targets=targets, weight=weight)),
    )


# --- presets ---

base_shape_config = BaseShapeConfig()

base = RewardFn(terminal=win_loss_terminal, shape=make_base_shape(base_shape_config))

# Named presets used in tactics.py
win_loss = RewardFn(terminal=win_loss_terminal, shape=identity_shape)
prize_pressure = RewardFn(terminal=win_loss_terminal, shape=prize_pressure_shape)
