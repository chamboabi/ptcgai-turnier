"""Compound reward for the Mega Abomasnow ex + Kyogre deck."""

from dataclasses import dataclass

from cg.api import Observation

from rewards.core import (
    RewardFn,
    RewardShapeFn,
    board,
    card_id,
    find,
    high_deck_energy_shape,
    win_loss_terminal,
    compose_shapes,
)
from rewards.shapes import base_shape_config, make_base_shape


@dataclass
class AbomasnowConfig:
    main_id: int = 723       # Mega Abomasnow ex
    main_target: int = 2     # energy it wants; surplus flows to Kyogre
    secondary_id: int = 721  # Kyogre (soaks overflow energy)
    secondary_max: int = 3
    pre_id: int = 722        # Snover (pre-evolution)
    pre_target: int = 2
    main_value: float = 1.0
    secondary_value: float = 0.7
    pre_value: float = 0.7
    overflow_penalty: float = 0.5
    weight: float = 0.2


def make_abomasnow_shape(config: AbomasnowConfig) -> RewardShapeFn:
    """Energy priority waterfall: main 723 -> secondary 721 -> pre-evo 722."""
    c = config

    def shape(obs: Observation, your_index: int, nn_value: float) -> float:
        you = obs.current.players[your_index]
        b = board(you)
        alive = [p for p in b if p is not None]

        total_value = 0.0
        earned = 0.0

        main = find(b, c.main_id)
        main_satisfied = False
        main_overflow = 0
        if main is not None:
            e = len(main.energyCards)
            score = min(e, c.main_target) / c.main_target
            total_value += c.main_value
            earned += c.main_value * score
            main_satisfied = e >= c.main_target
            main_overflow = max(0, e - c.main_target)

        sec = find(b, c.secondary_id)
        if sec is not None:
            se = len(sec.energyCards)
            score = min(se, c.secondary_max) / c.secondary_max
            total_value += c.secondary_value
            earned += c.secondary_value * score
            room = max(0, c.secondary_max - se)
            if main_overflow > 0 and room > 0:
                waste = min(main_overflow, room) / c.secondary_max
                earned -= c.secondary_value * c.overflow_penalty * waste

        pre = find(b, c.pre_id)
        if pre is not None:
            only_one = len(alive) == 1
            hand_ids = {card_id(h) for h in (you.hand or [])}
            can_evolve = (c.main_id in hand_ids) and not getattr(pre, "appearThisTurn", False)
            pre_active = only_one or (can_evolve and main_satisfied)
            if pre_active:
                score = min(len(pre.energyCards), c.pre_target) / c.pre_target
                total_value += c.pre_value
                earned += c.pre_value * score

        if total_value <= 0:
            return nn_value
        signal = max(0.0, earned / total_value)
        return nn_value * (1.0 - c.weight) + signal * c.weight

    return shape


@dataclass
class AbomasnowGetConfig:
    main_id: int = 723
    pre_id: int = 722
    secondary_id: int = 721
    main_value: float = 1.0
    pre_value: float = 0.6
    secondary_value: float = 0.5
    secondary_value_late: float = 1.2
    late_prizes: int = 3
    hand_credit: float = 0.5
    weight: float = 0.1


def make_abomasnow_get_shape(config: AbomasnowGetConfig) -> RewardShapeFn:
    """Hunt for the Abomasnow line (pivots to Snover when 723 is in hand) + Kyogre."""
    c = config

    def shape(obs: Observation, your_index: int, nn_value: float) -> float:
        you = obs.current.players[your_index]
        opp = obs.current.players[1 - your_index]
        board_ids = {p.id for p in board(you) if p is not None}
        hand_ids = {card_id(h) for h in (you.hand or [])}

        total_value = 0.0
        earned = 0.0

        total_value += c.main_value
        if c.main_id in board_ids:
            earned += c.main_value
        elif c.main_id in hand_ids:
            if c.pre_id in board_ids:
                earned += c.main_value * 0.8
            elif c.pre_id in hand_ids:
                earned += c.main_value * c.hand_credit
        elif c.pre_id in board_ids:
            earned += c.main_value * c.hand_credit
        elif c.pre_id in hand_ids:
            earned += c.main_value * c.hand_credit * 0.5

        opp_prizes_left = len(opp.prize)
        sec_value = c.secondary_value_late if opp_prizes_left >= c.late_prizes else c.secondary_value
        total_value += sec_value
        if c.secondary_id in board_ids:
            earned += sec_value
        elif c.secondary_id in hand_ids:
            earned += sec_value * c.hand_credit

        if total_value <= 0:
            return nn_value
        return nn_value * (1.0 - c.weight) + (earned / total_value) * c.weight

    return shape


# --- presets ---

_abomasnow_config = AbomasnowConfig()

abomasnow_get_config = AbomasnowGetConfig(
    main_id=_abomasnow_config.main_id,
    pre_id=_abomasnow_config.pre_id,
    secondary_id=_abomasnow_config.secondary_id,
    hand_credit=0.5,
    weight=0.1,
)

abomasnow = RewardFn(
    terminal=win_loss_terminal,
    shape=compose_shapes(
        make_abomasnow_get_shape(abomasnow_get_config),
        make_abomasnow_shape(_abomasnow_config),
        high_deck_energy_shape,
        make_base_shape(base_shape_config),
    ),
)
