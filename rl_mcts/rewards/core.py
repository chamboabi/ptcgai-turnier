"""Primitive building blocks shared by all reward files.

Import from here when writing a new compound reward module:
    from rewards.core import (
        RewardFn, RewardShapeFn, RewardTerminalFn,
        compose_shapes,
        win_loss_terminal, prize_pressure_shape, ...
    )
"""

from dataclasses import dataclass
from typing import Callable

from cg.api import CardType, Observation, all_attack, all_card_data

ENERGY_IDS = frozenset(
    cd.cardId for cd in all_card_data() if cd.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)
)
ATTACK_BY_ID = {a.attackId: a for a in all_attack()}
ENERGY_NEEDED = {
    cd.cardId: max(
        (len(ATTACK_BY_ID[aid].energies) for aid in cd.attacks if aid in ATTACK_BY_ID),
        default=0,
    )
    for cd in all_card_data()
}

RewardTerminalFn = Callable[[Observation, int], float]
RewardShapeFn = Callable[[Observation, int, float], float]


@dataclass
class RewardFn:
    terminal: RewardTerminalFn
    shape: RewardShapeFn


# --- terminals ---


def win_loss_terminal(obs: Observation, your_index: int) -> float:
    result = obs.current.result
    if result == 2:
        return 0.0
    return 1.0 if result == your_index else -1.0


def fast_win_terminal(obs: Observation, your_index: int) -> float:
    result = obs.current.result
    if result == 2:
        return 0.0
    sign = 1.0 if result == your_index else -1.0
    turn_factor = max(0.0, 1.0 - obs.current.turn / 60.0)
    return sign * (0.8 + 0.2 * turn_factor)


# --- primitive shapes ---


def identity_shape(obs: Observation, your_index: int, nn_value: float) -> float:
    return nn_value


def prize_pressure_shape(obs: Observation, your_index: int, nn_value: float, weight: float = 0.1) -> float:
    state = obs.current
    your_prizes_left = len(state.players[your_index].prize)
    opp_prizes_left = len(state.players[1 - your_index].prize)
    prize_adv = (opp_prizes_left - your_prizes_left) / 6.0
    return nn_value * (1.0 - weight) + prize_adv * weight


# normalization cap for total attached energy across your board
ATTACHED_ENERGY_CAP = 8

# normalization cap for opponent energy sent to discard (KO'd loaded Pokemon)
OPP_DISCARDED_ENERGY_CAP = 8

# normalization cap for total damage dealt across the opponent's board (HP)
DAMAGE_CAP = 300

# total energy cards in YOUR decklist; override per-deck if needed
DECK_ENERGY_TOTAL = 12


def attached_energy_shape(obs: Observation, your_index: int, nn_value: float, weight: float = 0.1) -> float:
    you = obs.current.players[your_index]
    attached = sum(len(p.energyCards) for p in board(you) if p is not None)
    loaded = min(1.0, attached / ATTACHED_ENERGY_CAP)
    return nn_value * (1.0 - weight) + loaded * weight


def opp_discarded_energy_shape(obs: Observation, your_index: int, nn_value: float, weight: float = 0.1) -> float:
    opp = obs.current.players[1 - your_index]
    trashed = count_energy(opp.discard)
    signal = min(1.0, trashed / OPP_DISCARDED_ENERGY_CAP)
    return nn_value * (1.0 - weight) + signal * weight


def damage_shape(obs: Observation, your_index: int, nn_value: float, weight: float = 0.1) -> float:
    opp = obs.current.players[1 - your_index]
    dealt = sum(max(0, p.maxHp - p.hp) for p in board(opp) if p is not None)
    signal = min(1.0, dealt / DAMAGE_CAP)
    return nn_value * (1.0 - weight) + signal * weight


def high_deck_energy_shape(obs: Observation, your_index: int, nn_value: float, weight: float = 0.1) -> float:
    you = obs.current.players[your_index]
    seen = count_energy(you.hand) + count_energy(you.discard)
    for poke in board(you):
        if poke is not None:
            seen += count_energy(poke.energyCards)
    energy_left = max(0, DECK_ENERGY_TOTAL - seen)
    reserve = energy_left / DECK_ENERGY_TOTAL
    return nn_value * (1.0 - weight) + reserve * weight


# --- helpers ---


def card_id(card) -> int:
    return card if isinstance(card, int) else card.id


def count_energy(cards: list | None) -> int:
    return sum(1 for c in (cards or []) if card_id(c) in ENERGY_IDS)


def board(player) -> list:
    active = player.active[0] if player.active else None
    return [active, *player.bench]


def find(b, cid):
    for p in b:
        if p is not None and p.id == cid:
            return p
    return None


# --- composition ---


def compose_shapes(*shapes: RewardShapeFn) -> RewardShapeFn:
    """Chain shapes: each one's output feeds the next as nn_value."""
    def shape(obs: Observation, your_index: int, nn_value: float) -> float:
        v = nn_value
        for s in shapes:
            v = s(obs, your_index, v)
        return v
    return shape
