from dataclasses import dataclass
from typing import Callable

from cg.api import Observation

RewardTerminalFn = Callable[[Observation, int], float]
RewardShapeFn = Callable[[Observation, int, float], float]


@dataclass
class RewardFn:
    terminal: RewardTerminalFn
    shape: RewardShapeFn


# --- terminals ---


def _win_loss_terminal(obs: Observation, your_index: int) -> float:
    result = obs.current.result
    if result == 2:
        return 0.0
    return 1.0 if result == your_index else -1.0


def _fast_win_terminal(obs: Observation, your_index: int) -> float:
    result = obs.current.result
    if result == 2:
        return 0.0
    sign = 1.0 if result == your_index else -1.0
    # faster win = higher reward; cap at 60 turns
    turn_factor = max(0.0, 1.0 - obs.current.turn / 60.0)
    return sign * (0.8 + 0.2 * turn_factor)


# --- shapes ---


def _identity_shape(obs: Observation, your_index: int, nn_value: float) -> float:
    return nn_value


def _prize_pressure_shape(obs: Observation, your_index: int, nn_value: float) -> float:
    state = obs.current
    your_prizes_left = len(state.players[your_index].prize)
    opp_prizes_left = len(state.players[1 - your_index].prize)
    # positive when opponent has more prizes left (= you took more)
    prize_adv = (opp_prizes_left - your_prizes_left) / 6.0
    return nn_value * 0.9 + prize_adv * 0.1


# --- presets ---

win_loss = RewardFn(terminal=_win_loss_terminal, shape=_identity_shape)
prize_pressure = RewardFn(terminal=_win_loss_terminal, shape=_prize_pressure_shape)
fast_win = RewardFn(terminal=_fast_win_terminal, shape=_identity_shape)
