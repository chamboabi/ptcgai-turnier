"""Evee (evoli) reward shape preset.

Functions stay defined in rewards.core; here they are imported, bound to any
deck-specific ids, and assigned weights to build the evee compound reward.

Absolute-state shapes (energy on a pokemon, a card being in play) are scored as a
*delta* from a baseline snapshotted at the search root, so board state that already
stood before the turn does not keep paying on every MCTS leaf. Use
`make_evee_shape(obs0, your_index)` once per agent call to bake in that baseline.
"""

import functools

from cg.api import EnergyType
from rewards.core import (
    Observation,
    attach_energy_type_shape,
    big_hand_penalty_shape,
    make_compound_shape,
    opp_evolved_devolves_shape,
    opp_hand_discard_shape,
    race_vs_ex_shape,
    search_card_shape,
)

# ============================================================
#  Deck-specific ids — fill in
# ============================================================

ATTACH_ADDITIONAL_ENERGY_ID = 112
ADDITIONAL_ENERGY = EnergyType.DARKNESS

# Card to race into play as the counter, only while the opponent's active is ex.
RACE_TARGET_ID = 330

# Shapes that take a poke_id / energy type are pre-bound here so they match the
# generic shape signature (obs, your_index, nn_value, weight=...).
attach_eevee_energy = functools.partial(
    attach_energy_type_shape,
    poke_id=ATTACH_ADDITIONAL_ENERGY_ID,
    energy_type_code=ADDITIONAL_ENERGY,
)
race_eevee = functools.partial(race_vs_ex_shape, poke_id=RACE_TARGET_ID)

# ============================================================
#  Weights — fill in
# ============================================================

WEIGHTS = {
    attach_eevee_energy: 0.01,
    race_eevee: 0.2,
    big_hand_penalty_shape: 0.001,
    search_card_shape: 0.0001,
    opp_hand_discard_shape: 0.001,
    opp_evolved_devolves_shape: 0.1,
}

# Shapes scored as a delta from a per-turn baseline (read absolute field state).
# big_hand / search / opp_hand_discard / opp_evolved_devolves are state-target or
# per-step-log shapes, already deltas -> no baseline.
_NEEDS_BASELINE = frozenset(
    {
        attach_eevee_energy,
        race_eevee,
    }
)


# ============================================================
#  Compound reward
# ============================================================


def make_evee_shape(obs0: Observation, your_index: int):
    """Build the evee reward shape for one agent call, freezing baselines from obs0.

    Call once at the search root each turn; reuse the returned fn for every MCTS
    leaf eval. Re-snapshot next turn.
    """
    return make_compound_shape(WEIGHTS, _NEEDS_BASELINE, obs0, your_index)
