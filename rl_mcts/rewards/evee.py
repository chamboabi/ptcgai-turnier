"""Evee (evoli) reward shape preset.

Functions stay defined in rewards.core; here they are imported and assigned
weights to build the evee compound reward. Fill in WEIGHTS below.
"""

from rewards.core import compose_shapes  # noqa: F401  (re-export)
from rewards.core import (
    attach_energy_type_shape,
    big_hand_penalty_shape,
    opp_hand_discard_shape,
    opp_target_devolves_shape,
    race_card_shape,
    search_card_shape,
)

# ============================================================
#  Weights — fill in
# ============================================================

WEIGHTS = {
    attach_energy_type_shape: 0.01,
    race_card_shape: 0.2,
    big_hand_penalty_shape: 0.001,
    search_card_shape: 0.0001,
    opp_hand_discard_shape: 0.001,
    opp_target_devolves_shape: 0.1,
}
