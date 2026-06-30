"""Base reward shape preset.

Functions stay defined in rewards.core; here they are imported and assigned
weights to build the `base` compound reward. Fill in WEIGHTS below.
"""

from rewards.core import compose_shapes  # noqa: F401  (re-export)
from rewards.core import (
    attached_energy_shape,
    attack_energy_match_shape,
    damage_taken_uncapped_shape,
    damage_uncapped_shape,
    opp_discarded_energy_shape,
    play_card_penalty_shape,
    small_opp_hand_shape,
    win_game_shape,
)

# ============================================================
#  Weights — fill in
# ============================================================

WEIGHTS = {
    win_game_shape: 1.0,
    attached_energy_shape: 0.001,
    attack_energy_match_shape: 0.001,
    damage_uncapped_shape: 0.1,
    damage_taken_uncapped_shape: 0.1,
    opp_discarded_energy_shape: 0.1,
    small_opp_hand_shape: 0.1,
    play_card_penalty_shape: 0.0001,
}
