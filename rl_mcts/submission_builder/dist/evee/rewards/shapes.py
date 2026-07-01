"""Base reward shape preset.

Functions stay defined in rewards.core; here they are imported and assigned
weights to build the `base` compound reward.

Absolute-state shapes (energy on board, damage, opp hand/discard) must be scored
as a *delta* from a baseline snapshotted at the search root, otherwise board state
that already existed before the turn keeps paying on every MCTS leaf. Use
`make_base_shape(obs0, your_index)` once per agent call to bake in that baseline.
"""

from rewards.core import (
    Observation,
    damage_taken_uncapped_shape,
    damage_uncapped_shape,
    make_compound_shape,
    opp_discarded_energy_shape,
    play_card_penalty_shape,
    prize_pressure_shape,
    small_opp_hand_shape,
    win_game_shape,
)

# ============================================================
#  Weights — fill in
# ============================================================

WEIGHTS = {
    win_game_shape: 1.0,
    prize_pressure_shape: 0.3,
    damage_uncapped_shape: 0.1,
    damage_taken_uncapped_shape: 0.1,
    opp_discarded_energy_shape: 0.1,
    small_opp_hand_shape: 0.1,
    play_card_penalty_shape: 0.0001,
}
# attack_energy_match_shape needs poke_id / attack_index (no defaults) -> it can't
# run in a generic base. Put it in a deck preset where those ids are known.

# Shapes scored as a delta from a per-turn baseline (read absolute board state).
# Everything else is left as-is: win/loss outcome and per-step log events are
# already deltas, so they need no baseline. prize_pressure is baselined so taking a
# prize pays even when the KO'd mon leaves the board (board-damage signal vanishes).
_NEEDS_BASELINE = frozenset(
    {
        prize_pressure_shape,
        damage_uncapped_shape,
        damage_taken_uncapped_shape,
        opp_discarded_energy_shape,
        small_opp_hand_shape,
    }
)


# ============================================================
#  Compound reward
# ============================================================


def make_base_shape(obs0: Observation, your_index: int):
    """Build the base reward shape for one agent call, freezing baselines from obs0.

    Call once at the search root each turn; reuse the returned fn for every MCTS
    leaf eval. Re-snapshot next turn.
    """
    return make_compound_shape(WEIGHTS, _NEEDS_BASELINE, obs0, your_index)
