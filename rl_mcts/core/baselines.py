"""Per-step baseline refresh shared by base-shape and derived reward shapes."""

from __future__ import annotations

import rewards


def set_base_baselines(cur: dict) -> None:
    """Set the shared base-shape hand baselines from the ACTING player's view.

    Both `base` and shapes built on top of it read rewards.base_shape_config,
    and only one agent acts per step, so refreshing it from the current
    perspective each step keeps the hand-disruption / hand-build deltas
    correct for whoever moves.
    """
    yi = cur["yourIndex"]
    players = cur["players"]
    rewards.base_shape_config.disruption.baseline = players[1 - yi]["handCount"]
    rewards.base_shape_config.build.baseline = players[yi]["handCount"]
