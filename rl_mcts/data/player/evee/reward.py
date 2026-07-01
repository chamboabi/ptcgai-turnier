"""Player-package reward entrypoint for evee.

Thin re-export: the real shape logic still lives in rewards/evee.py (shared
with submission_builder, which bundles the whole rewards/ package into the
tournament dist). Once that pipeline is updated to read from data/player/
packages directly, this can become the canonical definition instead of a
wrapper.
"""

from rewards.evee import make_evee_shape as make_shape

__all__ = ["make_shape"]
