"""Shared filesystem locations used by run scripts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent
CUSTOMDECKS = ROOT / "data" / "decks" / "customdecks"
DECKLISTS = ROOT / "data" / "decks" / "decklists"
REPLAYS = ROOT / "replays"
