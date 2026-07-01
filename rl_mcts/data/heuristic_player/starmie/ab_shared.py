from __future__ import annotations

from pathlib import Path
from typing import Any

from ab_decktools import read_deck_csv
from ab_opttools import legalize_choice, obj_get


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DECK_PATH = PROJECT_ROOT / "submission" / "deck.csv"


def is_initial_observation(obs_dict: Any) -> bool:
    return obj_get(obs_dict, "select") is None


def load_agent_deck(deck: list[int] | None = None, deck_path: str | Path | None = None) -> list[int]:
    if deck is not None:
        return [int(card_id) for card_id in deck]
    return read_deck_csv(deck_path or DEFAULT_DECK_PATH)


def finish_choice(obs_dict: Any, choices: list[int] | None) -> list[int]:
    return legalize_choice(obs_dict, choices or [])
