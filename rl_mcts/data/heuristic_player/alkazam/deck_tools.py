from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable


def read_deck_csv(path: str | Path) -> list[int]:
    """Read a one-card-id-per-row Kaggle deck CSV."""
    deck_path = Path(path)
    if not deck_path.exists():
        raise FileNotFoundError(f"Deck file not found: {deck_path}")

    deck: list[int] = []
    with deck_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if not row or not row[0].strip():
                continue
            try:
                deck.append(int(row[0].strip()))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid card ID in {deck_path} at row {row_number}: {row[0]!r}"
                ) from exc
    return deck


def write_deck_csv(deck: Iterable[int], path: str | Path) -> None:
    """Write a deck in the format expected by the competition sample agent."""
    deck_path = Path(path)
    deck_path.parent.mkdir(parents=True, exist_ok=True)
    with deck_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for card_id in deck:
            writer.writerow([int(card_id)])


def validate_deck_basic(deck: Iterable[int]) -> dict:
    """Basic structural deck validation without full Pokemon legality checks."""
    deck_list = list(deck)
    errors: list[str] = []

    if len(deck_list) != 60:
        errors.append(f"Deck must contain 60 cards; found {len(deck_list)}.")

    invalid_positions = [
        idx
        for idx, card_id in enumerate(deck_list)
        if not isinstance(card_id, int) or isinstance(card_id, bool)
    ]
    if invalid_positions:
        preview = ", ".join(str(idx) for idx in invalid_positions[:10])
        errors.append(f"Card IDs must be ints; invalid positions: {preview}.")

    counts = Counter(card_id for card_id in deck_list if isinstance(card_id, int))
    return {
        "valid": not errors,
        "length": len(deck_list),
        "is_60": len(deck_list) == 60,
        "all_ints": not invalid_positions,
        "counts": dict(sorted(counts.items())),
        "errors": errors,
    }
