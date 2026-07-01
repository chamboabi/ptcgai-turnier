"""Player setup: a deck + the function that picks its moves."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HEURISTIC_PLAYER_DIR = Path(__file__).parent.parent / "data" / "heuristic_player"


def load_deck(path: Path) -> list[int]:
    """Read a deck CSV. Handles both one-ID-per-line and comma-separated."""
    text = path.read_text().replace(",", "\n")
    return [int(tok) for tok in text.split() if tok.strip()]


@dataclass
class Player:
    deck: list[int]
    act: Callable[[dict], list[int]]  # obs -> selection
    name: str = ""


def load_player(deck: list[int] | str | Path, act: Callable[[dict], list[int]], name: str = "") -> Player:
    """Base player loader: resolve a deck (list of ids or CSV path) and pair it
    with an `act` function. Build agent/opponent-specific loaders on top by
    passing different `act` closures.
    """
    if isinstance(deck, (str, Path)):
        deck = load_deck(Path(deck))
    return Player(deck=deck, act=act, name=name)


def load_heuristic_player(folder: str | Path, name: str = "") -> Player:
    """Load a player from data/heuristic_player/<folder>/.

    The folder holds a .csv deck (any filename) and a main.py defining
    `agent(obs_dict: dict) -> list[int]`, used as the Player's act function.
    `folder` may be a name looked up under data/heuristic_player/, or a
    direct path to such a folder.
    """
    path = Path(folder)
    if not path.is_dir():
        path = HEURISTIC_PLAYER_DIR / folder
    if not path.is_dir():
        raise FileNotFoundError(f"heuristic player folder not found: {folder}")

    csv_files = sorted(path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"no .csv deck file in {path}")
    deck = load_deck(csv_files[0])

    main_path = path / "main.py"
    if not main_path.is_file():
        raise FileNotFoundError(f"no main.py in {path}")
    spec = importlib.util.spec_from_file_location(f"heuristic_player_{path.name}", main_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return load_player(deck, module.agent, name=name or path.name)
