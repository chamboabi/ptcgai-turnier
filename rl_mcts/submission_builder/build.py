"""Bundle a tournament submission.

A submission folder contains exactly:
    main.py     agent logic (self-contained; imports only torch + bundled cg/)
    deck.csv    60 card IDs, one per line
    cg/         the untouched game library
    model.pth   optional trained weights (main.py uses random init if absent)

Usage:
    from build import build_submission

    build_submission(
        deck="../data/decks/customdecks/abamasnow.csv",   # path or list[int]
        out_dir="dist/abomasnow",
        weights="../out/model_best.pth",                   # optional
        zip_it=True,
    )

Or from the shell:
    python build.py ../data/decks/customdecks/abamasnow.csv --out dist/abomasnow --weights ../out/model_best.pth
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).parent
SRC_CG = HERE / "cg"
SRC_MAIN = HERE / "main.py"
# extra modules main.py now imports (shaping + opponent-deck prediction)
SRC_REWARDS = HERE.parent / "rewards"
SRC_PREDICT = HERE.parent / "deck_predict.py"
SRC_ARCHETYPES = HERE.parent / "data" / "archetypes.json"


def _load_deck(deck) -> list[int]:
    """Accept a list of ints or a path to a CSV (one-per-line or comma-separated)."""
    if isinstance(deck, (list, tuple)):
        ids = [int(x) for x in deck]
    else:
        text = Path(deck).read_text().replace(",", "\n")
        ids = [int(tok) for tok in text.split() if tok.strip()]
    if len(ids) != 60:
        raise ValueError(f"Deck must have exactly 60 cards, got {len(ids)}.")
    return ids


def build_submission(
    deck,
    out_dir: str | Path = HERE / "dist" / "submission",
    weights: str | Path | None = None,
    zip_it: bool = False,
) -> Path:
    """Assemble main.py + deck.csv + cg/ (+ optional model.pth) into out_dir.

    Returns the submission directory (or the .zip path if zip_it=True).
    """
    if not SRC_MAIN.exists():
        raise FileNotFoundError(f"Missing agent source: {SRC_MAIN}")
    if not SRC_CG.is_dir():
        raise FileNotFoundError(f"Missing cg/ library: {SRC_CG}")
    for src in (SRC_REWARDS, SRC_PREDICT, SRC_ARCHETYPES):
        if not src.exists():
            raise FileNotFoundError(f"Missing required asset: {src}")

    ids = _load_deck(deck)
    out_dir = Path(out_dir)

    # fresh build dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 1) main.py + bundled modules/assets it imports
    shutil.copy2(SRC_MAIN, out_dir / "main.py")
    shutil.copytree(SRC_REWARDS, out_dir / "rewards", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(SRC_PREDICT, out_dir / "deck_predict.py")
    shutil.copy2(SRC_ARCHETYPES, out_dir / "archetypes.json")

    # 2) deck.csv — one ID per line (main.py / example reader expects 60 lines)
    (out_dir / "deck.csv").write_text("\n".join(str(c) for c in ids) + "\n")

    # 3) cg/ untouched (skip caches so the copy stays clean)
    shutil.copytree(SRC_CG, out_dir / "cg", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # 4) optional weights
    if weights is not None:
        wsrc = Path(weights)
        if not wsrc.exists():
            raise FileNotFoundError(f"Weights file not found: {wsrc}")
        shutil.copy2(wsrc, out_dir / "model.pth")

    print(f"Built submission: {out_dir}")
    print(f"  main.py   ({(out_dir / 'main.py').stat().st_size} bytes)")
    print(f"  rewards/ + deck_predict.py + archetypes.json bundled")
    print(f"  deck.csv  (60 cards)")
    print(f"  cg/       ({sum(1 for _ in (out_dir / 'cg').rglob('*') if _.is_file())} files)")
    print(f"  model.pth {'included' if weights else 'OMITTED (random init)'}")

    if zip_it:
        archive = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
        print(f"Zipped: {archive}")
        return Path(archive)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Bundle a tournament submission.")
    ap.add_argument("deck", help="Path to a deck CSV (60 card IDs).")
    ap.add_argument("--out", default=str(HERE / "dist" / "submission"), help="Output directory.")
    ap.add_argument("--weights", default=None, help="Optional model .pth to embed.")
    ap.add_argument("--zip", action="store_true", help="Also produce a .zip.")
    args = ap.parse_args()
    build_submission(args.deck, out_dir=args.out, weights=args.weights, zip_it=args.zip)


if __name__ == "__main__":
    main()
