"""
Archetype x signature-card heatmap.

Rows = archetypes, columns = the cards that most define each archetype
(union of each cluster's top defining cards). Cell color = relative usage
(mean copies normalized per card column, so a 4-of stands out regardless of
energy stacks); the printed number is the raw mean copy count.

Reads ../archetypes.json via deck_predict so it gets real card names.
Bundled alongside the deck data so it travels with it.

Usage (from anywhere):
    python data/decks/archetype_heatmap.py
    python data/decks/archetype_heatmap.py --top 6 --out PATH
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent                       # rl_mcts/
DEFAULT_MODEL = HERE.parent / "archetypes.json"      # data/archetypes.json
DEFAULT_OUT = HERE / "archetype_heatmap.png"

sys.path.insert(0, str(REPO_ROOT))                   # find deck_predict
import deck_predict as dp


def build(model, top_per_cluster: int):
    clusters = list(range(model.n_clusters))

    # Union of each cluster's top defining cards, in first-seen order.
    col_ids: list[int] = []
    for k in clusters:
        for cid, _name, _copies, _freq in model.describe_archetype(k, top_per_cluster):
            if cid not in col_ids:
                col_ids.append(cid)

    col_names = [
        model.card_meta[c].name if c in model.card_meta else str(c)
        for c in col_ids
    ]
    row_names = [model.archetype_names[k] for k in clusters]
    counts = [int((model.cluster_labels == k).sum()) for k in clusters]

    # mean copies + deck-frequency per (archetype, card)
    cidx = {c: i for i, c in enumerate(model.card_universe)}
    raw = np.zeros((len(clusters), len(col_ids)))
    freq = np.zeros((len(clusters), len(col_ids)))
    for r, k in enumerate(clusters):
        deck_mask = model.cluster_labels == k
        present = (model.matrix[deck_mask] > 0)            # (decks_in_k, all_cards)
        n_k = max(int(deck_mask.sum()), 1)
        for c, cid in enumerate(col_ids):
            j = cidx[cid]
            raw[r, c] = model.centroids[k][j]
            freq[r, c] = present[:, j].sum() / n_k

    # A card "belongs" to an archetype if it shows up in >=50% of its decks.
    # shared = how many archetypes use it -> 1 means signature, many means staple.
    shared = (freq >= 0.5).sum(axis=0)

    # Normalize per column so big energy stacks don't wash out singles.
    col_max = raw.max(axis=0, keepdims=True)
    col_max[col_max == 0] = 1.0
    norm = raw / col_max
    return row_names, counts, col_names, raw, norm, shared


def plot(row_names, counts, col_names, raw, norm, shared, out):
    n_rows, n_cols = raw.shape
    half = (n_rows + 1) // 2
    # Label color by how many archetypes share the card.
    def lab_color(s):
        if s >= half:
            return "#c0392b"        # staple — most archetypes use it
        if s >= 2:
            return "#e67e22"        # shared by a few
        return "#1a1a1a"            # signature — one archetype

    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.55), max(5, n_rows * 0.55)))
    ax.imshow(norm, aspect="auto", cmap="viridis")

    ax.set_xticks(range(n_cols))
    xlabels = [f"{nm}  ×{s}" if s >= 2 else nm for nm, s in zip(col_names, shared)]
    ax.set_xticklabels(xlabels, rotation=60, ha="right", fontsize=7)
    for tick, s in zip(ax.get_xticklabels(), shared):
        tick.set_color(lab_color(s))
        if s >= half:
            tick.set_fontweight("bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"{n}  (n={c})" for n, c in zip(row_names, counts)], fontsize=8)

    # Box the cells where a card is actually used (freq>=50%) so shared columns
    # are visible as a vertical run of boxes spanning several archetypes.
    for r in range(n_rows):
        for c in range(n_cols):
            v = raw[r, c]
            if v >= 0.05:
                ax.text(
                    c, r, f"{v:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if norm[r, c] < 0.6 else "black",
                )

    ax.set_title(
        "Archetype card composition — mean copies (cell color = per-card relative)\n"
        "card label: black = signature (1 archetype),  orange = shared by few,  "
        "bold red = staple (most archetypes); ×N = # archetypes using it",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--top", type=int, default=5, help="top defining cards per cluster")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    model = dp.load_model(args.model)
    row_names, counts, col_names, raw, norm, shared = build(model, args.top)
    plot(row_names, counts, col_names, raw, norm, shared, args.out)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
