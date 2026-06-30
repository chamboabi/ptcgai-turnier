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
from matplotlib.patches import Patch

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

    # Order columns by how many archetypes use the card (×N), descending.
    order = sorted(range(len(col_ids)), key=lambda c: (-shared[c], c))
    col_names = [col_names[c] for c in order]
    raw = raw[:, order]
    norm = norm[:, order]
    shared = shared[order]

    # Order rows by deck usage: most-played archetype on top.
    row_order = sorted(range(len(row_names)), key=lambda r: counts[r], reverse=True)
    row_names = [row_names[r] for r in row_order]
    counts = [counts[r] for r in row_order]
    raw = raw[row_order, :]
    norm = norm[row_order, :]
    return row_names, counts, col_names, raw, norm, shared


SIGNATURE_COLOR = "#1a1a1a"   # one archetype
SHARED_COLOR = "#e67e22"      # shared by a few
STAPLE_COLOR = "#c0392b"      # most archetypes use it


def plot(row_names, counts, col_names, raw, norm, shared, out):
    n_rows, n_cols = raw.shape
    half = (n_rows + 1) // 2
    # Label color by how many archetypes share the card.
    def lab_color(s):
        if s >= half:
            return STAPLE_COLOR
        if s >= 2:
            return SHARED_COLOR
        return SIGNATURE_COLOR

    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.55), max(5, n_rows * 0.55)))
    im = ax.imshow(norm, aspect="auto", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("per-card relative usage", fontsize=8)

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
                    color="white" if norm[r, c] >= 0.6 else "black",
                )

    ax.set_title(
        "Archetype card composition — mean copies (cell color = per-card relative)",
        fontsize=10,
    )

    legend_handles = [
        Patch(facecolor=SIGNATURE_COLOR, label="signature — 1 archetype"),
        Patch(facecolor=SHARED_COLOR, label="shared by a few archetypes"),
        Patch(facecolor=STAPLE_COLOR, label="staple — most archetypes (bold; ×N = # using it)"),
    ]
    ax.legend(
        handles=legend_handles,
        title="card label color",
        loc="upper left",
        bbox_to_anchor=(1.06, 1.0),
        fontsize=8,
        title_fontsize=8,
        frameon=True,
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
