"""
2D scatter of decklist archetypes.

Reads a saved archetype model (../archetypes.json) and projects each deck's
card-presence vector to 2D, colored by its KMeans cluster (archetype).
Tight blobs = clean archetypes; smears = fuzzy ones.

Bundled alongside the deck data so it travels with it.

Usage (from anywhere):
    python data/decks/archetype_scatter.py
    python data/decks/archetype_scatter.py --method pca
    python data/decks/archetype_scatter.py --model PATH --out PATH
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE.parent / "archetypes.json"          # data/archetypes.json
DEFAULT_OUT = HERE / "archetype_scatter.png"


def load(path: str):
    with open(path) as f:
        data = json.load(f)
    card_universe = [int(x) for x in data["card_universe"]]
    idx = {cid: i for i, cid in enumerate(card_universe)}
    decks = data["decks"]                       # list of {card_id: count}
    labels = np.array(data["cluster_labels"], dtype=int)
    names = {int(k): v for k, v in data["archetype_names"].items()}

    # Binary presence matrix — same space the clustering used.
    mat = np.zeros((len(decks), len(card_universe)), dtype=np.float32)
    for i, deck in enumerate(decks):
        for cid, cnt in deck.items():
            j = idx.get(int(cid))
            if j is not None and cnt > 0:
                mat[i, j] = 1.0
    return normalize(mat, norm="l2"), labels, names


def embed(mat: np.ndarray, method: str) -> np.ndarray:
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(mat)
    from sklearn.manifold import TSNE
    perplexity = min(30, max(5, (len(mat) - 1) // 3))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=42,
    ).fit_transform(mat)


def plot(xy: np.ndarray, labels: np.ndarray, names: dict, method: str, out: str):
    clusters = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab20" if len(clusters) > 10 else "tab10")

    fig, ax = plt.subplots(figsize=(12, 9))
    for n, k in enumerate(clusters):
        mask = labels == k
        label = f"{names.get(k, f'Cluster {k}')} ({int(mask.sum())})"
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            s=40, alpha=0.75, color=cmap(n % cmap.N),
            edgecolors="white", linewidths=0.4, label=label,
        )
        # Archetype name at cluster centroid.
        cx, cy = xy[mask, 0].mean(), xy[mask, 1].mean()
        ax.annotate(
            names.get(k, f"Cluster {k}"), (cx, cy),
            fontsize=9, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )

    ax.set_title(f"Deck archetypes — {method.upper()} ({len(labels)} decks)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--method", choices=["tsne", "pca"], default="tsne")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    mat, labels, names = load(args.model)
    xy = embed(mat, args.method)
    plot(xy, labels, names, args.method, args.out)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
