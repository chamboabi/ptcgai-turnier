"""Sweep KMeans cluster counts and render one shared-layout t-SNE comparison.

Builds an archetype model for every k in the range, saves each as
../archetypes_k{k}.json (auto-labeled by signature card), and draws a grid
where the t-SNE layout is identical across panels so the only thing that
changes is the coloring — making over/under-clustering obvious.

Bundled alongside the deck data so it travels with it.

Usage (from anywhere):
    python data/decks/archetype_k_sweep.py
    python data/decks/archetype_k_sweep.py --kmin 4 --kmax 27
    python data/decks/archetype_k_sweep.py --out PATH --no-save-json
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent                       # rl_mcts/
DECK_DIR = str(HERE / "decklists")
DATA_DIR = HERE.parent                               # data/

sys.path.insert(0, str(REPO_ROOT))
import deck_predict as dp

# 60-color palette so even large k has distinct colors.
PALETTE = (list(plt.get_cmap("tab20").colors)
           + list(plt.get_cmap("tab20b").colors)
           + list(plt.get_cmap("tab20c").colors))

# Generic cards that don't name an archetype — skip when auto-labeling.
STAPLE_HINTS = ("Energy", "Poké Pad", "Lillie", "Ultra Ball", "Buddy-Buddy",
                "Pokégear", "Boss", "Night Stretcher", "Dusk Ball", "Nest Ball",
                "Earthen Vessel", "Professor", "Iono", "Arven", "Counter Catcher")


def cluster_ellipse(pts, color, n_std=1.5):
    """Covariance confidence ellipse enclosing a cluster's points."""
    if len(pts) < 3:
        return None
    cov = np.cov(pts, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    return Ellipse(pts.mean(axis=0), w, h, angle=angle,
                   facecolor=color, edgecolor=color, alpha=0.12, lw=1.0, ls="--")


def signature(model, k):
    """Best archetype name for cluster k: top defining non-staple card."""
    for cid, name, _c, _f in model.describe_archetype(k, top_n=12):
        if not any(h in name for h in STAPLE_HINTS):
            return name
    cards = model.describe_archetype(k, top_n=1)
    return cards[0][1] if cards else f"Cluster {k}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmin", type=int, default=4)
    ap.add_argument("--kmax", type=int, default=27)
    ap.add_argument("--out", default=None, help="output png (default derives from range)")
    ap.add_argument("--no-save-json", action="store_true",
                    help="don't write per-k archetypes_k{k}.json files")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    ks = list(range(args.kmin, args.kmax + 1))
    out_png = args.out or str(HERE / f"archetype_k_sweep_{args.kmin}_{args.kmax}.png")

    # Base model just to grab the shared binary matrix + a single t-SNE layout.
    base = dp.build_model(DECK_DIR, n_clusters=ks[0])
    normed = normalize((base.matrix > 0).astype(np.float32), norm="l2")

    print("computing shared t-SNE layout ...")
    perplexity = min(30, max(5, (len(normed) - 1) // 3))
    xy = TSNE(n_components=2, perplexity=perplexity, init="pca",
              random_state=42).fit_transform(normed)

    ncols = 4
    nrows = -(-len(ks) // ncols)            # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
    axes = axes.ravel()

    for ax_i, k in enumerate(ks):
        model = dp.build_model(DECK_DIR, n_clusters=k)
        labels = model.cluster_labels
        names = {c: signature(model, c) for c in range(k)}
        for c in range(k):
            model.set_archetype_name(c, names[c])
        if not args.no_save_json:
            model.save(str(DATA_DIR / f"archetypes_k{k}.json"))
        print(f"k={k}: " +
              ", ".join(f"{names[c]}({int((labels == c).sum())})" for c in range(k)))

        ax = axes[ax_i]
        for c in range(k):
            m = labels == c
            color = PALETTE[c % len(PALETTE)]
            ell = cluster_ellipse(xy[m], color)
            if ell is not None:
                ax.add_patch(ell)
            ax.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.75, color=color,
                       edgecolors="white", linewidths=0.2)
            cx, cy = xy[m, 0].mean(), xy[m, 1].mean()
            ax.annotate(names[c], (cx, cy), fontsize=5, fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.65))
        ax.set_title(f"k = {k}", fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    for j in range(len(ks), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Archetype clustering — k = {args.kmin}..{args.kmax} (shared t-SNE layout)",
                 fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print("saved", out_png)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
