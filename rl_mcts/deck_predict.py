"""
Deck archetype prediction from tournament decklists.

Usage:
    # First run — cluster and save
    model = build_model("data/decks/decklists", n_clusters=10)
    model.describe_all_archetypes()           # inspect to label
    model.set_archetype_name(0, "Alakazam ex")
    model.save("data/archetypes.json")

    # Later — load without re-reading decklists
    model = load_model("data/archetypes.json")
    result = model.predict([190, 190, 169, 169])
"""

from collections import Counter
from dataclasses import dataclass
import glob
import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

DECK_SIZE = 60


@dataclass
class CardPrediction:
    card_id: int
    name: str
    probability: float        # fraction of compatible decks containing this card
    expected_copies: float    # weighted mean copy count


@dataclass
class PredictionResult:
    archetype_probs: dict[str, float]             # archetype_name -> probability
    card_predictions: list[CardPrediction]        # sorted by probability desc


def _load_decks(decklist_dir: str) -> tuple[list[Counter], list[str]]:
    decks, names = [], []
    for path in sorted(glob.glob(os.path.join(decklist_dir, "*.csv"))):
        with open(path) as f:
            content = f.read().strip()
        card_ids = [int(x) for x in content.split(",") if x.strip()]
        if len(card_ids) == DECK_SIZE:
            decks.append(Counter(card_ids))
            names.append(os.path.basename(path))
    return decks, names


def _build_matrix(decks: list[Counter], card_universe: list[int]) -> np.ndarray:
    idx = {cid: i for i, cid in enumerate(card_universe)}
    mat = np.zeros((len(decks), len(card_universe)), dtype=np.float32)
    for i, deck in enumerate(decks):
        for cid, count in deck.items():
            if cid in idx:
                mat[i, idx[cid]] = float(count)
    return mat


class ArchetypeModel:
    def __init__(
        self,
        decks: list[Counter],
        deck_names: list[str],
        card_meta: dict,          # {card_id: CardData}
        n_clusters: int,
    ):
        self.decks = decks
        self.deck_names = deck_names
        self.card_meta = card_meta

        all_ids: set[int] = set()
        for d in decks:
            all_ids.update(d.keys())
        self.card_universe: list[int] = sorted(all_ids)
        self.card_idx: dict[int, int] = {cid: i for i, cid in enumerate(self.card_universe)}

        self.matrix = _build_matrix(decks, self.card_universe)   # (n_decks, n_cards)

        # Cluster on binary presence vectors — captures WHICH cards, not copy counts
        binary = (self.matrix > 0).astype(np.float32)
        normed = normalize(binary, norm="l2")
        km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
        self.cluster_labels: np.ndarray = km.fit_predict(normed)
        self.n_clusters = n_clusters

        # Per-cluster mean count vectors
        self.centroids = np.zeros((n_clusters, len(self.card_universe)), dtype=np.float64)
        for k in range(n_clusters):
            mask = self.cluster_labels == k
            if mask.any():
                self.centroids[k] = self.matrix[mask].mean(axis=0)

        self.archetype_names: dict[int, str] = {k: f"Cluster {k}" for k in range(n_clusters)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model to JSON. Archetype names can be edited by hand."""
        data = {
            "n_clusters": self.n_clusters,
            "archetype_names": {str(k): v for k, v in self.archetype_names.items()},
            "card_universe": self.card_universe,
            "cluster_labels": self.cluster_labels.tolist(),
            "centroids": self.centroids.tolist(),
            "deck_names": self.deck_names,
            "decks": [{str(cid): cnt for cid, cnt in deck.items()} for deck in self.decks],
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def _from_dict(cls, data: dict, card_meta: dict) -> "ArchetypeModel":
        card_universe: list[int] = [int(x) for x in data["card_universe"]]
        decks = [Counter({int(cid): cnt for cid, cnt in d.items()}) for d in data["decks"]]
        deck_names: list[str] = data["deck_names"]
        n_clusters: int = data["n_clusters"]

        obj = object.__new__(cls)
        obj.decks = decks
        obj.deck_names = deck_names
        obj.card_meta = card_meta
        obj.card_universe = card_universe
        obj.card_idx = {cid: i for i, cid in enumerate(card_universe)}
        obj.matrix = _build_matrix(decks, card_universe)
        obj.n_clusters = n_clusters
        obj.cluster_labels = np.array(data["cluster_labels"], dtype=np.int32)
        obj.centroids = np.array(data["centroids"], dtype=np.float64)
        obj.archetype_names = {int(k): v for k, v in data["archetype_names"].items()}
        return obj

    # ------------------------------------------------------------------

    def set_archetype_name(self, cluster_id: int, name: str) -> None:
        self.archetype_names[cluster_id] = name

    def describe_archetype(self, cluster_id: int, top_n: int = 15) -> list[tuple[int, str, float, float]]:
        """Return top cards for a cluster as (card_id, name, mean_copies, deck_freq)."""
        centroid = self.centroids[cluster_id]
        mask = self.cluster_labels == cluster_id
        n_decks_in_cluster = mask.sum()
        freq = (self.matrix[mask] > 0).sum(axis=0) / max(n_decks_in_cluster, 1)
        score = centroid * freq                    # high if many copies AND high freq
        top_indices = np.argsort(score)[::-1][:top_n]
        result = []
        for i in top_indices:
            if centroid[i] > 0:
                cid = self.card_universe[i]
                name = self.card_meta[cid].name if cid in self.card_meta else str(cid)
                result.append((cid, name, float(centroid[i]), float(freq[i])))
        return result

    def describe_all_archetypes(self, top_n: int = 8) -> None:
        """Print a summary of all clusters to help with labeling."""
        for k in range(self.n_clusters):
            mask = self.cluster_labels == k
            cards = self.describe_archetype(k, top_n)
            print(f"\n--- Cluster {k} ({mask.sum()} decks) ---")
            for cid, name, mean_copies, freq in cards:
                print(f"  {name:<40} copies={mean_copies:.1f}  freq={freq:.0%}")

    def predict(self, known_cards: list[int]) -> PredictionResult:
        """
        Given a partial list of card IDs (repeat entries = copies),
        return archetype probabilities and card predictions.

        known_cards: e.g. [190, 190, 169] means 2x card 190, 1x card 169.
        """
        known_counts = Counter(known_cards)
        weights = self._score_decks(known_counts)

        # Archetype probabilities
        archetype_probs: dict[str, float] = {}
        for k in range(self.n_clusters):
            mask = self.cluster_labels == k
            archetype_probs[self.archetype_names[k]] = float(weights[mask].sum())

        # Card predictions for cards NOT already known
        preds: list[CardPrediction] = []
        for cid in self.card_universe:
            if cid in known_counts:
                continue
            col_idx = self.card_idx[cid]
            col = self.matrix[:, col_idx].astype(np.float64)
            prob = float((col > 0).astype(np.float64) @ weights)
            expected = float(col @ weights)
            if prob < 0.01:
                continue
            name = self.card_meta[cid].name if cid in self.card_meta else str(cid)
            preds.append(CardPrediction(cid, name, prob, expected))

        preds.sort(key=lambda c: c.probability, reverse=True)
        return PredictionResult(archetype_probs, preds)

    def _score_decks(self, known_counts: Counter) -> np.ndarray:
        """
        Weight each deck by compatibility with known_counts.
        weight_i = prod_{c,k in known_counts} min(deck_i[c], k) / k
        Falls back to uniform if no deck matches at all.
        """
        weights = np.ones(len(self.decks), dtype=np.float64)
        for cid, required in known_counts.items():
            if cid not in self.card_idx:
                weights *= 0.0
                break
            col = self.matrix[:, self.card_idx[cid]].astype(np.float64)
            weights *= np.minimum(col, required) / required

        total = weights.sum()
        if total == 0.0:
            # No deck fully matches — fall back to soft partial matching
            weights = np.ones(len(self.decks), dtype=np.float64)
            for cid, required in known_counts.items():
                if cid not in self.card_idx:
                    continue
                col = self.matrix[:, self.card_idx[cid]].astype(np.float64)
                weights *= (col + 0.5) / (required + 0.5)
            total = weights.sum()

        if total > 0:
            weights /= total
        else:
            weights[:] = 1.0 / len(self.decks)
        return weights


def _fetch_card_meta() -> dict:
    try:
        from cg.api import all_card_data
        return {c.cardId: c for c in all_card_data()}
    except Exception:
        return {}


def build_model(
    decklist_dir: str = "data/decks/decklists",
    n_clusters: int = 10,
) -> ArchetypeModel:
    """Cluster decklists into archetypes. Save result with model.save()."""
    decks, deck_names = _load_decks(decklist_dir)
    if not decks:
        raise ValueError(f"No valid 60-card decklists found in {decklist_dir}")
    return ArchetypeModel(decks, deck_names, _fetch_card_meta(), n_clusters)


def load_model(path: str) -> ArchetypeModel:
    """Load a previously saved model. No decklists re-read, no re-clustering."""
    with open(path) as f:
        data = json.load(f)
    return ArchetypeModel._from_dict(data, _fetch_card_meta())
