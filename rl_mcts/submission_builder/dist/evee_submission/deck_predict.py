"""Slim, load/predict-only opponent-deck predictor for the bundled submission.

Mirrors deck_predict.ArchetypeModel.predict() but drops all clustering/training
code AND the numpy dependency — the submission ships with torch only, so this is
pure Python. The data is tiny (a few hundred decks), so plain loops are fast
enough to run once per turn.

Load a model saved by deck_predict.ArchetypeModel.save() and call predict() with
the opponent's known card IDs to get per-card P(in deck) and E[copies].
"""

from collections import Counter
from dataclasses import dataclass
import json


@dataclass
class CardPrediction:
    card_id: int
    probability: float        # fraction of compatible decks containing this card
    expected_copies: float    # weighted mean copy count


@dataclass
class PredictionResult:
    card_predictions: list[CardPrediction]   # sorted by probability desc


class ArchetypeModel:
    """Predict-only view over saved archetype data (no clustering, no numpy)."""

    def __init__(self, card_universe: list[int], decks: list[Counter]):
        self.card_universe = card_universe
        self.universe_set = set(card_universe)
        self.decks = decks

    def _score_decks(self, known_counts: Counter) -> list[float]:
        """Weight each deck by compatibility with known_counts (see deck_predict)."""
        n = len(self.decks)
        weights = [1.0] * n

        def _hard() -> float:
            total = 0.0
            for cid, required in known_counts.items():
                if cid not in self.universe_set:
                    return 0.0  # a card no deck has -> nothing matches
                for i, deck in enumerate(self.decks):
                    weights[i] *= min(deck.get(cid, 0), required) / required
            for w in weights:
                total += w
            return total

        total = _hard()
        if total == 0.0:
            # No deck fully matches — fall back to soft partial matching
            weights = [1.0] * n
            total = 0.0
            for cid, required in known_counts.items():
                if cid not in self.universe_set:
                    continue
                for i, deck in enumerate(self.decks):
                    weights[i] *= (deck.get(cid, 0) + 0.5) / (required + 0.5)
            total = sum(weights)

        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / n] * n
        return weights

    def predict(self, known_cards: list[int]) -> PredictionResult:
        """Given partial card IDs (repeats = copies), predict the rest of the deck."""
        known_counts = Counter(known_cards)
        weights = self._score_decks(known_counts)

        preds: list[CardPrediction] = []
        for cid in self.card_universe:
            if cid in known_counts:
                continue
            prob = 0.0
            expected = 0.0
            for i, deck in enumerate(self.decks):
                cnt = deck.get(cid, 0)
                if cnt > 0:
                    prob += weights[i]
                    expected += weights[i] * cnt
            if prob < 0.01:
                continue
            preds.append(CardPrediction(cid, prob, expected))

        preds.sort(key=lambda c: c.probability, reverse=True)
        return PredictionResult(preds)


def load_model(path: str) -> ArchetypeModel:
    """Load a model saved by deck_predict.ArchetypeModel.save() — predict only."""
    with open(path) as f:
        data = json.load(f)
    card_universe = [int(x) for x in data["card_universe"]]
    decks = [Counter({int(cid): cnt for cid, cnt in d.items()}) for d in data["decks"]]
    return ArchetypeModel(card_universe, decks)
