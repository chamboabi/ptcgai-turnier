"""
Interactive debug tool for deck_predict.py.

Run from the rl_mcts directory:
    python debug/deck_predict_debug.py

Commands:
    predict <card_id> [card_id ...]   - predict archetype + missing cards
    describe [cluster_id]             - describe one or all clusters
    build [n_clusters]                - rebuild model from decklists and save
    load                              - reload model from data/archetypes.json
    cards <query>                     - search card names
    label <cluster_id> <name>         - rename a cluster and save
    quit
"""

import sys
import os

# Allow running from debug/ subdir as well
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deck_predict import build_model, load_model

MODEL_PATH = "data/archetypes.json"
DECKLIST_DIR = "data/decks/decklists"


def fmt_prob(p: float) -> str:
    return f"{p * 100:5.1f}%"


def print_archetype_probs(probs: dict[str, float], top_n: int = 5) -> None:
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print("\nArchetype probabilities:")
    for name, prob in sorted_items:
        bar = "#" * int(prob * 30)
        print(f"  {name:<30} {fmt_prob(prob)}  {bar}")


def print_card_predictions(preds, top_n: int = 20) -> None:
    print(f"\nTop {top_n} predicted cards (not yet seen):")
    print(f"  {'ID':>6}  {'Name':<40} {'Prob':>6}  {'Exp copies':>10}")
    print("  " + "-" * 68)
    for cp in preds[:top_n]:
        print(f"  {cp.card_id:>6}  {cp.name:<40} {fmt_prob(cp.probability)}  {cp.expected_copies:>8.2f}x")


def search_cards(model, query: str) -> None:
    query_lower = query.lower()
    matches = [
        (cid, cd)
        for cid, cd in model.card_meta.items()
        if query_lower in cd.name.lower()
    ]
    if not matches:
        print(f"  No cards matching '{query}'")
        return
    print(f"  Matches for '{query}':")
    for cid, cd in sorted(matches, key=lambda x: x[1].name)[:20]:
        print(f"    {cid:>6}  {cd.name}")


def run(model):
    print(f"\nModel loaded: {model.n_clusters} clusters, {len(model.decks)} decks, {len(model.card_universe)} unique cards")
    print("Type 'help' for commands.\n")

    while True:
        try:
            line = input("deck_predict> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "exit", "q"):
            break

        elif cmd == "help":
            print(__doc__)

        elif cmd == "predict":
            if len(parts) < 2:
                print("  Usage: predict <card_id> [card_id ...]")
                continue
            try:
                card_ids = [int(x) for x in parts[1:]]
            except ValueError:
                print("  Card IDs must be integers.")
                continue
            result = model.predict(card_ids)
            known_names = []
            for cid in set(card_ids):
                name = model.card_meta[cid].name if cid in model.card_meta else str(cid)
                known_names.append(f"{cid} ({name})")
            print(f"\nKnown cards: {', '.join(known_names)}")
            print_archetype_probs(result.archetype_probs)
            print_card_predictions(result.card_predictions)

        elif cmd == "describe":
            if len(parts) >= 2:
                try:
                    cid = int(parts[1])
                    cards = model.describe_archetype(cid, top_n=20)
                    mask = model.cluster_labels == cid
                    print(f"\n--- {model.archetype_names[cid]} ({mask.sum()} decks) ---")
                    print(f"  {'ID':>6}  {'Name':<40} {'Copies':>6}  {'Freq':>6}")
                    print("  " + "-" * 64)
                    for card_id, name, copies, freq in cards:
                        print(f"  {card_id:>6}  {name:<40} {copies:>5.1f}  {fmt_prob(freq)}")
                except (ValueError, KeyError):
                    print(f"  Invalid cluster id: {parts[1]}")
            else:
                model.describe_all_archetypes(top_n=8)

        elif cmd == "build":
            n_clusters = int(parts[1]) if len(parts) >= 2 else 10
            print(f"Building model with {n_clusters} clusters from {DECKLIST_DIR} ...")
            model = build_model(DECKLIST_DIR, n_clusters=n_clusters)
            model.save(MODEL_PATH)
            print(f"Saved to {MODEL_PATH}")
            model.describe_all_archetypes(top_n=5)

        elif cmd == "load":
            print(f"Loading {MODEL_PATH} ...")
            model = load_model(MODEL_PATH)
            print(f"Loaded: {model.n_clusters} clusters, {len(model.decks)} decks")

        elif cmd == "cards":
            if len(parts) < 2:
                print("  Usage: cards <query>")
                continue
            search_cards(model, " ".join(parts[1:]))

        elif cmd == "label":
            if len(parts) < 3:
                print("  Usage: label <cluster_id> <name>")
                continue
            try:
                cid = int(parts[1])
                name = " ".join(parts[2:])
                model.set_archetype_name(cid, name)
                model.save(MODEL_PATH)
                print(f"  Cluster {cid} -> '{name}', saved.")
            except (ValueError, KeyError):
                print(f"  Invalid cluster id: {parts[1]}")

        else:
            print(f"  Unknown command: {cmd!r}. Type 'help'.")

    return model


if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH} ...")
        model = load_model(MODEL_PATH)
    else:
        print(f"No model at {MODEL_PATH}. Building from {DECKLIST_DIR} ...")
        model = build_model(DECKLIST_DIR, n_clusters=10)
        model.save(MODEL_PATH)
        print(f"Saved to {MODEL_PATH}")

    run(model)
