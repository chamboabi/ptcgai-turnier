"""Kaggle entry point: Alakazam (Powerful Hand) deck piloted by determinized rollout search.

On the initial observation, returns the deck. Otherwise: at a MAIN decision it runs a
time-bounded determinized search (the rule-based Alakazam agent as the rollout policy for
us, the generic KO-aware heuristic for the guessed opponent); everywhere else, and
whenever search has no time/signal, it falls back to the rule-based Alakazam agent.
Never raises; always returns a legal action."""
import csv
import os
import random
import sys
import time

# The Kaggle grader execs main.py with NO __file__ defined, so derive the agent dir
# safely and put it (plus the standard grader path) on sys.path BEFORE importing the
# bundled helper modules.
try:
    _DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _DIR = os.getcwd()
for _p in (_DIR, "/kaggle_simulations/agent", os.getcwd()):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from ab_opttools import legalize_choice, safe_choice  # noqa: E402
from alakazam_pilot import agent as _alakazam_agent  # noqa: E402  (flat agent(obs)->choice)
from heuristic_agent import make_agent as make_heuristic  # noqa: E402
import ab_search  # noqa: E402

BUDGET_S = 2.0             # per-decision search budget (runTimeout 2000s/episode,
                           # actTimeout=0 -> ~10x+ headroom even in long games)
EMERGENCY_DECK = [3] * 56 + [1030] * 4   # last resort: legal-ish (Staryu = a Basic)


def _read_deck():
    for path in (os.path.join(_DIR, "deck.csv"), "deck.csv",
                 "/kaggle_simulations/agent/deck.csv", os.path.join(os.getcwd(), "deck.csv")):
        try:
            if not os.path.exists(path):
                continue
            deck = []
            with open(path, "r", newline="", encoding="utf-8-sig") as fh:
                for row in csv.reader(fh):
                    if row and row[0].strip():
                        deck.append(int(row[0].strip()))
            if len(deck) == 60:
                return deck
        except Exception:
            continue
    return list(EMERGENCY_DECK)


_DECK = _read_deck()
_LOOKUPS = {}
_STATE = {}


def _lookups():
    if "card" not in _LOOKUPS:
        try:
            from cg.api import all_attack, all_card_data
            _LOOKUPS["card"] = {c.cardId: c for c in all_card_data()}
            _LOOKUPS["attack"] = {a.attackId: a for a in all_attack()}
        except Exception:
            _LOOKUPS["card"] = None
            _LOOKUPS["attack"] = None
    return _LOOKUPS["attack"], _LOOKUPS["card"]


def _make_our_pilot(deck):
    # The rule-based Alakazam agent is a flat callable agent(obs)->choice; it loads its
    # own deck/lookups at import time, so just hand it back as the pilot.
    return _alakazam_agent


def _make_opp_pilot(deck):
    al, cl = _lookups()
    return make_heuristic(deck, al, cl)


def _reset():
    from collections import Counter
    _STATE["pilot"] = _make_our_pilot(_DECK)
    _STATE["observed"] = Counter()
    _STATE["rng"] = random.Random(0xABBA)


def _update_observed(obs):
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    me = cur.get("yourIndex", 0)
    if (1 - me) < len(players):
        from collections import Counter
        for cid, c in Counter(ab_search._visible_ids(players[1 - me], False)).items():
            if _STATE["observed"][cid] < c:
                _STATE["observed"][cid] = c


def agent(obs):
    if not isinstance(obs, dict) or obs.get("select") is None:
        _reset()
        return list(_DECK)
    if "pilot" not in _STATE:
        _reset()
    pilot = _STATE["pilot"]
    try:
        _update_observed(obs)
        deadline = time.perf_counter() + BUDGET_S
        choice = ab_search.search_choice(obs, _DECK, _STATE["observed"], _make_opp_pilot,
                                         pilot, deadline, _STATE["rng"])
        if choice is not None:
            return legalize_choice(obs, choice)
        return legalize_choice(obs, pilot(obs))
    except Exception:
        try:
            return legalize_choice(obs, pilot(obs))
        except Exception:
            return safe_choice(obs)
