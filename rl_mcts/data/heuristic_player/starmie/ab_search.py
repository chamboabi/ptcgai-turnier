"""Determinized rollout search for the submission. Self-contained (stdlib + cg).

At a MAIN decision: guess the opponent's deck (archetype detection from revealed cards
-> embedded prior), then in the engine's native sandbox (search_begin/step) roll each
candidate option to game-end under random determinizations using the pilot as the
rollout policy, and pick the best win rate. Time-bounded: always falls back to the
pilot before the move clock runs out, so it can never time out."""
import ctypes
import json
import random
import time
from collections import Counter

from cg.sim import lib
from ab_opttools import get_context, get_options, obj_get

# Embedded archetype priors (best guess of an opponent's 60-card deck per archetype).
PRIORS = {
    "CRUSTLE": {1: 11, 7: 4, 230: 3, 235: 2, 344: 4, 345: 4, 1086: 4, 1094: 3, 1102: 2,
                1119: 2, 1120: 4, 1121: 4, 1124: 3, 1161: 1, 1182: 3, 1188: 2, 1227: 3, 1247: 1},
    "STARMIE": {3: 9, 17: 4, 666: 4, 1030: 3, 1031: 3, 1086: 4, 1097: 2, 1120: 4, 1121: 1,
                1122: 4, 1145: 4, 1159: 1, 1182: 1, 1189: 4, 1223: 2, 1225: 2, 1227: 4, 1229: 4},
    "LIGHTNING": {4: 22, 265: 3, 268: 3, 269: 3, 270: 3, 271: 3, 1086: 3, 1097: 2, 1110: 1,
                  1118: 1, 1121: 3, 1152: 2, 1227: 4, 1233: 4, 1254: 3},
    "DRAGAPULT": {119: 4, 120: 4, 121: 3, 140: 1, 184: 1, 235: 2, 1071: 1, 1079: 2, 1080: 1,
                  1086: 4, 1097: 2, 1120: 4, 1121: 4, 1152: 3, 1156: 1, 1182: 3, 1198: 4,
                  1210: 2, 1227: 4, 1256: 2, 2: 4, 5: 4},
    "LUCARIO": {673: 2, 674: 2, 675: 2, 676: 3, 677: 3, 678: 4, 1102: 4, 1123: 2, 1141: 4,
                1142: 4, 1152: 4, 1159: 1, 1182: 2, 1192: 4, 1227: 4, 1252: 2, 6: 13},
    "ABOMASNOW": {723: 4, 722: 2, 721: 2, 1206: 3, 1231: 2, 1219: 2, 1182: 2, 1121: 4, 3: 39},
    # Alakazam ex "Powerful Hand" (20 dmg x hand size). Real ladder build
    # (himihimi_alakazam.csv): Abra/Kadabra/Alakazam line + heavy draw to hoard a
    # big hand. Psychic energy (5). See [[alakazam-counter]].
    "ALAKAZAM": {5: 13, 741: 4, 742: 2, 743: 2, 751: 2, 971: 3, 1079: 3, 1086: 4,
                 1121: 4, 1124: 3, 1182: 4, 1192: 4, 1213: 2, 1224: 4, 1227: 2, 1236: 4},
}
# Signature card -> archetype (first match wins).
SIGNATURE = {345: "CRUSTLE", 344: "CRUSTLE", 1031: "STARMIE", 1030: "STARMIE",
             269: "LIGHTNING", 271: "LIGHTNING", 268: "LIGHTNING",
             119: "DRAGAPULT", 120: "DRAGAPULT", 678: "LUCARIO", 723: "ABOMASNOW",
             743: "ALAKAZAM", 742: "ALAKAZAM", 741: "ALAKAZAM"}
DEFAULT_ARCHETYPE = "CRUSTLE"  # most common + where search helps most

_AGENT_PTR = None


def _ptr():
    global _AGENT_PTR
    if _AGENT_PTR is None:
        _AGENT_PTR = lib.AgentStart()
    return _AGENT_PTR


def _arr(xs):
    return (ctypes.c_int * len(xs))(*xs)


def _sbegin(obs, kw):
    sbi = obs["search_begin_input"]
    bs = lib.SearchBegin(_ptr(), sbi.encode("ascii"), len(sbi),
                         _arr(kw["your_deck"]), _arr(kw["your_prize"]), _arr(kw["opponent_deck"]),
                         _arr(kw["opponent_prize"]), _arr(kw["opponent_hand"]), _arr(kw["opponent_active"]), 0)
    return json.loads(bs.decode())


def _sstep(sid, select):
    bs = lib.SearchStep(_ptr(), sid, _arr(select), len(select))
    return json.loads(bs.decode())


def detect_archetype(observed):
    """Return a matched archetype, or None if no signature card has been seen yet."""
    for cid in observed:
        if cid in SIGNATURE:
            return SIGNATURE[cid]
    return None


def guess_opp_deck(observed):
    """A 60-card decklist guess that is consistent with every card we've seen."""
    base = dict(PRIORS[detect_archetype(observed) or DEFAULT_ARCHETYPE])
    for cid, c in observed.items():          # never guess fewer than we've already seen
        if base.get(cid, 0) < c:
            base[cid] = c
    total = sum(base.values())
    while total > 60:                        # trim filler we have NOT observed
        for cid in sorted(base, key=lambda k: base[k], reverse=True):
            if base[cid] > observed.get(cid, 0):
                base[cid] -= 1; total -= 1
                break
        else:
            break
    if total < 60:                           # pad with Water energy
        base[3] = base.get(3, 0) + (60 - total)
    return [cid for cid, c in base.items() for _ in range(c)]


def _visible_ids(p, is_me):
    ids = []
    if is_me:
        for c in (p.get("hand") or []):
            if c: ids.append(c["id"])
    for c in (p.get("discard") or []):
        ids.append(c["id"])
    for area in ("active", "bench"):
        for pk in (p.get(area) or []):
            if not pk:
                continue
            ids.append(pk["id"])
            for grp in ("energyCards", "tools", "preEvolution"):
                for c in (pk.get(grp) or []):
                    ids.append(c["id"])
    for c in (p.get("prize") or []):
        if c: ids.append(c["id"])
    return ids


def _needed(obs):
    cur = obs["current"]; me = cur["yourIndex"]; opp = 1 - me
    mp, op = cur["players"][me], cur["players"][opp]
    deck_known = (obs.get("select") or {}).get("deck") is not None
    return mp, op, {
        "your_deck": 0 if deck_known else mp["deckCount"],
        "your_prize": len(mp.get("prize") or []),
        "opponent_deck": op["deckCount"],
        "opponent_prize": len(op.get("prize") or []),
        "opponent_hand": op["handCount"],
        "opponent_active": 1 if (op.get("active") and op["active"][0] is None) else 0,
    }


def _determinize(obs, my_deck, opp_deck, rng):
    mp, op, need = _needed(obs)
    pm = list((Counter(my_deck) - Counter(_visible_ids(mp, True))).elements()); rng.shuffle(pm)
    po = list((Counter(opp_deck) - Counter(_visible_ids(op, False))).elements()); rng.shuffle(po)
    dc, npz_m = need["your_deck"], need["your_prize"]
    odc, ohc, npz_o, oa = (need["opponent_deck"], need["opponent_hand"],
                           need["opponent_prize"], need["opponent_active"])
    kw = dict(your_deck=pm[:dc], your_prize=pm[dc:dc + npz_m],
              opponent_deck=po[:odc], opponent_hand=po[odc:odc + ohc],
              opponent_prize=po[odc + ohc:odc + ohc + npz_o],
              opponent_active=po[odc + ohc + npz_o:odc + ohc + npz_o + oa])
    if any(len(kw[k]) != need[k] for k in need):   # never feed the C lib bad lengths
        return None
    return kw


def _rollout(state, my_index, my_pilot, opp_pilot, max_steps=220):
    sid = state["searchId"]; ob = state["observation"]
    for _ in range(max_steps):
        cur = ob.get("current") or {}
        res = cur.get("result", -1)
        if res in (0, 1, 2):
            return res
        who = cur.get("yourIndex", 0)
        choice = (my_pilot if who == my_index else opp_pilot)(ob)
        st = _sstep(sid, choice).get("state")
        if st is None:
            return cur.get("result", -1)
        sid = st["searchId"]; ob = st["observation"]
    return -1


def search_choice(obs, my_deck, observed, make_pilot, my_pilot, deadline, rng, max_cand=6):
    """Return the chosen option (list[int]) via time-bounded determinized rollouts, or
    None to fall back to the pilot (no time / unsafe determinization / no signal)."""
    if (get_context(obs) or "").upper() != "MAIN" or not obs.get("search_begin_input"):
        return None
    options = get_options(obs)
    if len(options) <= 1:
        return None
    if (obs.get("current") or {}).get("turn", 0) < 3:
        return None
    if _determinize(obs, my_deck, guess_opp_deck(observed), rng) is None:
        return None  # count-mismatch guard (still skip rather than feed the C lib bad lengths)
    opp_deck = guess_opp_deck(observed)
    my_index = obs["current"]["yourIndex"]
    base = my_pilot(obs)
    base_idx = base[0] if isinstance(base, list) and len(base) == 1 else 0
    cand = [base_idx] + [i for i in range(len(options)) if i != base_idx]
    cand = cand[:max_cand]
    opp_pilot = make_pilot(opp_deck)
    wins = {c: 0 for c in cand}; n = {c: 0 for c in cand}
    while time.perf_counter() < deadline:
        kw = _determinize(obs, my_deck, opp_deck, rng)
        if kw is None:
            break
        root = _sbegin(obs, kw).get("state")
        if root is None:
            break
        for c in cand:
            if time.perf_counter() >= deadline:
                break
            child = _sstep(root["searchId"], [c]).get("state")
            if child is None:
                continue
            res = _rollout(child, my_index, my_pilot, opp_pilot)
            n[c] += 1
            if res == my_index:
                wins[c] += 1
        lib.SearchEnd(_ptr())
    if not any(n.values()):
        return None
    best = max(cand, key=lambda c: (wins[c] / n[c] if n[c] else -1, c == base_idx))
    return [best]
