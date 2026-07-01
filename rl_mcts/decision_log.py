"""Dump one MCTS decision (per Evee move) to a folder as JSON + readable text.

`mcts_agent` fills a `debug_out` dict when passed one; the functions here decode
that raw data (Option dataclasses, card ids, search stats) into human-readable
form and write two files per decision:

    turn{NN}_step{NNN}.json   machine-readable, full detail
    turn{NN}_step{NNN}.txt    human summary

Captured per decision: the chosen action, every candidate action with its NN
policy prior / NN value / search visits+mean, the opponent belief (archetype
probabilities, per-card predictions, the concretely sampled opponent deck / hand
/ prize), and the predicted principal variation (what the agent expects BOTH
players to do next).
"""

import json
from pathlib import Path

from cg.api import AreaType, OptionType, SelectContext, SelectType


# ============================================================
#  Option / action decoding
# ============================================================

def _area(a) -> str:
    """AreaType int -> short name, or '?' when absent."""
    if a is None:
        return "?"
    try:
        return AreaType(int(a)).name.lower()
    except ValueError:
        return f"area{a}"


def _card_name(names: dict[int, str], cid) -> str | None:
    """Card name for an id, or None when the id is missing / a special (0)."""
    if not cid:
        return None
    return names.get(cid, f"#{cid}")


def _resolve(state, player_index, area, index, names: dict[int, str]) -> str | None:
    """Card/Pokémon name sitting at (player, area, index) in a game State.

    Lets us name PLAY / CARD options that carry only an area+index (no cardId),
    using the concrete cards in the state — including the opponent's SAMPLED hand
    during search. Returns None for facedown / deck / out-of-range slots.
    """
    if state is None or player_index is None or index is None:
        return None
    try:
        p = state.players[player_index]
    except (IndexError, TypeError):
        return None
    a = None if area is None else int(area)
    slots = {
        int(AreaType.HAND): p.hand,
        int(AreaType.DISCARD): p.discard,
        int(AreaType.PRIZE): p.prize,
        int(AreaType.ACTIVE): p.active,
        int(AreaType.BENCH): p.bench,
    }.get(a)
    if not slots or not (0 <= index < len(slots)):
        return None
    entry = slots[index]
    if entry is None:  # facedown
        return None
    return _card_name(names, entry.id)


def describe_option(opt, names: dict[int, str], state=None, actor: int | None = None) -> str:
    """One readable string for a single select Option.

    When `state` (+ `actor`, the acting player) is given, resolves card names for
    options that only reference an area+index rather than a cardId.
    """
    t = int(opt.type)
    nm = _card_name(names, opt.cardId)
    OT = OptionType

    def loc(pi, area, index):
        """name @ area[index], resolving the name from state when needed."""
        name = _resolve(state, pi, area, index, names)
        tag = f"{_area(area)}[{index}]"
        return f"{name} ({tag})" if name else tag

    if t == OT.END:
        return "END turn"
    if t == OT.ATTACK:
        return f"ATTACK (attackId {opt.attackId})"
    if t == OT.RETREAT:
        return "retreat"
    if t == OT.PLAY:
        name = _resolve(state, actor, AreaType.HAND, opt.index, names)
        return f"play {name} (hand[{opt.index}])" if name else f"play hand[{opt.index}]"
    if t == OT.ATTACH:
        src = loc(actor, opt.area, opt.index)
        dst = loc(actor, opt.inPlayArea, opt.inPlayIndex)
        return f"attach {src} -> {dst}"
    if t == OT.EVOLVE:
        evo = nm or _resolve(state, actor, opt.area, opt.index, names)
        dst = loc(actor, opt.inPlayArea, opt.inPlayIndex)
        return f"evolve {dst} -> {evo}" if evo else f"evolve {dst}"
    if t == OT.ABILITY:
        name = nm or _resolve(state, actor, opt.area, opt.index, names)
        return f"ability {name}" if name else f"ability @ {_area(opt.area)}[{opt.index}]"
    if t == OT.DISCARD:
        name = nm or _resolve(state, actor, opt.area, opt.index, names)
        return f"discard {name}" if name else f"discard @ {_area(opt.area)}[{opt.index}]"
    if t == OT.CARD:
        name = nm or _resolve(state, opt.playerIndex, opt.area, opt.index, names)
        where = f"{_area(opt.area)}[{opt.index}] (p{opt.playerIndex})"
        return f"{name} @ {where}" if name else f"card @ {where}"
    if t in (OT.TOOL_CARD, OT.ENERGY_CARD):
        name = nm or _resolve(state, opt.playerIndex, opt.area, opt.index, names)
        where = f"{_area(opt.area)}[{opt.index}]"
        return f"attached {name} @ {where}" if name else f"attached @ {where}"
    if t == OT.ENERGY:
        return f"energy x{opt.count} @ {_area(opt.area)}[{opt.index}]"
    if t == OT.NUMBER:
        return f"number {opt.number}"
    if t == OT.YES:
        return "YES"
    if t == OT.NO:
        return "NO"
    if t == OT.SKILL:
        return f"skill {nm}" if nm else "skill"
    if t == OT.SPECIAL_CONDITION:
        return f"special-condition {opt.specialConditionType}"
    return f"opt(type={t})"


def describe_action(select: list[int], options: list, names: dict[int, str],
                    state=None, actor: int | None = None) -> str:
    """Decode a multi-select (indices into `options`) into ' + '-joined text."""
    if not select:
        return "(no-op / empty select)"
    parts = []
    for i in select:
        if 0 <= i < len(options):
            parts.append(" ".join(describe_option(options[i], names, state, actor).split()))
        else:
            parts.append(f"opt#{i}?")
    return " + ".join(parts)


def _select_label(dbg: dict) -> str:
    """Readable 'what is being selected' label from the select type/context."""
    try:
        st = SelectType(int(dbg["select_type"])).name
    except (ValueError, KeyError):
        st = str(dbg.get("select_type"))
    try:
        sc = SelectContext(int(dbg["select_context"])).name
    except (ValueError, KeyError):
        sc = str(dbg.get("select_context"))
    return f"{st} / {sc}"


# ============================================================
#  Serialization
# ============================================================

def _card_counts(ids: list[int], names: dict[int, str]) -> list[dict]:
    """Collapse a card-id list to [{id,name,count}] sorted by count desc."""
    counts: dict[int, int] = {}
    for cid in ids:
        counts[cid] = counts.get(cid, 0) + 1
    out = [
        {"id": cid, "name": names.get(cid, f"#{cid}"), "count": c}
        for cid, c in counts.items()
    ]
    out.sort(key=lambda r: -r["count"])
    return out


def _decision_dict(dbg: dict, names: dict[int, str]) -> dict:
    """Build the JSON-serializable view of one decision from raw debug data."""
    options = dbg["options"]
    state = dbg["state"]
    actor = dbg["yourIndex"]

    candidates = []
    for c in dbg["candidates"]:
        candidates.append({
            "action": describe_action(c["select"], options, names, state, actor),
            "select": c["select"],
            "nn_policy_prior": round(c["prob"], 4),
            "nn_value": None if c["nn_value"] is None else round(c["nn_value"], 4),
            "visits": c["visit"],
            "search_mean_value": None if c["mean_value"] is None else round(c["mean_value"], 4),
            "expanded": c["expanded"],
            "policy_target": round(c["policy_target"], 4),
        })
    # Most-visited first: mirrors how the move is actually chosen.
    candidates.sort(key=lambda r: -r["visits"])

    belief = dbg["belief"]
    belief_out = {
        "archetype_probs": (
            None if belief["archetype_probs"] is None
            else {k: round(v, 4) for k, v in sorted(
                belief["archetype_probs"].items(), key=lambda kv: -kv[1])}
        ),
        "card_predictions": belief["card_predictions"],
        "known_opponent_cards": _card_counts(belief["known_cards"], names),
        "sampled_opponent_deck": _card_counts(belief["sampled_deck"], names),
        "sampled_opponent_hand": _card_counts(belief["sampled_hand"], names),
        "sampled_opponent_prize": _card_counts(belief["sampled_prize"], names),
    }

    line = []
    for ply in dbg["predicted_line"]:
        line.append({
            "actor": "Evee" if ply["yourIndex"] == dbg["yourIndex"] else "opponent",
            "player_index": ply["yourIndex"],
            "action": describe_action(ply["select"], ply["options"], names,
                                      ply["state"], ply["yourIndex"]),
            "value": round(ply["value"], 4),
        })

    return {
        "turn": dbg["turn"],
        "turn_action_count": dbg["turnActionCount"],
        "acting_player_index": dbg["yourIndex"],
        "selecting": _select_label(dbg),
        "root_value": round(dbg["root_value"], 4),
        "chosen_action": describe_action(dbg["chosen_select"], options, names, state, actor),
        "candidates": candidates,
        "opponent_belief": belief_out,
        "predicted_line": line,
    }


def _render_text(d: dict) -> str:
    """Human-readable summary from the serialized decision dict."""
    L = []
    L.append(f"Turn {d['turn']}  (action #{d['turn_action_count']}, P{d['acting_player_index']} Evee)")
    L.append(f"Selecting: {d['selecting']}")
    L.append(f"Root value: {d['root_value']:+.4f}")
    L.append(f"CHOSEN: {d['chosen_action']}")
    L.append("")
    L.append("Candidate actions (most-visited first):")
    L.append(f"  {'visits':>6}  {'mean':>8}  {'nn_val':>8}  {'prior':>6}  action")
    for c in d["candidates"]:
        mean = "   --   " if c["search_mean_value"] is None else f"{c['search_mean_value']:+.4f}"
        nnv = "   --   " if c["nn_value"] is None else f"{c['nn_value']:+.4f}"
        L.append(f"  {c['visits']:>6}  {mean:>8}  {nnv:>8}  {c['nn_policy_prior']:>6.3f}  {c['action']}")
    L.append("")

    b = d["opponent_belief"]
    L.append("Opponent belief:")
    if b["archetype_probs"]:
        top = list(b["archetype_probs"].items())[:5]
        L.append("  archetype: " + ", ".join(f"{k} {v:.0%}" for k, v in top))
    else:
        L.append("  archetype: (no model attached — belief is UNKNOWN cards)")
    if b["card_predictions"]:
        L.append("  likely cards (P, E[copies]):")
        for p in b["card_predictions"][:15]:
            L.append(f"    {p['probability']:.2f}  {p['expected_copies']:.2f}x  {p['name']}")
    L.append("  known (visible) opponent cards:")
    for r in b["known_opponent_cards"][:20]:
        L.append(f"    {r['count']}x {r['name']}")
    L.append("  sampled opponent hand (this search):")
    for r in b["sampled_opponent_hand"]:
        L.append(f"    {r['count']}x {r['name']}")
    L.append("")

    L.append("Predicted line (principal variation):")
    if d["predicted_line"]:
        for i, ply in enumerate(d["predicted_line"]):
            L.append(f"  {i+1}. [{ply['actor']}] {ply['action']}  (v={ply['value']:+.4f})")
    else:
        L.append("  (root leaf — no expanded continuation)")
    return "\n".join(L) + "\n"


def write_decision(folder: Path, idx: int, dbg: dict, names: dict[int, str]) -> Path:
    """Write one decision as JSON + txt into `folder`; return the JSON path."""
    folder.mkdir(parents=True, exist_ok=True)
    d = _decision_dict(dbg, names)
    stem = f"turn{d['turn']:02d}_step{idx:03d}"
    json_path = folder / f"{stem}.json"
    txt_path = folder / f"{stem}.txt"
    json_path.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    txt_path.write_text(_render_text(d))
    return json_path
