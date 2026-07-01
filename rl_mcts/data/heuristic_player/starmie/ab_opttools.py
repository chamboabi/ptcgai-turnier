from __future__ import annotations

import re
from typing import Any, Callable


OPTION_TYPE_NAMES = {
    0: "NUMBER",
    1: "YES",
    2: "NO",
    3: "CARD",
    4: "TOOL_CARD",
    5: "ENERGY_CARD",
    6: "ENERGY",
    7: "PLAY",
    8: "ATTACH",
    9: "EVOLVE",
    10: "ABILITY",
    11: "DISCARD",
    12: "RETREAT",
    13: "ATTACK",
    14: "END",
    15: "SKILL",
    16: "SPECIAL_CONDITION",
}

CONTEXT_NAMES = {
    0: "MAIN",
    1: "SETUP_ACTIVE_POKEMON",
    2: "SETUP_BENCH_POKEMON",
    4: "TO_ACTIVE",
    5: "TO_BENCH",
    6: "TO_FIELD",
    7: "TO_HAND",
    30: "DISCARD_ENERGY",
    22: "ATTACH_TO",
    35: "ATTACK",
    38: "DRAW_COUNT",
    41: "IS_FIRST",
}

AREA_NAMES = {
    1: "DECK",
    2: "HAND",
    3: "DISCARD",
    4: "ACTIVE",
    5: "BENCH",
}


def obj_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _raw_name(value: Any, value_map: dict[int, str] | None = None) -> str:
    if value is None:
        return ""
    if hasattr(value, "name"):
        return str(value.name)
    if isinstance(value, int) and value_map and value in value_map:
        return value_map[value]
    text = str(value)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def _token(value: Any, value_map: dict[int, str] | None = None) -> str:
    return re.sub(r"[^a-z0-9]+", "", _raw_name(value, value_map).lower())


def _matches(value: Any, expected: str, value_map: dict[int, str] | None = None) -> bool:
    return _token(value, value_map) == _token(expected)


def _select(obs_dict: Any) -> Any:
    return obj_get(obs_dict, "select")


def get_options(obs_dict: Any) -> list[Any]:
    """Return the current legal option objects or dictionaries."""
    select = _select(obs_dict)
    if select is None:
        return []
    return list(obj_get(select, "option", []) or [])


def get_context(obs_dict: Any) -> str | None:
    """Return a stable context name such as MAIN, IS_FIRST, or DRAW_COUNT."""
    select = _select(obs_dict)
    if select is None:
        return None
    context = obj_get(select, "context")
    return _raw_name(context, CONTEXT_NAMES) or None


def option_type_name(option: Any) -> str:
    """Return a stable option type name such as PLAY, ATTACH, ATTACK, or END."""
    return _raw_name(obj_get(option, "type"), OPTION_TYPE_NAMES).upper()


def current_player_index(obs_dict: Any) -> int:
    current = obj_get(obs_dict, "current", {}) or {}
    return int(obj_get(current, "yourIndex", 0) or 0)


def current_player_state(obs_dict: Any) -> Any | None:
    current = obj_get(obs_dict, "current", {}) or {}
    players = obj_get(current, "players", []) or []
    idx = current_player_index(obs_dict)
    if 0 <= idx < len(players):
        return players[idx]
    return None


def hand_card_for_play_option(obs_dict: Any, option: Any) -> Any | None:
    """Map a PLAY option back to the visible card object in the current hand."""
    if not _matches(obj_get(option, "type"), "PLAY", OPTION_TYPE_NAMES):
        return None
    hand_index = obj_get(option, "index")
    if hand_index is None:
        return None
    player = current_player_state(obs_dict)
    hand = obj_get(player, "hand", []) or []
    if 0 <= int(hand_index) < len(hand):
        return hand[int(hand_index)]
    return None


def hand_card_id_for_play_option(obs_dict: Any, option: Any) -> int | None:
    """Return the Card ID for a PLAY option when the hand is visible."""
    card = hand_card_for_play_option(obs_dict, option)
    card_id = obj_get(card, "id")
    return int(card_id) if card_id is not None else None


def selection_bounds(obs_dict: Any) -> tuple[int, int]:
    select = _select(obs_dict)
    options = get_options(obs_dict)
    if select is None:
        return (0, 0)
    min_count = int(obj_get(select, "minCount", 0) or 0)
    max_count = int(obj_get(select, "maxCount", min_count) or 0)
    max_count = min(max_count, len(options))
    min_count = max(0, min(min_count, max_count))
    return min_count, max_count


def legalize_choice(obs_dict: Any, choices: list[int] | None) -> list[int]:
    """Clamp a proposed selection to legal indices and count bounds."""
    options = get_options(obs_dict)
    min_count, max_count = selection_bounds(obs_dict)
    if max_count == 0:
        return []

    seen: set[int] = set()
    legal: list[int] = []
    for choice in choices or []:
        if isinstance(choice, bool) or not isinstance(choice, int):
            continue
        if 0 <= choice < len(options) and choice not in seen:
            legal.append(choice)
            seen.add(choice)
        if len(legal) >= max_count:
            break

    if len(legal) < min_count:
        for idx in range(len(options)):
            if idx not in seen:
                legal.append(idx)
                seen.add(idx)
            if len(legal) >= min_count:
                break

    return legal[:max_count]


def legal_first(obs_dict: Any) -> list[int]:
    """Choose the first legal options needed to satisfy minCount."""
    min_count, _ = selection_bounds(obs_dict)
    return legalize_choice(obs_dict, list(range(min_count)))


def find_options_by_type(options: list[Any], type_name: str) -> list[tuple[int, Any]]:
    """Return (index, option) pairs with a matching OptionType name."""
    return [
        (idx, option)
        for idx, option in enumerate(options)
        if _matches(obj_get(option, "type"), type_name, OPTION_TYPE_NAMES)
    ]


def choose_end(options: list[Any]) -> int | None:
    matches = find_options_by_type(options, "END")
    return matches[0][0] if matches else None


def _attack_score(option: Any, attack: Any | None) -> float:
    if attack is None:
        return 0.0
    damage = int(obj_get(attack, "damage", 0) or 0)
    text = str(obj_get(attack, "text", "") or "").lower()
    energy_count = len(obj_get(attack, "energies", []) or [])
    score = float(damage) - energy_count
    if "discard" in text:
        score -= 20
    if "prevent" in text and "damage" in text:
        score += 25
    if "draw" in text:
        score += 8
    return score


def choose_attack(
    options: list[Any],
    attack_lookup: dict[int, Any] | None = None,
    scoring_fn: Callable[[Any, Any | None], float] | None = None,
) -> int | None:
    """Choose the highest-scoring attack option."""
    attacks = find_options_by_type(options, "ATTACK")
    if not attacks:
        return None

    lookup = attack_lookup or {}
    scorer = scoring_fn or _attack_score
    best_idx: int | None = None
    best_score = float("-inf")
    for idx, option in attacks:
        attack_id = obj_get(option, "attackId")
        attack = lookup.get(int(attack_id)) if attack_id is not None and lookup else None
        score = scorer(option, attack)
        if score > best_score:
            best_idx = idx
            best_score = score
    return best_idx


def choose_attach_to_active(options: list[Any]) -> int | None:
    """Prefer Attach options targeting the Active spot."""
    attaches = find_options_by_type(options, "ATTACH")
    if not attaches:
        return None
    for idx, option in attaches:
        if _matches(obj_get(option, "inPlayArea"), "ACTIVE", AREA_NAMES):
            return idx
    return attaches[0][0]


def choose_number_max(options: list[Any]) -> int | None:
    numbers = find_options_by_type(options, "NUMBER")
    if not numbers:
        return None
    return max(numbers, key=lambda item: int(obj_get(item[1], "number", 0) or 0))[0]


def safe_choice(obs_dict: Any) -> list[int]:
    """Return a legal fallback selection for any non-initial observation."""
    options = get_options(obs_dict)
    if not options:
        return []

    context = _token(get_context(obs_dict))
    if context == _token("DRAW_COUNT"):
        number_idx = choose_number_max(options)
        if number_idx is not None:
            return legalize_choice(obs_dict, [number_idx])

    if context == _token("MAIN"):
        end_idx = choose_end(options)
        if end_idx is not None:
            return legalize_choice(obs_dict, [end_idx])

    min_count, _ = selection_bounds(obs_dict)
    if min_count == 0:
        return []
    return legalize_choice(obs_dict, list(range(min_count)))
