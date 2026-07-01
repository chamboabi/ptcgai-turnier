from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared import finish_choice, is_initial_observation, load_agent_deck
from option_tools import (
    AREA_NAMES,
    choose_attach_to_active,
    choose_end,
    choose_number_max,
    current_player_index,
    find_options_by_type,
    get_context,
    get_options,
    hand_card_id_for_play_option,
    obj_get,
    option_type_name,
    safe_choice,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBMISSION_DIR = PROJECT_ROOT / "submission"
BASIC_WATER_ENERGY_ID = 3
CRUSTLE_ID = 345
DWEBBLE_ID = 344
BUDEW_ID = 235
PECHARUNT_ID = 230

_DEFAULT_ATTACK_LOOKUP: dict[int, Any] | None = None
_DEFAULT_CARD_LOOKUP: dict[int, Any] | None = None


@dataclass(frozen=True)
class ScoredAction:
    index: int
    score: float
    reason: str
    detail: dict


def _load_cg_metadata(submission_dir: str | Path = DEFAULT_SUBMISSION_DIR) -> tuple[dict[int, Any], dict[int, Any]]:
    """Load attack and card metadata from the bundled cg API if available."""
    global _DEFAULT_ATTACK_LOOKUP, _DEFAULT_CARD_LOOKUP
    if _DEFAULT_ATTACK_LOOKUP is not None and _DEFAULT_CARD_LOOKUP is not None:
        return _DEFAULT_ATTACK_LOOKUP, _DEFAULT_CARD_LOOKUP

    try:
        import importlib
        import sys

        submission_path = Path(submission_dir).resolve()
        inserted = False
        if str(submission_path) not in sys.path:
            sys.path.insert(0, str(submission_path))
            inserted = True
        try:
            api = importlib.import_module("cg.api")
            _DEFAULT_ATTACK_LOOKUP = {int(a.attackId): a for a in api.all_attack()}
            _DEFAULT_CARD_LOOKUP = {int(c.cardId): c for c in api.all_card_data()}
        finally:
            if inserted:
                try:
                    sys.path.remove(str(submission_path))
                except ValueError:
                    pass
    except Exception:
        _DEFAULT_ATTACK_LOOKUP = {}
        _DEFAULT_CARD_LOOKUP = {}

    return _DEFAULT_ATTACK_LOOKUP, _DEFAULT_CARD_LOOKUP


def load_default_metadata(submission_dir: str | Path = DEFAULT_SUBMISSION_DIR) -> tuple[dict[int, Any], dict[int, Any]]:
    """Public helper for scripts that want the same cg metadata as the agent."""
    return _load_cg_metadata(submission_dir)


def _current(obs_dict: Any) -> Any:
    return obj_get(obs_dict, "current")


def _player_state(obs_dict: Any, player_index: int) -> Any | None:
    current = _current(obs_dict)
    players = obj_get(current, "players", []) or []
    if 0 <= player_index < len(players):
        return players[player_index]
    return None


def _your_index(obs_dict: Any) -> int:
    return current_player_index(obs_dict)


def _opponent_index(obs_dict: Any) -> int:
    return 1 - _your_index(obs_dict)


def _pilot_name(pilot_profile: Any | None) -> str:
    return str(getattr(pilot_profile, "name", "") or "").lower()


def _active_pokemon(obs_dict: Any, player_index: int) -> Any | None:
    player = _player_state(obs_dict, player_index)
    active = obj_get(player, "active", []) or []
    return active[0] if active else None


def _opponent_active_hp(obs_dict: Any) -> int | None:
    active = _active_pokemon(obs_dict, _opponent_index(obs_dict))
    hp = obj_get(active, "hp")
    return int(hp) if hp is not None else None


def _your_deck_count(obs_dict: Any) -> int | None:
    you = _player_state(obs_dict, _your_index(obs_dict))
    value = obj_get(you, "deckCount")
    return int(value) if value is not None else None


def _your_hand_count(obs_dict: Any) -> int:
    you = _player_state(obs_dict, _your_index(obs_dict))
    hand = obj_get(you, "hand", None)
    if hand is not None:
        return len(hand)
    return int(obj_get(you, "handCount", 0) or 0)


def _bench_count(obs_dict: Any, player_index: int) -> int:
    player = _player_state(obs_dict, player_index)
    return len(obj_get(player, "bench", []) or [])


def _attached_energy_count(pokemon: Any | None) -> int:
    return len(obj_get(pokemon, "energies", []) or [])


def _player_energy_count(obs_dict: Any, player_index: int) -> int:
    active_count = _attached_energy_count(_active_pokemon(obs_dict, player_index))
    player = _player_state(obs_dict, player_index)
    bench_count = sum(_attached_energy_count(pokemon) for pokemon in obj_get(player, "bench", []) or [])
    return active_count + bench_count


def _in_play_count(obs_dict: Any, player_index: int) -> int:
    active_count = 1 if _active_pokemon(obs_dict, player_index) is not None else 0
    return active_count + _bench_count(obs_dict, player_index)


def _in_play_ids(obs_dict: Any, player_index: int | None = None) -> set[int]:
    if player_index is None:
        player_index = _your_index(obs_dict)
    ids: set[int] = set()
    active = _active_pokemon(obs_dict, player_index)
    active_id = obj_get(active, "id")
    if active_id is not None:
        ids.add(int(active_id))
    player = _player_state(obs_dict, player_index)
    for pokemon in obj_get(player, "bench", []) or []:
        card_id = obj_get(pokemon, "id")
        if card_id is not None:
            ids.add(int(card_id))
    return ids


def _select_deck(obs_dict: Any) -> list[Any]:
    select = obj_get(obs_dict, "select")
    return obj_get(select, "deck", []) or []


def _gain_option_card_id(obs_dict: Any, option: Any) -> int | None:
    """Resolve the Card ID a CARD option refers to in a search/setup selection."""
    card_id = obj_get(option, "cardId")
    if card_id is not None:
        return int(card_id)
    area = AREA_NAMES.get(obj_get(option, "area"), obj_get(option, "area"))
    index = obj_get(option, "index")
    if index is None:
        return None
    try:
        index = int(index)
    except (TypeError, ValueError):
        return None
    if area == "DECK":
        deck = _select_deck(obs_dict)
        if 0 <= index < len(deck) and deck[index] is not None:
            value = obj_get(deck[index], "id")
            return int(value) if value is not None else None
    if area == "HAND":
        player = _player_state(obs_dict, _your_index(obs_dict))
        hand = obj_get(player, "hand", []) or []
        if 0 <= index < len(hand) and hand[index] is not None:
            value = obj_get(hand[index], "id")
            return int(value) if value is not None else None
    return None


def _gain_target_value(
    obs_dict: Any,
    card_id: int | None,
    context: str,
    pilot_profile: Any | None,
    card_lookup: dict[int, Any],
) -> float:
    """Rank cards a search/setup effect can take into hand or onto the board."""
    in_play = _in_play_ids(obs_dict)
    setup_ids = set(getattr(pilot_profile, "preferred_setup_ids", ()) or ())
    if card_id == DWEBBLE_ID:
        return 100.0 if DWEBBLE_ID not in in_play else 72.0
    if card_id == CRUSTLE_ID:
        if context.startswith("TO_HAND") or context == "TOHAND":
            return 92.0 if (DWEBBLE_ID in in_play and CRUSTLE_ID not in in_play) else 34.0
        return 30.0
    if card_id in setup_ids:
        return 64.0
    card = card_lookup.get(int(card_id)) if card_id is not None else None
    if _is_basic_pokemon(card):
        return 55.0
    if _is_pokemon(card):
        return 26.0
    if _is_energy_card(card):
        return 16.0
    return 12.0


def _choose_gain_targets(
    obs_dict: Any,
    context: str,
    options: list[Any],
    pilot_profile: Any | None,
    card_lookup: dict[int, Any],
) -> list[int]:
    """Take the best options for a beneficial search/setup context (never decline)."""
    if not options:
        return []
    return sorted(
        range(len(options)),
        key=lambda idx: _gain_target_value(
            obs_dict, _gain_option_card_id(obs_dict, options[idx]), context, pilot_profile, card_lookup
        ),
        reverse=True,
    )


def _has_in_play_card(obs_dict: Any, player_index: int, card_ids: set[int]) -> bool:
    active = _active_pokemon(obs_dict, player_index)
    active_id = obj_get(active, "id")
    if active_id is not None and int(active_id) in card_ids:
        return True
    player = _player_state(obs_dict, player_index)
    for pokemon in obj_get(player, "bench", []) or []:
        card_id = obj_get(pokemon, "id")
        if card_id is not None and int(card_id) in card_ids:
            return True
    return False


def _bench_max(obs_dict: Any, player_index: int) -> int:
    player = _player_state(obs_dict, player_index)
    return int(obj_get(player, "benchMax", 5) or 5)


def _discard_card_ids(obs_dict: Any, player_index: int) -> list[int]:
    player = _player_state(obs_dict, player_index)
    discard = obj_get(player, "discard", []) or []
    ids: list[int] = []
    for card in discard:
        card_id = obj_get(card, "id")
        if card_id is not None:
            ids.append(int(card_id))
    return ids


def _pokemon_tool_ids(pokemon: Any | None) -> list[int]:
    tools = obj_get(pokemon, "tools", []) or []
    ids: list[int] = []
    for card in tools:
        card_id = obj_get(card, "id")
        if card_id is not None:
            ids.append(int(card_id))
    return ids


def _card_name(card_data: Any | None, fallback: str = "unknown") -> str:
    return str(obj_get(card_data, "name", fallback) or fallback)


def _enum_name(value: Any) -> str:
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def _is_pokemon(card_data: Any | None) -> bool:
    return _enum_name(obj_get(card_data, "cardType", "")).upper() in {"POKEMON", "0"}


def _is_basic_pokemon(card_data: Any | None) -> bool:
    return _is_pokemon(card_data) and bool(obj_get(card_data, "basic", False))


def _is_energy_card(card_data: Any | None) -> bool:
    return obj_get(card_data, "cardType") in {5, 6}


def _is_tool_card(card_data: Any | None) -> bool:
    return obj_get(card_data, "cardType") == 2


def _is_stadium_card(card_data: Any | None) -> bool:
    return obj_get(card_data, "cardType") == 4


def _skill_text(card_data: Any | None) -> str:
    skills = obj_get(card_data, "skills", []) or []
    parts = []
    for skill in skills:
        parts.append(str(obj_get(skill, "name", "") or ""))
        parts.append(str(obj_get(skill, "text", "") or ""))
    return " ".join(parts).lower()


def _attack_text(attack: Any | None) -> str:
    return str(obj_get(attack, "text", "") or "").lower()


def _opponent_active_is_ex(obs_dict: Any, card_lookup: dict[int, Any]) -> bool:
    active = _active_pokemon(obs_dict, _opponent_index(obs_dict))
    card_id = obj_get(active, "id")
    card = card_lookup.get(int(card_id)) if card_id is not None else None
    return bool(obj_get(card, "ex", False) or obj_get(card, "megaEx", False))


def _attack_pierces_active_effects(attack: Any | None) -> bool:
    """True for attacks that 'isn't affected by ... effects on your opponent's Active'.

    These attacks (Nebula Beam, Dudunsparce ex, Mega Lopunny ex, ...) bypass a
    Crustle-style damage-prevention ability and so still KO the wall.
    """
    text = _attack_text(attack)
    return "affected by" in text and "effects on your opponent" in text


def _opponent_active_walls_ex(obs_dict: Any, card_lookup: dict[int, Any]) -> bool:
    """True if the opponent's Active prevents all damage from our ex (Crustle-style)."""
    active = _active_pokemon(obs_dict, _opponent_index(obs_dict))
    card_id = obj_get(active, "id")
    if card_id is None:
        return False
    if int(card_id) == CRUSTLE_ID:
        return True
    card = card_lookup.get(int(card_id))
    for skill in obj_get(card, "skills", []) or []:
        txt = (str(obj_get(skill, "name", "") or "") + " " + str(obj_get(skill, "text", "") or "")).lower()
        if "prevent all damage" in txt and "ex" in txt:
            return True
    return False


def _our_active_is_ex(obs_dict: Any, card_lookup: dict[int, Any]) -> bool:
    active = _active_pokemon(obs_dict, _your_index(obs_dict))
    card_id = obj_get(active, "id")
    card = card_lookup.get(int(card_id)) if card_id is not None else None
    return bool(obj_get(card, "ex", False) or obj_get(card, "megaEx", False))


def _active_has_maximum_belt(obs_dict: Any) -> bool:
    active = _active_pokemon(obs_dict, _your_index(obs_dict))
    return 1158 in _pokemon_tool_ids(active)


def _opponent_bench_pokemon(obs_dict: Any) -> list[Any]:
    player = _player_state(obs_dict, _opponent_index(obs_dict))
    return list(obj_get(player, "bench", []) or [])


def _pokemon_hp(pokemon: Any | None) -> int | None:
    hp = obj_get(pokemon, "hp")
    return int(hp) if hp is not None else None


def _any_bench_koable(obs_dict: Any, damage: float) -> bool:
    """True if some benched opponent Pokemon has HP <= damage (a gust converts to a KO)."""
    if damage < 50:
        return False
    for pokemon in _opponent_bench_pokemon(obs_dict):
        hp = _pokemon_hp(pokemon)
        if hp is not None and 0 < hp <= damage:
            return True
    return False


def _gust_enables_ko(obs_dict: Any, best_attack: "ScoredAction | None", card_lookup: dict[int, Any]) -> bool:
    """True if a benched opponent Pokemon is KO-able by our best attack.

    Only meaningful once we have skipped the KO-of-active branch, i.e. the
    current Active is not KO-able. Used by rush pilots to decide whether playing
    Boss/Catcher actually converts into a Prize this turn.
    """
    if best_attack is None:
        return False
    return _any_bench_koable(obs_dict, float(best_attack.detail.get("damage", 0) or 0))


def _reachable_printed_damage(obs_dict: Any, card_lookup: dict[int, Any], attack_lookup: dict[int, Any]) -> float:
    """Max printed damage our Active could do to a (non-immune) gusted-up target.

    Uses printed damage of attacks we can currently pay for, ignoring the active
    Pokemon's damage-prevention. Lets a wall-blocked ex still gust + KO the frail
    Bench (Crustle's support Pokemon are not protected, only Crustle is)."""
    active = _active_pokemon(obs_dict, _your_index(obs_dict))
    card_id = obj_get(active, "id")
    card = card_lookup.get(int(card_id)) if card_id is not None else None
    attached = _attached_energy_count(active)
    best = 0.0
    for attack_id in obj_get(card, "attacks", []) or []:
        attack = attack_lookup.get(int(attack_id))
        cost = len(obj_get(attack, "energies", []) or [])
        if cost <= attached:
            best = max(best, float(int(obj_get(attack, "damage", 0) or 0)))
    return best


def _active_best_attack_damage(
    obs_dict: Any,
    deck: list[int] | None,
    attack_lookup: dict[int, Any],
    card_lookup: dict[int, Any],
) -> float:
    """Estimate the best damage our current Active can do (used for gust targeting)."""
    active = _active_pokemon(obs_dict, _your_index(obs_dict))
    card_id = obj_get(active, "id")
    card = card_lookup.get(int(card_id)) if card_id is not None else None
    best = 0.0
    for attack_id in obj_get(card, "attacks", []) or []:
        attack = attack_lookup.get(int(attack_id))
        best = max(best, _estimated_attack_damage(obs_dict, attack, deck, card_lookup))
    return best


def _best_gust_target_index(obs_dict: Any, options: list[Any], best_damage: float, card_lookup: dict[int, Any]) -> int | None:
    """Pick the opponent-bench option to drag up: a KO-able Pokemon, preferring ex.

    Used when a gust card (Boss/Catcher) asks which benched Pokemon to switch in.
    Ranks KO-able targets first (2-prize ex highest), then lowest HP.
    """
    best_idx: int | None = None
    best_key: tuple | None = None
    for idx, option in enumerate(options):
        area = obj_get(option, "inPlayArea", obj_get(option, "area"))
        index = obj_get(option, "inPlayIndex", obj_get(option, "index"))
        pokemon = _in_play_pokemon(obs_dict, _opponent_index(obs_dict), area, index)
        if pokemon is None:
            continue
        hp = _pokemon_hp(pokemon)
        if hp is None:
            continue
        card_id = obj_get(pokemon, "id")
        card = card_lookup.get(int(card_id)) if card_id is not None else None
        is_ex = bool(obj_get(card, "ex", False) or obj_get(card, "megaEx", False))
        koable = best_damage > 0 and hp <= best_damage
        # Higher tuple sorts first: KO-able beats not, ex beats non-ex, then lower HP.
        key = (1 if koable else 0, 1 if (koable and is_ex) else 0, -hp)
        if best_key is None or key > best_key:
            best_key = key
            best_idx = idx
    return best_idx


def _in_play_pokemon(obs_dict: Any, player_index: int, area: Any, index: Any) -> Any | None:
    area_name = AREA_NAMES.get(area, area)
    try:
        index_int = int(index or 0)
    except (TypeError, ValueError):
        index_int = 0
    if area_name == "ACTIVE":
        return _active_pokemon(obs_dict, player_index)
    elif area_name == "BENCH":
        player = _player_state(obs_dict, player_index)
        bench = obj_get(player, "bench", []) or []
        return bench[index_int] if 0 <= index_int < len(bench) else None
    return None


def _in_play_pokemon_id(obs_dict: Any, player_index: int, area: Any, index: Any) -> int | None:
    pokemon = _in_play_pokemon(obs_dict, player_index, area, index)
    card_id = obj_get(pokemon, "id")
    return int(card_id) if card_id is not None else None


def _hand_card_id_for_option(obs_dict: Any, option: Any) -> int | None:
    if AREA_NAMES.get(obj_get(option, "area"), obj_get(option, "area")) != "HAND":
        return None
    hand_index = obj_get(option, "index")
    if hand_index is None:
        return None
    you = _player_state(obs_dict, _your_index(obs_dict))
    hand = obj_get(you, "hand", []) or []
    try:
        card = hand[int(hand_index)]
    except (TypeError, ValueError, IndexError):
        return None
    card_id = obj_get(card, "id")
    return int(card_id) if card_id is not None else None


def _estimated_attack_damage(
    obs_dict: Any,
    attack: Any | None,
    deck: list[int] | None,
    card_lookup: dict[int, Any],
) -> float:
    if attack is None:
        return 0.0

    # Play around a Crustle-style wall: if our Active is an ex and the opponent's
    # Active prevents all damage from ex, a normal attack does 0 — unless it
    # "isn't affected by effects on your opponent's Active" (Nebula Beam-style),
    # which pierces the wall. Zeroing the walled attacks makes the bot build to
    # and fire the piercer (or pivot to gusting the bench) instead of bashing it.
    if (
        _our_active_is_ex(obs_dict, card_lookup)
        and _opponent_active_walls_ex(obs_dict, card_lookup)
        and not _attack_pierces_active_effects(attack)
    ):
        return 0.0

    damage = float(int(obj_get(attack, "damage", 0) or 0))
    text = _attack_text(attack)
    your_index = _your_index(obs_dict)

    if "20 damage for each basic {w} energy card in your discard pile" in text:
        water_in_discard = _discard_card_ids(obs_dict, your_index).count(BASIC_WATER_ENERGY_ID)
        damage = max(damage, float(20 * water_in_discard))

    if "discard the top 6 cards of your deck" in text and "100 damage for each basic {w} energy" in text:
        deck_count = _your_deck_count(obs_dict) or 0
        if deck_count > 0:
            if deck:
                water_density = deck.count(BASIC_WATER_ENERGY_ID) / len(deck)
            else:
                water_density = 0.45
            expected_hits = min(6, deck_count) * water_density
            damage = max(damage, float(100 * expected_hits))

    if "20 more damage for each benched" in text:
        your_bench = _bench_count(obs_dict, _your_index(obs_dict))
        opponent_bench = _bench_count(obs_dict, _opponent_index(obs_dict))
        damage = max(damage, float(20 + 20 * (your_bench + opponent_bench)))

    if "does 100 damage to 1 of your opponent" in text:
        damage = max(damage, 100.0)

    # Alakazam "Powerful Hand": place 2 damage counters (=20 HP) on the opponent's
    # Active for each card in the attacker's hand. Printed damage is 0, so without
    # this the heuristic would read Alakazam as a 0-damage attacker and never fire
    # it -- making it a useless training opponent. damage = 20 x current hand size.
    if "damage counters" in text and "for each card in your hand" in text:
        hand = _your_hand_count(obs_dict)
        damage = max(damage, float(20 * hand))

    if _active_has_maximum_belt(obs_dict) and _opponent_active_is_ex(obs_dict, card_lookup):
        damage += 50

    # Weakness: if our attacker's type matches the opponent Active's Weakness, the
    # engine doubles the damage. Teaching the pilot this lets it value a Metal
    # attacker into a Metal-weak wall (e.g. Mega Abomasnow ex, 350 HP, Metal-weak)
    # at its true KO damage instead of the printed number.
    if damage > 0:
        attacker = _active_pokemon(obs_dict, your_index)
        attacker_id = obj_get(attacker, "id")
        attacker_card = card_lookup.get(int(attacker_id)) if attacker_id is not None else None
        opp_active = _active_pokemon(obs_dict, _opponent_index(obs_dict))
        opp_id = obj_get(opp_active, "id")
        opp_card = card_lookup.get(int(opp_id)) if opp_id is not None else None
        atk_type = obj_get(attacker_card, "energyType")
        weak = obj_get(opp_card, "weakness")
        try:
            if atk_type is not None and weak is not None and int(atk_type) == int(weak):
                damage *= 2
        except (TypeError, ValueError):
            pass

    return damage


def _score_attack(
    obs_dict: Any,
    option: Any,
    attack: Any | None,
    deck: list[int] | None,
    card_lookup: dict[int, Any],
    pilot_profile: Any | None = None,
) -> ScoredAction:
    attack_id = obj_get(option, "attackId")
    text = _attack_text(attack)
    damage = _estimated_attack_damage(obs_dict, attack, deck, card_lookup)
    printed_damage = int(obj_get(attack, "damage", 0) or 0) if attack is not None else 0
    energy_count = len(obj_get(attack, "energies", []) or []) if attack is not None else 0
    opponent_hp = _opponent_active_hp(obs_dict)
    deck_count = _your_deck_count(obs_dict)
    is_ko = opponent_hp is not None and damage >= opponent_hp and damage > 0

    score = damage - energy_count * 3
    reasons = [f"damage={damage:.1f}"]
    if is_ko:
        score += 1000
        reasons.append(f"KO active hp={opponent_hp}")
    if "discard" in text and "energy" in text and not is_ko:
        score -= 45
        reasons.append("penalty: energy discard")
    elif "discard" in text and not is_ko:
        score -= 18
        reasons.append("penalty: discard cost")
    if "discard the top" in text and "deck" in text:
        if deck_count is not None and deck_count <= 10:
            score -= 180
            reasons.append("penalty: low-deck self-mill")
        elif not is_ko:
            score -= 25
            reasons.append("penalty: self-mill")
    if "takes 30 less damage" in text or ("prevent" in text and "damage" in text):
        score += 45
        reasons.append("defensive effect")
    if "draw" in text:
        score += 8
        reasons.append("draw effect")
    if "evolves from this pokémon" in text and "search your deck" in text:
        score += 58
        reasons.append("setup evolution attack")
    if "can't play any item" in text:
        score += 48
        reasons.append("item lock")
    if "can't retreat" in text:
        score += 28
        reasons.append("trap effect")
    if "poisoned" in text or "confused" in text:
        score += 12
        reasons.append("special condition")
    active = _active_pokemon(obs_dict, _your_index(obs_dict))
    active_id = obj_get(active, "id")
    if pilot_profile and active_id is not None:
        if int(active_id) in set(getattr(pilot_profile, "preferred_main_attacker_ids", ())):
            if is_ko or damage >= 120:
                score += 18
                reasons.append(f"pilot main attacker: {pilot_profile.name}")
            elif damage >= 80:
                score += 6
                reasons.append(f"pilot main attacker medium damage: {pilot_profile.name}")
            else:
                score -= 8
                reasons.append("pilot main attacker needs setup")
        priorities = getattr(pilot_profile, "attack_priorities", None) or {}
        if attack_id in priorities:
            score += float(priorities[attack_id])
            reasons.append("pilot attack priority")

    if printed_damage == 0 and damage == 0:
        score -= 8
        reasons.append("no known damage")

    return ScoredAction(
        index=-1,
        score=score,
        reason=", ".join(reasons),
        detail={
            "kind": "attack",
            "attackId": attack_id,
            "attacker_id": active_id,
            "attack": str(obj_get(attack, "name", attack_id)),
            "damage": damage,
            "opponent_hp": opponent_hp,
            "ko": is_ko,
        },
    )


def _score_attacks(
    obs_dict: Any,
    options: list[Any],
    attack_lookup: dict[int, Any],
    card_lookup: dict[int, Any],
    deck: list[int] | None,
    pilot_profile: Any | None = None,
) -> list[ScoredAction]:
    scored: list[ScoredAction] = []
    for idx, option in find_options_by_type(options, "ATTACK"):
        attack_id = obj_get(option, "attackId")
        attack = attack_lookup.get(int(attack_id)) if attack_id is not None else None
        action = _score_attack(obs_dict, option, attack, deck, card_lookup, pilot_profile=pilot_profile)
        scored.append(
            ScoredAction(
                index=idx,
                score=action.score,
                reason=action.reason,
                detail=action.detail,
            )
        )
    return sorted(scored, key=lambda action: action.score, reverse=True)


def _score_play(
    obs_dict: Any,
    idx: int,
    option: Any,
    card_lookup: dict[int, Any],
    pilot_profile: Any | None = None,
) -> ScoredAction | None:
    card_id = hand_card_id_for_play_option(obs_dict, option)
    if card_id is None:
        return None

    card = card_lookup.get(card_id)
    name = _card_name(card, str(card_id))
    name_lower = name.lower()
    text = _skill_text(card)
    hand_count = _your_hand_count(obs_dict)
    bench_count = _bench_count(obs_dict, _your_index(obs_dict))
    bench_max = _bench_max(obs_dict, _your_index(obs_dict))
    active = _active_pokemon(obs_dict, _your_index(obs_dict))
    active_tools = _pokemon_tool_ids(active)
    score = 0.0
    purpose = "unknown"
    reasons: list[str] = []
    pilot = _pilot_name(pilot_profile)
    playstyle = str(getattr(pilot_profile, "playstyle", "") or "")

    main_attackers = set(getattr(pilot_profile, "preferred_main_attacker_ids", ()) if pilot_profile else ())

    if pilot == "crustle" and "crushing hammer" in name_lower:
        opponent_energy = _player_energy_count(obs_dict, _opponent_index(obs_dict))
        if opponent_energy <= 0:
            return None
        score = 74
        purpose = "energy_denial"
        reasons.append(f"opponent energy={opponent_energy}")
    elif pilot == "crustle" and ("boss" in name_lower or "catcher" in name_lower):
        opponent_bench = _bench_count(obs_dict, _opponent_index(obs_dict))
        if opponent_bench <= 0:
            return None
        score = 63 if "boss" in name_lower else 55
        purpose = "gust_trap"
        reasons.append(f"gust bench={opponent_bench}")
    elif pilot == "crustle" and "neutralization zone" in name_lower:
        score = 82 if _has_in_play_card(obs_dict, _your_index(obs_dict), {CRUSTLE_ID, DWEBBLE_ID}) else 58
        purpose = "wall_stadium"
        reasons.append("anti-ex stadium")
    elif pilot == "crustle" and "buddy-buddy poffin" in name_lower:
        score = 78 if not _has_in_play_card(obs_dict, _your_index(obs_dict), {DWEBBLE_ID}) else 44
        purpose = "search"
        reasons.append("find Dwebble")
    elif pilot == "crustle" and "bug catching set" in name_lower:
        score = 74 if not _has_in_play_card(obs_dict, _your_index(obs_dict), {DWEBBLE_ID, CRUSTLE_ID}) else 54
        purpose = "search"
        reasons.append("find Grass Pokemon/Energy")
    elif pilot == "crustle" and "dusk ball" in name_lower:
        score = 58 if bench_count < 3 else 38
        purpose = "search"
        reasons.append("find Pokemon")
    elif pilot == "crustle" and "energy search" in name_lower:
        score = 52
        purpose = "search_energy"
        reasons.append("find Grass Energy")
    elif playstyle == "rush" and ("boss" in name_lower or "catcher" in name_lower):
        opponent_bench = _bench_count(obs_dict, _opponent_index(obs_dict))
        if opponent_bench <= 0:
            return None
        # Moderate score: the decision to actually gust is gated in MAIN on
        # whether it converts into a KO this turn (see _gust_enables_ko).
        score = 60 if "boss" in name_lower else 52
        purpose = "gust"
        reasons.append(f"rush gust bench={opponent_bench}")
    elif playstyle == "rush" and _is_stadium_card(card):
        # Damage/utility stadiums (e.g. Postwick) only get played when no attack,
        # gust, or strong setup is available, so the moderate score is fine.
        score = 46
        purpose = "stadium"
        reasons.append("rush stadium")
    elif "lillie" in name_lower and "pearl" in name_lower:
        if not _has_in_play_card(obs_dict, _your_index(obs_dict), main_attackers or {272}):
            return None
        score = 66
        purpose = "tool"
        reasons.append("protect Lillie's Pokemon")
    elif "mega signal" in name_lower:
        score = 68 if bench_count < 2 else 46
        purpose = "search"
        reasons.append("search Mega Evolution")
    elif "cyrano" in name_lower:
        score = 72 if bench_count < 2 or hand_count <= 4 else 45
        purpose = "search"
        reasons.append("search Pokemon ex")
    elif "lillie" in name_lower or "determination" in name_lower:
        if hand_count <= 3:
            score = 78
        elif hand_count <= 5:
            score = 62
        elif hand_count <= 7:
            score = 38
        else:
            score = 8
        purpose = "draw"
        reasons.append(f"refresh hand={hand_count}")
    elif "waitress" in name_lower:
        score = 70 if bench_count < 2 else 55
        purpose = "energy_accel"
        reasons.append("energy acceleration")
    elif "maximum belt" in name_lower:
        if 1158 in active_tools:
            score = -20
            reasons.append("active already has Maximum Belt")
        else:
            score = 58
            purpose = "tool"
            reasons.append("damage tool")
    elif _is_basic_pokemon(card) and bench_count < bench_max:
        score = 54 if bench_count == 0 else 42
        purpose = "bench_basic"
        reasons.append("bench Basic Pokemon")
        if pilot_profile and card_id in set(getattr(pilot_profile, "bench_early_ids", ())):
            score += 18
            reasons.append(f"pilot bench priority: {pilot_profile.name}")
        if pilot_profile and card_id in set(getattr(pilot_profile, "avoid_bench_ids", ())):
            score -= 30
            reasons.append(f"pilot bench caution: {pilot_profile.name}")

    if score == 0:
        if "draw" in text and hand_count <= 5:
            score = 52
            purpose = "draw"
            reasons.append("generic draw")
        elif "search your deck" in text and bench_count < 3:
            score = 50
            purpose = "search"
            reasons.append("generic setup search")
        elif "look at" in text and ("pokemon" in text or "pokémon" in text) and "hand" in text and bench_count < 3:
            # Dusk Ball-style "look at the bottom/top N cards, take a Pokemon".
            score = 48
            purpose = "search"
            reasons.append("generic look-and-take Pokemon search")
        elif "attach a basic energy" in text:
            score = 48
            purpose = "energy_accel"
            reasons.append("generic energy acceleration")

    if score <= 0:
        return None

    return ScoredAction(
        index=idx,
        score=score,
        reason=", ".join(reasons),
        detail={
            "kind": "play",
            "purpose": purpose,
            "cardId": card_id,
            "card": name,
            "hand_count": hand_count,
            "bench_count": bench_count,
        },
    )


def _score_plays(
    obs_dict: Any,
    options: list[Any],
    card_lookup: dict[int, Any],
    pilot_profile: Any | None = None,
) -> list[ScoredAction]:
    scored: list[ScoredAction] = []
    for idx, option in find_options_by_type(options, "PLAY"):
        action = _score_play(obs_dict, idx, option, card_lookup, pilot_profile=pilot_profile)
        if action is not None:
            scored.append(action)
    return sorted(scored, key=lambda action: action.score, reverse=True)


def _score_attach(
    obs_dict: Any,
    options: list[Any],
    card_lookup: dict[int, Any],
    pilot_profile: Any | None = None,
) -> ScoredAction | None:
    scored: list[ScoredAction] = []
    main_attackers = set(getattr(pilot_profile, "preferred_main_attacker_ids", ()) if pilot_profile else ())
    preferred_energy = set(getattr(pilot_profile, "preferred_energy_ids", ()) if pilot_profile else ())
    pilot = _pilot_name(pilot_profile)
    playstyle = str(getattr(pilot_profile, "playstyle", "") or "")
    for idx, option in find_options_by_type(options, "ATTACH"):
        target_id = _in_play_pokemon_id(
            obs_dict,
            _your_index(obs_dict),
            obj_get(option, "inPlayArea"),
            obj_get(option, "inPlayIndex"),
        )
        target_pokemon = _in_play_pokemon(
            obs_dict,
            _your_index(obs_dict),
            obj_get(option, "inPlayArea"),
            obj_get(option, "inPlayIndex"),
        )
        target_energy_count = len(obj_get(target_pokemon, "energies", []) or [])
        area_name = AREA_NAMES.get(obj_get(option, "inPlayArea"), obj_get(option, "inPlayArea"))
        is_active = area_name == "ACTIVE"
        active = _active_pokemon(obs_dict, _your_index(obs_dict))
        active_energy_count = len(obj_get(active, "energies", []) or [])
        attached_card_id = _hand_card_id_for_option(obs_dict, option)
        attached_card = card_lookup.get(attached_card_id) if attached_card_id is not None else None
        attached_name = _card_name(attached_card, str(attached_card_id or "unknown")).lower()

        if _is_tool_card(attached_card):
            if "lillie" in attached_name and "pearl" in attached_name:
                if target_id in main_attackers:
                    scored.append(
                        ScoredAction(
                            index=idx,
                            score=82,
                            reason=f"protect Lillie's attacker target={target_id}",
                            detail={
                                "kind": "attach_tool",
                                "target_id": target_id,
                                "attached_card_id": attached_card_id,
                            },
                        )
                    )
                continue
            if "maximum belt" in attached_name and is_active:
                scored.append(
                    ScoredAction(
                        index=idx,
                        score=62,
                        reason="damage tool to active",
                        detail={
                            "kind": "attach_tool",
                            "target_id": target_id,
                            "attached_card_id": attached_card_id,
                        },
                    )
                )
                continue
            if playstyle == "rush" and (
                "choice band" in attached_name
                or "choice belt" in attached_name
                or "amulet" in attached_name
                or "cape" in attached_name  # Hero's Cape (+100 HP survivability)
                or attached_name.endswith(" band")
            ):
                if is_active or target_id in main_attackers:
                    scored.append(
                        ScoredAction(
                            index=idx,
                            score=70,
                            reason=f"rush damage tool to attacker target={target_id}",
                            detail={
                                "kind": "attach_tool",
                                "target_id": target_id,
                                "attached_card_id": attached_card_id,
                            },
                        )
                    )
                continue
            if pilot == "crustle" and "handheld fan" in attached_name:
                if target_id == CRUSTLE_ID and is_active:
                    score = 78
                    reason = "energy disruption tool to active Crustle"
                elif is_active:
                    score = 52
                    reason = "energy disruption tool to active"
                else:
                    score = 16
                    reason = "low-value bench tool"
                scored.append(
                    ScoredAction(
                        index=idx,
                        score=score,
                        reason=reason,
                        detail={
                            "kind": "attach_tool",
                            "target_id": target_id,
                            "attached_card_id": attached_card_id,
                        },
                    )
                )
                continue
            continue

        if attached_card is not None and not _is_energy_card(attached_card):
            continue

        score = 40.0
        reasons = []
        if is_active:
            score += 16
            reasons.append("target active")
            if active_energy_count <= 0:
                score += 10
            elif active_energy_count == 1:
                score += 6
            elif active_energy_count >= 3:
                score -= 24
        if target_id in main_attackers:
            desired_energy = 3 if pilot == "crustle" and target_id == CRUSTLE_ID else 2
            reasons.append(f"pilot main attacker target={target_id}")
            if attached_card_id in preferred_energy:
                score += 72
                reasons.append(f"preferred energy={attached_card_id}")
            elif target_energy_count >= 1:
                score += 18
                reasons.append("colorless follow-up energy")
            else:
                score -= 10
                reasons.append(f"off-type first energy={attached_card_id}")
            if target_energy_count >= desired_energy:
                score -= 70
                reasons.append("main attacker already powered")
        elif pilot_profile and main_attackers:
            score -= 6
        if (
            pilot == "crustle"
            and target_id == PECHARUNT_ID
            and target_energy_count >= 2
            and not (_in_play_ids(obs_dict) & {DWEBBLE_ID, CRUSTLE_ID})
        ):
            score -= 60
            reasons.append("avoid over-loading Pecharunt with no Dwebble/Crustle in play")
        if obj_get(_current(obs_dict), "energyAttached", False):
            score -= 20
        scored.append(
            ScoredAction(
                index=idx,
                score=score,
                reason=", ".join(reasons) or "attach energy",
                detail={
                    "kind": "attach",
                    "target_id": target_id,
                    "attached_card_id": attached_card_id,
                    "active_energy": active_energy_count,
                    "target_energy": target_energy_count,
                },
            )
        )
    if scored:
        return sorted(scored, key=lambda action: action.score, reverse=True)[0]

    idx = choose_attach_to_active(options)
    if idx is None:
        return None
    return ScoredAction(index=idx, score=10, reason="fallback attach", detail={"kind": "attach"})


def _score_evolves(options: list[Any]) -> list[ScoredAction]:
    scored: list[ScoredAction] = []
    for idx, option in find_options_by_type(options, "EVOLVE"):
        in_play_area = obj_get(option, "inPlayArea")
        in_play_index = obj_get(option, "inPlayIndex")
        is_active = AREA_NAMES.get(in_play_area, in_play_area) == "ACTIVE" and in_play_index == 0
        scored.append(
            ScoredAction(
                index=idx,
                score=88 if is_active else 62,
                reason="evolve active" if is_active else "evolve board",
                detail={"kind": "evolve", "active": is_active},
            )
        )
    return sorted(scored, key=lambda action: action.score, reverse=True)


def _choose_no(options: list[Any]) -> int | None:
    matches = find_options_by_type(options, "NO")
    return matches[0][0] if matches else None


def _choose_active_target(options: list[Any]) -> int | None:
    for idx, option in enumerate(options):
        if AREA_NAMES.get(obj_get(option, "area"), obj_get(option, "area")) == "ACTIVE":
            return idx
        if AREA_NAMES.get(obj_get(option, "inPlayArea"), obj_get(option, "inPlayArea")) == "ACTIVE":
            return idx
    return 0 if options else None


def _choose_preferred_target(obs_dict: Any, options: list[Any], pilot_profile: Any | None = None) -> int | None:
    main_attackers = set(getattr(pilot_profile, "preferred_main_attacker_ids", ()) if pilot_profile else ())
    if main_attackers:
        for idx, option in enumerate(options):
            area = obj_get(option, "area", obj_get(option, "inPlayArea"))
            index = obj_get(option, "index", obj_get(option, "inPlayIndex"))
            target_id = _in_play_pokemon_id(obs_dict, _your_index(obs_dict), area, index)
            if target_id in main_attackers:
                return idx
    return _choose_active_target(options)


def _format_option(obs_dict: Any, idx: int, option: Any, attack_lookup: dict[int, Any], card_lookup: dict[int, Any]) -> dict:
    option_type = option_type_name(option)
    summary = {"index": idx, "type": option_type}
    if option_type == "ATTACK":
        attack_id = obj_get(option, "attackId")
        attack = attack_lookup.get(int(attack_id)) if attack_id is not None else None
        summary.update(
            {
                "attackId": attack_id,
                "attack": str(obj_get(attack, "name", attack_id)),
                "damage": obj_get(attack, "damage"),
            }
        )
    elif option_type == "PLAY":
        card_id = hand_card_id_for_play_option(obs_dict, option)
        card = card_lookup.get(card_id) if card_id is not None else None
        summary.update({"cardId": card_id, "card": _card_name(card, "unknown")})
    elif option_type == "ATTACH":
        summary.update(
            {
                "inPlayArea": AREA_NAMES.get(obj_get(option, "inPlayArea"), obj_get(option, "inPlayArea")),
                "inPlayIndex": obj_get(option, "inPlayIndex"),
            }
        )
    return summary


def _record_debug(
    debug_log: list[dict] | None,
    obs_dict: Any,
    options: list[Any],
    choice: list[int],
    reason: str,
    attack_lookup: dict[int, Any],
    card_lookup: dict[int, Any],
    scores: list[ScoredAction] | None = None,
) -> None:
    if debug_log is None:
        return
    current = _current(obs_dict) or {}
    chosen_options = [
        _format_option(obs_dict, idx, options[idx], attack_lookup, card_lookup)
        for idx in choice
        if isinstance(idx, int) and 0 <= idx < len(options)
    ]
    chosen_score = None
    if choice and scores:
        for score in scores:
            if score.index == choice[0]:
                chosen_score = round(score.score, 3)
                break
    debug_log.append(
        {
            "turn": obj_get(current, "turn"),
            "player": obj_get(current, "yourIndex"),
            "context": get_context(obs_dict),
            "options": [
                _format_option(obs_dict, idx, option, attack_lookup, card_lookup)
                for idx, option in enumerate(options)
            ],
            "choice": choice,
            "chosenOptionIndex": choice[0] if choice else None,
            "chosenOptionType": chosen_options[0]["type"] if chosen_options else None,
            "chosenOptions": chosen_options,
            "chosenScore": chosen_score,
            "reason": reason,
            "scores": [
                {"index": score.index, "score": round(score.score, 3), "reason": score.reason, **score.detail}
                for score in (scores or [])
            ],
        }
    )


def _finish(
    obs_dict: Any,
    choices: list[int],
    reason: str,
    options: list[Any],
    attack_lookup: dict[int, Any],
    card_lookup: dict[int, Any],
    debug_log: list[dict] | None,
    scores: list[ScoredAction] | None = None,
) -> list[int]:
    legal = finish_choice(obs_dict, choices)
    _record_debug(debug_log, obs_dict, options, legal, reason, attack_lookup, card_lookup, scores)
    return legal


def agent(
    obs_dict: Any,
    deck: list[int] | None = None,
    attack_lookup: dict[int, Any] | None = None,
    card_lookup: dict[int, Any] | None = None,
    debug_log: list[dict] | None = None,
    pilot_profile: Any | None = None,
) -> list[int]:
    """KO-aware heuristic v1 with conservative setup sequencing."""
    if is_initial_observation(obs_dict):
        return load_agent_deck(deck)

    default_attacks, default_cards = _load_cg_metadata()
    attack_lookup = attack_lookup if attack_lookup is not None else default_attacks
    card_lookup = card_lookup if card_lookup is not None else default_cards

    options = get_options(obs_dict)
    context = (get_context(obs_dict) or "").upper()
    playstyle = str(getattr(pilot_profile, "playstyle", "") or "")

    # Gust target select (raw context 3): a gust card is asking which benched
    # opponent Pokemon to drag into the Active Spot. Rush pilots pick the most
    # KO-able target (2-prize ex first) instead of the default first option.
    if context == "3" and playstyle == "rush":
        best_damage = _active_best_attack_damage(obs_dict, deck, attack_lookup, card_lookup)
        target_idx = _best_gust_target_index(obs_dict, options, best_damage, card_lookup)
        if target_idx is not None:
            return _finish(
                obs_dict,
                [target_idx],
                "rush gust target: most KO-able opponent Pokemon",
                options,
                attack_lookup,
                card_lookup,
                debug_log,
            )

    if context in {"IS_FIRST", "ISFIRST"}:
        no_idx = _choose_no(options)
        if no_idx is not None:
            return _finish(obs_dict, [no_idx], "go second", options, attack_lookup, card_lookup, debug_log)

    if context in {"DRAW_COUNT", "DRAWCOUNT"}:
        number_idx = choose_number_max(options)
        if number_idx is not None:
            return _finish(obs_dict, [number_idx], "draw max", options, attack_lookup, card_lookup, debug_log)

    if context in {
        "SETUP_BENCH_POKEMON",
        "SETUPBENCHPOKEMON",
        "TO_HAND",
        "TOHAND",
        "TO_BENCH",
        "TOBENCH",
        "TO_FIELD",
        "TOFIELD",
        "TO_ACTIVE",
        "TOACTIVE",
    }:
        return _finish(
            obs_dict,
            _choose_gain_targets(obs_dict, context, options, pilot_profile, card_lookup),
            f"search/setup: take best {context.lower()} targets (Dwebble/line first)",
            options,
            attack_lookup,
            card_lookup,
            debug_log,
        )

    if context in {"ATTACH_TO", "ATTACHTO"}:
        target_idx = _choose_preferred_target(obs_dict, options, pilot_profile)
        if target_idx is not None:
            return _finish(
                obs_dict,
                [target_idx],
                "attach effect to active",
                options,
                attack_lookup,
                card_lookup,
                debug_log,
            )

    if context == "MAIN":
        attack_scores = _score_attacks(obs_dict, options, attack_lookup, card_lookup, deck, pilot_profile=pilot_profile)
        play_scores = _score_plays(obs_dict, options, card_lookup, pilot_profile=pilot_profile)
        evolve_scores = _score_evolves(options)
        attach_score = _score_attach(obs_dict, options, card_lookup, pilot_profile=pilot_profile)
        end_idx = choose_end(options)
        all_scores = attack_scores + play_scores + evolve_scores + ([attach_score] if attach_score else [])

        ko_attacks = [score for score in attack_scores if score.detail.get("ko")]
        if ko_attacks:
            best = ko_attacks[0]
            return _finish(
                obs_dict,
                [best.index],
                f"take KO attack: {best.reason}",
                options,
                attack_lookup,
                card_lookup,
                debug_log,
                all_scores,
            )

        best_attack = attack_scores[0] if attack_scores else None
        best_play = play_scores[0] if play_scores else None
        best_evolve = evolve_scores[0] if evolve_scores else None
        bench_count = _bench_count(obs_dict, _your_index(obs_dict))
        in_play_count = _in_play_count(obs_dict, _your_index(obs_dict))
        pilot = _pilot_name(pilot_profile)
        gust_play = next((s for s in play_scores if s.detail.get("purpose") == "gust"), None)

        if best_evolve and best_evolve.score >= 60:
            return _finish(
                obs_dict,
                [best_evolve.index],
                f"evolve before non-KO line: {best_evolve.reason}",
                options,
                attack_lookup,
                card_lookup,
                debug_log,
                all_scores,
            )

        if best_play and best_play.detail.get("purpose") == "bench_basic" and bench_count == 0:
            return _finish(
                obs_dict,
                [best_play.index],
                f"bench insurance before attacking: {best_play.reason}",
                options,
                attack_lookup,
                card_lookup,
                debug_log,
                all_scores,
            )

        if (
            pilot == "crustle"
            and best_play
            and best_play.detail.get("purpose") == "bench_basic"
            and in_play_count < 3
        ):
            return _finish(
                obs_dict,
                [best_play.index],
                f"crustle preserve backup basic: {best_play.reason}",
                options,
                attack_lookup,
                card_lookup,
                debug_log,
                all_scores,
            )

        if (
            pilot_profile
            and best_play
            and best_play.detail.get("purpose") == "bench_basic"
            and in_play_count < 3
            and (best_attack is None or best_attack.score < 70)
        ):
            return _finish(
                obs_dict,
                [best_play.index],
                f"pilot bench before weak attack: {best_play.reason}",
                options,
                attack_lookup,
                card_lookup,
                debug_log,
                all_scores,
            )

        if (
            best_play
            and best_play.detail.get("purpose") in {"search", "energy_accel"}
            and bench_count == 0
            and (best_attack is None or best_attack.score < 90)
        ):
            return _finish(
                obs_dict,
                [best_play.index],
                f"setup empty bench before attacking: {best_play.reason}",
                options,
                attack_lookup,
                card_lookup,
                debug_log,
                all_scores,
            )

        if best_play and best_play.score >= 70 and (best_attack is None or best_attack.score < 20):
            return _finish(
                obs_dict,
                [best_play.index],
                f"use setup play: {best_play.reason}",
                options,
                attack_lookup,
                card_lookup,
                debug_log,
                all_scores,
            )

        if (
            pilot == "crustle"
            and best_play
            and best_play.detail.get("purpose") in {"energy_denial", "gust_trap", "wall_stadium"}
            and (best_attack is None or best_attack.score < 150)
        ):
            return _finish(
                obs_dict,
                [best_play.index],
                f"crustle disruption before non-KO attack: {best_play.reason}",
                options,
                attack_lookup,
                card_lookup,
                debug_log,
                all_scores,
            )

        if (
            pilot_profile
            and best_play
            and best_play.detail.get("purpose") in {"search", "draw", "energy_accel"}
            and (best_attack is None or best_attack.score < 45)
        ):
            return _finish(
                obs_dict,
                [best_play.index],
                f"pilot setup before weak attack: {best_play.reason}",
                options,
                attack_lookup,
                card_lookup,
                debug_log,
                all_scores,
            )

        if (
            pilot_profile
            and attach_score
            and attach_score.score >= 80
            and (best_attack is None or best_attack.score < 90)
        ):
            return _finish(
                obs_dict,
                [attach_score.index],
                f"pilot build best attacker before weak attack: {attach_score.reason}",
                options,
                attack_lookup,
                card_lookup,
                debug_log,
                all_scores,
            )

        # Gust-the-bench: fire when our best (effective) attack can KO a benched
        # target, OR when our Active is wall-blocked (e.g. ex vs Crustle) but our
        # printed damage would still KO a frail Bench Pokemon if we drag it up.
        gust_damage = best_attack.detail.get("damage", 0) if best_attack else 0
        if playstyle == "rush" and _our_active_is_ex(obs_dict, card_lookup) and _opponent_active_walls_ex(obs_dict, card_lookup):
            gust_damage = max(float(gust_damage or 0), _reachable_printed_damage(obs_dict, card_lookup, attack_lookup))
        if (
            playstyle == "rush"
            and gust_play is not None
            and _any_bench_koable(obs_dict, float(gust_damage or 0))
        ):
            return _finish(
                obs_dict,
                [gust_play.index],
                f"rush gust to enable KO: {gust_play.reason}",
                options,
                attack_lookup,
                card_lookup,
                debug_log,
                all_scores,
            )

        if best_attack and best_attack.score >= 5:
            return _finish(
                obs_dict,
                [best_attack.index],
                f"best non-KO attack: {best_attack.reason}",
                options,
                attack_lookup,
                card_lookup,
                debug_log,
                all_scores,
            )

        if attach_score and attach_score.score >= 40:
            return _finish(
                obs_dict,
                [attach_score.index],
                f"build energy: {attach_score.reason}",
                options,
                attack_lookup,
                card_lookup,
                debug_log,
                all_scores,
            )

        if best_play and best_play.score >= 45:
            return _finish(
                obs_dict,
                [best_play.index],
                f"late useful play: {best_play.reason}",
                options,
                attack_lookup,
                card_lookup,
                debug_log,
                all_scores,
            )

        if attach_score and attach_score.score > 0:
            return _finish(
                obs_dict,
                [attach_score.index],
                f"fallback attach: {attach_score.reason}",
                options,
                attack_lookup,
                card_lookup,
                debug_log,
                all_scores,
            )

        if best_attack:
            return _finish(
                obs_dict,
                [best_attack.index],
                f"fallback attack over end: {best_attack.reason}",
                options,
                attack_lookup,
                card_lookup,
                debug_log,
                all_scores,
            )

        if end_idx is not None:
            return _finish(obs_dict, [end_idx], "end: no useful action", options, attack_lookup, card_lookup, debug_log, all_scores)

    choice = safe_choice(obs_dict)
    _record_debug(debug_log, obs_dict, options, choice, "safe fallback", attack_lookup, card_lookup)
    return choice


def make_agent(
    deck: list[int],
    attack_lookup: dict[int, Any] | None = None,
    card_lookup: dict[int, Any] | None = None,
    debug_log: list[dict] | None = None,
    pilot_profile: Any | None = None,
    deck_name: str | None = None,
):
    if pilot_profile is None:
        try:
            from deck_pilots import select_pilot_profile

            pilot_profile = select_pilot_profile(deck, deck_name=deck_name)
        except Exception:
            pilot_profile = None

    def _agent(obs_dict: Any) -> list[int]:
        return agent(
            obs_dict,
            deck=deck,
            attack_lookup=attack_lookup,
            card_lookup=card_lookup,
            debug_log=debug_log,
            pilot_profile=pilot_profile,
        )

    return _agent
