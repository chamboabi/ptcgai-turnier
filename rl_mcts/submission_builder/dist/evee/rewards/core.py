"""Primitive building blocks shared by all reward files.

Import from here when writing a new compound reward module:
    from rewards.core import (
        RewardFn, RewardShapeFn, RewardTerminalFn,
        compose_shapes,
        win_loss_terminal, prize_pressure_shape, ...
    )
"""

import functools
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from cg.api import AreaType, CardData, CardType, LogType, Observation, all_attack, all_card_data

ENERGY_IDS = frozenset(
    cd.cardId for cd in all_card_data() if cd.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)
)
ATTACK_BY_ID = {a.attackId: a for a in all_attack()}
CARD_BY_ID: CardData = {cd.cardId: cd for cd in all_card_data()}
ENERGY_NEEDED = {
    cd.cardId: max(
        (len(ATTACK_BY_ID[aid].energies) for aid in cd.attacks if aid in ATTACK_BY_ID),
        default=0,
    )
    for cd in all_card_data()
}

RewardTerminalFn = Callable[[Observation, int], float]
RewardShapeFn = Callable[[Observation, int, float], float]
# Builds a per-turn shape from the search-root obs (freezes baselines). See
# make_compound_shape / rewards.shapes.make_base_shape.
ShapeFactoryFn = Callable[[Observation, int], RewardShapeFn]


@dataclass
class RewardFn:
    terminal: RewardTerminalFn
    shape: RewardShapeFn
    # Optional: when set, mcts_agent rebuilds `shape` from this at each search
    # root so absolute shapes score deltas from a per-turn baseline.
    shape_factory: ShapeFactoryFn | None = None


# ============================================================
#  Terminals
# ============================================================


def win_loss_terminal(obs: Observation, your_index: int) -> float:
    """+1 win, -1 loss, 0 draw."""
    result = obs.current.result
    if result == 2:
        return 0.0
    return 1.0 if result == your_index else -1.0


def fast_win_terminal(obs: Observation, your_index: int) -> float:
    """Win/loss scaled toward earlier turns: faster wins pay more, slower less."""
    result = obs.current.result
    if result == 2:
        return 0.0
    sign = 1.0 if result == your_index else -1.0
    turn_factor = max(0.0, 1.0 - obs.current.turn / 60.0)
    return sign * (0.8 + 0.2 * turn_factor)


# ============================================================
#  Misc shapes
# ============================================================


def identity_shape(obs: Observation, your_index: int, nn_value: float) -> float:
    """Pass nn_value through unchanged (no shaping)."""
    return nn_value


def win_game_shape(obs: Observation, your_index: int, nn_value: float, weight: float = 1.0) -> float:
    """Add a flat bonus when you have won the game; symmetric penalty on a loss.

    Draws and unfinished games pass nn_value through unchanged.
    """
    result = obs.current.result
    if result == your_index:
        return nn_value + weight
    if result == 1 - your_index:
        return nn_value - weight
    return nn_value


# ============================================================
#  Prize shapes
# ============================================================


def prize_pressure_shape(obs: Observation, your_index: int, nn_value: float, weight: float = 0.1) -> float:
    """Blend nn_value with your prize-card lead (fewer prizes left vs opponent)."""
    state = obs.current
    your_prizes_left = len(state.players[your_index].prize)
    opp_prizes_left = len(state.players[1 - your_index].prize)
    prize_adv = (opp_prizes_left - your_prizes_left) / 6.0
    return nn_value * (1.0 - weight) + prize_adv * weight


# ============================================================
#  Energy shapes
# ============================================================


def attached_energy_shape(
    obs: Observation, your_index: int, nn_value: float, weight: float = 0.1, cap: int = 8
) -> float:
    """Blend nn_value with total energy attached across your board, capped at `cap`."""
    you = obs.current.players[your_index]
    attached = sum(len(p.energyCards) for p in board(you) if p is not None)
    loaded = min(1.0, attached / cap)
    return nn_value * (1.0 - weight) + loaded * weight


def attached_energy_uncapped_shape(obs: Observation, your_index: int, nn_value: float, weight: float = 0.1) -> float:
    """Add a flat bonus per energy attached across your board (no cap)."""
    you = obs.current.players[your_index]
    attached = sum(len(p.energyCards) for p in board(you) if p is not None)
    return nn_value + attached * weight


def attach_energy_type_shape(
    obs: Observation, your_index: int, nn_value: float, poke_id: int, energy_type_code: int, weight: float = 0.1
) -> float:
    """Reward each matching-type energy attached to the given pokemon on your board."""
    you = obs.current.players[your_index]
    matched = 0
    for poke in board(you):
        if poke is None or poke.id != poke_id:
            continue
        matched += sum(1 for e in poke.energyCards if energy_type(e) == energy_type_code)
    return nn_value + matched * weight


def attach_energy_capped_shape(
    obs: Observation,
    your_index: int,
    nn_value: float,
    poke_id: int,
    energy_type_code: int,
    cap: int,
    weight: float = 0.1,
) -> float:
    """Reward matching-type energy attached to the given pokemon, counted up to cap."""
    you = obs.current.players[your_index]
    matched = 0
    for poke in board(you):
        if poke is None or poke.id != poke_id:
            continue
        matched += sum(1 for e in poke.energyCards if energy_type(e) == energy_type_code)
    return nn_value + min(matched, cap) * weight


# ============================================================
#  Attack-energy shapes
# ============================================================


COLORLESS = 0


def attack_energy_match_shape(
    obs: Observation, your_index: int, nn_value: float, poke_id: int, attack_index: int, weight: float = 0.1
) -> float:
    """Reward each attached energy that fills a slot of the pokemon's chosen attack cost.

    Specific types matched first; colorless slots filled by any leftover energy.
    Caps at the attack's cost (over-attaching past the need pays nothing).
    """
    cd = CARD_BY_ID.get(poke_id)
    if cd is None or not (0 <= attack_index < len(cd.attacks)):
        return nn_value
    atk = ATTACK_BY_ID.get(cd.attacks[attack_index])
    if atk is None:
        return nn_value
    poke = find(board(obs.current.players[your_index]), poke_id)
    if poke is None:
        return nn_value

    remaining = Counter(energy_type(e) for e in poke.energyCards)
    matched = 0
    colorless = 0
    for need in atk.energies:
        if need == COLORLESS:
            colorless += 1
        elif remaining.get(need, 0) > 0:
            remaining[need] -= 1
            matched += 1
    leftover = sum(c for c in remaining.values() if c > 0)
    matched += min(colorless, leftover)
    return nn_value + matched * weight


def attack_energy_overload_shape(
    obs: Observation, your_index: int, nn_value: float, poke_id: int, attack_index: int, weight: float = 0.1
) -> float:
    """Like attack_energy_match_shape but the specific types are uncapped.

    Energy matching a needed specific (non-colorless) type always counts, including
    extras past the attack cost. Colorless slots are filled by leftover energy of any
    other type, but only up to the number of colorless slots (colorless is not overloaded).
    """
    cd = CARD_BY_ID.get(poke_id)
    if cd is None or not (0 <= attack_index < len(cd.attacks)):
        return nn_value
    atk = ATTACK_BY_ID.get(cd.attacks[attack_index])
    if atk is None:
        return nn_value
    poke = find(board(obs.current.players[your_index]), poke_id)
    if poke is None:
        return nn_value

    specific = {need for need in atk.energies if need != COLORLESS}
    colorless_slots = sum(1 for need in atk.energies if need == COLORLESS)
    matched = 0
    leftover = 0
    for e in poke.energyCards:
        et = energy_type(e)
        if et is None:
            continue
        if et in specific:
            matched += 1
        else:
            leftover += 1
    matched += min(colorless_slots, leftover)
    return nn_value + matched * weight


# ============================================================
#  Damage shapes
# ============================================================


def damage_capped_shape(
    obs: Observation, your_index: int, nn_value: float, weight: float = 0.1, cap: int = 300
) -> float:
    """Blend nn_value with total damage dealt across opponent's board, capped at `cap`."""
    opp = obs.current.players[1 - your_index]
    dealt = sum(max(0, p.maxHp - p.hp) for p in board(opp) if p is not None)
    signal = min(1.0, dealt / cap)
    return nn_value * (1.0 - weight) + signal * weight


def damage_uncapped_shape(obs: Observation, your_index: int, nn_value: float, weight: float = 0.1) -> float:
    """Add a flat bonus per HP of damage dealt across opponent's board (no cap)."""
    opp = obs.current.players[1 - your_index]
    dealt = sum(max(0, p.maxHp - p.hp) for p in board(opp) if p is not None)
    return nn_value + dealt * weight


def damage_taken_shape(
    obs: Observation, your_index: int, nn_value: float, weight: float = 0.1, cap: int = 300
) -> float:
    """Blend nn_value with how little damage your own board has taken (more HP intact -> higher)."""
    you = obs.current.players[your_index]
    taken = sum(max(0, p.maxHp - p.hp) for p in board(you) if p is not None)
    signal = 1.0 - min(1.0, taken / cap)
    return nn_value * (1.0 - weight) + signal * weight


def damage_taken_uncapped_shape(obs: Observation, your_index: int, nn_value: float, weight: float = 0.1) -> float:
    """Subtract a flat penalty per HP of damage taken across your own board (no cap)."""
    you = obs.current.players[your_index]
    taken = sum(max(0, p.maxHp - p.hp) for p in board(you) if p is not None)
    return nn_value - taken * weight


# ============================================================
#  Opponent shapes
# ============================================================


def opp_discarded_energy_shape(
    obs: Observation, your_index: int, nn_value: float, weight: float = 0.1, cap: int = 8
) -> float:
    """Blend nn_value with opponent energy sent to discard (KO'd loaded mons), capped at `cap`."""
    opp = obs.current.players[1 - your_index]
    trashed = count_energy(opp.discard)
    signal = min(1.0, trashed / cap)
    return nn_value * (1.0 - weight) + signal * weight


# ============================================================
#  Niche shapes (probably only used for one pokemon/deck)
# ============================================================


def high_deck_energy_shape(
    obs: Observation, your_index: int, nn_value: float, weight: float = 0.1, deck_energy_total: int = 12
) -> float:
    """Blend nn_value with energy still left in deck (rewards keeping reserves un-drawn)."""
    you = obs.current.players[your_index]
    seen = count_energy(you.hand) + count_energy(you.discard)
    for poke in board(you):
        if poke is not None:
            seen += count_energy(poke.energyCards)
    energy_left = max(0, deck_energy_total - seen)
    reserve = energy_left / deck_energy_total
    return nn_value * (1.0 - weight) + reserve * weight


# ============================================================
#  Board shapes
# ============================================================


def race_card_shape(obs: Observation, your_index: int, nn_value: float, poke_id: int, weight: float = 0.1) -> float:
    """Reward getting a specific card into play on your field (race it down, keep it out).

    Flat bonus while that card is present among your active/bench pokemon.
    """
    present = find(board(obs.current.players[your_index]), poke_id) is not None
    return nn_value + (weight if present else 0.0)


def opp_active_is_ex(obs: Observation, your_index: int) -> bool:
    """True if the opponent's active pokemon is a Pokemon ex (includes Mega ex)."""
    active = obs.current.players[1 - your_index].active
    if not active or active[0] is None:
        return False
    cd = CARD_BY_ID.get(active[0].id)
    if cd is not None and cd.ex:
        print("I see a EX card called:", cd.name)

    return cd is not None and cd.ex


def race_vs_ex_shape(obs: Observation, your_index: int, nn_value: float, poke_id: int, weight: float = 0.1) -> float:
    """Reward racing a specific card into play, but only while the opponent's
    active is a Pokemon ex (the counter only matters against an ex)."""
    if not opp_active_is_ex(obs, your_index):
        return nn_value
    return race_card_shape(obs, your_index, nn_value, poke_id, weight)


# ============================================================
#  Hand shapes
# ============================================================


def hand_size_range_shape(
    obs: Observation, your_index: int, nn_value: float, lo: int = 4, hi: int = 6, weight: float = 0.1
) -> float:
    """Reward keeping your hand size within [lo, hi] inclusive (avoid flooding or emptying out)."""
    hand = obs.current.players[your_index].handCount
    return nn_value + (weight if lo <= hand <= hi else 0.0)


def small_opp_hand_shape(
    obs: Observation, your_index: int, nn_value: float, weight: float = 0.1, cap: int = 10
) -> float:
    """Reward the opponent having a small hand (starved of resources).

    Blends nn_value with 1 - oppHandCount/cap: empty opp hand -> full signal, cap+ cards -> 0.
    """
    opp_hand = obs.current.players[1 - your_index].handCount
    signal = 1.0 - min(1.0, opp_hand / cap)
    return nn_value * (1.0 - weight) + signal * weight


def big_hand_penalty_shape(
    obs: Observation, your_index: int, nn_value: float, weight: float = 0.1, threshold: int = 6
) -> float:
    """Penalize holding a big hand on your own turn (cards you failed to play out).

    Subtracts weight per card above threshold. Only applies while it is your turn.
    """
    if turn_owner(obs.current) != your_index:
        return nn_value
    hand = obs.current.players[your_index].handCount
    excess = max(0, hand - threshold)
    return nn_value - excess * weight


def play_card_penalty_shape(obs: Observation, your_index: int, nn_value: float, weight: float = 0.1) -> float:
    """Penalize each card you play this step (discourages over-committing resources).

    Counts this step's PLAY log events by you; subtracts weight per card played.
    """
    played = sum(1 for log in obs.logs if log.type == LogType.PLAY and log.playerIndex == your_index)
    return nn_value - played * weight


def search_card_shape(
    obs: Observation, your_index: int, nn_value: float, target: int | None = None, weight: float = 0.1
) -> float:
    """Reward searching a card out of your deck into hand (tutor effects, not normal draws).

    Counts this step's MOVE_CARD log events from your DECK to HAND. If target is given,
    only that card id counts; otherwise any searched card counts.
    """
    got = sum(
        1
        for log in obs.logs
        if log.type == LogType.MOVE_CARD
        and log.playerIndex == your_index
        and log.fromArea == AreaType.DECK
        and log.toArea == AreaType.HAND
        and (target is None or log.cardId == target)
    )
    return nn_value + got * weight


def opp_hand_discard_shape(obs: Observation, your_index: int, nn_value: float, weight: float = 0.1) -> float:
    """Reward forcing the opponent to discard hand cards on your own turn (disruption).

    Counts this step's log events moving an opponent card from their HAND to DISCARD.
    Only applies while it is your turn.
    """
    if turn_owner(obs.current) != your_index:
        return nn_value
    opp = 1 - your_index
    discarded = sum(
        1
        for log in obs.logs
        if log.playerIndex == opp and log.fromArea == AreaType.HAND and log.toArea == AreaType.DISCARD
    )
    return nn_value + discarded * weight


def opp_target_leaves_field_shape(
    obs: Observation, your_index: int, nn_value: float, poke_id: int, weight: float = 0.1
) -> float:
    """Reward when the target opponent pokemon leaves the field (KO'd, scooped, returned).

    Counts this step's log events moving that card off ACTIVE/BENCH to a non-field area.
    Active<->bench switches stay on the field and do not count.
    """
    opp = 1 - your_index
    field = (AreaType.ACTIVE, AreaType.BENCH)
    left = sum(
        1
        for log in obs.logs
        if log.playerIndex == opp and log.cardId == poke_id and log.fromArea in field and log.toArea not in field
    )
    return nn_value + left * weight


def opp_target_devolves_shape(
    obs: Observation, your_index: int, nn_value: float, poke_id: int, weight: float = 0.1
) -> float:
    """Reward devolving the target opponent pokemon (knock it back a stage).

    Counts this step's DEVOLVE log events on the opponent matching the target card.
    """
    opp = 1 - your_index
    devolved = sum(
        1
        for log in obs.logs
        if log.type == LogType.DEVOLVE and log.playerIndex == opp and poke_id in (log.cardId, log.cardIdBefore)
    )
    return nn_value + devolved * weight


def opp_evolved_devolves_shape(obs: Observation, your_index: int, nn_value: float, weight: float = 0.1) -> float:
    """Reward devolving any evolved opponent pokemon (Stage 1 or Stage 2).

    Counts this step's DEVOLVE log events on the opponent whose devolved card is an
    evolved (Stage 1 / Stage 2) pokemon. More than one devolve in a step each count.
    """
    opp = 1 - your_index
    count = 0
    for log in obs.logs:
        if log.type != LogType.DEVOLVE or log.playerIndex != opp:
            continue
        candidates = (log.cardIdBefore, log.cardId, log.cardIdTarget)
        if any((cd := CARD_BY_ID.get(cid)) is not None and (cd.stage1 or cd.stage2) for cid in candidates):
            count += 1
    return nn_value + count * weight


# ============================================================
#  Helpers
# ============================================================


def turn_owner(state) -> int | None:
    """Index of the player whose turn it is, or None if undetermined.

    Odd turns belong to the starting player, even turns to the other.
    """
    if state.firstPlayer < 0 or state.turn < 1:
        return None
    return state.firstPlayer if state.turn % 2 == 1 else 1 - state.firstPlayer


def card_id(card) -> int:
    """Card id from a card object or a raw int id."""
    return card if isinstance(card, int) else card.id


def count_energy(cards: list | None) -> int:
    """Count energy cards in a list (objects or raw ids); None -> 0."""
    return sum(1 for c in (cards or []) if card_id(c) in ENERGY_IDS)


def energy_type(card) -> int | None:
    """Energy type code of an energy card (accepts object or raw card id). None if not an energy."""
    if card is None:
        return None
    cid = card if isinstance(card, int) else card_id(card)
    if cid not in ENERGY_IDS:
        return None
    cd = CARD_BY_ID.get(cid)
    return cd.energyType if cd is not None else None


def attacks_of(poke) -> list:
    """Attack objects for a pokemon. Accepts a board/hand object or a raw card id."""
    if poke is None:
        return []
    cid = poke if isinstance(poke, int) else card_id(poke)
    cd = CARD_BY_ID.get(cid)
    if cd is None:
        return []
    return [ATTACK_BY_ID[aid] for aid in cd.attacks if aid in ATTACK_BY_ID]


def board(player) -> list:
    """Player's in-play pokemon: [active, *bench]. Active may be None."""
    active = player.active[0] if player.active else None
    return [active, *player.bench]


def find(b, cid):
    """First pokemon on board `b` with matching card id, or None."""
    for p in b:
        if p is not None and p.id == cid:
            return p
    return None


# ============================================================
#  Composition
# ============================================================


def compose_shapes(*shapes: RewardShapeFn) -> RewardShapeFn:
    """Chain shapes: each one's output feeds the next as nn_value."""

    def shape(obs: Observation, your_index: int, nn_value: float) -> float:
        v = nn_value
        for s in shapes:
            v = s(obs, your_index, v)
        return v

    return shape


def _shape_contribution(shape_fn: RewardShapeFn, obs: Observation, your_index: int, weight: float) -> float:
    """The signal*weight a shape adds on its own (nn_value=0 isolates it).

    Works for both shape forms: blend (nn*(1-w)+sig*w) and additive (nn+amt*w)
    both collapse to the weighted signal when nn_value is 0.
    """
    return shape_fn(obs, your_index, 0.0, weight=weight)


def _delta_shape(shape_fn: RewardShapeFn, weight: float, baselines: dict[int, float]) -> RewardShapeFn:
    """Wrap an absolute shape to score current contribution minus a frozen baseline.

    `baselines` holds one frozen contribution per player index, because mcts shapes
    each node from the side-to-move's perspective (which flips during search); the
    delta is taken against the baseline for that same side.
    """

    def shape(obs: Observation, your_index: int, nn_value: float) -> float:
        now = shape_fn(obs, your_index, 0.0, weight=weight)
        return nn_value + (now - baselines[your_index])

    return shape


def make_compound_shape(weights, needs_baseline, obs0: Observation, your_index: int) -> RewardShapeFn:
    """Build a compound shape for one agent call, freezing baselines from obs0.

    `weights` maps each shape fn to its weight; `needs_baseline` is the subset of
    those keys that read absolute board state and must be scored as a delta from a
    baseline snapshotted at the search root. Per-step-log and outcome shapes are
    already deltas, so they are left as plain weighted shapes.

    Baselines are frozen for BOTH player perspectives (the search shapes opponent
    nodes from the opponent's side-to-move view), so the `your_index` argument is
    unused here and kept only to satisfy the ShapeFactoryFn signature.

    Call once per turn at the search root; reuse the returned fn for every MCTS
    leaf eval. Re-snapshot next turn.
    """
    parts = []
    for fn, w in weights.items():
        if fn in needs_baseline:
            baselines = {i: _shape_contribution(fn, obs0, i, w) for i in (0, 1)}
            parts.append(_delta_shape(fn, w, baselines))
        else:
            parts.append(functools.partial(fn, weight=w))
    return compose_shapes(*parts)
