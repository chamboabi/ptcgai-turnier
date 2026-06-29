from dataclasses import dataclass, field
from typing import Callable

from cg.api import CardType, Observation, all_attack, all_card_data

# card IDs that are energy (basic or special). Built once at import.
_ENERGY_IDS = frozenset(
    cd.cardId for cd in all_card_data() if cd.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)
)

# card ID -> energy needed for its costliest attack. Built once at import.
_ATTACK_BY_ID = {a.attackId: a for a in all_attack()}
_ENERGY_NEEDED = {
    cd.cardId: max(
        (len(_ATTACK_BY_ID[aid].energies) for aid in cd.attacks if aid in _ATTACK_BY_ID),
        default=0,
    )
    for cd in all_card_data()
}

RewardTerminalFn = Callable[[Observation, int], float]
RewardShapeFn = Callable[[Observation, int, float], float]


@dataclass
class RewardFn:
    terminal: RewardTerminalFn
    shape: RewardShapeFn


# --- terminals ---


def _win_loss_terminal(obs: Observation, your_index: int) -> float:
    result = obs.current.result
    if result == 2:
        return 0.0
    return 1.0 if result == your_index else -1.0


def _fast_win_terminal(obs: Observation, your_index: int) -> float:
    result = obs.current.result
    if result == 2:
        return 0.0
    sign = 1.0 if result == your_index else -1.0
    # faster win = higher reward; cap at 60 turns
    turn_factor = max(0.0, 1.0 - obs.current.turn / 60.0)
    return sign * (0.8 + 0.2 * turn_factor)


# --- shapes ---


def _identity_shape(obs: Observation, your_index: int, nn_value: float) -> float:
    return nn_value


def _prize_pressure_shape(obs: Observation, your_index: int, nn_value: float) -> float:
    state = obs.current
    your_prizes_left = len(state.players[your_index].prize)
    opp_prizes_left = len(state.players[1 - your_index].prize)
    # positive when opponent has more prizes left (= you took more)
    prize_adv = (opp_prizes_left - your_prizes_left) / 6.0
    return nn_value * 0.9 + prize_adv * 0.1


def _card_id(card) -> int:
    """Accept a raw card-ID int or a Card/Pokemon object, return its ID."""
    return card if isinstance(card, int) else card.id


def _count_energy(cards: list | None) -> int:
    """Count energy cards. Accepts lists of Card objects or raw card-ID ints.

    Hidden zones (e.g. the opponent's hand) come through as None -> count 0.
    """
    return sum(1 for c in (cards or []) if _card_id(c) in _ENERGY_IDS)


def _board(player) -> list:
    """Pokemon in play: active (if any) + bench.

    The API exposes `active` as a length-0/1 list (`list[Pokemon | None]`), so
    unwrap it to a single Pokemon-or-None before chaining with the bench.
    """
    active = player.active[0] if player.active else None
    return [active, *player.bench]


# total energy cards in YOUR decklist. Set this to match your deck.
DECK_ENERGY_TOTAL = 12


def _high_deck_energy_shape(obs: Observation, your_index: int, nn_value: float) -> float:
    """Higher value when more energy cards remain in your deck.

    Deck contents are hidden (only deckCount is known), so count the energy
    you CAN see — hand, discard, attached to your Pokemon — and subtract from
    your decklist's known energy total to estimate energy left in deck.
    """
    you = obs.current.players[your_index]

    seen = _count_energy(you.hand) + _count_energy(you.discard)
    for poke in _board(you):
        if poke is not None:
            seen += _count_energy(poke.energyCards)

    energy_left = max(0, DECK_ENERGY_TOTAL - seen)
    # 1.0 when all energy still in deck, 0.0 when none left
    reserve = energy_left / DECK_ENERGY_TOTAL

    return nn_value * 0.9 + reserve * 0.1


# normalization cap for total attached energy across your board
ATTACHED_ENERGY_CAP = 8


def _attached_energy_shape(obs: Observation, your_index: int, nn_value: float) -> float:
    """Higher value the more energy is attached to your Pokemon.

    Counts energy attached to active + bench, normalized by a board cap.
    """
    you = obs.current.players[your_index]

    attached = 0
    for poke in _board(you):
        if poke is not None:
            attached += len(poke.energyCards)

    loaded = min(1.0, attached / ATTACHED_ENERGY_CAP)

    return nn_value * 0.9 + loaded * 0.1


# normalization cap for opponent energy sent to discard (KO'd loaded Pokemon)
OPP_DISCARDED_ENERGY_CAP = 8


def _opp_discarded_energy_shape(obs: Observation, your_index: int, nn_value: float) -> float:
    """Higher value the more energy you've sent to the opponent's discard.

    When you KO a Pokemon, the energy attached to it is discarded too. So
    energy piling up in the opponent's discard means you destroyed their
    investment -> KO'ing a 3-energy Starmie scores higher than a bare one.
    """
    opp = obs.current.players[1 - your_index]
    trashed = _count_energy(opp.discard)
    signal = min(1.0, trashed / OPP_DISCARDED_ENERGY_CAP)
    return nn_value * 0.9 + signal * 0.1


# normalization cap for total damage dealt across the opponent's board (HP)
DAMAGE_CAP = 300


def _damage_shape(obs: Observation, your_index: int, nn_value: float) -> float:
    """Higher value the more damage sits on the opponent's board.

    Sums raw damage (maxHp - hp) over the opp's active + bench, normalized by a
    HP cap. Unlike target_card_shape (per-card fraction), this rewards ABSOLUTE
    damage -> hitting harder always scores more, pushing the search to maximize
    damage output. KO'd Pokemon leave the board, so prize/KO credit should come
    from _prize_pressure_shape alongside this.
    """
    opp = obs.current.players[1 - your_index]

    dealt = 0
    for poke in _board(opp):
        if poke is not None:
            dealt += max(0, poke.maxHp - poke.hp)

    signal = min(1.0, dealt / DAMAGE_CAP)
    return nn_value * 0.9 + signal * 0.1


@dataclass
class AttackerConfig:
    """Card IDs you treat as attackers, mapped to how much you value powering them.

    Mutate `main` / `secondary` during the game to shift the search's focus
    (e.g. promote a card from secondary to main once your plan changes).
    """

    main: dict[int, float] = field(default_factory=dict)
    secondary: dict[int, float] = field(default_factory=dict)
    # card ID -> energy needed, overrides the auto costliest-attack default
    needed: dict[int, int] = field(default_factory=dict)
    weight: float = 0.1  # how much shaping bends the NN value


def _attacker_progress(poke, values: dict[int, float], needed_override: dict[int, int]) -> tuple[float, float]:
    """Return (value, value*progress) for a Pokemon if it's a listed attacker."""
    if poke is None or poke.id not in values:
        return 0.0, 0.0
    needed = needed_override.get(poke.id, _ENERGY_NEEDED.get(poke.id, 0))
    if needed <= 0:
        progress = 1.0  # attacks for free / unknown cost = already "ready"
    else:
        progress = min(1.0, len(poke.energyCards) / needed)
    val = values[poke.id]
    return val, val * progress


def make_attacker_energy_shape(config: AttackerConfig) -> RewardShapeFn:
    """Build a shape that rewards energy on the RIGHT attacker.

    Each attacker contributes value*progress where progress = attached/needed.
    Once the main attacker is fully powered its term maxes out, so further
    energy only helps via the secondary term -> search shifts to the backup.
    Hold the returned config and mutate it to change targets mid-game.
    """

    def shape(obs: Observation, your_index: int, nn_value: float) -> float:
        you = obs.current.players[your_index]
        board = _board(you)

        total_value = 0.0
        earned = 0.0
        for values in (config.main, config.secondary):
            if not values:
                continue
            # best matching Pokemon per group (handles multiple copies on board)
            best_val, best_earned = 0.0, 0.0
            for poke in board:
                val, e = _attacker_progress(poke, values, config.needed)
                if val > 0 and e >= best_earned:
                    best_val, best_earned = val, e
            total_value += best_val
            earned += best_earned

        if total_value <= 0:
            return nn_value
        signal = earned / total_value  # 0..1
        return nn_value * (1.0 - config.weight) + signal * config.weight

    return shape


@dataclass
class GetAttackerConfig:
    """Card IDs of attackers you want in play, mapped to how much you value each.

    Rewards FINDING the attacker: credit when it sits on your board, partial
    credit when it's still in hand (closer than buried in deck/prize). Distinct
    from make_attacker_energy_shape, which rewards energy ON an attacker already
    in play -> use this to push the search to dig for and land the attacker, that
    one to power it up. Mutate `targets` live to change who you're hunting for.
    """

    targets: dict[int, float] = field(default_factory=dict)
    hand_credit: float = 0.5  # fraction of value earned while only in hand
    weight: float = 0.1


def make_get_attacker_shape(config: GetAttackerConfig) -> RewardShapeFn:
    """Build a shape that rewards getting your main attacker in hand or play.

    Per target: full value once it's on your board, `hand_credit` * value while
    it's only in hand, nothing if it's still hidden (deck/prize/discard). Board
    beats hand, hand beats unseen -> search progresses toward landing it.
    """

    def shape(obs: Observation, your_index: int, nn_value: float) -> float:
        if not config.targets:
            return nn_value
        you = obs.current.players[your_index]
        board_ids = {p.id for p in _board(you) if p is not None}
        hand_ids = {_card_id(c) for c in (you.hand or [])}

        total_value = 0.0
        earned = 0.0
        for cid, val in config.targets.items():
            total_value += val
            if cid in board_ids:
                earned += val
            elif cid in hand_ids:
                earned += val * config.hand_credit

        if total_value <= 0:
            return nn_value
        signal = earned / total_value  # 0..1
        return nn_value * (1.0 - config.weight) + signal * config.weight

    return shape


@dataclass
class BenchKeepConfig:
    """Card IDs you want to keep safe on the bench, mapped to how much you value it.

    Mutate `keep` during the game to change which Pokemon to protect.
    """

    keep: dict[int, float] = field(default_factory=dict)
    weight: float = 0.1


def make_bench_keep_shape(config: BenchKeepConfig) -> RewardShapeFn:
    """Build a shape that rewards keeping target Pokemon on your bench.

    Credit per target only when it sits on the bench (not active, not gone).
    """

    def shape(obs: Observation, your_index: int, nn_value: float) -> float:
        if not config.keep:
            return nn_value
        you = obs.current.players[your_index]
        bench_ids = {p.id for p in you.bench if p is not None}

        total = sum(config.keep.values())
        earned = sum(val for cid, val in config.keep.items() if cid in bench_ids)

        signal = earned / total  # 0..1
        return nn_value * (1.0 - config.weight) + signal * config.weight

    return shape


@dataclass
class AbomasnowConfig:
    main_id: int = 723  # Mega Abomasnow ex (main attacker)
    main_target: int = 2  # energy it wants
    main_max: int = 3  # energy it can still use
    secondary_id: int = 721  # Kyogre (soaks leftover energy)
    secondary_max: int = 3
    pre_id: int = 722  # Snover (pre-evolution of main)
    pre_target: int = 2  # energy to pre-load before evolving
    main_value: float = 1.0
    secondary_value: float = 0.5
    pre_value: float = 0.7
    weight: float = 0.1


def _find(board, cid):
    for p in board:
        if p is not None and p.id == cid:
            return p
    return None


def make_abomasnow_shape(config: AbomasnowConfig) -> RewardShapeFn:
    """Energy priority for the Abomasnow team.

    Waterfall (highest value first, normalized so leftover flows down):
      723 main  -> wants 2 energy, small extra credit for a 3rd.
      721 secondary -> soaks energy not needed by the others.
      722 Snover -> credited ONLY if it's your only Pokemon, OR it can evolve
                    into 723 next turn (723 in hand, Snover settled) AND the
                    main attacker already has enough energy.
    """
    c = config

    def shape(obs: Observation, your_index: int, nn_value: float) -> float:
        you = obs.current.players[your_index]
        board = _board(you)
        alive = [p for p in board if p is not None]

        total_value = 0.0
        earned = 0.0

        # --- main 723: target 2, bonus for 3rd ---
        main = _find(board, c.main_id)
        main_satisfied = False
        if main is not None:
            e = len(main.energyCards)
            base = min(e, c.main_target) / c.main_target
            span = max(1, c.main_max - c.main_target)
            extra = max(0, min(e, c.main_max) - c.main_target) / span
            score = base * 0.9 + extra * 0.1
            total_value += c.main_value
            earned += c.main_value * score
            main_satisfied = e >= c.main_target

        # --- secondary 721: soak overflow ---
        sec = _find(board, c.secondary_id)
        if sec is not None:
            score = min(len(sec.energyCards), c.secondary_max) / c.secondary_max
            total_value += c.secondary_value
            earned += c.secondary_value * score

        # --- pre-evo 722: only under its condition ---
        pre = _find(board, c.pre_id)
        if pre is not None:
            only_one = len(alive) == 1
            hand_ids = {_card_id(h) for h in (you.hand or [])}
            can_evolve = (c.main_id in hand_ids) and not getattr(pre, "appearThisTurn", False)
            pre_active = only_one or (can_evolve and main_satisfied)
            if pre_active:
                score = min(len(pre.energyCards), c.pre_target) / c.pre_target
                total_value += c.pre_value
                earned += c.pre_value * score

        if total_value <= 0:
            return nn_value
        signal = earned / total_value
        return nn_value * (1.0 - c.weight) + signal * c.weight

    return shape


@dataclass
class AbomasnowGetConfig:
    """Hunt logic for landing the Abomasnow attacker line.

    Drives the SEARCH toward the RIGHT Pokemon on board, not just any copy:
      - Mega Abomasnow ex (723) is the goal, but it can't be played from hand
        directly -> it evolves from Snover (722). So when 723 is already in hand,
        the search should switch to getting Snover onto the board (deck/hand ->
        board), since that's what actually unlocks the evolution. Crediting 723
        sitting dead in hand would tell the search it's done when it isn't.
      - Kyogre (721) is the backup attacker. When the opponent still has many
        prize cards left (long game), weight Kyogre higher so the search keeps a
        second threat coming online instead of over-committing to Abomasnow.
    """

    main_id: int = 723  # Mega Abomasnow ex (evolution, can't be played raw)
    pre_id: int = 722  # Snover (pre-evo; must be on board to evolve into 723)
    secondary_id: int = 721  # Kyogre (backup attacker)
    main_value: float = 1.0
    pre_value: float = 0.6
    secondary_value: float = 0.5  # Kyogre weight in a normal/short game
    secondary_value_late: float = 1.2  # Kyogre weight when opp prizes left >= late_prizes
    late_prizes: int = 3  # opp prize cards remaining at/above which Kyogre matters more
    hand_credit: float = 0.5  # fraction of value for being in hand vs on board
    weight: float = 0.1


def make_abomasnow_get_shape(config: AbomasnowGetConfig) -> RewardShapeFn:
    """Reward progress toward landing the Abomasnow attacker line + Kyogre backup.

    Replaces the generic get-attacker shape for the Abomasnow team so it can:
      1. Hunt Snover (722) onto the board once Mega Abomasnow ex (723) is in hand,
         because 723 evolves from Snover and can't be played raw.
      2. Up-weight Kyogre (721) when the opponent still holds many prizes.
    """
    c = config

    def shape(obs: Observation, your_index: int, nn_value: float) -> float:
        you = obs.current.players[your_index]
        opp = obs.current.players[1 - your_index]
        board_ids = {p.id for p in _board(you) if p is not None}
        hand_ids = {_card_id(h) for h in (you.hand or [])}

        total_value = 0.0
        earned = 0.0

        # --- main attacker line: 723 evolves from Snover 722 ---
        total_value += c.main_value
        if c.main_id in board_ids:
            earned += c.main_value  # landed: fully evolved on board
        elif c.main_id in hand_ids:
            # 723 in hand can't be played raw -> need Snover on board to evolve.
            # Credit Snover progress instead of the dead 723 in hand.
            if c.pre_id in board_ids:
                earned += c.main_value * 0.8  # ready to evolve next turn
            elif c.pre_id in hand_ids:
                earned += c.main_value * c.hand_credit  # both pieces in hand
            # else: neither piece placed -> 0, keep digging for Snover
        elif c.pre_id in board_ids:
            earned += c.main_value * c.hand_credit  # Snover down, still need 723
        elif c.pre_id in hand_ids:
            earned += c.main_value * c.hand_credit * 0.5  # only Snover, in hand

        # --- backup attacker Kyogre 721 (heavier in a long, high-prize game) ---
        opp_prizes_left = len(opp.prize)
        sec_value = c.secondary_value_late if opp_prizes_left >= c.late_prizes else c.secondary_value
        total_value += sec_value
        if c.secondary_id in board_ids:
            earned += sec_value
        elif c.secondary_id in hand_ids:
            earned += sec_value * c.hand_credit

        if total_value <= 0:
            return nn_value
        signal = earned / total_value
        return nn_value * (1.0 - c.weight) + signal * c.weight

    return shape


@dataclass
class TargetCardConfig:
    """Opponent card IDs you want to remove, mapped to how much you value it.

    Swap `targets` per matchup/archetype. Mutate it live to retarget mid-game.
    """

    targets: dict[int, float] = field(default_factory=dict)
    weight: float = 0.1


def make_target_card_shape(config: TargetCardConfig) -> RewardShapeFn:
    """Reward damaging / KO'ing specific opponent Pokemon.

    Per target: credit = value * progress, where progress is
      damage taken (1 - hp/maxHp) while on board, or 1.0 once KO'd (in discard).
    Targets that haven't shown up (deck/hand/prize) don't count.
    """

    def shape(obs: Observation, your_index: int, nn_value: float) -> float:
        if not config.targets:
            return nn_value
        opp = obs.current.players[1 - your_index]
        on_board = {p.id: p for p in _board(opp) if p is not None}
        discard = {_card_id(c) for c in (opp.discard or [])}

        total_value = 0.0
        earned = 0.0
        for cid, val in config.targets.items():
            if cid in on_board:
                poke = on_board[cid]
                progress = 0.0
                if poke.maxHp > 0:
                    progress = max(0.0, 1.0 - poke.hp / poke.maxHp)
                total_value += val
                earned += val * progress
            elif cid in discard:  # KO'd
                total_value += val
                earned += val

        if total_value <= 0:
            return nn_value
        signal = earned / total_value
        return nn_value * (1.0 - config.weight) + signal * config.weight

    return shape


@dataclass
class HandDisruptionConfig:
    """Reward NET cards removed from the opponent's hand this turn.

    Credit is the drop from `baseline` (opp handCount captured at the start of
    your turn) down to the current handCount, capped at `target`. Hitting the
    target maxes the term; removing more adds nothing. e.g. opp has 8, you knock
    them to 4 -> removed 4; target 4 -> full credit, target 2 -> also full.

    Set `baseline` live each turn before searching (mutate-config idiom):
        cfg.baseline = obs.current.players[1 - your_index].handCount
    If baseline is None, falls back to the current handCount (-> 0 credit).
    """

    target: int = 2  # net cards you aim to strip from opp hand
    baseline: int | None = None  # opp handCount at the start of your turn; set live
    weight: float = 0.1


def make_hand_disruption_shape(config: HandDisruptionConfig) -> RewardShapeFn:
    """Build a shape that rewards reducing the opponent's hand toward `target`.

    Opponent hand contents are hidden, but `handCount` is visible. signal =
    clamp(baseline - handCount, 0, target) / target.
    """

    def shape(obs: Observation, your_index: int, nn_value: float) -> float:
        opp = obs.current.players[1 - your_index]
        base = config.baseline if config.baseline is not None else opp.handCount
        target = max(1, config.target)
        removed = base - opp.handCount
        # -1..1: +1 at target removed, -1 if you grew their hand by `target`.
        signal = max(-1.0, min(removed, target) / target)
        return nn_value * (1.0 - config.weight) + signal * config.weight

    return shape


@dataclass
class HandBuildConfig:
    """Reward NET cards GAINED in YOUR hand this turn (inverse of disruption).

    Credit is the rise from `baseline` (your handCount at the start of your
    turn) up to the current handCount, capped at `target`. Hitting the target
    maxes the term; drawing more adds nothing. Growing nothing = 0; SHRINKING
    your own hand by `target` = -1.

    Set `baseline` live each turn before searching:
        cfg.baseline = obs.current.players[your_index].handCount
    If baseline is None, falls back to the current handCount (-> 0 credit).
    """

    target: int = 2  # net cards you aim to add to your hand
    baseline: int | None = None  # your handCount at the start of your turn; set live
    weight: float = 0.1


def make_hand_build_shape(config: HandBuildConfig) -> RewardShapeFn:
    """Build a shape that rewards growing YOUR hand toward `target`.

    signal = clamp(handCount - baseline, -target, target) / target.
    """

    def shape(obs: Observation, your_index: int, nn_value: float) -> float:
        you = obs.current.players[your_index]
        base = config.baseline if config.baseline is not None else you.handCount
        target = max(1, config.target)
        gained = you.handCount - base
        # -1..1: +1 at target gained, -1 if you shrank your own hand by `target`.
        signal = max(-1.0, min(gained, target) / target)
        return nn_value * (1.0 - config.weight) + signal * config.weight

    return shape


def hand_build_reward(target: int = 2, weight: float = 0.1) -> "RewardFn":
    """A BARE hand-build RewardFn for per-archetype tactics."""
    return RewardFn(
        terminal=_win_loss_terminal,
        shape=make_hand_build_shape(HandBuildConfig(target=target, weight=weight)),
    )


def hand_disruption_reward(target: int = 2, weight: float = 0.1) -> "RewardFn":
    """A BARE hand-disruption RewardFn for per-archetype tactics.

    Holds just the disruption delta (no base shape). apply_tactic composes it
    onto the agent's base reward. Mutate the config's `baseline` each turn via
    the returned shape's closure -- or build the config yourself for live access.
    """
    return RewardFn(
        terminal=_win_loss_terminal,
        shape=make_hand_disruption_shape(HandDisruptionConfig(target=target, weight=weight)),
    )


def get_attacker_reward(targets: dict[int, float], hand_credit: float = 0.5, weight: float = 0.1) -> "RewardFn":
    """A BARE get-attacker RewardFn for per-archetype tactics.

    Holds just the find-the-attacker delta (no base shape). apply_tactic composes
    it onto the agent's base reward, so the base is never dropped.
    """
    return RewardFn(
        terminal=_win_loss_terminal,
        shape=make_get_attacker_shape(GetAttackerConfig(targets=targets, hand_credit=hand_credit, weight=weight)),
    )


def target_reward(targets: dict[int, float], weight: float = 0.1) -> "RewardFn":
    """A BARE target-only RewardFn for per-archetype tactics.

    Holds just the focus-fire delta (no base shape). apply_tactic composes it
    onto the agent's base reward, so the base (e.g. abomasnow) is never dropped.
    """
    return RewardFn(
        terminal=_win_loss_terminal,
        shape=make_target_card_shape(TargetCardConfig(targets=targets, weight=weight)),
    )


def compose_shapes(*shapes: RewardShapeFn) -> RewardShapeFn:
    """Chain shapes: each one's output feeds the next as nn_value."""

    def shape(obs: Observation, your_index: int, nn_value: float) -> float:
        v = nn_value
        for s in shapes:
            v = s(obs, your_index, v)
        return v

    return shape


@dataclass
class BaseShapeConfig:
    """Configs for the live-baseline shapes inside the base shape.

    Hold this object and set the two baselines each turn BEFORE searching:
        cfg.disruption.baseline = obs.current.players[1 - your_index].handCount
        cfg.build.baseline      = obs.current.players[your_index].handCount
    """

    disruption: HandDisruptionConfig = field(default_factory=HandDisruptionConfig)
    build: HandBuildConfig = field(default_factory=HandBuildConfig)


def make_base_shape(config: BaseShapeConfig | None = None) -> RewardShapeFn:
    """General-purpose base shape: chains six signals onto the NN value.

    Order (each feeds the next as nn_value, diluting by its own weight):
      1. _attached_energy_shape       -> energy loaded on your board
      2. _opp_discarded_energy_shape  -> energy you blew up in opp discard
      3. _damage_shape                -> damage dealt to the opp's board
      4. hand disruption              -> cards stripped from opp hand this turn
      5. hand build                   -> cards added to your hand this turn
      6. _prize_pressure_shape        -> prize lead

    Pass a BaseShapeConfig and mutate its `.disruption.baseline` /
    `.build.baseline` each turn so the hand deltas read correctly.
    """
    cfg = config if config is not None else BaseShapeConfig()
    return compose_shapes(
        _attached_energy_shape,
        _opp_discarded_energy_shape,
        _damage_shape,
        make_hand_disruption_shape(cfg.disruption),
        make_hand_build_shape(cfg.build),
        _prize_pressure_shape,
    )


# --- presets ---
# general base reward. Hold base_shape_config and set its baselines each turn.
base_shape_config = BaseShapeConfig()
base = RewardFn(terminal=_win_loss_terminal, shape=make_base_shape(base_shape_config))
_abomasnow_config = AbomasnowConfig()
# hunt for the attacker line: Snover -> Mega Abomasnow ex, plus Kyogre backup.
# Pivots to Snover when 723 is in hand; up-weights Kyogre when opp has 3+ prizes.
abomasnow_get = AbomasnowGetConfig(
    main_id=_abomasnow_config.main_id,
    pre_id=_abomasnow_config.pre_id,
    secondary_id=_abomasnow_config.secondary_id,
    hand_credit=0.5,
    weight=0.1,
)
abomasnow = RewardFn(
    terminal=_win_loss_terminal,
    shape=compose_shapes(
        make_abomasnow_get_shape(abomasnow_get),
        make_abomasnow_shape(_abomasnow_config),
        _high_deck_energy_shape,
        make_base_shape(base_shape_config),
    ),
)
