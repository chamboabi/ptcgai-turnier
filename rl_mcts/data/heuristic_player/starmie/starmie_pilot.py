"""Mega Starmie ex pilot — encodes the #1 player's (Yushin Ito) control-tempo heuristics
on top of the generic KO-aware heuristic engine:

  * CONSISTENCY + Nebula Beam (210, pierces walls/weakness) -> handled by the generic
    engine (evolve Staryu->Starmie, load 3 energy, attack).
  * CRUSHING HAMMER energy denial -> play it (Item, no per-turn cap) to strip the
    opponent's Active attacker's Energy whenever it has some and we have no lethal.
  * WALLY'S COMPASSION emergency heal -> full-heal the Active Mega ex when it's about to
    be KO'd. NOTE: Wally strips the healed Pokemon's Energy to hand, so without a
    re-acceleration engine (Cinderace) it costs a turn -- only worth it as a last resort
    to deny a Prize, never when we have lethal or the attacker is healthy.

Everything else delegates to src.agents.heuristic_agent (which already pilots the core
Starmie line well: 95/82/55/57 vs Alakazam/Dragapult/Lucario/Crustle in deck testing)."""
from __future__ import annotations

from typing import Any

import heuristic_agent as H
from shared import is_initial_observation, load_agent_deck
from option_tools import (
    current_player_index,
    find_options_by_type,
    get_context,
    get_options,
    hand_card_id_for_play_option,
    obj_get,
)

MEGA_STARMIE = 1031
CRUSHING_HAMMER = 1120
WALLY = 1229
HEAL_HP_THRESHOLD = 130  # active Mega ex this low = in 1-shot range -> emergency heal


def _active(obs, idx):
    cur = obj_get(obs, "current")
    players = obj_get(cur, "players", []) or []
    if 0 <= idx < len(players):
        a = obj_get(players[idx], "active", []) or []
        return a[0] if a and a[0] is not None else None
    return None


def _energies(pk):
    return len(obj_get(pk, "energies", []) or []) if pk else 0


def _play_option_for(obs, options, card_id):
    for idx, opt in find_options_by_type(options, "PLAY"):
        if hand_card_id_for_play_option(obs, opt) == card_id:
            return idx
    return None


def _best_attack_lethal(obs, options, attack_lookup, card_lookup):
    """True if some attack we can pay for KOs the opponent's Active this turn."""
    scored = H._score_attacks(obs, options, attack_lookup, card_lookup, None)
    return any(s.detail.get("ko") for s in scored)


def make_agent(deck, attack_lookup=None, card_lookup=None, debug_log=None, deck_name=None, **_):
    al, cl = H._load_cg_metadata()
    attack_lookup = attack_lookup or al
    card_lookup = card_lookup or cl

    def agent(obs: Any):
        if is_initial_observation(obs):
            return load_agent_deck(deck)
        if (get_context(obs) or "").upper() == "MAIN":
            options = get_options(obs)
            me = current_player_index(obs)
            opp = 1 - me
            my_active = _active(obs, me)
            opp_active = _active(obs, opp)
            lethal = _best_attack_lethal(obs, options, attack_lookup, card_lookup)

            # --- Crushing Hammer: deny the opponent's loaded Active (no lethal of our own) ---
            if not lethal and _energies(opp_active) >= 1:
                idx = _play_option_for(obs, options, CRUSHING_HAMMER)
                if idx is not None:
                    return H._finish(obs, [idx], "starmie: crushing hammer energy denial",
                                     options, attack_lookup, card_lookup, debug_log)

            # --- Wally's Compassion: emergency full-heal a dying Mega ex (no lethal) ---
            if (not lethal and my_active is not None
                    and int(obj_get(my_active, "id", -1) or -1) == MEGA_STARMIE
                    and int(obj_get(my_active, "hp", 999) or 999) <= HEAL_HP_THRESHOLD):
                idx = _play_option_for(obs, options, WALLY)
                if idx is not None:
                    return H._finish(obs, [idx], "starmie: emergency Wally heal",
                                     options, attack_lookup, card_lookup, debug_log)

        # everything else -> the generic KO-aware engine
        return H.agent(obs, deck=deck, attack_lookup=attack_lookup,
                       card_lookup=card_lookup, debug_log=debug_log)

    return agent
