"""Self-contained tournament agent.

Bundled by submission_builder/build.py into a folder holding:
    main.py     (this file)
    deck.csv    (60 card IDs, one per line)
    cg/         (untouched game library)
    model.pth   (optional trained weights; random init if absent)

Entry point the arena calls each turn: agent(obs_dict) -> list[int].
On the initial selection obs.select is None -> return the 60-card deck.
"""

import math
import os
import random
from dataclasses import replace

import torch
import torch.nn
import torch.nn.functional

import deck_predict
import rewards

from cg.api import (
    AreaType,
    Card,
    LogType,
    Observation,
    OptionType,
    PlayerState,
    Pokemon,
    SearchState,
    SelectContext,
    all_attack,
    all_card_data,
    search_begin,
    search_end,
    search_step,
    to_observation_class,
)

# --- config (inlined from config.json so no extra files are needed) ---
D_MODEL = 128
NUM_HEADS = 2
D_FEEDFORWARD = 256
NUM_LAYERS_ENCODER = 1
NUM_LAYERS_DECODER = 1

NUM_WORDS_ENCODER = 24
ENCODER_SIZE = 22000
DECODER_MAIN_FEATURE = 8
DECODER_ATTACK_OFFSET = 14

# MCTS knobs. SEARCH_COUNT trades strength for time; raise for stronger play if
# the per-turn time budget allows (repo training used 3000).
SEARCH_COUNT = 3000
MAX_ACTION_COMBINATIONS = 64
UCB_EXPLORATION = 0.4
POLICY_TEMPERATURE = 10.0
UNVISITED_PENALTY = 0.03

_UNKNOWN = 1072  # filler card id for hidden opponent cards

# --- card data derived sizes ---
all_card = all_card_data()
card_count = max(all_card, key=lambda c: c.cardId).cardId + 1
attack_count = max(all_attack(), key=lambda a: a.attackId).attackId + 1
decoder_card_offset = DECODER_ATTACK_OFFSET + attack_count
decoder_size = decoder_card_offset + (1 + DECODER_MAIN_FEATURE + SelectContext.RECOVER_SPECIAL_CONDITION) * card_count


# --- sparse input builder ---
class SparseVector:
    def __init__(self):
        self.index: list[int] = []
        self.value: list[float] = []
        self.offset: list[int] = []
        self.pos = 0

    def add(self, index: int, value):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos + index)
            self.value.append(value)

    def add_pos(self, pos: int):
        self.pos += pos

    def add_single(self, value):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos)
            self.value.append(value)
        self.pos += 1

    def word_start(self):
        self.offset.append(len(self.index))


# --- model ---
class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_feedforward):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(d_model, num_heads)
        self.fc1 = torch.nn.Linear(d_model, d_feedforward)
        self.fc2 = torch.nn.Linear(d_feedforward, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x, encoder_out):
        y, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
        res = self.norm1(x + y)
        y = self.fc1(res)
        y = torch.nn.functional.relu(y)
        y = self.fc2(y)
        return self.norm2(res + y)


class MyModel(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_feedforward, num_layers_encoder, num_layers_decoder):
        super().__init__()
        self.d_model = d_model
        self.encoder_bag = torch.nn.EmbeddingBag(ENCODER_SIZE, d_model, mode="sum")
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, 0)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers_encoder, enable_nested_tensor=False)
        self.encoder_fc = torch.nn.Linear(d_model, 1)
        self.decoder_bag = torch.nn.EmbeddingBag(decoder_size, d_model, mode="sum")
        self.decoder = torch.nn.ModuleList()
        for _ in range(num_layers_decoder):
            self.decoder.append(DecoderLayer(d_model, num_heads, d_feedforward))
        self.decoder_fc = torch.nn.Linear(d_model, 1)

    def forward(self, index_encoder, value_encoder, offset_encoder, index_decoder, value_decoder, offset_decoder):
        v = self.encoder_bag(index_encoder, offset_encoder, value_encoder)
        v = v.reshape(-1, NUM_WORDS_ENCODER, self.d_model).transpose(0, 1)
        batch_size = v.size(1)
        encoder_out = self.encoder(v)
        v = self.encoder_fc(encoder_out)
        v = torch.tanh(v.mean(0))

        p = self.decoder_bag(index_decoder, offset_decoder, value_decoder)
        p = p.reshape(batch_size, -1, self.d_model).transpose(0, 1)
        for layer in self.decoder:
            p = layer(p, encoder_out)
        p = self.decoder_fc(p)
        p = p.transpose(0, 1).view(batch_size, -1)
        p = torch.tanh(p)
        return (v, p)


# --- encoder feature construction ---
def add_card(sv, card):
    if card is not None:
        sv.add(card.id, 1)
    sv.add_pos(card_count)


def add_cards(sv, cards, value):
    if cards is not None:
        for card in cards:
            sv.add(card.id, value)
    sv.add_pos(card_count)


def add_pokemon(sv, poke):
    if poke is None:
        sv.add_single(1)
        sv.add_pos(1 + 3 * card_count)
    else:
        sv.add_single(0)
        sv.add_single(poke.hp / 400)
        add_card(sv, poke)
        add_cards(sv, poke.tools, 1.0)
        add_cards(sv, poke.energyCards, 0.5)


def add_player(sv, ps):
    sv.add_single(ps.deckCount / 60)
    sv.add_single(len(ps.discard) / 60)
    sv.add_single(ps.handCount / 8)
    sv.add_single(len(ps.bench) / 5)
    sv.add(len(ps.prize), 1)
    sv.add_pos(7)
    sv.add_single(ps.poisoned)
    sv.add_single(ps.burned)
    sv.add_single(ps.asleep)
    sv.add_single(ps.paralyzed)
    sv.add_single(ps.confused)
    add_cards(sv, ps.discard, 0.25)


def get_card(obs, area, index, player_index):
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


def get_encoder_input(obs, your_deck):
    your_index = obs.current.yourIndex
    state = obs.current
    sv = SparseVector()
    for i in range(2):
        ps = state.players[i ^ your_index]
        for j in range(8):
            sv.word_start()
            pos = sv.pos
            if j < len(ps.bench):
                add_pokemon(sv, ps.bench[j])
            else:
                add_pokemon(sv, None)
            if j != 7:
                sv.pos = pos
    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        if 0 < len(ps.active):
            add_pokemon(sv, ps.active[0])
        else:
            add_pokemon(sv, None)
    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        add_player(sv, ps)
    sv.word_start()
    add_cards(sv, state.players[your_index].hand, 0.25)
    sv.word_start()
    for cid in your_deck:
        sv.add(cid, 0.25)
    sv.add_pos(card_count)
    sv.word_start()
    add_cards(sv, state.stadium, 1.0)
    sv.word_start()
    sv.add_single(1)
    sv.add_single(state.turn / 10)
    sv.add_single(state.firstPlayer == your_index)
    return sv


# --- decoder feature construction ---
def decoder_main(sv, feature_index, card):
    if card is not None:
        sv.add(decoder_card_offset + feature_index * card_count + card.id, 1)


def decoder_card_id(sv, context, card_id):
    sv.add(decoder_card_offset + (DECODER_MAIN_FEATURE + context) * card_count + card_id, 1)


def decoder_card(sv, context, card):
    if card is not None:
        decoder_card_id(sv, context, card.id)


def get_decoder_input(obs, actions):
    sv = SparseVector()
    your_index = obs.current.yourIndex
    ps = obs.current.players[your_index]
    context = obs.select.context
    for action in actions:
        sv.word_start()
        if len(action) == 0:
            sv.add(0, 1)
            continue
        for i in action:
            o = obs.select.option[i]
            match o.type:
                case OptionType.END:
                    sv.add(1, 1)
                case OptionType.YES:
                    sv.add(2, 1)
                case OptionType.NO:
                    sv.add(3, 1)
                case OptionType.SPECIAL_CONDITION:
                    sv.add(4 + o.specialConditionType, 1)
                case OptionType.NUMBER:
                    sv.add(9 + min(o.number, 4), 1)
                case OptionType.ATTACK:
                    sv.add(DECODER_ATTACK_OFFSET + o.attackId, 1)
                case OptionType.PLAY:
                    decoder_main(sv, 0, ps.hand[o.index])
                case OptionType.ATTACH:
                    decoder_main(sv, 1, get_card(obs, o.area, o.index, your_index))
                    decoder_main(sv, 2, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
                case OptionType.EVOLVE:
                    decoder_main(sv, 3, get_card(obs, o.area, o.index, your_index))
                    decoder_main(sv, 4, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
                case OptionType.ABILITY:
                    decoder_main(sv, 5, get_card(obs, o.area, o.index, your_index))
                case OptionType.DISCARD:
                    decoder_main(sv, 6, get_card(obs, o.area, o.index, your_index))
                case OptionType.RETREAT:
                    decoder_main(sv, 7, ps.active[0])
                case OptionType.CARD:
                    decoder_card(sv, context, get_card(obs, o.area, o.index, o.playerIndex))
                case OptionType.TOOL_CARD:
                    card = get_card(obs, o.area, o.index, o.playerIndex)
                    decoder_card(sv, context, card.tools[o.toolIndex])
                case OptionType.ENERGY_CARD | OptionType.ENERGY:
                    card = get_card(obs, o.area, o.index, o.playerIndex)
                    decoder_card(sv, context, card.energyCards[o.energyIndex])
                case OptionType.SKILL:
                    decoder_card_id(sv, context, o.cardId)
    return sv


def eval_nn(sv_enc, sv_dec, model):
    device = next(model.parameters()).device
    value, policy = model(
        torch.tensor(sv_enc.index, dtype=torch.int32, device=device),
        torch.tensor(sv_enc.value, dtype=torch.float32, device=device),
        torch.tensor(sv_enc.offset, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.index, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.value, dtype=torch.float32, device=device),
        torch.tensor(sv_dec.offset, dtype=torch.int32, device=device),
    )
    return (value.tolist()[0][0], policy.tolist()[0])


# --- MCTS ---
class Child:
    def __init__(self, select, prob):
        self.node = None
        self.select = select
        self.prob = prob


class Node:
    def __init__(self, parent, state):
        self.value = -2.0
        self.total = 0.0
        self.visit = 0
        self.parent = parent
        self.children: list[Child] = []
        self.state = state

    def backprop(self, value):
        self.total += value
        self.visit += 1
        if self.parent is not None:
            self.parent.backprop(value)


# --- opponent deck prediction (mirror of mcts._sample_opponent_deck) ---
def _opponent_known_cards(opp: PlayerState) -> list[int]:
    known: list[int] = []

    def _add_pokemon(p) -> None:
        known.append(p.id)
        for c in p.preEvolution:
            known.append(c.id)
        for c in p.energyCards:
            known.append(c.id)
        for c in p.tools:
            known.append(c.id)

    for p in opp.active:
        if p is not None:
            _add_pokemon(p)
    for p in opp.bench:
        _add_pokemon(p)
    for c in opp.discard:
        known.append(c.id)
    for c in opp.prize:
        if c is not None:
            known.append(c.id)
    return known


def _sample_opponent_deck(archetype_model, opp: PlayerState):
    """Return (deck, prize, hand) card-ID lists using per-card predictions.

    Bernoulli-sample inclusion from P(card in deck), then add E[copies | in deck]
    rounded copies. Falls back to _UNKNOWN filler when no archetype model is loaded.
    """
    deck_n = opp.deckCount
    prize_n = len(opp.prize)
    hand_n = opp.handCount
    total = deck_n + prize_n + hand_n

    if archetype_model is None:
        return [_UNKNOWN] * deck_n, [1] * prize_n, [1] * hand_n

    known = _opponent_known_cards(opp)
    result = archetype_model.predict(known)

    pool: list[int] = []
    for pred in result.card_predictions:
        if random.random() < pred.probability:
            conditional_mean = pred.expected_copies / max(pred.probability, 1e-9)
            pool.extend([pred.card_id] * max(1, round(conditional_mean)))

    random.shuffle(pool)
    pool = pool[:total]
    pool += [_UNKNOWN] * (total - len(pool))
    random.shuffle(pool)

    return pool[:deck_n], pool[deck_n : deck_n + prize_n], pool[deck_n + prize_n : total]


def create_node(parent, search_state, your_index, deck, model, reward_fn):
    node = Node(parent, search_state)
    obs = search_state.observation
    state = obs.current
    if state.result >= 0:
        node.value = reward_fn.terminal(obs, your_index)
        node.backprop(node.value)
    else:
        actions = []
        indices = list(range(obs.select.maxCount))
        for _ in range(MAX_ACTION_COMBINATIONS):
            actions.append(indices.copy())
            for i in range(len(indices)):
                index = len(indices) - i - 1
                if indices[index] < len(obs.select.option) - i - 1:
                    indices[index] += 1
                    for j in range(index + 1, len(indices)):
                        indices[j] = indices[j - 1] + 1
                    break
            else:
                break

        sv_enc = get_encoder_input(obs, deck)
        sv_dec = get_decoder_input(obs, actions)
        value, policy = eval_nn(sv_enc, sv_dec, model)
        # shape bends the NN value before it is backed up through the tree, so it
        # steers MCTS toward good intermediate states (same as training mcts.py).
        shaped = reward_fn.shape(obs, your_index, value)
        v = shaped if state.yourIndex == your_index else -shaped
        node.value = v
        node.backprop(v)

        prob_sum = 0.0
        for i in range(len(policy)):
            p = math.exp(policy[i] * POLICY_TEMPERATURE)
            node.children.append(Child(actions[i], p))
            prob_sum += p
        for c in node.children:
            c.prob /= prob_sum
    return node


def mcts_agent(obs_dict, deck, model, reward_fn, archetype_model):
    obs = to_observation_class(obs_dict)
    your_index = obs.current.yourIndex
    state = obs.current
    opp = state.players[1 - your_index]

    # Freeze per-turn baselines from the search root: rebuild the shape once here so
    # absolute reward shapes score deltas caused by this move, not standing board
    # state. Plain (factory-less) rewards pass through unchanged. (See rl_mcts/mcts.py.)
    if reward_fn.shape_factory is not None:
        reward_fn = replace(reward_fn, shape=reward_fn.shape_factory(obs, your_index))

    opp_deck, opp_prize, opp_hand = _sample_opponent_deck(archetype_model, opp)
    search_state = search_begin(
        obs,
        your_deck=random.sample(deck, state.players[your_index].deckCount),
        your_prize=random.sample(deck, len(state.players[your_index].prize)),
        opponent_deck=opp_deck,
        opponent_prize=opp_prize,
        opponent_hand=opp_hand,
        opponent_active=[_UNKNOWN] if len(opp.active) > 0 and opp.active[0] is None else [],
    )
    root = create_node(None, search_state, your_index, deck, model, reward_fn)

    for _ in range(SEARCH_COUNT):
        current = root
        while True:
            value = -1e9
            c = UCB_EXPLORATION * math.sqrt(current.visit)
            next_child = None
            for child in current.children:
                visit = 0
                if child.node is None:
                    v = current.total / current.visit
                else:
                    v = child.node.total / child.node.visit
                    visit = child.node.visit
                if current.state.observation.current.yourIndex != your_index:
                    v = -v
                v += c * child.prob / (1 + visit)
                if value < v:
                    value = v
                    next_child = child

            if next_child is None:
                break
            if next_child.node is None:
                step_state = search_step(current.state.searchId, next_child.select)
                next_child.node = create_node(current, step_state, your_index, deck, model, reward_fn)
                break
            current = next_child.node
            if current.state.observation.current.result >= 0:
                current.backprop(current.value)
                break

    max_child = None
    max_visit = -1
    for child in root.children:
        if child.node is not None and max_visit < child.node.visit:
            max_child = child
            max_visit = child.node.visit

    search_end()
    if max_child is None:
        # no expanded child (e.g. forced/degenerate state) — fall back to a legal pick
        return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)
    return max_child.select


# --- bundled-asset loading (deck + weights live next to main.py) ---
def _asset_path(name: str) -> str:
    # __file__ is undefined when Kaggle exec()s the agent, so derive dir defensively.
    base = globals().get("__file__")
    if base:
        here = os.path.join(os.path.dirname(os.path.abspath(base)), name)
        if os.path.exists(here):
            return here
    kaggle = "/kaggle_simulations/agent/" + name
    if os.path.exists(kaggle):
        return kaggle
    return os.path.abspath(name)


def read_deck_csv() -> list[int]:
    path = _asset_path("deck.csv")
    with open(path) as f:
        text = f.read().replace(",", "\n")
    return [int(tok) for tok in text.split() if tok.strip()][:60]


def _build_model() -> MyModel:
    model = MyModel(D_MODEL, NUM_HEADS, D_FEEDFORWARD, NUM_LAYERS_ENCODER, NUM_LAYERS_DECODER)
    wpath = _asset_path("model.pth")
    if os.path.exists(wpath):
        model.load_state_dict(torch.load(wpath, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()


def _load_archetype_model():
    """Load the bundled opponent-deck predictor, or None if absent (-> filler)."""
    path = _asset_path("archetypes.json")
    if not os.path.exists(path):
        return None
    try:
        return deck_predict.load_model(path)
    except Exception:
        return None  # never let a bad asset break the agent


_DECK = read_deck_csv()
_MODEL = _build_model()
_ARCH = _load_archetype_model()
# Evee deck -> evee shape (per-turn baseline built via shape_factory in mcts_agent).
_REWARD = rewards.evee


# --- optional turn-path logging (artifact you can pull after a match) ---
# Set env var AGENT_LOG=/path/to/file to append a readable per-turn action path.
# Off by default, so real-arena runs (read-only FS, discarded stdout) are unaffected.
_LOG_FILE = os.environ.get("AGENT_LOG")
_CARD_NAMES = {c.cardId: c.name for c in all_card}
_TRACE = {"lines": [], "owner": None, "turn": 0}


def _log_name(cid) -> str:
    if cid is None:
        return "?"
    return _CARD_NAMES.get(cid, f"#{cid}")


def _format_log(log: dict):
    """One readable line for an interesting log event, or None to skip noise."""
    t = log.get("type")
    n = lambda key: _log_name(log.get(key))
    if t == LogType.DRAW:
        return f"draw {n('cardId')}"
    if t == LogType.PLAY:
        return f"play {n('cardId')}"
    if t == LogType.ATTACH:
        return f"attach {n('cardId')} -> {n('cardIdTarget')}"
    if t == LogType.EVOLVE:
        return f"evolve {n('cardIdTarget')} -> {n('cardId')}"
    if t == LogType.DEVOLVE:
        return f"devolve {n('cardIdTarget')} -> {n('cardId')}"
    if t == LogType.SWITCH:
        return f"switch active {n('cardIdActive')} <-> bench {n('cardIdBench')}"
    if t == LogType.CHANGE:
        return f"change {n('cardIdBefore')} -> {n('cardIdAfter')}"
    if t == LogType.ATTACK:
        return f"ATTACK with {n('cardId')} (attackId {log.get('attackId')})"
    if t == LogType.HP_CHANGE:
        v = log.get("value")
        if v:
            return f"hp {n('cardId')} {v:+d}"
    return None


def _write_turn() -> None:
    if _TRACE["owner"] is None:
        return
    try:
        with open(_LOG_FILE, "a") as f:
            f.write(f"=== P{_TRACE['owner']} turn {_TRACE['turn']} path ===\n")
            for ln in _TRACE["lines"]:
                f.write(f"    {ln}\n")
            if not _TRACE["lines"]:
                f.write("    (no actions)\n")
    except OSError:
        pass  # never let logging break the agent


def _trace_logs(obs_dict: dict) -> None:
    """Accumulate realized actions from obs.logs; flush a turn at each TURN_END.

    Each agent() call carries the logs since our last selection, so this sees
    our own turns AND the opponent's turns in between -> full match path.
    """
    if not _LOG_FILE:
        return
    turn = (obs_dict.get("current") or {}).get("turn", _TRACE["turn"])
    for log in obs_dict.get("logs") or []:
        t = log.get("type")
        if t == LogType.TURN_START:
            _TRACE["lines"] = []
            _TRACE["owner"] = log.get("playerIndex")
            _TRACE["turn"] = turn
        elif t == LogType.TURN_END:
            _write_turn()
            _TRACE["lines"] = []
            _TRACE["owner"] = None
        else:
            line = _format_log(log)
            if line is not None:
                _TRACE["lines"].append(line)


def agent(obs_dict: dict) -> list[int]:
    """Arena entry point. Returns option indices, or the deck on initial select."""
    _trace_logs(obs_dict)
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return _DECK
    with torch.inference_mode():
        return mcts_agent(obs_dict, _DECK, _MODEL, _REWARD, _ARCH)
