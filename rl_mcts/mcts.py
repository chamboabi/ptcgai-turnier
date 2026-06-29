import math
import random
from collections import Counter

from cg.api import PlayerState, SearchState, search_begin, search_end, search_step, to_observation_class

from agent import Agent
from model import SparseVector, get_decoder_input, get_encoder_input, eval_nn

_UNKNOWN = 1072


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


def _sample_opponent_deck(
    agent: Agent, opp: PlayerState
) -> tuple[list[int], list[int], list[int]]:
    """Return (deck, prize, hand) card-ID lists using per-card predictions.

    Each CardPrediction carries:
      probability     = P(card is in deck)
      expected_copies = E[copies] unconditional (= probability * E[copies | in deck])

    We Bernoulli-sample inclusion, then add E[copies | in deck] rounded copies.
    """
    deck_n = opp.deckCount
    prize_n = len(opp.prize)
    hand_n = opp.handCount
    total = deck_n + prize_n + hand_n

    if agent.archetype_model is None:
        return [_UNKNOWN] * deck_n, [1] * prize_n, [1] * hand_n

    known = _opponent_known_cards(opp)
    result = agent.archetype_model.predict(known)

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


class LearnSample:
    def __init__(self, value: float, policy: list[float], sv_enc: SparseVector, sv_dec: SparseVector):
        self.value = value
        self.policy = policy
        self.sv_enc = sv_enc
        self.sv_dec = sv_dec


class Child:
    node: "Node | None"
    select: list[int]
    prob: float

    def __init__(self, select: list[int], prob: float):
        self.node = None
        self.select = select
        self.prob = prob


class Node:
    value: float
    total: float
    visit: int
    parent: "Node | None"
    children: list[Child]
    state: SearchState

    def __init__(self, parent: "Node | None", state: SearchState):
        self.value = -2.0
        self.total = 0.0
        self.visit = 0
        self.parent = parent
        self.children = []
        self.state = state

    def backprop(self, value: float):
        self.total += value
        self.visit += 1
        if self.parent is not None:
            self.parent.backprop(value)


def create_node(
    parent: Node | None, search_state: SearchState, your_index: int, agent: Agent
) -> tuple[Node, LearnSample | None]:
    node = Node(parent, search_state)
    cfg = agent.mcts_cfg

    obs = search_state.observation
    state = obs.current
    if state.result >= 0:
        node.value = agent.reward_fn.terminal(obs, your_index)
        node.backprop(node.value)
        sample = None
    else:
        actions = []
        indices = list(range(obs.select.maxCount))
        for _ in range(cfg.max_action_combinations):
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

        sv_enc = get_encoder_input(obs, agent.deck)
        sv_dec = get_decoder_input(obs, actions)
        value, policy = eval_nn(sv_enc, sv_dec, agent.model)

        # shape affects tree search; raw value goes into LearnSample for learning
        shaped = agent.reward_fn.shape(obs, your_index, value)
        v = shaped if state.yourIndex == your_index else -shaped
        node.value = v
        node.backprop(v)

        prob_sum = 0.0
        for i in range(len(policy)):
            p = math.exp(policy[i] * cfg.policy_temperature)
            node.children.append(Child(actions[i], p))
            prob_sum += p
        for c in node.children:
            c.prob /= prob_sum
        sample = LearnSample(value, policy, sv_enc, sv_dec)

    return (node, sample)


def mcts_agent(obs_dict: dict, agent: Agent) -> tuple[list[int], LearnSample]:
    obs = to_observation_class(obs_dict)
    your_index = obs.current.yourIndex
    state = obs.current
    opp = state.players[1 - your_index]
    opp_deck, opp_prize, opp_hand = _sample_opponent_deck(agent, opp)
    search_state = search_begin(
        obs,
        your_deck=random.sample(agent.deck, state.players[your_index].deckCount),
        your_prize=random.sample(agent.deck, len(state.players[your_index].prize)),
        opponent_deck=opp_deck,
        opponent_prize=opp_prize,
        opponent_hand=opp_hand,
        opponent_active=[_UNKNOWN] if len(opp.active) > 0 and opp.active[0] is None else [],
    )
    root, sample = create_node(None, search_state, your_index, agent)
    cfg = agent.mcts_cfg

    for _ in range(cfg.search_count):
        current = root
        while True:
            value = -1e9
            c = cfg.ucb_exploration * math.sqrt(current.visit)
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

            if next_child.node is None:
                search_state = search_step(current.state.searchId, next_child.select)
                next_child.node, _ = create_node(current, search_state, your_index, agent)
                break
            else:
                current = next_child.node
                if current.state.observation.current.result >= 0:
                    current.backprop(current.value)
                    break

    max_child = None
    max_visit = -1
    min_value = 10.0
    for child in root.children:
        if child.node is not None:
            if max_visit < child.node.visit:
                max_child = child
                max_visit = child.node.visit
            v = child.node.total / child.node.visit
            if min_value > v:
                min_value = v

    sample.value = root.total / root.visit
    for i in range(len(root.children)):
        child = root.children[i]
        v = sample.value
        if child.node is None:
            v = min_value - v - cfg.unvisited_penalty
        else:
            v = child.node.total / child.node.visit - v
        sample.policy[i] = max(-1.0, min(1.0, v))

    search_end()
    return (max_child.select, sample)
