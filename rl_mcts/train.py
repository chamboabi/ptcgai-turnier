import os
import random
import sys

import torch
import torch.optim

from cg.api import to_observation_class
from cg.game import battle_finish, battle_select, battle_start

import config as cfg
from agent import Agent, MCTSConfig
from mcts import LearnSample, mcts_agent
from model import MyModel, SparseVector
from rewards import win_loss


class LearnInput:
    index: list[int]
    value: list[float]
    offset: list[int]

    def __init__(self):
        self.index = []
        self.value = []
        self.offset = []

    def add(self, sv: SparseVector):
        count = len(self.index)
        self.index.extend(sv.index)
        self.value.extend(sv.value)
        for o in sv.offset:
            self.offset.append(o + count)


def random_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)


def progress(count: int, text: str):
    current = 0
    while True:
        percent = 100 * current // count
        sys.stderr.write(f"\r{text} {percent}%   ")
        sys.stderr.flush()
        if current >= count:
            sys.stderr.write("\n")
            sys.stderr.flush()
            break
        yield current
        current += 1


sample_deck = [
    721, 721,
    722, 722, 722, 722,
    723, 723, 723, 723,
    1092,
    1121, 1121,
    1145, 1145,
    1163, 1163,
    1219, 1219, 1219, 1219,
    1227, 1227, 1227, 1227,
    1262, 1262,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3,
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel(
    cfg.model["d_model"],
    cfg.model["num_heads"],
    cfg.model["d_feedforward"],
    cfg.model["num_layers_encoder"],
    cfg.model["num_layers_decoder"],
)
model = model.to(device)

mcts_cfg = MCTSConfig(
    search_count=cfg.mcts["search_count"],
    max_action_combinations=cfg.mcts["max_action_combinations"],
    ucb_exploration=cfg.mcts["ucb_exploration"],
    policy_temperature=cfg.mcts["policy_temperature"],
    unvisited_penalty=cfg.mcts["unvisited_penalty"],
)

agent = Agent(deck=sample_deck, model=model, mcts_cfg=mcts_cfg, reward_fn=win_loss)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training["learning_rate"])
loss_fn_enc = torch.nn.HuberLoss(delta=cfg.training["huber_delta_encoder"])
loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=cfg.training["huber_delta_decoder"])
os.makedirs("out", exist_ok=True)

best_winrate = -1.0
best_model_path = "out/model_best.pth"
ROLLBACK_THRESHOLD = cfg.training.get("rollback_threshold", 0.1)

for counter in range(cfg.training["outer_iterations"]):
    torch.save(model.state_dict(), "out/model" + str(counter) + ".pth")
    sample_list: list[LearnSample] = []

    model.eval()
    with torch.inference_mode():
        results = [0, 0, 0]

        for i in progress(cfg.training["eval_games"], "Evaluating... "):
            obs, start_data = battle_start(agent.deck, agent.deck)
            if start_data.errorPlayer >= 0:
                error = "Deck error."
                if start_data.errorType == 1:
                    error = "The deck contains invalid card ID."
                elif start_data.errorType == 2:
                    error = (
                        "You can include up to four cards with the same name in the deck, excluding basic Energy cards."
                    )
                elif start_data.errorType == 3:
                    error = "There are no Basic Pokémon in the deck."
                elif start_data.errorType == 4:
                    error = "You can include only one Ace Spec card in the deck."
                raise ValueError(error)
            your_index = i % 2
            while True:
                if obs["current"]["result"] >= 0:
                    break
                if obs["current"]["yourIndex"] == your_index:
                    selected, _ = mcts_agent(obs, agent)
                else:
                    selected = random_agent(obs)
                obs = battle_select(selected)

            battle_finish()

            if obs["current"]["result"] == 2:
                results[2] += 1
            elif obs["current"]["result"] == your_index:
                results[0] += 1
            else:
                results[1] += 1

        total_decisive = results[0] + results[1]
        winrate = results[0] / max(1, total_decisive)
        print("Evaluation win rate " + str(100 * results[0] // max(1, total_decisive)) + "%", flush=True)

        if winrate > best_winrate:
            best_winrate = winrate
            torch.save(model.state_dict(), best_model_path)
            print(f"New best: {best_winrate:.1%} — checkpoint saved.", flush=True)
        elif winrate < best_winrate - ROLLBACK_THRESHOLD:
            print(
                f"Win rate dropped to {winrate:.1%} (best {best_winrate:.1%}), rolling back.",
                flush=True,
            )
            model.load_state_dict(torch.load(best_model_path))
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training["learning_rate"])

        for _ in progress(cfg.training["collect_games"], "Training Data Collecting... "):
            obs, _ = battle_start(agent.deck, agent.deck)
            samples: list[list[LearnSample]] = [[], []]
            while True:
                if obs["current"]["result"] >= 0:
                    break
                selected, sample = mcts_agent(obs, agent)
                samples[obs["current"]["yourIndex"]].append(sample)
                obs = battle_select(selected)

            battle_finish()

            final_obs = to_observation_class(obs)
            for i in range(2):
                LAMBDA = cfg.training["td_lambda"]
                value = agent.reward_fn.terminal(final_obs, i)

                for sample in reversed(samples[i]):
                    label = (value + sample.value) * 0.5
                    value = value * LAMBDA + sample.value * (1.0 - LAMBDA)
                    sample.value = label
                    sample_list.append(sample)

    print("Training Start.")
    model.train()
    random.shuffle(sample_list)
    BATCH_SIZE = cfg.training["batch_size"]
    batch_count = len(sample_list) // BATCH_SIZE
    for i in range(batch_count):
        input_enc = LearnInput()
        input_dec = LearnInput()
        mask = []
        label_enc = []
        label_dec = []
        start = BATCH_SIZE * i
        for j in range(start, start + BATCH_SIZE):
            sample = sample_list[j]
            input_enc.add(sample.sv_enc)
            input_dec.add(sample.sv_dec)
            label_enc.append(sample.value)
            label_dec.extend(sample.policy)
            for _ in range(len(sample.policy)):
                mask.append(1.0)
            for _ in range(agent.mcts_cfg.max_action_combinations - len(sample.policy)):
                mask.append(0.0)
                label_dec.append(0.0)
                input_dec.offset.append(len(input_dec.index))

        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)
        mask_tensor = mask_tensor.view(BATCH_SIZE, -1)
        label_tensor_enc = torch.tensor(label_enc, dtype=torch.float32, device=device)
        label_tensor_enc = label_tensor_enc.view(BATCH_SIZE, -1)
        label_tensor_dec = torch.tensor(label_dec, dtype=torch.float32, device=device)
        label_tensor_dec = label_tensor_dec.view(BATCH_SIZE, -1)

        optimizer.zero_grad()

        out_enc, out_dec = model(
            torch.tensor(input_enc.index, dtype=torch.int32, device=device),
            torch.tensor(input_enc.value, dtype=torch.float32, device=device),
            torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
            torch.tensor(input_dec.index, dtype=torch.int32, device=device),
            torch.tensor(input_dec.value, dtype=torch.float32, device=device),
            torch.tensor(input_dec.offset, dtype=torch.int32, device=device),
        )

        loss_enc = loss_fn_enc(out_enc, label_tensor_enc)
        loss_dec = loss_fn_dec(out_dec, label_tensor_dec)
        loss_dec = loss_dec * mask_tensor
        loss_dec = loss_dec.sum() / float(BATCH_SIZE)
        loss = loss_enc + loss_dec

        loss.backward()
        optimizer.step()
    print("Training Finish.")
