import json
import os

with open(os.path.join(os.path.dirname(__file__), "config.json")) as f:
    _cfg = json.load(f)

model = _cfg["model"]
mcts = _cfg["mcts"]
training = _cfg["training"]
